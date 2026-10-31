import Foundation
import Metal

/// Tokens per simdgroup in the batched (prefill) matmul kernels.
private let prefillTT = 8
private let rowsPerThreadgroup = 8 // QROWS * QNSG

/// A linear layer, either MLX-quantized or dense.
final class Linear {
    let inDim: Int
    let outDim: Int
    private let weight: Tensor
    private var scales: Tensor?
    private var biases: Tensor?
    private var bias: Tensor?
    private var bits = 0
    private var groupSize = 0
    private let pso1: MTLComputePipelineState
    private let psoN: MTLComputePipelineState
    private var biasPSO: MTLComputePipelineState?
    let description: String

    init(_ store: WeightStore, _ prefix: String, inDim: Int) throws {
        let gpu = GPU.shared
        weight = try store.require(prefix + ".weight")
        self.inDim = inDim
        outDim = weight.shape[0]
        if let s = store.get(prefix + ".scales") {
            guard let b = store.get(prefix + ".biases") else { throw EngineError.missingTensor(prefix + ".biases") }
            scales = s
            biases = b
            let packedBytes = weight.shape[1] * weight.dtype.size
            bits = packedBytes * 8 / inDim
            groupSize = inDim / s.shape[1]
            guard [2, 3, 4, 5, 6, 8].contains(bits), groupSize % 8 == 0, inDim % 8 == 0 else {
                throw EngineError.unsupported("\(prefix): \(bits)-bit / group \(groupSize) quantization")
            }
            guard let sfx = s.dtype.suffix else { throw EngineError.unsupported("\(prefix): scales dtype \(s.dtype)") }
            pso1 = try gpu.pipeline("qmm_\(bits)_\(sfx)_1")
            psoN = try gpu.pipeline("qmm_\(bits)_\(sfx)_\(prefillTT)")
            description = "q\(bits)g\(groupSize)"
        } else {
            guard let sfx = weight.dtype.suffix, weight.shape[1] == inDim else {
                throw EngineError.unsupported("\(prefix): weight dtype \(weight.dtype) shape \(weight.shape)")
            }
            guard inDim % 4 == 0 else { throw EngineError.unsupported("\(prefix): input dim \(inDim)") }
            pso1 = try gpu.pipeline("dmm_\(sfx)_1")
            psoN = try gpu.pipeline("dmm_\(sfx)_\(prefillTT)")
            description = weight.dtype.rawValue.lowercased()
        }
        if let b = store.get(prefix + ".bias"), let sfx = b.dtype.suffix {
            bias = b
            biasPSO = try gpu.pipeline("add_bias_\(sfx)")
        }
    }

    /// y[t, :outDim] (row stride yStride, starting at yOffset bytes) = x[t] · Wᵀ
    func encode(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, xOffset: Int = 0, xStride: Int,
                y: MTLBuffer, yOffset: Int = 0, yStride: Int, T: Int) {
        let tt = T == 1 ? 1 : prefillTT
        enc.setComputePipelineState(T == 1 ? pso1 : psoN)
        enc.set(weight, 0)
        if let s = scales, let b = biases {
            enc.set(s, 1)
            enc.set(b, 2)
        }
        enc.set(x, xOffset, 3)
        enc.set(y, yOffset, 4)
        enc.params(MMParams(K: Int32(inDim), O: Int32(outDim), T: Int32(T), gs: Int32(groupSize),
                            xStride: Int32(xStride), yStride: Int32(yStride)), 5)
        enc.dispatchThreadgroups(MTLSize(width: (outDim + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
                                         height: (T + tt - 1) / tt, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        if let bias, let biasPSO {
            enc.setComputePipelineState(biasPSO)
            enc.set(y, yOffset, 0)
            enc.set(bias, 1)
            enc.params(EWParams(n: 0, stride: Int32(yStride), cols: Int32(outDim)), 2)
            enc.dispatchThreads(MTLSize(width: outDim, height: T, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(256, outDim), height: 1, depth: 1))
        }
    }

    // Embedding-table access (row lookup) for quantized or dense weights.
    func encodeLookup(_ enc: MTLComputeCommandEncoder, ids: MTLBuffer, out: MTLBuffer, T: Int) throws {
        if let s = scales, let b = biases, let sfx = s.dtype.suffix {
            enc.setComputePipelineState(try GPU.shared.pipeline("qembed_\(bits)_\(sfx)"))
            enc.set(weight, 0); enc.set(s, 1); enc.set(b, 2)
            enc.set(ids, 0, 3); enc.set(out, 0, 4)
            enc.params(MMParams(K: Int32(inDim), O: Int32(outDim), T: Int32(T), gs: Int32(groupSize), xStride: 0, yStride: 0), 5)
            enc.dispatchThreads(MTLSize(width: inDim / 8, height: T, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(256, inDim / 8), height: 1, depth: 1))
        } else {
            enc.setComputePipelineState(try GPU.shared.pipeline("dembed_\(weight.dtype.suffix!)"))
            enc.set(weight, 0); enc.set(ids, 0, 3); enc.set(out, 0, 4)
            enc.params(MMParams(K: Int32(inDim), O: Int32(outDim), T: Int32(T), gs: 0, xStride: 0, yStride: 0), 5)
            enc.dispatchThreads(MTLSize(width: inDim, height: T, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(256, inDim), height: 1, depth: 1))
        }
    }
}

final class RMSNorm {
    let weight: Tensor
    let pso: MTLComputePipelineState
    let eps: Float

    init(_ store: WeightStore, _ name: String, eps: Float) throws {
        weight = try store.require(name)
        guard let sfx = weight.dtype.suffix else { throw EngineError.unsupported("\(name) dtype") }
        pso = try GPU.shared.pipeline("rms_norm_\(sfx)")
        self.eps = eps
    }

    /// Normalises `rows` vectors of `dim`; with headsPerRow > 1 each token row holds several heads.
    func encode(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, xOffset: Int = 0, y: MTLBuffer, yOffset: Int = 0,
                dim: Int, rows: Int, headsPerRow: Int = 1, rowStride: Int) {
        enc.setComputePipelineState(pso)
        enc.set(x, xOffset, 0)
        enc.set(y, yOffset, 1)
        enc.set(weight, 2)
        enc.params(NormParams(dim: Int32(dim), headsPerRow: Int32(headsPerRow), rowStride: Int32(rowStride), eps: eps), 3)
        let threads = dim >= 1024 ? 256 : (dim >= 256 ? 128 : 32)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }
}

final class TransformerLayer {
    let inputNorm: RMSNorm
    let postNorm: RMSNorm
    let qNorm: RMSNorm?
    let kNorm: RMSNorm?
    let qkv: [Linear]      // either [qkv_proj] or [q, k, v]
    let o: Linear
    let gateUp: [Linear]   // either [gate_up_proj] or [gate, up]
    let down: Linear
    let kCache: MTLBuffer
    let vCache: MTLBuffer

    init(_ s: WeightStore, _ i: Int, _ c: ModelConfig, maxSeq: Int) throws {
        let p = "model.layers.\(i)"
        inputNorm = try RMSNorm(s, "\(p).input_layernorm.weight", eps: c.rmsNormEps)
        postNorm = try RMSNorm(s, "\(p).post_attention_layernorm.weight", eps: c.rmsNormEps)
        qNorm = s.has("\(p).self_attn.q_norm.weight") ? try RMSNorm(s, "\(p).self_attn.q_norm.weight", eps: c.rmsNormEps) : nil
        kNorm = s.has("\(p).self_attn.k_norm.weight") ? try RMSNorm(s, "\(p).self_attn.k_norm.weight", eps: c.rmsNormEps) : nil
        if s.has("\(p).self_attn.qkv_proj.weight") {
            qkv = [try Linear(s, "\(p).self_attn.qkv_proj", inDim: c.hiddenSize)]
        } else {
            qkv = try ["q_proj", "k_proj", "v_proj"].map { try Linear(s, "\(p).self_attn.\($0)", inDim: c.hiddenSize) }
        }
        o = try Linear(s, "\(p).self_attn.o_proj", inDim: c.numHeads * c.headDim)
        if s.has("\(p).mlp.gate_up_proj.weight") {
            gateUp = [try Linear(s, "\(p).mlp.gate_up_proj", inDim: c.hiddenSize)]
        } else {
            gateUp = try ["gate_proj", "up_proj"].map { try Linear(s, "\(p).mlp.\($0)", inDim: c.hiddenSize) }
        }
        down = try Linear(s, "\(p).mlp.down_proj", inDim: c.intermediateSize)
        let cacheBytes = c.numKVHeads * maxSeq * c.headDim * 2
        kCache = GPU.shared.buffer(cacheBytes)
        vCache = GPU.shared.buffer(cacheBytes)
    }
}

/// A decoder-only transformer running entirely on Metal.
public final class Transformer {
    public let config: ModelConfig
    public let maxSeq: Int
    let store: WeightStore
    let embed: Linear
    let layers: [TransformerLayer]
    let norm: RMSNorm
    let lmHead: Linear
    let quantDescription: String

    // Scratch buffers sized for `maxBatch` tokens.
    let maxBatch = 512
    private let ids: MTLBuffer
    private let h: MTLBuffer
    private let xn: MTLBuffer
    private let qkvBuf: MTLBuffer
    private let attnBuf: MTLBuffer
    private let guBuf: MTLBuffer
    private let mlpBuf: MTLBuffer
    private let tmp: MTLBuffer
    let logits: MTLBuffer
    private let invFreq: MTLBuffer

    private let ropePSO: MTLComputePipelineState
    private let kvPSO: MTLComputePipelineState
    private let attnPSO: MTLComputePipelineState
    private let siluPSO: MTLComputePipelineState
    private let addPSO: MTLComputePipelineState

    private let qkvCols: Int
    public var vocabSize: Int { lmHead.outDim }
    public var weightBytes: Int { store.totalBytes }

    public init(directory: URL, contextLength: Int) throws {
        let gpu = GPU.shared
        let c = try ModelConfig(directory: directory)
        config = c
        let seq = max(256, min(contextLength, max(c.maxPositionEmbeddings, 256)))
        maxSeq = seq
        let store = try WeightStore(directory: directory, device: gpu.device)
        self.store = store
        guard c.headDim % 32 == 0, c.headDim / 32 <= 8, c.numHeads % c.numKVHeads == 0 else {
            throw EngineError.unsupported("head_dim \(c.headDim)")
        }
        embed = try Linear(store, "model.embed_tokens", inDim: c.hiddenSize)
        layers = try (0..<c.numLayers).map { try TransformerLayer(store, $0, c, maxSeq: seq) }
        norm = try RMSNorm(store, "model.norm.weight", eps: c.rmsNormEps)
        if store.has("lm_head.weight") {
            lmHead = try Linear(store, "lm_head", inDim: c.hiddenSize)
        } else {
            lmHead = embed
        }
        quantDescription = layers.first?.down.description ?? ""

        qkvCols = (c.numHeads + 2 * c.numKVHeads) * c.headDim
        let H = c.hiddenSize, I = c.intermediateSize, B = maxBatch
        ids = gpu.buffer(B * 4)
        h = gpu.buffer(B * H * 4)
        xn = gpu.buffer(B * H * 4)
        qkvBuf = gpu.buffer(B * qkvCols * 4)
        attnBuf = gpu.buffer(B * c.numHeads * c.headDim * 4)
        guBuf = gpu.buffer(B * 2 * I * 4)
        mlpBuf = gpu.buffer(B * I * 4)
        tmp = gpu.buffer(B * H * 4)
        logits = gpu.buffer(lmHead.outDim * 4)
        let f = c.inverseFrequencies()
        invFreq = gpu.device.makeBuffer(bytes: f, length: max(16, f.count * 4), options: .storageModeShared)!

        ropePSO = try gpu.pipeline("rope")
        kvPSO = try gpu.pipeline("kv_store")
        attnPSO = try gpu.pipeline("attention_\(c.headDim / 32)")
        siluPSO = try gpu.pipeline("silu_mul")
        addPSO = try gpu.pipeline("add_inplace")
    }

    /// Runs `tokens` at positions offset..<offset+count, updating the KV cache.
    /// Returns a pointer to the logits of the last token (valid until the next call).
    public func forward(_ tokens: [Int32], offset: Int) throws -> UnsafeMutablePointer<Float> {
        guard offset + tokens.count <= maxSeq else { throw EngineError.contextFull }
        var start = 0
        while start < tokens.count {
            let n = min(maxBatch, tokens.count - start)
            let last = start + n == tokens.count
            try step(Array(tokens[start..<start + n]), offset: offset + start, computeLogits: last)
            start += n
        }
        return logits.contents().bindMemory(to: Float.self, capacity: lmHead.outDim)
    }

    private func step(_ tokens: [Int32], offset: Int, computeLogits: Bool) throws {
        let c = config
        let T = tokens.count
        let H = c.hiddenSize, D = c.headDim, I = c.intermediateSize
        let nH = c.numHeads, nKV = c.numKVHeads
        tokens.withUnsafeBytes { raw in ids.contents().copyMemory(from: raw.baseAddress!, byteCount: raw.count) }

        guard let cb = GPU.shared.queue.makeCommandBufferWithUnretainedReferences(),
              let enc = cb.makeComputeCommandEncoder(dispatchType: .serial) else {
            throw EngineError.noMetalDevice
        }
        try embed.encodeLookup(enc, ids: ids, out: h, T: T)

        let kCol = nH * D, vCol = (nH + nKV) * D
        for layer in layers {
            layer.inputNorm.encode(enc, x: h, y: xn, dim: H, rows: T, rowStride: H)

            if layer.qkv.count == 1 {
                layer.qkv[0].encode(enc, x: xn, xStride: H, y: qkvBuf, yStride: qkvCols, T: T)
            } else {
                layer.qkv[0].encode(enc, x: xn, xStride: H, y: qkvBuf, yOffset: 0, yStride: qkvCols, T: T)
                layer.qkv[1].encode(enc, x: xn, xStride: H, y: qkvBuf, yOffset: kCol * 4, yStride: qkvCols, T: T)
                layer.qkv[2].encode(enc, x: xn, xStride: H, y: qkvBuf, yOffset: vCol * 4, yStride: qkvCols, T: T)
            }
            if let qn = layer.qNorm, let kn = layer.kNorm {
                qn.encode(enc, x: qkvBuf, y: qkvBuf, dim: D, rows: T * nH, headsPerRow: nH, rowStride: qkvCols)
                kn.encode(enc, x: qkvBuf, xOffset: kCol * 4, y: qkvBuf, yOffset: kCol * 4,
                          dim: D, rows: T * nKV, headsPerRow: nKV, rowStride: qkvCols)
            }

            enc.setComputePipelineState(ropePSO)
            enc.set(qkvBuf, 0, 0)
            enc.set(invFreq, 0, 1)
            enc.params(RopeParams(D: Int32(D), rowStride: Int32(qkvCols), offset: Int32(offset),
                                  traditional: c.ropeTraditional ? 1 : 0, rotDim: Int32(c.rotaryDim)), 2)
            enc.dispatchThreads(MTLSize(width: c.rotaryDim / 2, height: nH + nKV, depth: T),
                                threadsPerThreadgroup: MTLSize(width: min(64, c.rotaryDim / 2), height: 1, depth: 1))

            enc.setComputePipelineState(kvPSO)
            enc.set(qkvBuf, 0, 0)
            enc.set(layer.kCache, 0, 1)
            enc.set(layer.vCache, 0, 2)
            enc.params(KVParams(D: Int32(D), nKV: Int32(nKV), rowStride: Int32(qkvCols), kCol: Int32(kCol),
                                vCol: Int32(vCol), offset: Int32(offset), maxSeq: Int32(maxSeq)), 3)
            enc.dispatchThreads(MTLSize(width: D, height: nKV, depth: T),
                                threadsPerThreadgroup: MTLSize(width: min(128, D), height: 1, depth: 1))

            enc.setComputePipelineState(attnPSO)
            enc.set(qkvBuf, 0, 0)
            enc.set(layer.kCache, 0, 1)
            enc.set(layer.vCache, 0, 2)
            enc.set(attnBuf, 0, 3)
            enc.params(AttnParams(D: Int32(D), nH: Int32(nH), nKV: Int32(nKV), qStride: Int32(qkvCols),
                                  maxSeq: Int32(maxSeq), offset: Int32(offset), scale: 1 / Float(D).squareRoot(),
                                  outStride: Int32(nH * D)), 4)
            enc.dispatchThreadgroups(MTLSize(width: nH, height: T, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            layer.o.encode(enc, x: attnBuf, xStride: nH * D, y: tmp, yStride: H, T: T)
            add(enc, h, tmp, T * H)

            layer.postNorm.encode(enc, x: h, y: xn, dim: H, rows: T, rowStride: H)
            if layer.gateUp.count == 1 {
                layer.gateUp[0].encode(enc, x: xn, xStride: H, y: guBuf, yStride: 2 * I, T: T)
            } else {
                layer.gateUp[0].encode(enc, x: xn, xStride: H, y: guBuf, yOffset: 0, yStride: 2 * I, T: T)
                layer.gateUp[1].encode(enc, x: xn, xStride: H, y: guBuf, yOffset: I * 4, yStride: 2 * I, T: T)
            }
            enc.setComputePipelineState(siluPSO)
            enc.set(guBuf, 0, 0)
            enc.set(mlpBuf, 0, 1)
            enc.params(EWParams(n: 0, stride: Int32(2 * I), cols: Int32(I)), 2)
            enc.dispatchThreads(MTLSize(width: I, height: T, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            layer.down.encode(enc, x: mlpBuf, xStride: I, y: tmp, yStride: H, T: T)
            add(enc, h, tmp, T * H)
        }

        if computeLogits {
            norm.encode(enc, x: h, xOffset: (T - 1) * H * 4, y: xn, dim: H, rows: 1, rowStride: H)
            lmHead.encode(enc, x: xn, xStride: H, y: logits, yStride: lmHead.outDim, T: 1)
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { throw EngineError.kernelCompile("GPU error: \(err.localizedDescription)") }
    }

    private func add(_ enc: MTLComputeCommandEncoder, _ a: MTLBuffer, _ b: MTLBuffer, _ n: Int) {
        enc.setComputePipelineState(addPSO)
        enc.set(a, 0, 0)
        enc.set(b, 0, 1)
        enc.params(EWParams(n: Int32(n), stride: 0, cols: 0), 2)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}
