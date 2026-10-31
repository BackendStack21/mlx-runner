import Foundation
import Metal

/// Tokens per simdgroup in the small-batch matmul kernels.
private let prefillTT = 8
private let rowsPerThreadgroup = 8 // QROWS * QNSG
private let qmvSimdgroups = 2 // QVNSG
private let qmvRowsPerThreadgroup = 4 * qmvSimdgroups
private let qmvThreads = 32 * qmvSimdgroups

/// A linear layer, either MLX-quantized or dense.
final class Linear {
    let inDim: Int
    let outDim: Int
    let weight: Tensor
    private(set) var scales: Tensor?
    private(set) var biases: Tensor?
    private(set) var bias: Tensor?
    private(set) var bits = 0
    private(set) var groupSize = 0
    private let pso1: MTLComputePipelineState          // decode (T == 1)
    private let psoN: MTLComputePipelineState          // small batches (2..<16 tokens)
    private var gemmPSO: MTLComputePipelineState?      // prefill tiles (simdgroup matrices)
    private(set) var qmvPSO: MTLComputePipelineState?  // fused decode kernel (bias folded in)
    private var biasPSO: MTLComputePipelineState?
    let description: String

    convenience init(_ store: WeightStore, _ prefix: String, inDim: Int, outDim: Int? = nil) throws {
        let s = store.get(prefix + ".scales")
        let b = store.get(prefix + ".biases")
        if s != nil && b == nil { throw EngineError.missingTensor(prefix + ".biases") }
        try self.init(weight: try store.require(prefix + ".weight"), scales: s, biases: b,
                      bias: store.get(prefix + ".bias"), inDim: inDim, outDim: outDim, name: prefix)
    }

    init(weight: Tensor, scales: Tensor?, biases: Tensor?, bias: Tensor?, inDim: Int, outDim expectOut: Int? = nil, name prefix: String) throws {
        let gpu = GPU.shared
        // Shapes come from an untrusted file: validate them so no kernel can index out of bounds.
        func bad(_ what: String) -> EngineError { .invalidFile("\(prefix): \(what)") }
        guard weight.shape.count == 2, weight.shape[0] > 0, weight.shape[1] > 0, inDim > 0 else { throw bad("weight must be a 2-D tensor") }
        if let e = expectOut, weight.shape[0] != e { throw bad("expected \(e) output rows, found \(weight.shape[0])") }
        if let bias, bias.shape != [weight.shape[0]] { throw bad("bias shape \(bias.shape)") }
        if let s = scales, let b = biases {
            guard s.shape.count == 2, s.shape[0] == weight.shape[0], s.shape[1] > 0, inDim % s.shape[1] == 0, b.shape == s.shape,
                  (weight.shape[1] * weight.dtype.size * 8) % inDim == 0 else { throw bad("inconsistent quantization tensors") }
        } else if weight.shape[1] != inDim {
            throw bad("expected \(inDim) input columns, found \(weight.shape[1])")
        }
        self.weight = weight
        self.inDim = inDim
        outDim = weight.shape[0]
        self.bias = bias
        if let s = scales, let b = biases {
            self.scales = s
            self.biases = b
            let packedBytes = weight.shape[1] * weight.dtype.size
            bits = packedBytes * 8 / inDim
            groupSize = inDim / s.shape[1]
            guard [2, 3, 4, 5, 6, 8].contains(bits), groupSize % 8 == 0, inDim % 8 == 0 else {
                throw EngineError.unsupported("\(prefix): \(bits)-bit / group \(groupSize) quantization")
            }
            guard let sfx = s.dtype.suffix else { throw EngineError.unsupported("\(prefix): scales dtype \(s.dtype)") }
            pso1 = try gpu.pipeline("qmm_\(bits)_\(sfx)_1")
            let qmvAlign = bits == 4 ? 32 : 16  // the 4-bit fast path consumes 32 values per lane
            if inDim % qmvAlign == 0 && groupSize % qmvAlign == 0 && (bias == nil || bias?.dtype == s.dtype) {
                qmvPSO = try gpu.pipeline("qmv_\(bits)_\(sfx)")
            }
            if inDim % 32 == 0 { gemmPSO = try gpu.pipeline("qgemm_\(bits)_\(sfx)") }
            psoN = try gpu.pipeline("qmm_\(bits)_\(sfx)_\(prefillTT)")
            description = "q\(bits)g\(groupSize)"
        } else {
            guard let sfx = weight.dtype.suffix, weight.shape[1] == inDim else {
                throw EngineError.unsupported("\(prefix): weight dtype \(weight.dtype) shape \(weight.shape)")
            }
            guard inDim % 4 == 0 else { throw EngineError.unsupported("\(prefix): input dim \(inDim)") }
            pso1 = try gpu.pipeline("dmm_\(sfx)_1")
            if inDim % 32 == 0 { gemmPSO = try gpu.pipeline("dgemm_\(sfx)") }
            psoN = try gpu.pipeline("dmm_\(sfx)_\(prefillTT)")
            description = weight.dtype.rawValue.lowercased()
        }
        if let b = bias, let sfx = b.dtype.suffix {
            biasPSO = try gpu.pipeline("add_bias_\(sfx)")
        }
    }

    /// Can these layers be computed by one fused decode dispatch?
    static func fusable(_ parts: [Linear]) -> Bool {
        guard let f = parts.first, parts.count <= 3, let s0 = f.scales else { return false }
        let hasBias = f.bias != nil
        return parts.allSatisfy { p in
            p.qmvPSO != nil && p.bits == f.bits && p.groupSize == f.groupSize && p.inDim == f.inDim
                && p.scales?.dtype == s0.dtype && p.outDim % rowsPerThreadgroup == 0 && (p.bias != nil) == hasBias
        }
    }

    /// One dispatch computing [A | B | C] rows for a single token (T == 1).
    static func encodeFusedDecode(_ enc: MTLComputeCommandEncoder, _ parts: [Linear], x: MTLBuffer, xOffset: Int = 0,
                                  y: MTLBuffer, yOffset: Int = 0) {
        let f = parts[0]
        enc.setComputePipelineState(f.qmvPSO!)
        for slot in 0..<3 {
            let p = parts[min(slot, parts.count - 1)]
            let base = slot == 0 ? 0 : 3 + slot * 3   // 0, 6, 9
            enc.set(p.weight, base)
            enc.set(p.scales!, base + 1)
            enc.set(p.biases!, base + 2)
            enc.set(p.bias ?? p.scales!, 12 + slot)
        }
        enc.set(x, xOffset, 3)
        enc.set(y, yOffset, 4)
        let total = parts.reduce(0) { $0 + $1.outDim }
        enc.params(MMParams(K: Int32(f.inDim), O: Int32(total), T: 1, gs: Int32(f.groupSize), xStride: 0, yStride: 0,
                            o1: Int32(parts[0].outDim), o2: Int32(parts.count > 1 ? parts[1].outDim : 0),
                            hasBias: f.bias != nil ? 1 : 0), 5)
        enc.dispatchThreadgroups(MTLSize(width: (total + qmvRowsPerThreadgroup - 1) / qmvRowsPerThreadgroup, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: qmvThreads, height: 1, depth: 1))
    }

    /// y[t, :outDim] (row stride yStride, starting at yOffset bytes) = x[t] · Wᵀ (+ bias)
    func encode(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, xOffset: Int = 0, xStride: Int,
                y: MTLBuffer, yOffset: Int = 0, yStride: Int, T: Int) {
        if T == 1, qmvPSO != nil {
            Linear.encodeFusedDecode(enc, [self], x: x, xOffset: xOffset, y: y, yOffset: yOffset)
            return
        }
        let gemm = T >= 16 ? gemmPSO : nil
        let tt = T == 1 ? 1 : prefillTT
        enc.setComputePipelineState(gemm ?? (T == 1 ? pso1 : psoN))
        enc.set(weight, 0)
        if let s = scales, let b = biases {
            enc.set(s, 1)
            enc.set(b, 2)
        }
        enc.set(x, xOffset, 3)
        enc.set(y, yOffset, 4)
        enc.params(MMParams(K: Int32(inDim), O: Int32(outDim), T: Int32(T), gs: Int32(groupSize),
                            xStride: Int32(xStride), yStride: Int32(yStride)), 5)
        if gemm != nil {
            enc.dispatchThreadgroups(MTLSize(width: (outDim + 31) / 32, height: (T + 31) / 32, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else {
            enc.dispatchThreadgroups(MTLSize(width: (outDim + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
                                             height: (T + tt - 1) / tt, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }
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
    let addPSO: MTLComputePipelineState
    let eps: Float

    init(_ store: WeightStore, _ name: String, eps: Float, dim: Int) throws {
        weight = try store.require(name)
        guard weight.shape == [dim] else { throw EngineError.invalidFile("\(name): expected shape [\(dim)], found \(weight.shape)") }
        guard let sfx = weight.dtype.suffix else { throw EngineError.unsupported("\(name) dtype") }
        pso = try GPU.shared.pipeline("rms_norm_\(sfx)")
        addPSO = try GPU.shared.pipeline("add_rms_norm_\(sfx)")
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
    let qkvFusable: Bool
    let gateUpFusable: Bool
    let qkvPostPSO: MTLComputePipelineState
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
        let H = c.hiddenSize, D = c.headDim, I = c.intermediateSize
        inputNorm = try RMSNorm(s, "\(p).input_layernorm.weight", eps: c.rmsNormEps, dim: H)
        postNorm = try RMSNorm(s, "\(p).post_attention_layernorm.weight", eps: c.rmsNormEps, dim: H)
        qNorm = s.has("\(p).self_attn.q_norm.weight") ? try RMSNorm(s, "\(p).self_attn.q_norm.weight", eps: c.rmsNormEps, dim: D) : nil
        kNorm = s.has("\(p).self_attn.k_norm.weight") ? try RMSNorm(s, "\(p).self_attn.k_norm.weight", eps: c.rmsNormEps, dim: D) : nil
        if s.has("\(p).self_attn.qkv_proj.weight") {
            qkv = [try Linear(s, "\(p).self_attn.qkv_proj", inDim: H, outDim: (c.numHeads + 2 * c.numKVHeads) * D)]
        } else {
            qkv = try [("q_proj", c.numHeads), ("k_proj", c.numKVHeads), ("v_proj", c.numKVHeads)].map {
                try Linear(s, "\(p).self_attn.\($0.0)", inDim: H, outDim: $0.1 * D)
            }
        }
        o = try Linear(s, "\(p).self_attn.o_proj", inDim: c.numHeads * D, outDim: H)
        if s.has("\(p).mlp.gate_up_proj.weight") {
            gateUp = [try Linear(s, "\(p).mlp.gate_up_proj", inDim: H, outDim: 2 * I)]
        } else {
            gateUp = try ["gate_proj", "up_proj"].map { try Linear(s, "\(p).mlp.\($0)", inDim: H, outDim: I) }
        }
        down = try Linear(s, "\(p).mlp.down_proj", inDim: I, outDim: H)
        qkvFusable = qkv.count > 1 && Linear.fusable(qkv)
        gateUpFusable = gateUp.count > 1 && Linear.fusable(gateUp)
        qkvPostPSO = try GPU.shared.pipeline("qkv_post_\(qNorm?.weight.dtype.suffix ?? "f32")")
        let cacheBytes = c.numKVHeads * maxSeq * c.headDim * 2
        kCache = try GPU.shared.buffer(cacheBytes)
        vCache = try GPU.shared.buffer(cacheBytes)
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
    let xn: MTLBuffer
    let qkvBuf: MTLBuffer
    let attnBuf: MTLBuffer
    let guBuf: MTLBuffer
    let mlpBuf: MTLBuffer
    let tmp: MTLBuffer
    let logits: MTLBuffer
    private let invFreq: MTLBuffer

    private let attnDecodePSO: (MTLComputePipelineState, Int)
    private let attnPrefillPSO: (MTLComputePipelineState, Int)
    private let prefillQB: Int
    private let siluPSO: MTLComputePipelineState
    private let addPSO: MTLComputePipelineState

    private let qkvCols: Int
    public private(set) var lastGPUTime: Double = 0
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
        norm = try RMSNorm(store, "model.norm.weight", eps: c.rmsNormEps, dim: c.hiddenSize)
        if store.has("lm_head.weight") {
            lmHead = try Linear(store, "lm_head", inDim: c.hiddenSize)
        } else {
            lmHead = embed
        }
        quantDescription = layers.first?.down.description ?? ""

        qkvCols = (c.numHeads + 2 * c.numKVHeads) * c.headDim
        let H = c.hiddenSize, I = c.intermediateSize, B = maxBatch
        ids = try gpu.buffer(B * 4)
        h = try gpu.buffer(B * H * 4)
        xn = try gpu.buffer(B * H * 4)
        qkvBuf = try gpu.buffer(B * qkvCols * 4)
        attnBuf = try gpu.buffer(B * c.numHeads * c.headDim * 4)
        guBuf = try gpu.buffer(B * 2 * I * 4)
        mlpBuf = try gpu.buffer(B * I * 4)
        tmp = try gpu.buffer(B * H * 4)
        logits = try gpu.buffer(lmHead.outDim * 4)
        let f = c.inverseFrequencies()
        invFreq = gpu.device.makeBuffer(bytes: f, length: max(16, f.count * 4), options: .storageModeShared)!

        let e = c.headDim / 32
        let decodeSG = e >= 8 ? 16 : 32
        attnDecodePSO = (try gpu.pipeline("attention_\(e)_\(decodeSG)"), decodeSG * 32)
        if [32, 64, 96, 128].contains(c.headDim) {
            attnPrefillPSO = (try gpu.pipeline("flash_prefill_\(c.headDim / 8)"), 128)
            prefillQB = 32
        } else {
            attnPrefillPSO = (try gpu.pipeline("attention_prefill_\(e)"), 256)
            prefillQB = e > 4 ? 4 : 8
        }
        siluPSO = try gpu.pipeline("silu_mul")
        addPSO = try gpu.pipeline("add_inplace")
    }

    /// Runs `tokens` at positions offset..<offset+count, updating the KV cache.
    /// Returns a pointer to the logits of the last token (valid until the next call).
    public func forward(_ tokens: [Int32], offset: Int, isCancelled: (() -> Bool)? = nil) throws -> UnsafeMutablePointer<Float> {
        guard offset + tokens.count <= maxSeq else { throw EngineError.contextFull }
        let rows = embed.outDim
        guard tokens.allSatisfy({ $0 >= 0 && Int($0) < rows }) else {
            throw EngineError.invalidFile("token id outside the embedding table (\(rows) rows)")
        }
        var start = 0
        while start < tokens.count {
            if start > 0, isCancelled?() == true { throw EngineError.cancelled }
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
        layers[0].inputNorm.encode(enc, x: h, y: xn, dim: H, rows: T, rowStride: H)

        let kCol = nH * D, vCol = (nH + nKV) * D
        let decode = T == 1
        let (attn, attnThreads) = decode ? attnDecodePSO : attnPrefillPSO
        for (li, layer) in layers.enumerated() {
            // Attention block: [q|k|v] projection → norm/rope/cache → attention → o_proj
            if decode && layer.qkvFusable {
                Linear.encodeFusedDecode(enc, layer.qkv, x: xn, y: qkvBuf)
            } else if layer.qkv.count == 1 {
                layer.qkv[0].encode(enc, x: xn, xStride: H, y: qkvBuf, yStride: qkvCols, T: T)
            } else {
                layer.qkv[0].encode(enc, x: xn, xStride: H, y: qkvBuf, yOffset: 0, yStride: qkvCols, T: T)
                layer.qkv[1].encode(enc, x: xn, xStride: H, y: qkvBuf, yOffset: kCol * 4, yStride: qkvCols, T: T)
                layer.qkv[2].encode(enc, x: xn, xStride: H, y: qkvBuf, yOffset: vCol * 4, yStride: qkvCols, T: T)
            }

            enc.setComputePipelineState(layer.qkvPostPSO)
            enc.set(qkvBuf, 0, 0)
            enc.set(invFreq, 0, 1)
            enc.set(layer.kCache, 0, 2)
            enc.set(layer.vCache, 0, 3)
            if let qn = layer.qNorm, let kn = layer.kNorm {
                enc.set(qn.weight, 4); enc.set(kn.weight, 5)
            } else {
                enc.set(invFreq, 0, 4); enc.set(invFreq, 0, 5)
            }
            enc.params(QKVPostParams(D: Int32(D), nH: Int32(nH), nKV: Int32(nKV), rowStride: Int32(qkvCols),
                                     offset: Int32(offset), maxSeq: Int32(maxSeq), traditional: c.ropeTraditional ? 1 : 0,
                                     rotDim: Int32(c.rotaryDim), eps: c.rmsNormEps, hasNorm: layer.qNorm != nil ? 1 : 0), 6)
            enc.dispatchThreadgroups(MTLSize(width: nH + 2 * nKV, height: T, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: D, height: 1, depth: 1))

            enc.setComputePipelineState(attn)
            enc.set(qkvBuf, 0, 0)
            enc.set(layer.kCache, 0, 1)
            enc.set(layer.vCache, 0, 2)
            enc.set(attnBuf, 0, 3)
            enc.params(AttnParams(D: Int32(D), nH: Int32(nH), nKV: Int32(nKV), qStride: Int32(qkvCols),
                                  maxSeq: Int32(maxSeq), offset: Int32(offset), scale: c.ropeAttentionFactor * c.ropeAttentionFactor / Float(D).squareRoot(),
                                  outStride: Int32(nH * D), T: Int32(T)), 4)
            enc.dispatchThreadgroups(MTLSize(width: nH, height: decode ? 1 : (T + prefillQB - 1) / prefillQB, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: attnThreads, height: 1, depth: 1))

            layer.o.encode(enc, x: attnBuf, xStride: nH * D, y: tmp, yStride: H, T: T)
            layer.postNorm.encodeAdd(enc, h: h, residual: tmp, y: xn, dim: H, rows: T)

            // MLP block: [gate|up] → SiLU(gate)·up → down_proj
            if decode && layer.gateUpFusable {
                Linear.encodeFusedDecode(enc, layer.gateUp, x: xn, y: guBuf)
            } else if layer.gateUp.count == 1 {
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

            // Residual add fused with the next norm (next layer's input norm, or the final norm).
            if li + 1 < layers.count {
                layers[li + 1].inputNorm.encodeAdd(enc, h: h, residual: tmp, y: xn, dim: H, rows: T)
            } else if computeLogits {
                norm.encodeAdd(enc, h: h, residual: tmp, y: xn, dim: H, rows: T)
            }
        }

        if computeLogits {
            lmHead.encode(enc, x: xn, xOffset: (T - 1) * H * 4, xStride: H, y: logits, yStride: lmHead.outDim, T: 1)
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        lastGPUTime = cb.gpuEndTime - cb.gpuStartTime
        if let err = cb.error { throw EngineError.kernelCompile("GPU error: \(err.localizedDescription)") }
    }

    func add(_ enc: MTLComputeCommandEncoder, _ a: MTLBuffer, _ b: MTLBuffer, _ n: Int) {
        enc.setComputePipelineState(addPSO)
        enc.set(a, 0, 0)
        enc.set(b, 0, 1)
        enc.params(EWParams(n: Int32(n), stride: 0, cols: 0), 2)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}

extension Transformer {
    /// Developer micro-benchmarks: matvec bandwidth and per-dispatch overhead.
    public func profile() throws -> String {
        let q = GPU.shared.queue
        func time(_ n: Int, _ body: (MTLComputeCommandEncoder) -> Void) -> Double {
            let cb = q.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder(dispatchType: .serial)!
            for _ in 0..<n { body(enc) }
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            return (cb.gpuEndTime - cb.gpuStartTime) / Double(n)
        }
        var out = ""
        let H = config.hiddenSize
        func bytes(_ l: Linear) -> Double { Double(l.weight.nbytes + (l.scales?.nbytes ?? 0) + (l.biases?.nbytes ?? 0)) }
        // Matvecs over every layer's distinct weights, so nothing is served from cache.
        func sweep(_ name: String, _ parts: (TransformerLayer) -> [Linear], x: MTLBuffer) {
            let total = layers.reduce(0.0) { acc, l in acc + parts(l).reduce(0.0) { $0 + bytes($1) } }
            let t = time(3) { enc in
                for l in self.layers {
                    let ps = parts(l)
                    if ps.count > 1, Linear.fusable(ps) { Linear.encodeFusedDecode(enc, ps, x: x, y: self.guBuf) }
                    else { for p in ps { p.encode(enc, x: x, xStride: p.inDim, y: self.guBuf, yStride: p.outDim, T: 1) } }
                }
            }
            out += String(format: "%-8@ matvec x%d: %6.2f ms  %6.1f GB/s\n", name as NSString, layers.count, t * 1000, total / t / 1e9)
        }
        sweep("qkv", { $0.qkv }, x: xn)
        sweep("o_proj", { [$0.o] }, x: attnBuf)
        sweep("gate_up", { $0.gateUp }, x: xn)
        sweep("down", { [$0.down] }, x: mlpBuf)
        let tLM = time(20) { lmHead.encode($0, x: xn, xStride: H, y: logits, yStride: lmHead.outDim, T: 1) }
        out += String(format: "lm_head matvec: %.3f ms  (%.0f GB/s, cache-warm)\n", tLM * 1000, bytes(lmHead) / tLM / 1e9)
        let l0 = layers[0]
        let tAdd = time(500) { enc in self.add(enc, tmp, xn, H) }
        out += String(format: "tiny dependent dispatch: %.1f us\n", tAdd * 1e6)
        // Prefill composition at T = 512: all projection GEMMs of one layer.
        let T = 512, I = config.intermediateSize
        let tGemm = time(10) { enc in
            for q in l0.qkv { q.encode(enc, x: self.xn, xStride: H, y: self.qkvBuf, yStride: q.outDim, T: T) }
            l0.o.encode(enc, x: self.attnBuf, xStride: l0.o.inDim, y: self.tmp, yStride: H, T: T)
            for g in l0.gateUp { g.encode(enc, x: self.xn, xStride: H, y: self.guBuf, yStride: g.outDim, T: T) }
            l0.down.encode(enc, x: self.mlpBuf, xStride: I, y: self.tmp, yStride: H, T: T)
        }
        let flops = 2.0 * Double(T) * Double(l0.qkv.reduce(0) { $0 + $1.outDim * H } + l0.o.outDim * l0.o.inDim + l0.gateUp.reduce(0) { $0 + $1.outDim * H } + H * I)
        out += String(format: "layer GEMMs at T=512: %.2f ms (%.2f TFLOPS) x %d layers = %.0f ms\n", tGemm * 1000, flops / tGemm / 1e12, layers.count, tGemm * 1000 * Double(layers.count))
        return out
    }
}

extension RMSNorm {
    /// h += residual; y = norm(h)  — one dispatch.
    func encodeAdd(_ enc: MTLComputeCommandEncoder, h: MTLBuffer, residual: MTLBuffer, y: MTLBuffer, dim: Int, rows: Int) {
        enc.setComputePipelineState(addPSO)
        enc.set(h, 0, 0)
        enc.set(residual, 0, 1)
        enc.set(y, 0, 2)
        enc.set(weight, 3)
        enc.params(NormParams(dim: Int32(dim), headsPerRow: 1, rowStride: Int32(dim), eps: eps), 4)
        let threads = dim >= 1024 ? 256 : (dim >= 256 ? 128 : 32)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }
}
