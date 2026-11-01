import Foundation
import Metal

/// Tokens per simdgroup in the small-batch matmul kernels.
private let prefillTT = 8
private let rowsPerThreadgroup = 8 // QROWS * QNSG
private let qmvSimdgroups = 2 // QVNSG
private let qmvRowsPerThreadgroup = 4 * qmvSimdgroups
private let qmvThreads = 32 * qmvSimdgroups
private let qmvbRowsPerSimdgroup = 4 // QVBR

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
    private var gemm8PSO: MTLComputePipelineState?     // 5...8 tokens: qgemm on an 8-token tile
    private var gemm16PSO: MTLComputePipelineState?    // 9...16 tokens: qgemm on a 16-token tile
    private(set) var qmvPSO: MTLComputePipelineState?  // fused decode kernel (bias folded in)
    private(set) var qmvGatePSO: MTLComputePipelineState?  // decode with y = act(y) · e
    private(set) var qmvbPSO: MTLComputePipelineState?  // 4-bit, 2...4 tokens (speculative verification)
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
                // 4-bit: a remainder of 128/256/512 after the 1024-wide passes gets its own kernel.
                var tail = ""
                if bits == 4, [128, 256, 512].contains(inDim % 1024) { tail = "t\(inDim % 1024 / 32)_" }
                qmvPSO = try gpu.pipeline("qmv_\(tail)\(bits)_\(sfx)")
                qmvGatePSO = try gpu.pipeline("qmv_gate_\(tail)\(bits)_\(sfx)")
                if bits == 4 { qmvbPSO = try gpu.pipeline("qmvb4_\(sfx)_4") }
            }
            if inDim % 32 == 0 { gemmPSO = try gpu.pipeline("qgemm_\(bits)_\(sfx)") }
            if inDim % 64 == 0 {  // the short-prompt tiles step K by 64
                gemm8PSO = try gpu.pipeline("qgemms_\(bits)_\(sfx)_8")
                gemm16PSO = try gpu.pipeline("qgemms_\(bits)_\(sfx)_16")
            }
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

    /// The fused matvec kernels handle T tokens (1, or up to 8 for 4-bit weights).
    static func canFuse(_ parts: [Linear], T: Int) -> Bool {
        T == 1 ? parts.allSatisfy { $0.qmvPSO != nil } : (T <= Transformer.maxVerifyTokens && parts.allSatisfy { $0.qmvbPSO != nil })
    }

    /// One dispatch computing [A | B | C] rows for T tokens (T == 1, or up to maxVerifyTokens with 4-bit weights).
    static func encodeFusedDecode(_ enc: MTLComputeCommandEncoder, _ parts: [Linear], x: MTLBuffer, xOffset: Int = 0,
                                  xStride: Int = 0, y: MTLBuffer, yOffset: Int = 0, yStride: Int = 0, T: Int = 1) {
        let f = parts[0]
        enc.setComputePipelineState(T == 1 ? f.qmvPSO! : f.qmvbPSO!)
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
        enc.set(x, xOffset, 15)
        let total = parts.reduce(0) { $0 + $1.outDim }
        let rowsPerTG = T == 1 ? qmvRowsPerThreadgroup : qmvbRowsPerSimdgroup * qmvSimdgroups
        enc.params(MMParams(K: Int32(f.inDim), O: Int32(total), T: Int32(T), gs: Int32(f.groupSize), xStride: Int32(xStride), yStride: Int32(yStride),
                            o1: Int32(parts[0].outDim), o2: Int32(parts.count > 1 ? parts[1].outDim : 0),
                            hasBias: f.bias != nil ? 1 : 0), 5)
        enc.dispatchThreadgroups(MTLSize(width: (total + rowsPerTG - 1) / rowsPerTG, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: qmvThreads, height: 1, depth: 1))
    }

    /// Decode-only gated output: y[:outDim] = act(x · Wᵀ + bias) · e[:outDim] in one dispatch.
    func encodeGated(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, xOffset: Int = 0, y: MTLBuffer, yOffset: Int = 0,
                     e: MTLBuffer, eOffset: Int, act: Int32) {
        enc.setComputePipelineState(qmvGatePSO!)
        for slot in [0, 6, 9] {
            enc.set(weight, slot); enc.set(scales!, slot + 1); enc.set(biases!, slot + 2)
            enc.set(bias ?? scales!, 12 + slot / 3)
        }
        enc.set(x, xOffset, 3)
        enc.set(y, yOffset, 4)
        enc.set(e, eOffset, 15)
        enc.params(MMParams(K: Int32(inDim), O: Int32(outDim), T: 1, gs: Int32(groupSize), xStride: 0, yStride: 0,
                            o1: Int32(outDim), o2: 0, hasBias: bias != nil ? 1 : 0, act: act), 5)
        enc.dispatchThreadgroups(MTLSize(width: (outDim + qmvRowsPerThreadgroup - 1) / qmvRowsPerThreadgroup, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: qmvThreads, height: 1, depth: 1))
    }

    /// y[t, :outDim] (row stride yStride, starting at yOffset bytes) = x[t] · Wᵀ (+ bias)
    func encode(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, xOffset: Int = 0, xStride: Int,
                y: MTLBuffer, yOffset: Int = 0, yStride: Int, T: Int) {
        if Linear.canFuse([self], T: T) {
            Linear.encodeFusedDecode(enc, [self], x: x, xOffset: xOffset, xStride: xStride, y: y, yOffset: yOffset, yStride: yStride, T: T)
            return
        }
        // 5+ tokens: tiled GEMM beats the small-batch kernel. A full tile is 32 tokens; short prompts
        // use an 8- or 16-token tile instead of paying for 32 (same arithmetic per output element).
        var gemm: (pso: MTLComputePipelineState, rows: Int)?
        if T > Transformer.maxVerifyTokens {
            if T <= 8, let p = gemm8PSO { gemm = (p, 8) }
            else if T <= 16, let p = gemm16PSO { gemm = (p, 16) }
            else if let p = gemmPSO { gemm = (p, 32) }
        }
        let tt = T == 1 ? 1 : prefillTT
        enc.setComputePipelineState(gemm?.pso ?? (T == 1 ? pso1 : psoN))
        enc.set(weight, 0)
        if let s = scales, let b = biases {
            enc.set(s, 1)
            enc.set(b, 2)
        }
        enc.set(x, xOffset, 3)
        enc.set(y, yOffset, 4)
        enc.params(MMParams(K: Int32(inDim), O: Int32(outDim), T: Int32(T), gs: Int32(groupSize),
                            xStride: Int32(xStride), yStride: Int32(yStride)), 5)
        if let gemm {
            enc.dispatchThreadgroups(MTLSize(width: (outDim + 31) / 32, height: (T + gemm.rows - 1) / gemm.rows, depth: 1),
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
        encodeLookup(enc, pso: try lookupPipeline(), ids: ids, out: out, T: T)
    }

    /// The row-lookup pipeline for this table; resolve it once rather than on every step.
    func lookupPipeline() throws -> MTLComputePipelineState {
        if scales != nil, biases != nil, let sfx = scales?.dtype.suffix { return try GPU.shared.pipeline("qembed_\(bits)_\(sfx)") }
        guard let sfx = weight.dtype.suffix else { throw EngineError.unsupported("embedding dtype \(weight.dtype)") }
        return try GPU.shared.pipeline("dembed_\(sfx)")
    }

    /// `pso` must come from `lookupPipeline()` of this layer.
    func encodeLookup(_ enc: MTLComputeCommandEncoder, pso: MTLComputePipelineState, ids: MTLBuffer, idsOffset: Int = 0, out: MTLBuffer, T: Int) {
        enc.setComputePipelineState(pso)
        if let s = scales, let b = biases, s.dtype.suffix != nil {
            enc.set(weight, 0); enc.set(s, 1); enc.set(b, 2)
            enc.set(ids, idsOffset, 3); enc.set(out, 0, 4)
            enc.params(MMParams(K: Int32(inDim), O: Int32(outDim), T: Int32(T), gs: Int32(groupSize), xStride: 0, yStride: 0), 5)
            enc.dispatchThreads(MTLSize(width: inDim / 8, height: T, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(256, inDim / 8), height: 1, depth: 1))
        } else {
            enc.set(weight, 0); enc.set(ids, idsOffset, 3); enc.set(out, 0, 4)
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
    /// `norm_add_rms_norm` (Gemma's normalized residual).
    let normAddPSO: MTLComputePipelineState
    let eps: Float

    /// Threads per row: one float4 each, up to 1024 (dim 4096) and the pipelines' maximum.
    let threads: Int

    /// `offset`: the stored weights are offsets from 1 (Gemma); 1 + w is computed once, in f32
    /// as transformers does, into a buffer of its own.
    init(_ store: WeightStore, _ name: String, eps: Float, dim: Int, offset: Bool = false) throws {
        var w = try store.require(name)
        guard w.shape == [dim] else { throw EngineError.invalidFile("\(name): expected shape [\(dim)], found \(w.shape)") }
        guard dim % 4 == 0 else { throw EngineError.unsupported("\(name): dimension \(dim) is not a multiple of 4") }
        if offset {
            let values = try w.floats().map { 1 + $0 }
            let buf = try GPU.shared.buffer(dim * 4)
            values.withUnsafeBytes { buf.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            w = Tensor(buffer: buf, offset: 0, dtype: .f32, shape: [dim])
        }
        weight = w
        guard let sfx = weight.dtype.suffix else { throw EngineError.unsupported("\(name) dtype") }
        pso = try GPU.shared.pipeline("rms_norm_\(sfx)")
        addPSO = try GPU.shared.pipeline("add_rms_norm_\(sfx)")
        normAddPSO = try GPU.shared.pipeline("norm_add_rms_norm_\(sfx)")
        let maxThreads = min(pso.maxTotalThreadsPerThreadgroup, addPSO.maxTotalThreadsPerThreadgroup,
                             normAddPSO.maxTotalThreadsPerThreadgroup)
        threads = max(32, min(1024, maxThreads, dim / 4) / 32 * 32)
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
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }
}

final class TransformerLayer {
    let spec: ModelConfig.LayerSpec
    let qkvFusable: Bool
    let gateUpFusable: Bool
    let qkvPostPSO: MTLComputePipelineState
    let inputNorm: RMSNorm
    /// The norm in front of the MLP (`post_attention_layernorm`, or Gemma 2/3/4's `pre_feedforward_layernorm`).
    let mlpNorm: RMSNorm
    /// Gemma 2/3/4: norms of the attention and MLP outputs, applied before their residual adds.
    let attnOutNorm: RMSNorm?
    let mlpOutNorm: RMSNorm?
    let qNorm: RMSNorm?
    let kNorm: RMSNorm?
    /// [qkv_proj], [q, k, v], [q, k] (K = V) or [q] (KV-shared layer).
    let qkv: [Linear]
    let o: Linear
    let gateUp: [Linear]   // either [gate_up_proj] or [gate, up]
    let down: Linear
    /// MLP width (Gemma 4 doubles it in the KV-shared layers).
    let intermediate: Int
    /// Row stride of the [q | k | v] projection buffer.
    let qkvCols: Int
    /// nil for a KV-shared layer (it attends over `spec.kvSource`'s cache).
    let kCache: MTLBuffer?
    let vCache: MTLBuffer?
    /// Rows in the K/V cache: the whole context, or for a sliding-window layer the window plus
    /// one batch (position p lives at row p mod cacheLen; a batch's T new rows must not land
    /// on rows its own queries still read).
    let cacheLen: Int
    /// Gemma 4 per-layer input: gate (H → P), projection (P → H) and its norm.
    let pleGate: Linear?
    let pleProj: Linear?
    let pleNorm: RMSNorm?
    /// Gemma 4: the layer output is multiplied by this.
    let layerScalar: Float

    init(_ s: WeightStore, _ i: Int, _ c: ModelConfig, maxSeq: Int) throws {
        let p = "model.layers.\(i)"
        spec = c.layers[i]
        let H = c.hiddenSize, D = spec.headDim, nKV = spec.numKVHeads
        let off = c.normWeightOffset
        let eps = c.rmsNormEps
        inputNorm = try RMSNorm(s, "\(p).input_layernorm.weight", eps: eps, dim: H, offset: off)
        if c.sandwichNorms {
            attnOutNorm = try RMSNorm(s, "\(p).post_attention_layernorm.weight", eps: eps, dim: H, offset: off)
            mlpNorm = try RMSNorm(s, "\(p).pre_feedforward_layernorm.weight", eps: eps, dim: H, offset: off)
            mlpOutNorm = try RMSNorm(s, "\(p).post_feedforward_layernorm.weight", eps: eps, dim: H, offset: off)
        } else {
            mlpNorm = try RMSNorm(s, "\(p).post_attention_layernorm.weight", eps: eps, dim: H, offset: off)
            attnOutNorm = nil
            mlpOutNorm = nil
        }
        qNorm = s.has("\(p).self_attn.q_norm.weight") ? try RMSNorm(s, "\(p).self_attn.q_norm.weight", eps: eps, dim: D, offset: off) : nil
        kNorm = spec.hasOwnKV && s.has("\(p).self_attn.k_norm.weight") ? try RMSNorm(s, "\(p).self_attn.k_norm.weight", eps: eps, dim: D, offset: off) : nil
        if spec.hasOwnKV, (qNorm == nil) != (kNorm == nil) { throw EngineError.invalidFile("\(p): q_norm and k_norm must come together") }
        qkvCols = (c.numHeads + 2 * nKV) * D
        if !spec.hasOwnKV {
            qkv = [try Linear(s, "\(p).self_attn.q_proj", inDim: H, outDim: c.numHeads * D)]
        } else if s.has("\(p).self_attn.qkv_proj.weight") {
            qkv = [try Linear(s, "\(p).self_attn.qkv_proj", inDim: H, outDim: qkvCols)]
        } else {
            let parts = spec.kEqV ? [("q_proj", c.numHeads), ("k_proj", nKV)] : [("q_proj", c.numHeads), ("k_proj", nKV), ("v_proj", nKV)]
            qkv = try parts.map { try Linear(s, "\(p).self_attn.\($0.0)", inDim: H, outDim: $0.1 * D) }
        }
        o = try Linear(s, "\(p).self_attn.o_proj", inDim: c.numHeads * D, outDim: H)
        // Gemma 4's MLP width varies per layer: take it from the weights.
        let I0: Int? = c.isGemma4 ? nil : c.intermediateSize
        if s.has("\(p).mlp.gate_up_proj.weight") {
            gateUp = [try Linear(s, "\(p).mlp.gate_up_proj", inDim: H, outDim: I0.map { 2 * $0 })]
            guard gateUp[0].outDim % 2 == 0 else { throw EngineError.invalidFile("\(p).mlp.gate_up_proj: odd output size") }
            intermediate = gateUp[0].outDim / 2
        } else {
            let gate = try Linear(s, "\(p).mlp.gate_proj", inDim: H, outDim: I0)
            gateUp = [gate, try Linear(s, "\(p).mlp.up_proj", inDim: H, outDim: gate.outDim)]
            intermediate = gate.outDim
        }
        guard (1...262_144).contains(intermediate) else { throw EngineError.invalidFile("\(p): MLP width \(intermediate) is out of range") }
        down = try Linear(s, "\(p).mlp.down_proj", inDim: intermediate, outDim: H)
        if c.perLayerInputSize > 0 {
            let P = c.perLayerInputSize
            pleGate = try Linear(s, "\(p).per_layer_input_gate", inDim: H, outDim: P)
            pleProj = try Linear(s, "\(p).per_layer_projection", inDim: P, outDim: H)
            pleNorm = try RMSNorm(s, "\(p).post_per_layer_input_norm.weight", eps: eps, dim: H)
        } else {
            pleGate = nil; pleProj = nil; pleNorm = nil
        }
        // Only Gemma 4 has (and can apply) a layer scalar: it rides on the sandwich-norm kernel.
        if c.isGemma4, let t = s.get("\(p).layer_scalar") {
            guard t.count == 1, let v = try t.floats().first, v.isFinite else { throw EngineError.invalidFile("\(p).layer_scalar") }
            layerScalar = v
        } else {
            layerScalar = 1
        }
        qkvFusable = qkv.count > 1 && Linear.fusable(qkv)
        gateUpFusable = gateUp.count > 1 && Linear.fusable(gateUp)
        qkvPostPSO = try GPU.shared.pipeline("qkv_post_\(qNorm?.weight.dtype.suffix ?? "f32")")
        cacheLen = c.cacheRows(spec, context: maxSeq)
        if spec.hasOwnKV {
            let cacheBytes = nKV * cacheLen * D * 2
            kCache = try GPU.shared.buffer(cacheBytes)
            vCache = try GPU.shared.buffer(cacheBytes)
        } else {
            kCache = nil; vCache = nil
        }
    }
}

extension EngineError {
    /// The engine error for a failed command buffer: an out-of-memory failure gets an
    /// actionable message, anything else is reported as a GPU error.
    static func gpu(_ err: Error) -> EngineError {
        let d = err.localizedDescription
        if d.contains("OutOfMemory") || d.lowercased().contains("insufficient memory") {
            return .outOfMemory("the GPU ran out of memory while running the model. Lower the context length (Settings) or use a smaller model.")
        }
        return .kernelCompile("GPU error: \(d)")
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
    static let maxBatch = 512
    let maxBatch = Transformer.maxBatch
    /// Token ids by position (maxSeq + 1 entries): the CPU writes prompt tokens, the GPU sampler
    /// appends generated ones, and every step's embedding lookup reads its tokens from here.
    private let hist: MTLBuffer
    /// xorshift64 state and pool-width hint of the GPU sampler; per-position status of its
    /// results (0 ok, 1 non-finite logits); pool size and attempts of the last run (profiling).
    private let rngState: MTLBuffer
    private let sampleStatus: MTLBuffer
    private let sampleDiag: MTLBuffer
    private let samplePSO: MTLComputePipelineState
    private let sampleThreads: Int
    private let h: MTLBuffer
    let xn: MTLBuffer
    let qkvBuf: MTLBuffer
    let attnBuf: MTLBuffer
    let guBuf: MTLBuffer
    let mlpBuf: MTLBuffer
    let tmp: MTLBuffer
    let logits: MTLBuffer
    /// Logits for every row of a small batch (speculative verification).
    let logitsAll: MTLBuffer
    public static let maxVerifyTokens = 4
    /// Inverse RoPE frequencies per layer (layers with the same RoPE share a buffer).
    private let ropeFreqs: [MTLBuffer]

    /// Attention pipelines for one head size.
    struct AttnKernels {
        var decode: (MTLComputePipelineState, Int)
        var prefill: (MTLComputePipelineState, Int)
        var prefillQB: Int
        /// Long-context decode attention: `attention_split` (KV head × 8 key slices, GQA heads
        /// share each K/V read) + `attention_combine`. Bit-identical to `decode`; nil when the
        /// shape has no instantiation.
        var split: (split: MTLComputePipelineState, combine: MTLComputePipelineState)?
        /// Query-head sub-groups per KV head of the split kernel (grid depth).
        var splitGroups = 1
    }
    /// By head size (Gemma 4 mixes 256 and 512).
    private let attnKernels: [Int: AttnKernels]
    private let attnPartials: MTLBuffer?
    static let attnSplitVirtualSG = 32, attnSplitSG = 4
    /// Decode uses the split attention once the context (offset + 1) reaches this length.
    /// Defaults to `MLXR_ATTN_SPLIT_MIN` (or 256); `Int.max` disables it.
    var attnSplitMinContext: Int = Transformer.defaultSplitMinContext
    static let defaultSplitMinContext: Int = {
        if let s = ProcessInfo.processInfo.environment["MLXR_ATTN_SPLIT_MIN"], let v = Int(s), v > 0 { return v }
        return 256
    }()

    /// Gemma 4 per-layer input embeddings.
    struct PerLayerInputs {
        let size: Int                 // P (hidden_size_per_layer_input)
        let embed: Linear             // embed_tokens_per_layer: [vocab][layers · P]
        let embedPSO: MTLComputePipelineState
        let modelProj: Linear         // per_layer_model_projection: H → layers · P
        let norm: RMSNorm             // per_layer_projection_norm (P)
        let raw: MTLBuffer            // [T][layers · P] looked-up embeddings
        let inputs: MTLBuffer         // [T][layers · P] per-layer inputs
        let gateBuf: MTLBuffer        // [T][P]
        let addPSO: MTLComputePipelineState
        let gatePSO: MTLComputePipelineState
    }
    private let ple: PerLayerInputs?
    /// The MLP's gated activation: SiLU, or GELU (tanh / exact) for Gemma.
    private let siluPSO: MTLComputePipelineState
    private let addPSO: MTLComputePipelineState
    /// Gemma: embedding scale and final-logit soft-capping (nil when unused).
    private let scalePSO: MTLComputePipelineState?
    private let softcapPSO: MTLComputePipelineState?
    private let embedPSO: MTLComputePipelineState
    /// The next decode step (T = 1, at `offset`), encoded while the previous one ran on the GPU.
    /// Encoding does not depend on the token id (read from `hist` at execution time), only on
    /// the position, so it is committed only for an exact (T = 1, offset) match.
    private var prepared: (cb: MTLCommandBuffer, offset: Int)?
    /// Developer profiling only (`profile`): kernel classes left out of `encode`.
    var profileSkip: Set<String> = []

    public private(set) var lastGPUTime: Double = 0
    public var vocabSize: Int { lmHead.outDim }
    public var weightBytes: Int { store.totalBytes }
    /// Every projection has the batched 4-bit kernel, so verifying a few tokens is cheap.
    public var supportsBatchedVerify: Bool {
        Linear.canFuse([lmHead], T: 2) && layers.allSatisfy { l in
            Linear.canFuse(l.qkv, T: 2) && Linear.canFuse([l.o], T: 2) && Linear.canFuse(l.gateUp, T: 2) && Linear.canFuse([l.down], T: 2)
        }
    }

    public init(directory: URL, contextLength: Int) throws {
        let gpu = GPU.shared
        let c = try ModelConfig(directory: directory)
        config = c
        let seq = max(256, min(contextLength, max(c.maxPositionEmbeddings, 256)))
        maxSeq = seq
        let store = try WeightStore(directory: directory, device: gpu.device)
        self.store = store
        let headDims = Set(c.layers.map(\.headDim))
        for d in headDims {
            guard d % 32 == 0, d / 32 <= 8 || d == 512 else { throw EngineError.unsupported("head_dim \(d)") }
        }
        guard c.layers.count == c.numLayers, c.layers.allSatisfy({ c.numHeads % $0.numKVHeads == 0 }) else {
            throw EngineError.unsupported("attention head layout")
        }
        embed = try Linear(store, "model.embed_tokens", inDim: c.hiddenSize)
        layers = try (0..<c.numLayers).map { try TransformerLayer(store, $0, c, maxSeq: seq) }
        norm = try RMSNorm(store, "model.norm.weight", eps: c.rmsNormEps, dim: c.hiddenSize, offset: c.normWeightOffset)
        if store.has("lm_head.weight") {
            lmHead = try Linear(store, "lm_head", inDim: c.hiddenSize)
        } else {
            lmHead = embed
        }
        quantDescription = layers.first?.down.description ?? ""
        // The fused residual-norm kernels read two norm weights with one dtype.
        let normType = norm.weight.dtype
        for l in layers {
            let norms = [l.inputNorm, l.mlpNorm, l.attnOutNorm, l.mlpOutNorm, l.pleNorm].compactMap { $0 }
            guard norms.allSatisfy({ $0.weight.dtype == normType }) else { throw EngineError.unsupported("mixed norm weight dtypes") }
        }

        // Scratch buffers fit the largest layer.
        let qkvMax = layers.map(\.qkvCols).max()!, attnMax = layers.map { c.numHeads * $0.spec.headDim }.max()!
        let I = layers.map(\.intermediate).max()!
        let H = c.hiddenSize, B = maxBatch
        hist = try gpu.buffer((seq + 1) * 4)
        rngState = try gpu.buffer(16)
        sampleStatus = try gpu.buffer((seq + 1) * 4)
        sampleDiag = try gpu.buffer(16)
        samplePSO = try gpu.pipeline("sample_token")
        sampleThreads = min(1024, samplePSO.maxTotalThreadsPerThreadgroup) / 32 * 32
        h = try gpu.buffer(B * H * 4)
        xn = try gpu.buffer(B * H * 4)
        qkvBuf = try gpu.buffer(B * qkvMax * 4)
        attnBuf = try gpu.buffer(B * attnMax * 4)
        guBuf = try gpu.buffer(B * 2 * I * 4)
        mlpBuf = try gpu.buffer(B * I * 4)
        tmp = try gpu.buffer(B * H * 4)
        logits = try gpu.buffer(lmHead.outDim * 4)
        logitsAll = try gpu.buffer(Self.maxVerifyTokens * lmHead.outDim * 4)
        var freqBuffers: [[Float]: MTLBuffer] = [:]
        ropeFreqs = c.layers.map { l in
            if let b = freqBuffers[l.invFreq] { return b }
            let b = gpu.device.makeBuffer(bytes: l.invFreq, length: max(16, l.invFreq.count * 4), options: .storageModeShared)!
            freqBuffers[l.invFreq] = b
            return b
        }

        var kernels: [Int: AttnKernels] = [:]
        var partialBytes = 0
        // Soft-capped attention (Gemma 2) has its own instantiations (head sizes 128 and 256).
        let cap = c.attnLogitSoftcap > 0 ? "_cap" : ""
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            do { return try gpu.pipeline(name) } catch {
                throw cap.isEmpty ? error : EngineError.unsupported("attention soft-capping with head_dim \(name)")
            }
        }
        for D in headDims {
            let e = D / 32
            let decodeSG = e >= 16 ? 8 : e >= 8 ? 16 : 32
            var k: AttnKernels
            if [32, 64, 96, 128, 256].contains(D) {
                k = AttnKernels(decode: (try pipeline("attention_\(e)_\(decodeSG)\(cap)"), decodeSG * 32),
                                prefill: (try pipeline("flash_prefill_\(D / 8)\(cap)"), 128), prefillQB: 32)
            } else {
                k = AttnKernels(decode: (try pipeline("attention_\(e)_\(decodeSG)\(cap)"), decodeSG * 32),
                                prefill: (try pipeline("attention_prefill_\(e)\(cap)"), 256), prefillQB: e >= 16 ? 2 : e > 4 ? 4 : 8)
            }
            // The split kernel reproduces attention_E_32 exactly, so for head sizes up to 128 it
            // needs the same virtual simdgroup count; G = query heads per KV head must have an
            // instantiation (every layer of this head size must have the same G). Head sizes
            // 256 / 512 take the query heads in sub-groups of up to 2 (grid depth; 4 heads of 256
            // per simdgroup measured 5% slower at 1K context on Gemma 3 1B); their decode kernel
            // has 16 / 8 simdgroups, so the split result is not bit-identical there.
            let Gs = Set(c.layers.filter { $0.headDim == D }.map { c.numHeads / $0.numKVHeads })
            let V = Self.attnSplitVirtualSG
            let maxG = e == 2 ? 8 : e <= 4 ? 4 : 2
            let gSub = Gs.first.map { e <= 4 ? $0 : min($0, maxG) } ?? 0
            if (decodeSG == V && (2...4).contains(e)) || e >= 8, Gs.count == 1, let G = Gs.first, (1...maxG).contains(gSub), G % gSub == 0,
               let split = try? gpu.pipeline("attention_split_\(e)_\(gSub)\(cap)"),
               let combine = try? gpu.pipeline("attention_combine_\(e)"),
               split.maxTotalThreadsPerThreadgroup >= Self.attnSplitSG * 32,
               // floats: m + l + acc[D] per (head, virtual simdgroup)
               case let (rows, o1) = c.numHeads.multipliedReportingOverflow(by: V), !o1,
               case let (perRow, o2) = D.addingReportingOverflow(2), !o2,
               case let (floats, o3) = rows.multipliedReportingOverflow(by: perRow), !o3,
               case let (bytes, o4) = floats.multipliedReportingOverflow(by: 4), !o4,
               rows <= Int(Int32.max) / max(1, perRow) {
                k.split = (split, combine)
                k.splitGroups = G / gSub
                partialBytes = max(partialBytes, bytes)
            }
            kernels[D] = k
        }
        attnPartials = partialBytes > 0 ? try gpu.buffer(partialBytes) : nil
        attnKernels = kernels

        if c.perLayerInputSize > 0 {
            let P = c.perLayerInputSize, LP = P * c.numLayers
            let table = try Linear(store, "model.embed_tokens_per_layer", inDim: LP)
            // The sampler can produce any lm_head row, and every id indexes this table too.
            guard table.outDim >= embed.outDim else {
                throw EngineError.invalidFile("embed_tokens_per_layer has \(table.outDim) rows for a \(embed.outDim)-token vocabulary")
            }
            ple = PerLayerInputs(size: P, embed: table, embedPSO: try table.lookupPipeline(),
                                 modelProj: try Linear(store, "model.per_layer_model_projection", inDim: H, outDim: LP),
                                 // The reference normalises proj(h) / sqrt(H): RMSNorm(x / sqrt(H), eps) == RMSNorm(x, eps · H).
                                 norm: try RMSNorm(store, "model.per_layer_projection_norm.weight", eps: c.rmsNormEps * Float(H), dim: P),
                                 raw: try gpu.buffer(B * LP * 4), inputs: try gpu.buffer(B * LP * 4), gateBuf: try gpu.buffer(B * P * 4),
                                 addPSO: try gpu.pipeline("add_scaled"), gatePSO: try gpu.pipeline("gelu_tanh_gate"))
        } else {
            ple = nil
        }
        switch c.activation {
        case .silu: siluPSO = try gpu.pipeline("silu_mul")
        case .geluTanh: siluPSO = try gpu.pipeline("gelu_tanh_mul")
        case .gelu: siluPSO = try gpu.pipeline("gelu_mul")
        }
        addPSO = try gpu.pipeline("add_inplace")
        scalePSO = c.embedScale != nil ? try gpu.pipeline("scale_inplace") : nil
        softcapPSO = c.finalLogitSoftcap > 0 ? try gpu.pipeline("softcap_inplace") : nil
        embedPSO = try embed.lookupPipeline()
    }

    /// Runs `tokens` at positions offset..<offset+count, updating the KV cache.
    /// Returns a pointer to the logits of the last token (valid until the next call).
    /// Runs the tokens at positions offset..<offset+count. Returns the last row's logits, or with
    /// `allLogits` (count <= maxVerifyTokens) a row-major [count][vocab] block.
    public func forward(_ tokens: [Int32], offset: Int, allLogits: Bool = false, isCancelled: (() -> Bool)? = nil) throws -> UnsafeMutablePointer<Float> {
        guard !allLogits || tokens.count <= Self.maxVerifyTokens else { throw EngineError.invalidOutput("verify batch too large") }
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
            try step(Array(tokens[start..<start + n]), offset: offset + start, computeLogits: last, allLogits: allLogits)
            start += n
        }
        if allLogits { return logitsAll.contents().bindMemory(to: Float.self, capacity: tokens.count * lmHead.outDim) }
        return logits.contents().bindMemory(to: Float.self, capacity: lmHead.outDim)
    }

    private func step(_ tokens: [Int32], offset: Int, computeLogits: Bool, allLogits: Bool = false) throws {
        let T = tokens.count
        // A prepared buffer is exactly what encode(T: 1, offset:, computeLogits: true, allLogits: false)
        // returns now, so using it changes nothing on the GPU. Any other call shape discards it.
        var ready: MTLCommandBuffer?
        if T == 1, computeLogits, !allLogits, let p = prepared, p.offset == offset { ready = p.cb }
        prepared = nil
        let cb = try ready ?? encode(T: T, offset: offset, computeLogits: computeLogits, allLogits: allLogits)
        // Nothing is in flight (the previous step was waited for), so `hist` is free to write.
        writeHistory(tokens, at: offset)
        cb.commit()
        // Overlap CPU encoding of the likely next decode step (the following position) with this
        // step's GPU work. It is only committed after this buffer completes, on an exact match.
        if computeLogits, offset + T < maxSeq, let next = try? encode(T: 1, offset: offset + T, computeLogits: true, allLogits: false) {
            prepared = (next, offset + T)
        }
        cb.waitUntilCompleted()
        lastGPUTime = cb.gpuEndTime - cb.gpuStartTime
        if let err = cb.error {
            prepared = nil
            throw EngineError.gpu(err)
        }
    }

    private func writeHistory(_ tokens: [Int32], at offset: Int) {
        tokens.withUnsafeBytes { raw in (hist.contents() + offset * 4).copyMemory(from: raw.baseAddress!, byteCount: raw.count) }
    }

    // MARK: Pipelined decoding with GPU token selection

    /// One decode step committed to the GPU; `finish` reads back its token.
    public struct DecodeStep {
        let cb: MTLCommandBuffer
        /// Position of the token this step processed; the sampled token lands at `offset + 1`.
        public let offset: Int
    }

    /// How much of a previous run's KV cache (positions 0..<previousLength) a new run sharing its
    /// first `common` tokens can keep. Sliding-window layers hold only the latest rows, so once
    /// the previous run went on for more than a batch past the divergence point, the rows the
    /// resumed run would read have been overwritten: the prompt is processed from the start.
    public func reusablePrefix(_ common: Int, previousLength: Int) -> Int {
        for l in layers where l.spec.hasOwnKV && l.cacheLen < maxSeq {
            if previousLength - common > l.cacheLen - config.slidingWindow + 1 { return 0 }
        }
        return common
    }

    /// Seeds the GPU sampler's xorshift64 generator. Call before `prefill` (nothing may be in flight).
    public func seedSampler(_ seed: UInt64) {
        rngState.contents().storeBytes(of: seed == 0 ? 0x9E3779B97F4A7C15 : seed, as: UInt64.self)
        rngState.contents().storeBytes(of: Float(8), toByteOffset: 8, as: Float.self)   // pool-width hint
    }

    /// Runs the prompt tokens at offset..<offset+count and selects the first token from the last row's
    /// logits on the GPU. Returns it (also stored at hist[offset + count], the position it will be
    /// processed at). Must not be called while a `DecodeStep` is in flight.
    public func prefill(_ tokens: [Int32], offset: Int, sample: SampleParams, isCancelled: (() -> Bool)? = nil) throws -> Int {
        guard !tokens.isEmpty, offset + tokens.count < maxSeq else { throw EngineError.contextFull }
        let rows = embed.outDim
        guard tokens.allSatisfy({ $0 >= 0 && Int($0) < rows }) else {
            throw EngineError.invalidFile("token id outside the embedding table (\(rows) rows)")
        }
        prepared = nil
        var start = 0
        while start < tokens.count {
            if start > 0, isCancelled?() == true { throw EngineError.cancelled }
            let n = min(maxBatch, tokens.count - start)
            let last = start + n == tokens.count
            let at = offset + start
            let cb = try encode(T: n, offset: at, computeLogits: last, allLogits: false, sample: last ? sample : nil)
            writeHistory(Array(tokens[start..<start + n]), at: at)
            cb.commit()
            cb.waitUntilCompleted()
            lastGPUTime = cb.gpuEndTime - cb.gpuStartTime
            if let err = cb.error { throw EngineError.gpu(err) }
            start += n
        }
        return try sampledToken(at: offset + tokens.count)
    }

    /// Commits the decode step for the token at hist[offset] (written by the CPU or by the previous
    /// step's sampler); its sampled token goes to hist[offset + 1]. Returns without waiting: commit the
    /// following step before calling `finish` and the GPU runs both back to back. Steps must be
    /// committed and finished in position order, at most two in flight.
    public func beginDecodeStep(offset: Int, sample: SampleParams) throws -> DecodeStep {
        guard offset >= 0, offset + 1 < maxSeq else { throw EngineError.contextFull }
        prepared = nil
        let cb = try encode(T: 1, offset: offset, computeLogits: true, allLogits: false, sample: sample)
        cb.commit()
        return DecodeStep(cb: cb, offset: offset)
    }

    /// Waits for the step and returns the token it sampled (hist[offset + 1]).
    public func finish(_ step: DecodeStep) throws -> Int {
        step.cb.waitUntilCompleted()
        lastGPUTime = step.cb.gpuEndTime - step.cb.gpuStartTime
        if let err = step.cb.error { throw EngineError.gpu(err) }
        return try sampledToken(at: step.offset + 1)
    }

    /// The last computed logits row (valid until the next step).
    public var logitsPointer: UnsafeMutablePointer<Float> { logits.contents().bindMemory(to: Float.self, capacity: lmHead.outDim) }

    /// The token the sampler stored at `pos` (and its status). Both are per position, so a step
    /// that has already been committed for the following position cannot overwrite them.
    private func sampledToken(at pos: Int) throws -> Int {
        guard sampleStatus.contents().load(fromByteOffset: pos * 4, as: Int32.self) == 0 else {
            throw EngineError.invalidOutput("non-finite logits")
        }
        let t = Int(hist.contents().load(fromByteOffset: pos * 4, as: Int32.self))
        guard t >= 0, t < embed.outDim else { throw EngineError.invalidOutput("sampled token \(t) outside the vocabulary") }
        return t
    }

    /// Encodes one forward pass over T tokens at positions offset..<offset+T. Token ids are read from
    /// `hist` when the buffer executes, so the encoding depends only on (T, offset, computeLogits, allLogits).
    /// With `sample`, the last row's logits are turned into a token on the GPU (see `sample_token`).
    private func encode(T: Int, offset: Int, computeLogits: Bool, allLogits: Bool, sample: SampleParams? = nil) throws -> MTLCommandBuffer {
        let c = config
        let H = c.hiddenSize, nH = c.numHeads
        guard T > 0, T <= maxBatch, offset >= 0, offset + T <= maxSeq else { throw EngineError.contextFull }
        if sample != nil { guard computeLogits, !allLogits, offset + T < maxSeq else { throw EngineError.contextFull } }

        guard let cb = GPU.shared.queue.makeCommandBufferWithUnretainedReferences(),
              let enc = cb.makeComputeCommandEncoder(dispatchType: .serial) else {
            throw EngineError.noMetalDevice
        }
        embed.encodeLookup(enc, pso: embedPSO, ids: hist, idsOffset: offset * 4, out: h, T: T)
        if let scalePSO, let s = c.embedScale { scale(enc, scalePSO, h, T * H, s) }
        if let ple { encodePerLayerInputs(enc, ple, T: T, offset: offset) }
        // Profiling only: every flag is false in normal use.
        let skip = profileSkip
        let skipNorm = skip.contains("norm"), skipPost = skip.contains("post"), skipAttn = skip.contains("attn")
        let skipSilu = skip.contains("silu"), skipQKV = skip.contains("qkv"), skipO = skip.contains("o")
        let skipGateUp = skip.contains("gateup"), skipDown = skip.contains("down"), skipLMHead = skip.contains("lmhead")
        if !skipNorm { layers[0].inputNorm.encode(enc, x: h, y: xn, dim: H, rows: T, rowStride: H) }

        let decode = T == 1
        for (li, layer) in layers.enumerated() {
            let spec = layer.spec
            let D = spec.headDim, nKV = spec.numKVHeads, qkvCols = layer.qkvCols, I = layer.intermediate
            let kv = spec.hasOwnKV ? layer : layers[spec.kvSource]
            let kernels = attnKernels[D]!
            // Attention block: [q|k|v] projection → norm/rope/cache → attention → o_proj
            if skipQKV {
            } else if layer.qkvFusable && Linear.canFuse(layer.qkv, T: T) {
                Linear.encodeFusedDecode(enc, layer.qkv, x: xn, xStride: H, y: qkvBuf, yStride: qkvCols, T: T)
            } else {
                var col = 0   // q, k, v sections (or one fused qkv_proj)
                for part in layer.qkv {
                    part.encode(enc, x: xn, xStride: H, y: qkvBuf, yOffset: col * 4, yStride: qkvCols, T: T)
                    col += part.outDim
                }
            }

            if !skipPost {
            enc.setComputePipelineState(layer.qkvPostPSO)
            enc.set(qkvBuf, 0, 0)
            enc.set(ropeFreqs[li], 0, 1)
            enc.set(kv.kCache!, 0, 2)
            enc.set(kv.vCache!, 0, 3)
            if let qn = layer.qNorm {
                enc.set(qn.weight, 4); enc.set((layer.kNorm ?? qn).weight, 5)
            } else {
                enc.set(ropeFreqs[li], 0, 4); enc.set(ropeFreqs[li], 0, 5)
            }
            enc.params(QKVPostParams(D: Int32(D), nH: Int32(nH), nKV: Int32(nKV), rowStride: Int32(qkvCols),
                                     offset: Int32(offset), maxSeq: Int32(kv.cacheLen), traditional: c.ropeTraditional ? 1 : 0,
                                     rotDim: Int32(spec.rotDim), eps: c.rmsNormEps, hasNorm: layer.qNorm != nil ? 1 : 0,
                                     vNorm: c.valueNorm ? 1 : 0, kEqV: spec.kEqV ? 1 : 0), 6)
            // A KV-shared layer only prepares its queries.
            enc.dispatchThreadgroups(MTLSize(width: spec.hasOwnKV ? nH + 2 * nKV : nH, height: T, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: D, height: 1, depth: 1))
            }

            let attnParams = AttnParams(D: Int32(D), nH: Int32(nH), nKV: Int32(nKV), qStride: Int32(qkvCols),
                                        maxSeq: Int32(kv.cacheLen), offset: Int32(offset), scale: c.ropeAttentionFactor * c.ropeAttentionFactor * c.queryScale,
                                        outStride: Int32(nH * D), T: Int32(T), window: Int32(spec.sliding ? c.slidingWindow : 0),
                                        softcap: c.attnLogitSoftcap)
            if skipAttn {
            } else if decode, offset + 1 >= attnSplitMinContext, let (split, combine) = kernels.split, let partials = attnPartials {
                // Partials are reused by every layer: the serial encoder orders the dispatches.
                enc.setComputePipelineState(split)
                enc.set(qkvBuf, 0, 0)
                enc.set(kv.kCache!, 0, 1)
                enc.set(kv.vCache!, 0, 2)
                enc.set(partials, 0, 3)
                enc.params(attnParams, 4)
                enc.dispatchThreadgroups(MTLSize(width: nKV, height: Self.attnSplitVirtualSG / Self.attnSplitSG, depth: kernels.splitGroups),
                                         threadsPerThreadgroup: MTLSize(width: Self.attnSplitSG * 32, height: 1, depth: 1))
                enc.setComputePipelineState(combine)
                enc.set(partials, 0, 0)
                enc.set(attnBuf, 0, 1)
                enc.params(attnParams, 2)
                enc.dispatchThreadgroups(MTLSize(width: nH, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            } else {
                let (attn, attnThreads) = decode ? kernels.decode : kernels.prefill
                enc.setComputePipelineState(attn)
                enc.set(qkvBuf, 0, 0)
                enc.set(kv.kCache!, 0, 1)
                enc.set(kv.vCache!, 0, 2)
                enc.set(attnBuf, 0, 3)
                enc.params(attnParams, 4)
                enc.dispatchThreadgroups(MTLSize(width: nH, height: decode ? 1 : (T + kernels.prefillQB - 1) / kernels.prefillQB, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: attnThreads, height: 1, depth: 1))
            }

            if !skipO { layer.o.encode(enc, x: attnBuf, xStride: nH * D, y: tmp, yStride: H, T: T) }
            if !skipNorm { layer.mlpNorm.encodeAdd(enc, h: h, residual: tmp, normedBy: layer.attnOutNorm, y: xn, dim: H, rows: T) }

            // MLP block: [gate|up] → act(gate)·up → down_proj
            if skipGateUp {
            } else if layer.gateUpFusable && Linear.canFuse(layer.gateUp, T: T) {
                Linear.encodeFusedDecode(enc, layer.gateUp, x: xn, xStride: H, y: guBuf, yStride: 2 * I, T: T)
            } else if layer.gateUp.count == 1 {
                layer.gateUp[0].encode(enc, x: xn, xStride: H, y: guBuf, yStride: 2 * I, T: T)
            } else {
                layer.gateUp[0].encode(enc, x: xn, xStride: H, y: guBuf, yOffset: 0, yStride: 2 * I, T: T)
                layer.gateUp[1].encode(enc, x: xn, xStride: H, y: guBuf, yOffset: I * 4, yStride: 2 * I, T: T)
            }
            if !skipSilu {
            enc.setComputePipelineState(siluPSO)
            enc.set(guBuf, 0, 0)
            enc.set(mlpBuf, 0, 1)
            enc.params(EWParams(n: 0, stride: Int32(2 * I), cols: Int32(I)), 2)
            enc.dispatchThreads(MTLSize(width: I, height: T, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            if !skipDown { layer.down.encode(enc, x: mlpBuf, xStride: I, y: tmp, yStride: H, T: T) }

            // Residual add fused with the next norm (next layer's input norm, or the final norm).
            // The last layer's output only matters for the logits.
            let isLast = li + 1 == layers.count
            if skipNorm || (isLast && !computeLogits) { continue }
            let next = isLast ? norm : layers[li + 1].inputNorm
            var residualNorm = layer.mlpOutNorm
            if let ple, let gate = layer.pleGate, let proj = layer.pleProj, let pn = layer.pleNorm, let mon = layer.mlpOutNorm {
                // Gemma 4 per-layer input: h += norm(mlp); h += norm(proj(gelu(gate(h)) · ple[layer]))
                mon.encodeAdd(enc, h: h, residual: tmp, normedBy: mon, y: xn, dim: H, rows: T)   // xn is scratch here
                if decode, gate.qmvGatePSO != nil {
                    gate.encodeGated(enc, x: h, y: ple.gateBuf, e: ple.inputs, eOffset: li * ple.size * 4, act: 1)
                } else {
                    gate.encode(enc, x: h, xStride: H, y: ple.gateBuf, yStride: ple.size, T: T)
                    enc.setComputePipelineState(ple.gatePSO)
                    enc.set(ple.inputs, li * ple.size * 4, 1)
                    enc.set(ple.gateBuf, 0, 0)
                    enc.params(EWParams(n: 0, stride: Int32(ple.size * layers.count), cols: Int32(ple.size)), 2)
                    enc.dispatchThreads(MTLSize(width: ple.size, height: T, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: min(256, ple.size), height: 1, depth: 1))
                }
                proj.encode(enc, x: ple.gateBuf, xStride: ple.size, y: tmp, yStride: H, T: T)
                residualNorm = pn
            }
            next.encodeAdd(enc, h: h, residual: tmp, normedBy: residualNorm, hScale: layer.layerScalar, y: xn, dim: H, rows: T)
        }

        if skipLMHead {
        } else if computeLogits && allLogits {
            lmHead.encode(enc, x: xn, xStride: H, y: logitsAll, yStride: lmHead.outDim, T: T)
            if let softcapPSO { scale(enc, softcapPSO, logitsAll, T * lmHead.outDim, c.finalLogitSoftcap) }
        } else if computeLogits {
            lmHead.encode(enc, x: xn, xOffset: (T - 1) * H * 4, xStride: H, y: logits, yStride: lmHead.outDim, T: 1)
            if let softcapPSO { scale(enc, softcapPSO, logits, lmHead.outDim, c.finalLogitSoftcap) }
        }
        if var sp = sample {
            sp.V = Int32(min(Int(sp.V), lmHead.outDim))
            sp.pos = Int32(offset + T - 1)
            enc.setComputePipelineState(samplePSO)
            enc.set(logits, 0, 0)
            enc.set(hist, 0, 1)
            enc.set(rngState, 0, 2)
            enc.set(sampleStatus, 0, 3)
            enc.params(sp, 4)
            enc.set(sampleDiag, 0, 5)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: sampleThreads, height: 1, depth: 1))
        }
        enc.endEncoding()
        return cb
    }

    /// Gemma 4: per-layer inputs for T tokens into `ple.inputs` ([T][layers][P]):
    /// (norm(per_layer_model_projection(h) / sqrt(H)) + embed_tokens_per_layer[id] · sqrt(P)) / sqrt(2).
    /// (The 1/sqrt(H) projection scale is folded into the norm's eps, see `PerLayerInputs.norm`.)
    private func encodePerLayerInputs(_ enc: MTLComputeCommandEncoder, _ ple: PerLayerInputs, T: Int, offset: Int) {
        let LP = ple.size * layers.count
        ple.embed.encodeLookup(enc, pso: ple.embedPSO, ids: hist, idsOffset: offset * 4, out: ple.raw, T: T)
        ple.modelProj.encode(enc, x: h, xStride: config.hiddenSize, y: ple.inputs, yStride: LP, T: T)
        ple.norm.encode(enc, x: ple.inputs, y: ple.inputs, dim: ple.size, rows: T * layers.count, headsPerRow: layers.count, rowStride: LP)
        enc.setComputePipelineState(ple.addPSO)
        enc.set(ple.inputs, 0, 0)
        enc.set(ple.raw, 0, 1)
        enc.params(EWParams(n: Int32(T * LP), stride: 0, cols: 0), 2)
        enc.params(SIMD2<Float>(Float(ple.size).squareRoot(), Float(0.5).squareRoot()), 3)
        enc.dispatchThreads(MTLSize(width: T * LP, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// `scale_inplace` / `softcap_inplace` over the first n floats of a.
    private func scale(_ enc: MTLComputeCommandEncoder, _ pso: MTLComputePipelineState, _ a: MTLBuffer, _ n: Int, _ v: Float) {
        enc.setComputePipelineState(pso)
        enc.set(a, 0, 0)
        enc.params(EWParams(n: Int32(n), stride: 0, cols: 0), 1)
        enc.params(v, 2)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
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
        func sweep(_ name: String, _ parts: (TransformerLayer) -> [Linear], x: MTLBuffer, T: Int = 1) {
            let total = layers.reduce(0.0) { acc, l in acc + parts(l).reduce(0.0) { $0 + bytes($1) } }
            let t = time(3) { enc in
                for l in self.layers {
                    let ps = parts(l)
                    if ps.count > 1, Linear.fusable(ps), Linear.canFuse(ps, T: T) { Linear.encodeFusedDecode(enc, ps, x: x, xStride: ps[0].inDim, y: self.guBuf, yStride: ps.reduce(0) { $0 + $1.outDim }, T: T) }
                    else { for p in ps { p.encode(enc, x: x, xStride: p.inDim, y: self.guBuf, yStride: p.outDim, T: T) } }
                }
            }
            out += String(format: "%-8@ matvec x%d: %6.2f ms  %6.1f GB/s\n", "\(name)@T\(T)" as NSString, layers.count, t * 1000, total / t / 1e9)
        }
        sweep("qkv", { $0.qkv }, x: xn)
        sweep("o_proj", { [$0.o] }, x: attnBuf)
        sweep("gate_up", { $0.gateUp }, x: xn)
        sweep("down", { [$0.down] }, x: mlpBuf)
        sweep("gate_up", { $0.gateUp }, x: xn, T: 4)
        sweep("down", { [$0.down] }, x: mlpBuf, T: 4)
        let tLM = time(20) { lmHead.encode($0, x: xn, xStride: H, y: logits, yStride: lmHead.outDim, T: 1) }
        out += String(format: "lm_head matvec: %.3f ms  (%.0f GB/s, cache-warm)\n", tLM * 1000, bytes(lmHead) / tLM / 1e9)
        // Decode step breakdown: the step without each kernel class (its cost includes dispatch gaps).
        func stepTime(_ skip: Set<String>, T: Int = 1) -> Double {
            profileSkip = skip
            defer { profileSkip = [] }
            var best = Double.infinity
            for _ in 0..<(T == 1 ? 5 : 3) {
                let cb = try! encode(T: T, offset: T == 1 ? 16 : 0, computeLogits: true, allLogits: false)
                cb.commit(); cb.waitUntilCompleted()
                best = min(best, cb.gpuEndTime - cb.gpuStartTime)
            }
            return best
        }
        let full = stepTime([])
        out += String(format: "decode step @16: %.3f ms; without", full * 1000)
        for cls in ["qkv", "o", "gateup", "down", "lmhead", "post", "attn", "norm", "silu"] {
            out += String(format: " %@ -%.3f", cls as NSString, (full - stepTime([cls])) * 1000)
        }
        out += String(format: "; matvecs only %.3f ms\n", stepTime(["post", "attn", "norm", "silu"]) * 1000)
        // Prefill step breakdown at 512 tokens (GEMMs, attention and the elementwise kernels).
        let fullP = stepTime([], T: 512)
        out += String(format: "prefill step @512: %.1f ms; without", fullP * 1000)
        for cls in ["qkv", "o", "gateup", "down", "lmhead", "post", "attn", "norm", "silu"] {
            out += String(format: " %@ -%.1f", cls as NSString, (fullP - stepTime([cls], T: 512)) * 1000)
        }
        out += "\n"
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

extension Transformer {
    /// Developer micro-benchmark of the GPU token selection on the current logits (run a forward
    /// pass first): GPU time per dispatch, pool size and attempts for the given parameters.
    public func profileSampler(_ params: SamplingParameters, repeats: Int = 50) -> String {
        var sp = SampleParams(params, vocab: lmHead.outDim)
        sp.pos = 0
        seedSampler(1)
        // The penalty edits logits in place, so each dispatch works on a fresh copy; the copies'
        // own cost is measured separately and subtracted.
        let scratch = try! GPU.shared.buffer(logits.length)
        func run(sample: Bool) -> Double {
            let cb = GPU.shared.queue.makeCommandBuffer()!
            for _ in 0..<repeats {
                let blit = cb.makeBlitCommandEncoder()!
                blit.copy(from: logits, sourceOffset: 0, to: scratch, destinationOffset: 0, size: lmHead.outDim * 4)
                blit.endEncoding()
                guard sample else { continue }
                let enc = cb.makeComputeCommandEncoder(dispatchType: .serial)!
                enc.setComputePipelineState(samplePSO)
                enc.set(scratch, 0, 0); enc.set(hist, 0, 1); enc.set(rngState, 0, 2); enc.set(sampleStatus, 0, 3)
                enc.params(sp, 4)
                enc.set(sampleDiag, 0, 5)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: sampleThreads, height: 1, depth: 1))
                enc.endEncoding()
            }
            cb.commit(); cb.waitUntilCompleted()
            return (cb.gpuEndTime - cb.gpuStartTime) / Double(repeats)
        }
        let copies = run(sample: false)
        let both = run(sample: true)
        let diag = sampleDiag.contents().bindMemory(to: Int32.self, capacity: 2)
        let l = UnsafeBufferPointer(start: logitsPointer, count: lmHead.outDim)
        let m = l.max() ?? 0
        let near = l.reduce(0) { $0 + ($1 >= m - 8 ? 1 : 0) }, tied = l.reduce(0) { $0 + ($1 == m ? 1 : 0) }
        return String(format: "sampler T=%.1f k=%d: %.1f us per token (pool %d, attempts %d, %d threads; max %.2f, %d within 8, %d tied)", params.temperature, params.topK,
                      max(0, both - copies) * 1e6, diag[0], diag[1], sampleThreads, m, near, tied)
    }
}

extension RMSNorm {
    /// h += residual; y = norm(h)  — one dispatch. With `normedBy` (Gemma 2/3):
    /// h += normedBy(residual); y = norm(h), also one dispatch.
    func encodeAdd(_ enc: MTLComputeCommandEncoder, h: MTLBuffer, residual: MTLBuffer, normedBy rn: RMSNorm? = nil,
                   hScale: Float = 1, y: MTLBuffer, dim: Int, rows: Int) {
        if let rn {
            // Same weight dtype for both norms (the kernel is instantiated per dtype).
            precondition(rn.weight.dtype == weight.dtype, "residual norm dtype")
            enc.setComputePipelineState(normAddPSO)
            enc.set(rn.weight, 5)
            enc.params(hScale, 6)
        } else {
            precondition(hScale == 1, "layer scale needs a residual norm")
            enc.setComputePipelineState(addPSO)
        }
        enc.set(h, 0, 0)
        enc.set(residual, 0, 1)
        enc.set(y, 0, 2)
        enc.set(weight, 3)
        enc.params(NormParams(dim: Int32(dim), headsPerRow: 1, rowStride: Int32(dim), eps: eps), 4)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }
}
