import Foundation
import Metal

/// Tokens per simdgroup in the small-batch matmul kernels.
private let prefillTT = 8
private let rowsPerThreadgroup = 8 // QROWS * QNSG
private let qmvSimdgroups = 2 // QVNSG
private let qmvRowsPerThreadgroup = 4 * qmvSimdgroups
private let qmvThreads = 32 * qmvSimdgroups
private let qmvxSimdgroups = 4 // QXNSG
private let qmvxRowsPerThreadgroup = 8 * 2 * qmvxSimdgroups // 8 * QXRF * QXNSG
private let qmvtRowsPerThreadgroup = 4 * 2 // R * QTNSG (64 threads)

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
    private var gemm64PSO: MTLComputePipelineState?    // long prompts: 64 × 64 tiles
    private(set) var qmvPSO: MTLComputePipelineState?  // fused decode kernel (bias folded in)
    private(set) var qmvGatePSO: MTLComputePipelineState?  // decode with y = act(y) · e
    private(set) var qmvtPSO: MTLComputePipelineState?  // 4-bit, 2 tokens (speculative verification)
    private(set) var qmvxPSO: MTLComputePipelineState?  // 4-bit, 3...8 tokens (speculative verification, short prompts)
    /// Decode over interleaved (scale, bias) pairs (`qmvp`; see `makePairs`), nil until made.
    private(set) var pairs: MTLBuffer?
    private(set) var qmvpPSO: MTLComputePipelineState?
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
                if bits == 4 {
                    qmvxPSO = try gpu.pipeline("qmvx4_\(sfx)")
                    qmvtPSO = try gpu.pipeline("qmvt4_\(sfx)")
                }
            }
            if inDim % 32 == 0 {
                gemmPSO = try gpu.pipeline("qgemm_\(bits)_\(sfx)")
                gemm64PSO = try gpu.pipeline("qgemm64_\(bits)_\(sfx)")
            }
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

    /// Reorders the output rows in place: row r becomes the old row perm[r] (the weight, its
    /// quantization scales and biases, and a bias). Rows are independent, so the layer computes the
    /// same outputs in the new order. `perm` must be a permutation of 0..<outDim.
    func permuteRows(_ perm: [Int]) throws {
        guard perm.count == outDim, Set(perm).count == outDim, perm.allSatisfy({ (0..<outDim).contains($0) }) else {
            throw EngineError.invalidOutput("bad row permutation")
        }
        for t in [weight, scales, biases, bias].compactMap({ $0 }) {
            guard t.shape.first == outDim, t.offset >= 0, t.nbytes <= t.buffer.length - t.offset else { throw EngineError.invalidFile("\(description): tensor outside its file") }
            let rowBytes = t.nbytes / outDim
            let p = t.buffer.contents() + t.offset
            let old = Data(bytes: p, count: t.nbytes)
            old.withUnsafeBytes { src in
                for (r, from) in perm.enumerated() { (p + r * rowBytes).copyMemory(from: src.baseAddress! + from * rowBytes, byteCount: rowBytes) }
            }
        }
    }

    /// One layer computing [a | b] (b's rows after a's), when both are stored alike; nil otherwise.
    /// The tensors are copied into new buffers (meant for small projections).
    static func concatRows(_ a: Linear, _ b: Linear, name: String) throws -> Linear? {
        guard a.inDim == b.inDim, a.bias == nil, b.bias == nil, a.weight.dtype == b.weight.dtype,
              a.weight.shape.count == 2, a.weight.shape[1] == b.weight.shape[1],
              (a.scales == nil) == (b.scales == nil), a.scales?.dtype == b.scales?.dtype,
              a.scales?.shape.last == b.scales?.shape.last, a.biases?.dtype == b.biases?.dtype else { return nil }
        func cat(_ x: Tensor, _ y: Tensor) throws -> Tensor {
            guard x.offset >= 0, y.offset >= 0, x.nbytes <= x.buffer.length - x.offset, y.nbytes <= y.buffer.length - y.offset else {
                throw EngineError.invalidFile("\(name): tensor outside its file")
            }
            let buf = try GPU.shared.buffer(x.nbytes + y.nbytes)
            buf.contents().copyMemory(from: x.buffer.contents() + x.offset, byteCount: x.nbytes)
            (buf.contents() + x.nbytes).copyMemory(from: y.buffer.contents() + y.offset, byteCount: y.nbytes)
            return Tensor(buffer: buf, offset: 0, dtype: x.dtype, shape: [x.shape[0] + y.shape[0]] + Array(x.shape.dropFirst()))
        }
        var s: Tensor?, q: Tensor?
        if let sa = a.scales, let sb = b.scales, let ba = a.biases, let bb = b.biases { s = try cat(sa, sb); q = try cat(ba, bb) }
        return try Linear(weight: try cat(a.weight, b.weight), scales: s, biases: q, bias: nil, inDim: a.inDim,
                          outDim: a.outDim + b.outDim, name: name)
    }

    /// Developer override of the `qmvp` grid for K = 1024 (MLXR_QMVP_GRID).
    static let pairsGrid: Int? = Int(ProcessInfo.processInfo.environment["MLXR_QMVP_GRID"] ?? "")

    /// Interleaves the quantization scales and biases into (scale, bias) pairs per group, so decode reads
    /// them as one stream (`qmvp`, with outputs identical to `qmv`). The original arrays stay for the
    /// other kernels: costs as much memory again as the scales and biases (1/9 of 4-bit g64 weights).
    /// Only 4-bit layers without bias whose input size is a multiple of 1024; others are left as they are.
    func makePairs() throws {
        guard pairs == nil, bits == 4, bias == nil, inDim % 1024 == 0, groupSize % 32 == 0, qmvPSO != nil, outDim % 8 == 0,
              let s = scales, let b = biases, let sfx = s.dtype.suffix, s.dtype.size == b.dtype.size else { return }
        // K = 1024: qmvq1 (x in registers, the next block's loads issued before the current block's
        // arithmetic); MLXR_QMVQ=0 keeps qmvp1k. Other multiples of 1024: qmvp.
        let kernel = inDim != 1024 ? "qmvp_\(sfx)" : ProcessInfo.processInfo.environment["MLXR_QMVQ"] == "0" ? "qmvp1k_\(sfx)" : "qmvq1_\(sfx)"
        let size = s.dtype.size, n = s.count
        guard s.offset >= 0, b.offset >= 0, s.nbytes <= s.buffer.length - s.offset, b.nbytes <= b.buffer.length - b.offset else {
            throw EngineError.invalidFile("\(description): tensor outside its file")
        }
        let buf = try GPU.shared.buffer(2 * n * size)
        let sp = s.buffer.contents() + s.offset, bp = b.buffer.contents() + b.offset, out = buf.contents()
        for i in 0..<n {
            (out + 2 * i * size).copyMemory(from: sp + i * size, byteCount: size)
            (out + (2 * i + 1) * size).copyMemory(from: bp + i * size, byteCount: size)
        }
        pairs = buf
        qmvpPSO = try GPU.shared.pipeline(kernel)
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
        T == 1 ? parts.allSatisfy { $0.qmvPSO != nil } : (T <= Transformer.maxVerifyTokens && parts.allSatisfy { $0.qmvxPSO != nil })
    }

    /// One dispatch computing [A | B | C] rows for T tokens (T == 1, or up to maxVerifyTokens with 4-bit weights).
    static func encodeFusedDecode(_ enc: MTLComputeCommandEncoder, _ parts: [Linear], x: MTLBuffer, xOffset: Int = 0,
                                  xStride: Int = 0, y: MTLBuffer, yOffset: Int = 0, yStride: Int = 0, T: Int = 1) {
        let f = parts[0]
        // 2 tokens: scalar qmvt4 (per-token FMAs on weights unpacked once); 3...8: qmvx4 (8×8 MMA,
        // tokens padded to 8), whose cost hardly depends on T. Both are ALU-bound on an M2 Pro.
        let scalar = T == 2
        let paired = T == 1 && parts.allSatisfy { $0.qmvpPSO != nil }
        enc.setComputePipelineState(paired ? f.qmvpPSO! : T == 1 ? f.qmvPSO! : scalar ? f.qmvtPSO! : f.qmvxPSO!)
        for slot in 0..<3 {
            let p = parts[min(slot, parts.count - 1)]
            let base = slot == 0 ? 0 : 3 + slot * 3   // 0, 6, 9
            enc.set(p.weight, base)
            if paired { enc.set(p.pairs!, 0, base + 1); continue }
            enc.set(p.scales!, base + 1)
            enc.set(p.biases!, base + 2)
            enc.set(p.bias ?? p.scales!, 12 + slot)
        }
        enc.set(x, xOffset, 3)
        enc.set(y, yOffset, 4)
        enc.set(x, xOffset, 15)
        let total = parts.reduce(0) { $0 + $1.outDim }
        let rowsPerTG = T == 1 ? qmvRowsPerThreadgroup : scalar ? qmvtRowsPerThreadgroup : qmvxRowsPerThreadgroup
        let threads = T == 1 || scalar ? qmvThreads : 32 * qmvxSimdgroups
        enc.params(MMParams(K: Int32(f.inDim), O: Int32(total), T: Int32(T), gs: Int32(f.groupSize), xStride: Int32(xStride), yStride: Int32(yStride),
                            o1: Int32(parts[0].outDim), o2: Int32(parts.count > 1 ? parts[1].outDim : 0),
                            hasBias: f.bias != nil ? 1 : 0), 5)
        if paired {
            // K = 1024: a grid of at most max(4096, blocks / 16) threadgroups whose simdgroups keep x and loop
            // over row blocks, prefetching the next (qmvq1): the layer matrices get one block per simdgroup
            // (at most 1,028 threadgroups), lm_head 4,096 (~15 blocks each; 3,880 and 2,048 measured ~1%
            // slower in the full step); otherwise one block per simdgroup.
            let blocks = (total + rowsPerTG - 1) / rowsPerTG
            let grid = f.inDim == 1024 ? min(blocks, Linear.pairsGrid ?? max(4096, blocks / 16)) : blocks
            enc.dispatchThreadgroups(MTLSize(width: grid, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
            return
        }
        enc.dispatchThreadgroups(MTLSize(width: (total + rowsPerTG - 1) / rowsPerTG, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
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

    /// The 64 × 64 GEMM tile runs ~1.2× faster per row than the 32 × 32 one, but pads the batch to a
    /// multiple of 64 tokens and makes a quarter as many threadgroups: use it when it pads at most
    /// 1.2× more rows and still yields enough threadgroups to fill the GPU (M2 Pro measurements).
    static func useGemm64(T: Int, outDim: Int) -> Bool {
        let rows64 = (T + 63) / 64 * 64, rows32 = (T + 31) / 32 * 32
        return rows64 * 5 <= rows32 * 6 && (T + 63) / 64 * ((outDim + 63) / 64) >= 64
    }

    /// y[t, :outDim] (row stride yStride, starting at yOffset bytes) = x[t] · Wᵀ (+ bias)
    func encode(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, xOffset: Int = 0, xStride: Int,
                y: MTLBuffer, yOffset: Int = 0, yStride: Int, T: Int) {
        if Linear.canFuse([self], T: T) {
            Linear.encodeFusedDecode(enc, [self], x: x, xOffset: xOffset, xStride: xStride, y: y, yOffset: yOffset, yStride: yStride, T: T)
            return
        }
        // 5+ tokens (not handled by qmvx4 above): tiled GEMM beats the small-batch kernel. A full
        // tile is 32 tokens; short prompts use an 8- or 16-token tile instead of paying for 32,
        // long ones a 64 × 64 tile (same arithmetic per output element in all of them).
        var gemm: (pso: MTLComputePipelineState, rows: Int, cols: Int)?
        if T > 4 {
            if T <= 8, let p = gemm8PSO { gemm = (p, 8, 32) }
            else if T <= 16, let p = gemm16PSO { gemm = (p, 16, 32) }
            else if Linear.useGemm64(T: T, outDim: outDim), let p = gemm64PSO { gemm = (p, 64, 64) }
            else if let p = gemmPSO { gemm = (p, 32, 32) }
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
            enc.dispatchThreadgroups(MTLSize(width: (outDim + gemm.cols - 1) / gemm.cols, height: (T + gemm.rows - 1) / gemm.rows, depth: 1),
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

/// The token mixer of a Qwen 3.5 linear-attention layer (see the `gdn_*` kernels): input projections
/// laid out [q | k | v | z | b | a] per token, the causal convolution, the recurrent state and the
/// gated output norm. Its output projection is the layer's `o`.
final class GatedDeltaNet {
    let dims: ModelConfig.LinearAttention
    /// [in_proj_qkv, in_proj_z, in_proj_b ‖ in_proj_a], or with b and a apart when they cannot be merged.
    let proj: [Linear]
    let projFusable: Bool
    /// Floats per token row of the projection buffer (projCols rounded up to whole float4s).
    let stride: Int
    let convW: MTLBuffer     // [convDim] float4: the taps, right-aligned (w.w multiplies the current input)
    let negA: MTLBuffer      // [Hv] f32: -exp(A_log)
    let dtBias: MTLBuffer    // [Hv] f32
    let normW: MTLBuffer     // [Dv] f32
    let ring: MTLBuffer      // [4][convDim] f32: convolution inputs by position mod 4
    let state: MTLBuffer     // [Hv][Dv][Dk] f32

    static func stride(_ d: ModelConfig.LinearAttention) -> Int { (d.projCols + 3) / 4 * 4 }

    init(_ s: WeightStore, _ p: String, _ c: ModelConfig, _ d: ModelConfig.LinearAttention) throws {
        let gpu = GPU.shared, H = c.hiddenSize
        func bad(_ what: String) -> EngineError { .invalidFile("\(p): \(what)") }
        dims = d
        stride = Self.stride(d)
        let qkv = try Linear(s, "\(p).in_proj_qkv", inDim: H, outDim: d.convDim)
        let z = try Linear(s, "\(p).in_proj_z", inDim: H, outDim: d.valueDim)
        let b = try Linear(s, "\(p).in_proj_b", inDim: H, outDim: d.numValueHeads)
        let a = try Linear(s, "\(p).in_proj_a", inDim: H, outDim: d.numValueHeads)
        if let ba = try Linear.concatRows(b, a, name: "\(p).in_proj_ba") { proj = [qkv, z, ba] } else { proj = [qkv, z, b, a] }
        projFusable = proj.count == 3 && Linear.fusable(proj)

        // Convolution taps: [convDim, K, 1] (MLX) or [convDim, 1, K] (Hugging Face): the same memory.
        let w = try s.require("\(p).conv1d.weight")
        let K = d.convKernel
        guard w.shape.count == 3, w.shape[0] == d.convDim, w.count == d.convDim * K else { throw bad("conv1d.weight shape \(w.shape)") }
        let taps = try w.floats()
        var w4 = [Float](repeating: 0, count: 4 * d.convDim)
        for ch in 0..<d.convDim { for j in 0..<K { w4[4 * ch + 4 - K + j] = taps[ch * K + j] } }
        func f32(_ v: [Float]) throws -> MTLBuffer {
            let buf = try gpu.buffer(v.count * 4)
            v.withUnsafeBytes { buf.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            return buf
        }
        convW = try f32(w4)
        let Hv = d.numValueHeads
        func vector(_ name: String, _ n: Int) throws -> [Float] {
            let t = try s.require("\(p).\(name)")
            guard t.count == n, t.shape.filter({ $0 != 1 }).count <= 1 else { throw bad("\(name) shape \(t.shape)") }
            let v = try t.floats()
            guard v.allSatisfy(\.isFinite) else { throw bad("\(name) is not finite") }
            return v
        }
        negA = try f32(try vector("A_log", Hv).map { -exp($0) })
        dtBias = try f32(try vector("dt_bias", Hv))
        normW = try f32(try vector("norm.weight", d.valueHeadDim))
        ring = try gpu.buffer(ModelConfig.LinearAttention.convRing * d.convDim * 4)
        state = try gpu.buffer(Hv * d.valueHeadDim * d.keyHeadDim * 4)
    }
}

final class TransformerLayer {
    let spec: ModelConfig.LayerSpec
    /// Qwen 3.5 linear-attention layers (no q/k/v projections, KV cache or attention).
    let gdn: GatedDeltaNet?
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
        if spec.linear {
            guard let d = c.linearAttention else { throw EngineError.invalidFile("\(p): linear attention without its dimensions") }
            let g = try GatedDeltaNet(s, "\(p).linear_attn", c, d)
            gdn = g
            qNorm = nil
            kNorm = nil
            qkvCols = g.stride
            qkv = []
            o = try Linear(s, "\(p).linear_attn.out_proj", inDim: d.valueDim, outDim: H)
        } else {
            gdn = nil
            qNorm = s.has("\(p).self_attn.q_norm.weight") ? try RMSNorm(s, "\(p).self_attn.q_norm.weight", eps: eps, dim: D, offset: off) : nil
            kNorm = spec.hasOwnKV && s.has("\(p).self_attn.k_norm.weight") ? try RMSNorm(s, "\(p).self_attn.k_norm.weight", eps: eps, dim: D, offset: off) : nil
            if spec.hasOwnKV, (qNorm == nil) != (kNorm == nil) { throw EngineError.invalidFile("\(p): q_norm and k_norm must come together") }
            // Qwen 3.5: q_proj also yields a per-head output gate (rows [q_h | gate_h] per head). Its rows
            // are reordered to [all q | all gates], so the projection row is [q | gate | k | v].
            let gated = c.attnOutputGate
            qkvCols = TransformerLayer.qkvColumns(spec, c)
            if gated && (!spec.hasOwnKV || spec.kEqV || s.has("\(p).self_attn.qkv_proj.weight")) {
                throw EngineError.unsupported("\(p): gated attention with this projection layout")
            }
            if !spec.hasOwnKV {
                qkv = [try Linear(s, "\(p).self_attn.q_proj", inDim: H, outDim: c.numHeads * D)]
            } else if s.has("\(p).self_attn.qkv_proj.weight") {
                qkv = [try Linear(s, "\(p).self_attn.qkv_proj", inDim: H, outDim: qkvCols)]
            } else {
                let parts = spec.kEqV ? [("q_proj", c.numHeads), ("k_proj", nKV)]
                    : [("q_proj", c.numHeads * (gated ? 2 : 1)), ("k_proj", nKV), ("v_proj", nKV)]
                qkv = try parts.map { try Linear(s, "\(p).self_attn.\($0.0)", inDim: H, outDim: $0.1 * D) }
            }
            if gated {
                let n = c.numHeads * D
                try qkv[0].permuteRows((0..<2 * n).map { r in r < n ? r / D * 2 * D + r % D : (r - n) / D * 2 * D + D + (r - n) % D })
            }
            o = try Linear(s, "\(p).self_attn.o_proj", inDim: c.numHeads * D, outDim: H)
        }
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
        // A batch's new rows must never land on rows its own queries read (see cacheRows).
        guard cacheLen == maxSeq || cacheLen >= c.slidingWindow + ModelConfig.maxBatch else {
            throw EngineError.invalidFile("\(p): KV cache of \(cacheLen) rows for a window of \(c.slidingWindow)")
        }
        if spec.hasOwnKV {
            let cacheBytes = nKV * cacheLen * D * 2
            kCache = try GPU.shared.buffer(cacheBytes)
            vCache = try GPU.shared.buffer(cacheBytes)
        } else {
            kCache = nil; vCache = nil
        }
    }
}

extension Transformer {
    /// Decode matvecs over interleaved quantization pairs (`Linear.makePairs`): hybrid (Qwen 3.5) models;
    /// MLXR_PAIRS=0 turns it off.
    static func usesPairs(_ c: ModelConfig) -> Bool {
        c.isHybrid && c.bits == 4 && ProcessInfo.processInfo.environment["MLXR_PAIRS"] != "0"
    }
}

extension TransformerLayer {
    /// Row stride of a layer's projection buffer: [q | k | v] ([q | gate | k | v] with Qwen 3.5's
    /// output gate), or a linear-attention layer's input projections.
    static func qkvColumns(_ spec: ModelConfig.LayerSpec, _ c: ModelConfig) -> Int {
        if spec.linear, let d = c.linearAttention { return GatedDeltaNet.stride(d) }
        return ((c.attnOutputGate ? 2 : 1) * c.numHeads + 2 * spec.numKVHeads) * spec.headDim
    }
}

extension Transformer {
    /// Upper bound on the scratch buffers `init` allocates for a config (the load-time memory
    /// check adds it to the weights and the KV caches). A test keeps it above `scratchAllocated`.
    public static func scratchBytes(config c: ModelConfig) -> Int {
        let B = maxBatch, H = c.hiddenSize
        let I = c.intermediateSize * (c.isGemma4 ? 2 : 1)   // Gemma 4's KV-shared layers double the MLP
        let qkv = c.layers.map { TransformerLayer.qkvColumns($0, c) }.max() ?? 0
        let attn = max(c.layers.map { c.numHeads * $0.headDim }.max() ?? 0, c.linearAttention?.valueDim ?? 0)
        let maxD = c.layers.map(\.headDim).max() ?? 0
        var bytes = B * 4 * (3 * H + qkv + attn + 3 * I)                     // h, xn, tmp, qkv, attn, gate|up, mlp
        bytes += (1 + maxVerifyTokens) * max(c.vocabSize, 1) * 4             // logits, logitsAll
        bytes += maxVerifyTokens * c.numHeads * attnSplitVirtualSG * (maxD + 2) * 4   // split-attention partials
        if c.perLayerInputSize > 0 {
            let LP = c.perLayerInputSize * c.numLayers
            bytes += B * 4 * (2 * LP + c.perLayerInputSize)                   // per-layer inputs: raw, inputs, gate
        }
        return bytes
    }
}

extension EngineError {
    /// The engine error for a failed command buffer: an out-of-memory failure gets an
    /// actionable message, anything else is reported as a GPU error.
    static func gpu(_ err: Error) -> EngineError {
        let ns = err as NSError
        let d = err.localizedDescription
        if (ns.domain == MTLCommandBufferErrorDomain && ns.code == MTLCommandBufferError.outOfMemory.rawValue)
            || d.contains("OutOfMemory") || d.lowercased().contains("insufficient memory") {
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
    static let maxBatch = ModelConfig.maxBatch
    let maxBatch = Transformer.maxBatch
    /// One past the highest position any committed step has written to the KV caches. Ring
    /// caches alias positions, so a resumed run must know about every write, including those
    /// of a cancelled prefill or a rejected draft that the caller's own bookkeeping dropped.
    public private(set) var writtenHigh = 0
    /// Scratch bytes `init` actually allocated (checked against `scratchBytes(config:)` by a test).
    private(set) var scratchAllocated = 0
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
    /// `sample_blocks`: the first pass of the token selection, the maximum of every block of 256
    /// logits and its index ([blocks] floats, then [blocks] ints), read by `sample_token`.
    private let sampleBlocksPSO: MTLComputePipelineState
    private let sampleBlocks: MTLBuffer
    private let sampleBlockCount: Int
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
    public static let maxVerifyTokens = 8
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
        /// Sparse decode attention (`attention_sparse`, split's shape; combined by `split.combine`).
        /// Only looked up for a model loaded with sparse attention.
        var sparse: MTLComputePipelineState?
        /// Hybrid models' decode attention (`attention_hyb`), replacing `decode` / `split` for small batches.
        var hybrid: HybridAttn?
    }
    struct HybridAttn {
        /// Single-kernel path below `minSplit`; nil leaves short contexts to the generic decode kernel.
        var fused: MTLComputePipelineState?; var fusedSG: Int
        var split: MTLComputePipelineState; var splitTG: Int; var splitSG: Int
        var combine: MTLComputePipelineState; var combineSG: Int
        /// The kernels apply the sigmoid output gate themselves.
        var gated: Bool
        /// Context length from which the split kernel pair is used.
        var minSplit: Int
    }
    /// `attention_hyb` shape: virtual simdgroups V (NSG per threadgroup), U keys in flight, the fused
    /// kernel's simdgroups and keys in flight (0: the generic kernel serves short contexts), the
    /// context from which the split pair runs, and the combine kernel's simdgroups. Measured on an
    /// M2 Pro with Qwen3.5-0.8B: 64 × (8 × 8) with 2 keys in flight and an 8-simdgroup combine took
    /// decode @1024 from 253.9 to 256.0 tok/s (sampled 246.0 → 247.3), with 128 virtual simdgroups
    /// or the split pair at every context no better; the fused kernel (2 threadgroups) lost ~2% at
    /// 16 tokens against the generic kernel, so short contexts keep that one. Only shapes with an
    /// instantiation work. `MLXR_HYB_ATTN=V,NSG,U,fusedSG,fusedU,minSplit,combineSG` overrides it
    /// (experiments); "0" disables the kernels.
    static let hybridAttnShape: (V: Int, NSG: Int, U: Int, fusedSG: Int, fusedU: Int, minSplit: Int, CNSG: Int)? = {
        if let s = ProcessInfo.processInfo.environment["MLXR_HYB_ATTN"] {
            if s == "0" { return nil }
            let v = s.split(separator: ",").compactMap { Int($0) }
            if v.count == 7 { return (v[0], v[1], v[2], v[3], v[4], v[5], v[6]) }
        }
        return (64, 8, 2, 0, 0, 256, 8)
    }()
    /// The decode attention kernels in use, per head size (for `mlxr bench` / `run`).
    public var attentionSummary: String {
        attnKernels.keys.sorted().map { D in
            let k = attnKernels[D]!
            if let h = k.hybrid {
                let short = h.fused != nil ? "fused \(h.fusedSG) simdgroups" : "generic kernel"
                return "head_dim \(D): hybrid kernels (\(short) below \(h.minSplit) tokens; split \(h.splitTG) threadgroups × \(h.splitSG), combine \(h.combineSG)\(h.gated ? ", gate folded" : ""))"
            }
            return "head_dim \(D): generic decode kernel\(k.split != nil ? " + split from \(attnSplitMinContext) tokens" : "")"
        }.joined(separator: "; ")
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
    /// Qwen 3.5: the attention output gate, and the Gated DeltaNet kernels.
    private let gateSigmoidPSO: MTLComputePipelineState?
    struct GDNKernels {
        let step: MTLComputePipelineState   // decode: one dispatch per layer
        let lpr: Int                        // lanes per state row (step and scan)
        let conv, prep, scan, norm: MTLComputePipelineState   // a batch of tokens
    }
    private let gdnKernels: GDNKernels?
    static let gdnDefaultLPR = 8
    /// Hybrid models: a saved recurrent state (`snapshotRecurrent`) that `reusablePrefix` may return to,
    /// so a run sharing at least its positions resumes there. Dropped as soon as a step rewrites an
    /// earlier position (its K/V rows and tokens would no longer match).
    public var recurrentCheckpoint: RecurrentSnapshot?
    /// Hybrid models: the position the recurrent state has reached (the end of the last committed
    /// step), or -1 when it is unknown (a failed step). Unlike a KV cache it cannot be rewound, so a
    /// step must start there or at 0.
    public private(set) var recurrentPosition = 0
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

    /// Sparse decode attention (experimental, changes outputs; see `SparseAttention`): its budget,
    /// or nil for dense decode. It takes effect only when the model was loaded with sparse attention
    /// (`init(directory:contextLength:sparse:)`), which allocates and maintains the index; it can
    /// then be changed or turned off between steps.
    public var sparse: SparseAttention? { didSet { prepared = nil } }
    /// Has `init` allocated the sparse index (so `sparse` can take effect)?
    public var sparseAvailable: Bool { sparseScores != nil && sparseIndex.contains { $0 != nil } }
    /// Layers that decode with sparse attention: full-attention layers with an index, and KV-shared
    /// layers that read their source layer's (0 without sparse attention).
    public let sparseLayerCount: Int
    /// Why a model loaded with sparse attention decodes dense anyway (nil when it is on, or was not
    /// asked for): a context that cannot reach `minContext`, no eligible layer, or a setup failure.
    public let sparseOffReason: String?
    /// Per layer: pooled index keys [cacheLen / pool][nKV][D] (f16), for a full-attention layer with
    /// its own cache and a sparse kernel for its head size. KV-shared layers use their source's.
    private let sparseIndex: [MTLBuffer?]
    /// Block scores of the current layer ([cacheLen / pool + 1] floats) and its attended positions
    /// ([maxSeq] ints); every layer reuses them (the serial encoder orders the dispatches).
    private let sparseScores: MTLBuffer?
    private let sparseList: MTLBuffer?
    /// Entries between two KV heads' position lists in `sparseList` (the context length).
    private let sparseListStride: Int
    private let sparseIndexPSO: MTLComputePipelineState?
    private let sparseScorePSO: MTLComputePipelineState?
    private let sparseSelectPSO: MTLComputePipelineState?
    private let sparseSelectThreads: Int
    /// Tests: layer attentions encoded with the sparse kernels so far.
    private(set) var sparseLayerSteps = 0

    public private(set) var lastGPUTime: Double = 0
    public var vocabSize: Int { lmHead.outDim }
    public var weightBytes: Int { store.totalBytes }
    /// Every projection has the batched 4-bit kernel, so verifying a few tokens is cheap.
    public var supportsBatchedVerify: Bool {
        // Verification runs drafts that may be rejected, and a recurrent state cannot drop them.
        !config.isHybrid && Linear.canFuse([lmHead], T: 2) && layers.allSatisfy { l in
            Linear.canFuse(l.qkv, T: 2) && Linear.canFuse([l.o], T: 2) && Linear.canFuse(l.gateUp, T: 2) && Linear.canFuse([l.down], T: 2)
        }
    }

    /// `sparse` (by default from `MLXR_SPARSE_ATTN`: on unless it is 0) loads the model with sparse
    /// decode attention: see `SparseAttention`.
    public init(directory: URL, contextLength: Int, sparse: SparseAttention? = SparseAttention.fromEnvironment) throws {
        let gpu = GPU.shared
        var c = try ModelConfig(directory: directory)
        let seq = max(256, min(contextLength, max(c.maxPositionEmbeddings, 256)))
        maxSeq = seq
        let store = try WeightStore(directory: directory, device: gpu.device)
        self.store = store
        // Qwen 3.5 converted straight from Hugging Face (conv1d still [C, 1, K]) stores its RMSNorm
        // weights as offsets from 1, except the linear-attention output norm (mlx-lm's sanitize).
        if c.isHybrid, let i = c.layers.firstIndex(where: \.linear),
           let w = store.get("model.layers.\(i).linear_attn.conv1d.weight"), w.shape.count == 3, w.shape[2] != 1 {
            c.normWeightOffset = true
        }
        config = c
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
        if Transformer.usesPairs(c) {
            for l in layers {
                for lin in l.qkv + [l.o] + l.gateUp + [l.down] + (l.gdn?.proj ?? []) { try lin.makePairs() }
            }
            try lmHead.makePairs()
        }
        // The fused residual-norm kernels read two norm weights with one dtype.
        let normType = norm.weight.dtype
        for l in layers {
            let norms = [l.inputNorm, l.mlpNorm, l.attnOutNorm, l.mlpOutNorm, l.pleNorm].compactMap { $0 }
            guard norms.allSatisfy({ $0.weight.dtype == normType }) else { throw EngineError.unsupported("mixed norm weight dtypes") }
        }

        // Scratch buffers fit the largest layer.
        let qkvMax = layers.map(\.qkvCols).max()!
        let attnMax = max(layers.map { c.numHeads * $0.spec.headDim }.max()!, c.linearAttention?.valueDim ?? 0)
        let I = layers.map(\.intermediate).max()!
        let H = c.hiddenSize, B = maxBatch
        hist = try gpu.buffer((seq + 1) * 4)
        rngState = try gpu.buffer(16)
        sampleStatus = try gpu.buffer((seq + 1) * 4)
        sampleDiag = try gpu.buffer(32)
        samplePSO = try gpu.pipeline("sample_token")
        sampleThreads = min(1024, samplePSO.maxTotalThreadsPerThreadgroup) / 32 * 32
        sampleBlocksPSO = try gpu.pipeline("sample_blocks")
        sampleBlockCount = (lmHead.outDim + 255) / 256
        sampleBlocks = try gpu.buffer(2 * sampleBlockCount * 4)
        var scratch = 0
        func buffer(_ n: Int) throws -> MTLBuffer { scratch += n; return try gpu.buffer(n) }
        h = try buffer(B * H * 4)
        xn = try buffer(B * H * 4)
        qkvBuf = try buffer(B * qkvMax * 4)
        attnBuf = try buffer(B * attnMax * 4)
        guBuf = try buffer(B * 2 * I * 4)
        mlpBuf = try buffer(B * I * 4)
        tmp = try buffer(B * H * 4)
        logits = try buffer(lmHead.outDim * 4)
        logitsAll = try buffer(Self.maxVerifyTokens * lmHead.outDim * 4)
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
            let maxG = e == 2 ? 8 : e == 4 ? 5 : e <= 4 ? 4 : 2   // E = 4: G = 1...5 (G = 8 was not bit-identical)
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
                if sparse != nil { k.sparse = try? gpu.pipeline("attention_sparse_\(e)_\(gSub)\(cap)") }
            }
            // Hybrid models (Qwen 3.5): vectorized decode attention with the output gate folded in.
            // Its partials (NT rows per head) fit the split buffer (32 rows per head).
            if c.isHybrid, D == 256, cap.isEmpty, Gs == [4], c.numHeads % 2 == 0,
               let hyb = Self.hybridAttnShape {
                let g = c.attnOutputGate ? "_g" : ""
                let fused = hyb.fusedSG > 0 ? try? gpu.pipeline("attention_hyb_4_\(hyb.fusedSG)_\(hyb.fusedSG)_\(hyb.fusedU)\(g)") : nil
                if hyb.fusedSG == 0 || (fused.map { $0.maxTotalThreadsPerThreadgroup >= hyb.fusedSG * 32 } ?? false),
                   let split = try? gpu.pipeline("attention_hyb_4_\(hyb.V)_\(hyb.NSG)_\(hyb.U)\(g)"),
                   split.maxTotalThreadsPerThreadgroup >= hyb.NSG * 32,
                   let combine = try? gpu.pipeline("attention_hyb_combine_\(hyb.V / hyb.NSG)_\(hyb.CNSG)\(g)"),
                   case let (bytes, o) = c.numHeads.multipliedReportingOverflow(by: Self.attnSplitVirtualSG * (D + 2) * 4), !o {
                    k.hybrid = HybridAttn(fused: fused, fusedSG: hyb.fusedSG, split: split, splitTG: hyb.V / hyb.NSG, splitSG: hyb.NSG,
                                          combine: combine, combineSG: hyb.CNSG, gated: c.attnOutputGate, minSplit: hyb.minSplit)
                    partialBytes = max(partialBytes, bytes)
                }
            }
            kernels[D] = k
        }
        attnPartials = partialBytes > 0 ? try buffer(partialBytes * Self.maxVerifyTokens) : nil   // per query token
        attnKernels = kernels

        // Sparse decode attention: an index for every eligible layer, block scores and the position
        // lists (not scratch: `sparseBytes` accounts for them). Nothing is allocated when the context
        // can never reach `minContext`; a setup failure (a pipeline, an allocation, a thread limit)
        // leaves the model with dense decode rather than failing the load.
        var sparseLayers: [Bool] = []
        for l in layers {
            sparseLayers.append(Transformer.sparseEligible(l.spec, cacheLen: l.cacheLen, context: seq) && kernels[l.spec.headDim]?.sparse != nil)
        }
        struct SparseSetup {
            var index: [MTLBuffer?]; var scores: MTLBuffer; var list: MTLBuffer
            var indexPSO: MTLComputePipelineState; var scorePSO: MTLComputePipelineState; var selectPSO: MTLComputePipelineState
            var selectThreads: Int
        }
        var setup: SparseSetup?
        var offReason: String?
        if let sparse {
            if seq < sparse.minContext { offReason = "the context length (\(seq)) is below \(sparse.minContext) tokens, where it starts" }
            else if attnPartials == nil || !sparseLayers.contains(true) { offReason = "no layer of this model supports it" }
        }
        if let sparse, seq >= sparse.minContext, attnPartials != nil, sparseLayers.contains(true) {
            do {
                let pool = SparseAttention.pool
                var index: [MTLBuffer?] = []
                for (l, on) in zip(layers, sparseLayers) {
                    index.append(on ? try gpu.buffer(seq / pool * l.spec.numKVHeads * l.spec.headDim * 2) : nil)
                }
                let indexed = zip(layers.map(\.spec), sparseLayers).filter { $0.1 }.map(\.0)
                // One score row and one list per KV head (a shared selection uses row 0).
                let rows = max(1, indexed.map(\.numKVHeads).max() ?? 1)
                let indexPSO = try gpu.pipeline("sparse_index"), score = try gpu.pipeline("sparse_score"), select = try gpu.pipeline("sparse_select")
                let selectThreads = min(1024, select.maxTotalThreadsPerThreadgroup) / 32 * 32
                guard selectThreads >= 32, score.maxTotalThreadsPerThreadgroup >= Self.sparseScoreThreads,
                      indexPSO.maxTotalThreadsPerThreadgroup >= indexed.map(\.headDim).max() ?? 0 else {
                    throw EngineError.unsupported("sparse attention: kernel thread limits")
                }
                setup = SparseSetup(index: index, scores: try gpu.buffer(rows * (seq / pool + 1) * 4), list: try gpu.buffer(rows * seq * 4),
                                    indexPSO: indexPSO, scorePSO: score, selectPSO: select, selectThreads: selectThreads)
            } catch {
                setup = nil   // dense decode
                offReason = "it could not be set up on this GPU (\(error.localizedDescription))"
            }
        }
        sparseOffReason = offReason
        sparseIndex = setup?.index ?? [MTLBuffer?](repeating: nil, count: layers.count)
        sparseScores = setup?.scores
        sparseList = setup?.list
        sparseListStride = setup != nil ? seq : 0
        sparseIndexPSO = setup?.indexPSO
        sparseScorePSO = setup?.scorePSO
        sparseSelectPSO = setup?.selectPSO
        sparseSelectThreads = setup?.selectThreads ?? 0
        // Layers that decode sparse: those with an index, and KV-shared layers reading their source's.
        var sparseCount = 0
        for (li, layer) in layers.enumerated() where setup != nil && !layer.spec.linear {
            let spec = layer.spec, source = spec.hasOwnKV ? li : spec.kvSource, kv = layers[source].spec
            if !spec.sliding, kv.numKVHeads == spec.numKVHeads, kv.headDim == spec.headDim,
               sparseIndex[source] != nil, kernels[spec.headDim]?.sparse != nil { sparseCount += 1 }
        }
        sparseLayerCount = sparseCount
        self.sparse = sparse

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
                                 raw: try buffer(B * LP * 4), inputs: try buffer(B * LP * 4), gateBuf: try buffer(B * P * 4),
                                 addPSO: try gpu.pipeline("add_scaled"), gatePSO: try gpu.pipeline("gelu_tanh_gate"))
        } else {
            ple = nil
        }
        scratchAllocated = scratch
        switch c.activation {
        case .silu: siluPSO = try gpu.pipeline("silu_mul")
        case .geluTanh: siluPSO = try gpu.pipeline("gelu_tanh_mul")
        case .gelu: siluPSO = try gpu.pipeline("gelu_mul")
        }
        addPSO = try gpu.pipeline("add_inplace")
        gateSigmoidPSO = c.attnOutputGate ? try gpu.pipeline("sigmoid_gate") : nil
        if c.isHybrid {
            // Decode: 8 lanes per state row (1024 threads) unless MLXR_GDN_LPR says otherwise or the GPU
            // cannot run that many threads per threadgroup.
            var lpr = Int(ProcessInfo.processInfo.environment["MLXR_GDN_LPR"] ?? "") ?? Self.gdnDefaultLPR
            if lpr != 4 && lpr != 8 { lpr = Self.gdnDefaultLPR }
            var step = try gpu.pipeline("gdn_step_\(lpr)")
            if step.maxTotalThreadsPerThreadgroup < 128 * lpr { lpr = 4; step = try gpu.pipeline("gdn_step_4") }
            let scan = try gpu.pipeline("gdn_scan_\(lpr)")
            guard step.maxTotalThreadsPerThreadgroup >= 128 * lpr, scan.maxTotalThreadsPerThreadgroup >= 64 else {
                throw EngineError.unsupported("Gated DeltaNet kernels: thread limits")
            }
            gdnKernels = GDNKernels(step: step, lpr: lpr, conv: try gpu.pipeline("gdn_conv"), prep: try gpu.pipeline("gdn_prep"),
                                    scan: scan, norm: try gpu.pipeline("gdn_norm"))
        } else {
            gdnKernels = nil
        }
        scalePSO = c.embedScale != nil ? try gpu.pipeline("scale_inplace") : nil
        softcapPSO = c.finalLogitSoftcap > 0 ? try gpu.pipeline("softcap_inplace") : nil
        embedPSO = try embed.lookupPipeline()
    }

    /// Runs `tokens` at positions offset..<offset+count, updating the KV cache.
    /// Returns a pointer to the logits of the last token (valid until the next call).
    /// Runs the tokens at positions offset..<offset+count. Returns the last row's logits, or with
    /// `allLogits` (count <= maxVerifyTokens) a row-major [count][vocab] block.
    /// `dense`: no sparse attention even for a single token (speculative decoding, whose
    /// verification passes are dense). Several tokens always run dense, like a prefill.
    public func forward(_ tokens: [Int32], offset: Int, allLogits: Bool = false, dense: Bool = false,
                        isCancelled: (() -> Bool)? = nil) throws -> UnsafeMutablePointer<Float> {
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
            try step(Array(tokens[start..<start + n]), offset: offset + start, computeLogits: last, allLogits: allLogits,
                     dense: dense || tokens.count > 1)
            start += n
        }
        if allLogits { return logitsAll.contents().bindMemory(to: Float.self, capacity: tokens.count * lmHead.outDim) }
        return logits.contents().bindMemory(to: Float.self, capacity: lmHead.outDim)
    }

    private func step(_ tokens: [Int32], offset: Int, computeLogits: Bool, allLogits: Bool = false, dense: Bool = false) throws {
        let T = tokens.count
        // A prepared buffer is exactly what encode(T: 1, offset:, computeLogits: true, allLogits: false)
        // returns now, so using it changes nothing on the GPU. Any other call shape discards it.
        var ready: MTLCommandBuffer?
        if T == 1, computeLogits, !allLogits, !dense, let p = prepared, p.offset == offset { ready = p.cb }
        prepared = nil
        try checkRecurrent(offset)
        let cb = try ready ?? encode(T: T, offset: offset, computeLogits: computeLogits, allLogits: allLogits, dense: dense)
        // Nothing is in flight (the previous step was waited for), so `hist` is free to write.
        writeHistory(tokens, at: offset)
        cb.commit()
        writtenHigh = max(writtenHigh, offset + T)
        recurrentPosition = offset + T
        // Overlap CPU encoding of the likely next decode step (the following position) with this
        // step's GPU work. It is only committed after this buffer completes, on an exact match.
        if computeLogits, offset + T < maxSeq, let next = try? encode(T: 1, offset: offset + T, computeLogits: true, allLogits: false) {
            prepared = (next, offset + T)
        }
        cb.waitUntilCompleted()
        lastGPUTime = cb.gpuEndTime - cb.gpuStartTime
        if let err = cb.error {
            prepared = nil
            recurrentPosition = -1
            throw EngineError.gpu(err)
        }
    }

    /// Hybrid models: a step must continue the recurrent state where it is, or restart at 0. A step
    /// that rewrites positions before the checkpoint invalidates it.
    private func checkRecurrent(_ offset: Int) throws {
        if let cp = recurrentCheckpoint, offset < cp.position { recurrentCheckpoint = nil }
        guard config.isHybrid, offset != 0, offset != recurrentPosition else { return }
        throw EngineError.unsupported("the recurrent state is at position \(recurrentPosition) and cannot resume at \(offset)")
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

    /// How much of the KV cache a new run sharing its first `common` tokens with the previous
    /// writes can keep. Sliding-window layers hold only the latest rows, so once writing went on
    /// for more than a batch past the divergence point (`writtenHigh`, which counts cancelled
    /// prefills and rejected drafts too), the rows the resumed run would read are gone: the
    /// prompt is processed from the start. `previousLength` is the caller's own count, used
    /// when it is larger (it never should be).
    public func reusablePrefix(_ common: Int, previousLength: Int) -> Int {
        // A recurrent state holds exactly the positions processed so far: it is reusable when the new run
        // continues right where it stopped, or from the checkpoint (restored here) if it shares that much.
        if config.isHybrid {
            if common > 0 && common == recurrentPosition { return common }
            if let cp = recurrentCheckpoint, cp.position > 0, cp.position <= common {
                restoreRecurrent(cp)
                return cp.position
            }
            return 0
        }
        let written = max(previousLength, writtenHigh)
        for l in layers where l.spec.hasOwnKV && l.cacheLen < maxSeq {
            if written - common > l.cacheLen - config.slidingWindow + 1 { return 0 }
        }
        return common
    }

    /// Forget the write history: the next run starts at position 0 and overwrites everything
    /// it will read.
    public func resetWriteTracking() { writtenHigh = 0 }

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
            try checkRecurrent(at)
            let cb = try encode(T: n, offset: at, computeLogits: last, allLogits: false, sample: last ? sample : nil, dense: true)
            writeHistory(Array(tokens[start..<start + n]), at: at)
            cb.commit()
            writtenHigh = max(writtenHigh, at + n)
            recurrentPosition = at + n
            cb.waitUntilCompleted()
            lastGPUTime = cb.gpuEndTime - cb.gpuStartTime
            if let err = cb.error { recurrentPosition = -1; throw EngineError.gpu(err) }
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
        try checkRecurrent(offset)
        let cb = try encode(T: 1, offset: offset, computeLogits: true, allLogits: false, sample: sample)
        cb.commit()
        writtenHigh = max(writtenHigh, offset + 1)
        recurrentPosition = offset + 1
        return DecodeStep(cb: cb, offset: offset)
    }

    /// Writes a few tokens to positions offset..<offset+count and commits one step over them that
    /// selects the next token on the GPU, without waiting (draft models: catching up on the tokens
    /// the target accepted, followed by `beginDecodeStep`s queued back to back). Nothing may be in
    /// flight. The returned step's `offset` is the last token's position. A few tokens run as single-token
    /// steps (the earlier ones without logits), which the matvec kernels do faster than one batch; their
    /// buffers count as one step for the in-flight limit (they share no per-position sampler state).
    public func beginSteps(_ tokens: [Int32], offset: Int, sample: SampleParams) throws -> DecodeStep {
        guard !tokens.isEmpty, tokens.count <= maxBatch, offset >= 0, offset + tokens.count < maxSeq else { throw EngineError.contextFull }
        guard tokens.allSatisfy({ $0 >= 0 && Int($0) < embed.outDim }) else {
            throw EngineError.invalidFile("token id outside the embedding table (\(embed.outDim) rows)")
        }
        prepared = nil
        let n = tokens.count
        let single = n <= 4
        writeHistory(tokens, at: offset)   // read when the buffers execute
        var cb: MTLCommandBuffer?
        for i in single ? 0..<n : 0..<1 {
            let last = !single || i == n - 1
            let b: MTLCommandBuffer
            do {
                try checkRecurrent(offset + i)
                b = try encode(T: single ? 1 : n, offset: offset + i, computeLogits: last, allLogits: false, sample: last ? sample : nil)
            } catch { cb?.waitUntilCompleted(); throw error }
            b.commit()
            writtenHigh = max(writtenHigh, offset + i + (single ? 1 : n))
            recurrentPosition = offset + i + (single ? 1 : n)
            cb = b
        }
        return DecodeStep(cb: cb!, offset: offset + n - 1)
    }

    /// Waits for the step and returns the token it sampled (hist[offset + 1]).
    public func finish(_ step: DecodeStep) throws -> Int {
        step.cb.waitUntilCompleted()
        lastGPUTime = step.cb.gpuEndTime - step.cb.gpuStartTime
        if let err = step.cb.error { recurrentPosition = -1; throw EngineError.gpu(err) }
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
    /// `dense`: no sparse attention (prefill, even a one-token tail, and dense forwards).
    private func encode(T: Int, offset: Int, computeLogits: Bool, allLogits: Bool, sample: SampleParams? = nil,
                        dense: Bool = false) throws -> MTLCommandBuffer {
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
        // Up to maxVerifyTokens queries run as T decode steps (decode / split attention per token):
        // the prefill kernels give one threadgroup per head the whole context, so a verify pass
        // grew with the context length far faster than a decode step.
        let smallT = T <= Self.maxVerifyTokens
        // Sparse decode attention: one plan per step (the blocks every eligible layer scores).
        var sparsePlan: SparsePlan?
        if decode, !dense, sparseScores != nil, let sp = sparse { sparsePlan = sp.plan(position: offset) }
        for (li, layer) in layers.enumerated() {
            let spec = layer.spec
            let D = spec.headDim, nKV = spec.numKVHeads, qkvCols = layer.qkvCols, I = layer.intermediate
            if let g = layer.gdn {
                // Linear attention: projections → convolution, recurrence, gated norm (into attnBuf).
                encodeGatedDeltaNet(enc, g, T: T, offset: offset, skipProj: skipQKV, skipMixer: skipAttn)
            } else {
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
                                     vNorm: c.valueNorm ? 1 : 0, kEqV: spec.kEqV ? 1 : 0, gateHeads: c.attnOutputGate ? Int32(nH) : 0), 6)
            // A KV-shared layer only prepares its queries.
            enc.dispatchThreadgroups(MTLSize(width: spec.hasOwnKV ? nH + 2 * nKV : nH, height: T, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: D, height: 1, depth: 1))
            }
            // Sparse index: (re)build the blocks completed by the positions this step wrote. Earlier
            // positions are valid, so a rewritten position (a new turn, a rejected draft) rebuilds its block.
            if !skipPost, let index = sparseIndex[li], let pso = sparseIndexPSO {
                let pool = SparseAttention.pool, b0 = offset / pool, b1 = (offset + T) / pool
                if b1 > b0 {
                    enc.setComputePipelineState(pso)
                    enc.set(layer.kCache!, 0, 0)
                    enc.set(index, 0, 1)
                    enc.params(SparseParams(pool: Int32(pool), nKV: Int32(nKV), D: Int32(D), maxSeq: Int32(layer.cacheLen),
                                            b0: Int32(b0), nb: Int32(b1 - b0)), 2)
                    enc.dispatchThreadgroups(MTLSize(width: b1 - b0, height: nKV, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: D, height: 1, depth: 1))
                }
            }

            let attnParams = AttnParams(D: Int32(D), nH: Int32(nH), nKV: Int32(nKV), qStride: Int32(qkvCols),
                                        maxSeq: Int32(kv.cacheLen), offset: Int32(offset), scale: c.ropeAttentionFactor * c.ropeAttentionFactor * c.queryScale,
                                        outStride: Int32(nH * D), T: Int32(T), window: Int32(spec.sliding ? c.slidingWindow : 0),
                                        softcap: c.attnLogitSoftcap)
            var gateDone = false   // the attention kernel applied the output gate itself
            if skipAttn {
            } else if let plan = sparsePlan, !spec.sliding, kv.spec.numKVHeads == nKV, kv.spec.headDim == D,   // a KV-shared layer reads its source's index
                      let index = sparseIndex[spec.hasOwnKV ? li : spec.kvSource],
                      let attnSparse = kernels.sparse, let combine = kernels.split?.combine, let partials = attnPartials,
                      let scores = sparseScores, let list = sparseList, let scorePSO = sparseScorePSO, let selectPSO = sparseSelectPSO {
                // Sparse decode: score the index blocks, select the attended positions, attend over them.
                var sp = plan.params
                sp.nKV = Int32(nKV); sp.D = Int32(D); sp.maxSeq = Int32(kv.cacheLen); sp.nH = Int32(nH)
                let perHead = sparse?.perHead ?? true   // rows <= the buffers' rows: nKV is an eligible layer's
                sp.rows = Int32(perHead ? nKV : 1); sp.listStride = Int32(perHead ? sparseListStride : 0)
                enc.setComputePipelineState(scorePSO)
                enc.set(qkvBuf, 0, 0)
                enc.set(index, 0, 1)
                enc.set(scores, 0, 2)
                enc.params(sp, 3)
                enc.dispatchThreadgroups(MTLSize(width: (Int(sp.nb) + Self.sparseScoreBlocks - 1) / Self.sparseScoreBlocks, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: Self.sparseScoreThreads, height: 1, depth: 1))
                enc.setComputePipelineState(selectPSO)
                enc.set(scores, 0, 0)
                enc.set(list, 0, 1)
                enc.params(sp, 2)
                enc.dispatchThreadgroups(MTLSize(width: Int(sp.rows), height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: sparseSelectThreads, height: 1, depth: 1))
                enc.setComputePipelineState(attnSparse)
                enc.set(qkvBuf, 0, 0)
                enc.set(kv.kCache!, 0, 1)
                enc.set(kv.vCache!, 0, 2)
                enc.set(partials, 0, 3)
                enc.params(attnParams, 4)
                enc.set(list, 0, 5)
                enc.params(SIMD2<Int32>(Int32(plan.count), sp.listStride), 6)
                enc.dispatchThreadgroups(MTLSize(width: nKV, height: Self.attnSplitVirtualSG / Self.attnSplitSG, depth: kernels.splitGroups),
                                         threadsPerThreadgroup: MTLSize(width: Self.attnSplitSG * 32, height: 1, depth: 1))
                enc.setComputePipelineState(combine)
                enc.set(partials, 0, 0)
                enc.set(attnBuf, 0, 1)
                enc.params(attnParams, 2)
                enc.dispatchThreadgroups(MTLSize(width: nH, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                sparseLayerSteps += 1
            } else if smallT, let hyb = kernels.hybrid, qkvCols % 4 == 0, offset + 1 >= hyb.minSplit || hyb.fused != nil {
                // Hybrid models: split + combine from `minSplit` (the gate folded into the combine);
                // below it the fused kernel when configured, else the generic path with its gate dispatch.
                enc.set(qkvBuf, 0, 0)
                enc.set(kv.kCache!, 0, 1)
                enc.set(kv.vCache!, 0, 2)
                enc.params(attnParams, 4)
                if offset + 1 >= hyb.minSplit, let partials = attnPartials {
                    // Partials are reused by every layer: the serial encoder orders the dispatches.
                    enc.setComputePipelineState(hyb.split)
                    enc.set(partials, 0, 3)
                    enc.dispatchThreadgroups(MTLSize(width: nKV * T, height: hyb.splitTG, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: hyb.splitSG * 32, height: 1, depth: 1))
                    enc.setComputePipelineState(hyb.combine)
                    enc.set(partials, 0, 0)
                    enc.set(attnBuf, 0, 1)
                    enc.params(attnParams, 2)
                    enc.set(qkvBuf, 0, 3)
                    enc.dispatchThreadgroups(MTLSize(width: nH, height: T, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: hyb.combineSG * 32, height: 1, depth: 1))
                } else {
                    enc.setComputePipelineState(hyb.fused!)
                    enc.set(attnBuf, 0, 3)
                    enc.dispatchThreadgroups(MTLSize(width: nKV * T, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: hyb.fusedSG * 32, height: 1, depth: 1))
                }
                gateDone = hyb.gated
            } else if smallT, offset + 1 >= attnSplitMinContext, let (split, combine) = kernels.split, let partials = attnPartials {
                // Partials are reused by every layer: the serial encoder orders the dispatches.
                enc.setComputePipelineState(split)
                enc.set(qkvBuf, 0, 0)
                enc.set(kv.kCache!, 0, 1)
                enc.set(kv.vCache!, 0, 2)
                enc.set(partials, 0, 3)
                enc.params(attnParams, 4)
                enc.dispatchThreadgroups(MTLSize(width: nKV * T, height: Self.attnSplitVirtualSG / Self.attnSplitSG, depth: kernels.splitGroups),
                                         threadsPerThreadgroup: MTLSize(width: Self.attnSplitSG * 32, height: 1, depth: 1))
                enc.setComputePipelineState(combine)
                enc.set(partials, 0, 0)
                enc.set(attnBuf, 0, 1)
                enc.params(attnParams, 2)
                enc.dispatchThreadgroups(MTLSize(width: nH, height: T, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            } else {
                let (attn, attnThreads) = smallT ? kernels.decode : kernels.prefill
                enc.setComputePipelineState(attn)
                enc.set(qkvBuf, 0, 0)
                enc.set(kv.kCache!, 0, 1)
                enc.set(kv.vCache!, 0, 2)
                enc.set(attnBuf, 0, 3)
                enc.params(attnParams, 4)
                enc.dispatchThreadgroups(MTLSize(width: nH, height: smallT ? T : (T + kernels.prefillQB - 1) / kernels.prefillQB, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: attnThreads, height: 1, depth: 1))
            }
            if let gatePSO = gateSigmoidPSO, !skipAttn, !gateDone {
                // Qwen 3.5: attn · sigmoid(gate), the gate sitting after the q section of the projection row.
                enc.setComputePipelineState(gatePSO)
                enc.set(attnBuf, 0, 0)
                enc.set(qkvBuf, nH * D * 4, 1)
                enc.params(EWParams(n: 0, stride: Int32(qkvCols), cols: Int32(nH * D)), 2)
                enc.dispatchThreads(MTLSize(width: nH * D, height: T, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            }

            if !skipO { layer.o.encode(enc, x: attnBuf, xStride: layer.o.inDim, y: tmp, yStride: H, T: T) }
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
            encodeSample(enc, sp, logits: logits)
        }
        enc.endEncoding()
        return cb
    }

    /// Token selection on `logits` into hist[sp.pos + 1]: `sample_blocks` (the block maxima, the
    /// whole GPU) then `sample_token` (one threadgroup over the candidate blocks).
    private func encodeSample(_ enc: MTLComputeCommandEncoder, _ sp: SampleParams, logits: MTLBuffer) {
        let blocks = (Int(sp.V) + 255) / 256
        enc.setComputePipelineState(sampleBlocksPSO)
        enc.set(logits, 0, 0)
        enc.params(sp, 4)
        enc.set(sampleBlocks, 0, 6)
        enc.set(sampleBlocks, sampleBlockCount * 4, 7)
        enc.dispatchThreadgroups(MTLSize(width: (blocks + 7) / 8, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.setComputePipelineState(samplePSO)
        enc.set(logits, 0, 0)
        enc.set(hist, 0, 1)
        enc.set(rngState, 0, 2)
        enc.set(sampleStatus, 0, 3)
        enc.params(sp, 4)
        enc.set(sampleDiag, 0, 5)
        enc.set(sampleBlocks, 0, 6)
        enc.set(sampleBlocks, sampleBlockCount * 4, 7)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: sampleThreads, height: 1, depth: 1))
    }

    /// Qwen 3.5 linear attention for T tokens at `offset`: the input projections into `qkvBuf`, then the
    /// mixer (one `gdn_step` for a single token, `gdn_conv` / `gdn_prep` / `gdn_scan` / `gdn_norm` for a
    /// batch) writing the gated, normalized values into `attnBuf` ([T][Hv · Dv]).
    private func encodeGatedDeltaNet(_ enc: MTLComputeCommandEncoder, _ g: GatedDeltaNet, T: Int, offset: Int,
                                     skipProj: Bool = false, skipMixer: Bool = false) {
        let d = g.dims, H = config.hiddenSize, cols = g.stride
        if skipProj {
        } else if g.projFusable && Linear.canFuse(g.proj, T: T) {
            Linear.encodeFusedDecode(enc, g.proj, x: xn, xStride: H, y: qkvBuf, yStride: cols, T: T)
        } else {
            var col = 0
            for part in g.proj {
                part.encode(enc, x: xn, xStride: H, y: qkvBuf, yOffset: col * 4, yStride: cols, T: T)
                col += part.outDim
            }
        }
        guard !skipMixer, let k = gdnKernels else { return }
        let p = GDNParams(T: Int32(T), offset: Int32(offset), cols: Int32(cols), Hk: Int32(d.numKeyHeads), Hv: Int32(d.numValueHeads),
                          convDim: Int32(d.convDim), outStride: Int32(d.valueDim), eps: config.rmsNormEps)
        let D = d.valueHeadDim
        if T == 1 {
            enc.setComputePipelineState(k.step)
            enc.set(qkvBuf, 0, 0); enc.set(g.convW, 0, 1); enc.set(g.ring, 0, 2); enc.set(g.state, 0, 3)
            enc.set(g.negA, 0, 4); enc.set(g.dtBias, 0, 5); enc.set(g.normW, 0, 6); enc.set(attnBuf, 0, 7)
            enc.params(p, 8)
            enc.dispatchThreadgroups(MTLSize(width: d.numValueHeads, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: D * k.lpr, height: 1, depth: 1))
            return
        }
        enc.setComputePipelineState(k.conv)
        enc.set(qkvBuf, 0, 0); enc.set(g.convW, 0, 1); enc.set(g.ring, 0, 2)
        enc.params(p, 3)
        enc.dispatchThreads(MTLSize(width: d.convDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.setComputePipelineState(k.prep)
        enc.set(qkvBuf, 0, 0); enc.set(g.negA, 0, 1); enc.set(g.dtBias, 0, 2)
        enc.params(p, 3)
        enc.dispatchThreadgroups(MTLSize(width: d.numKeyHeads, height: T, depth: 1), threadsPerThreadgroup: MTLSize(width: 2 * D, height: 1, depth: 1))
        // Two simdgroups (2 · 32 / LPR state rows) per threadgroup.
        let rowsPerTG = 2 * 32 / k.lpr
        enc.setComputePipelineState(k.scan)
        enc.set(qkvBuf, 0, 0); enc.set(g.state, 0, 1)
        enc.params(p, 2)
        enc.dispatchThreadgroups(MTLSize(width: D / rowsPerTG, height: d.numValueHeads, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.setComputePipelineState(k.norm)
        enc.set(qkvBuf, 0, 0); enc.set(g.normW, 0, 1); enc.set(attnBuf, 0, 2)
        enc.params(p, 3)
        enc.dispatchThreadgroups(MTLSize(width: d.numValueHeads, height: T, depth: 1), threadsPerThreadgroup: MTLSize(width: D, height: 1, depth: 1))
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
        // Best of 3 command buffers after one discarded warm-up: the first touch of freshly
        // loaded buffers and the GPU clock ramp otherwise make the first sweep look 2-3x slower.
        func time(_ n: Int, _ body: (MTLComputeCommandEncoder) -> Void) -> Double {
            var best = Double.infinity
            for rep in 0..<4 {
                let cb = q.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder(dispatchType: .serial)!
                for _ in 0..<n { body(enc) }
                enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                if rep > 0 { best = min(best, (cb.gpuEndTime - cb.gpuStartTime) / Double(n)) }
            }
            return best
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
        for vT in [2, 4, Self.maxVerifyTokens] {
            sweep("gate_up", { $0.gateUp }, x: xn, T: vT)
            sweep("down", { [$0.down] }, x: mlpBuf, T: vT)
        }
        let tLM = time(20) { lmHead.encode($0, x: xn, xStride: H, y: logits, yStride: lmHead.outDim, T: 1) }
        out += String(format: "lm_head matvec: %.3f ms  (%.0f GB/s, cache-warm)\n", tLM * 1000, bytes(lmHead) / tLM / 1e9)
        // Decode step breakdown: the step without each kernel class (its cost includes dispatch gaps).
        func stepTime(_ skip: Set<String>, T: Int = 1, offset: Int = 16) -> Double {
            profileSkip = skip
            defer { profileSkip = [] }
            var best = Double.infinity
            for _ in 0..<(T == 1 ? 5 : 3) {
                let cb = try! encode(T: T, offset: T == 1 ? offset : 0, computeLogits: true, allLogits: false)
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
        if maxSeq > 1100 {
            let f1k = stepTime([], offset: 1024)
            out += String(format: "decode step @1024: %.3f ms; without attn -%.3f post -%.3f\n", f1k * 1000,
                          (f1k - stepTime(["attn"], offset: 1024)) * 1000, (f1k - stepTime(["post"], offset: 1024)) * 1000)
        }
        // Small-batch step (speculative verification, qmvx4) at T = 4 and 8.
        for vT in [4, Self.maxVerifyTokens] {
            let fullV = stepTime([], T: vT)
            out += String(format: "verify step T=%d: %.3f ms; without", vT, fullV * 1000)
            for cls in ["qkv", "o", "gateup", "down", "lmhead", "post", "attn", "norm", "silu"] {
                out += String(format: " %@ -%.3f", cls as NSString, (fullV - stepTime([cls], T: vT)) * 1000)
            }
            out += "\n"
        }
        // Prefill step breakdown at 512 tokens (GEMMs, attention and the elementwise kernels).
        let fullP = stepTime([], T: 512)
        out += String(format: "prefill step @512: %.1f ms; without", fullP * 1000)
        for cls in ["qkv", "o", "gateup", "down", "lmhead", "post", "attn", "norm", "silu"] {
            out += String(format: " %@ -%.1f", cls as NSString, (fullP - stepTime([cls], T: 512)) * 1000)
        }
        out += "\n"
        // Qwen 3.5: the Gated DeltaNet mixers of every linear-attention layer (each its own state).
        let mixers = layers.compactMap(\.gdn)
        if let d = mixers.first?.dims {
            let state = Double(mixers.count * 2 * d.numValueHeads * d.valueHeadDim * d.keyHeadDim * 4)
            let t1 = time(5) { enc in for g in mixers { self.encodeGatedDeltaNet(enc, g, T: 1, offset: 16, skipProj: true) } }
            out += String(format: "gdn_step x%d: %.3f ms (%.0f GB/s of state read + written)\n", mixers.count, t1 * 1000, state / t1 / 1e9)
            let tP = time(2) { enc in for g in mixers { self.encodeGatedDeltaNet(enc, g, T: 512, offset: 16, skipProj: true) } }
            out += String(format: "gdn mixers at T=512 x%d: %.2f ms\n", mixers.count, tP * 1000)
        }
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
    /// Running totals of the GPU sampler since the last call (profiling): pool-pass attempts and pool
    /// sizes summed over the sampled steps, and the largest attempt count. Nothing may be in flight.
    public func takeSampleDiagnostics() -> (attempts: Int, pool: Int, maxAttempts: Int) {
        let d = sampleDiag.contents().bindMemory(to: Int32.self, capacity: 8)
        defer { d[2] = 0; d[3] = 0; d[4] = 0 }
        return (Int(d[2]), Int(d[3]), Int(d[4]))
    }

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
                encodeSample(enc, sp, logits: scratch)
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

extension Transformer {
    /// Hybrid models: a copy of the recurrent state (every linear-attention layer's convolution ring
    /// and S) at `recurrentPosition`.
    public struct RecurrentSnapshot {
        public let position: Int
        fileprivate let data: [Data]
    }

    /// Copies the recurrent state, so `restoreRecurrent` can later return to this position (to rerun
    /// it: benchmarks, tests). Nothing may be in flight. The K/V caches need no copy: rows past the
    /// position are rewritten before they are read.
    public func snapshotRecurrent() -> RecurrentSnapshot {
        RecurrentSnapshot(position: recurrentPosition, data: recurrentBuffers.map { Data(bytes: $0.contents(), count: $0.length) })
    }

    /// Returns the recurrent state to a snapshot's position (see `snapshotRecurrent`). Nothing may be in flight.
    public func restoreRecurrent(_ s: RecurrentSnapshot) {
        for (b, d) in zip(recurrentBuffers, s.data) {
            d.withUnsafeBytes { b.contents().copyMemory(from: $0.baseAddress!, byteCount: min($0.count, b.length)) }
        }
        recurrentPosition = s.position
        prepared = nil
    }

    private var recurrentBuffers: [MTLBuffer] { layers.compactMap(\.gdn).flatMap { [$0.ring, $0.state] } }
}
