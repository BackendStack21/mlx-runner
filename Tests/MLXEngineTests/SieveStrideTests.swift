import Foundation
import Metal
import Testing
@testable import MLXEngine

/// Phase 0 of docs/sieve-architecture.md, hypothesis H3 (Stride): how often does a Qwen 3.5
/// checkpoint's multi-token-prediction head draft the token after next correctly? It measures and
/// reports; the one thing it asserts is that the CPU decoder block used for the head reproduces the
/// engine's own full-attention layers. Opt-in, and slow without optimisation:
///
///     MLXR_SIEVE_STRIDE=fcobos/Qwen3.5-0.8B-oQ4-mtp swift test -c release -Xswiftc -enable-testing --filter SieveStride
///
/// The export must keep the `mtp.` tensors (the mlx-community ones drop them). The engine leaves
/// them on disk, so this suite reads the safetensors file itself.
///
/// - `MLXR_SIEVE_STRIDE_CHUNKS` (16): 512-token chunks per domain. The domains are this repository's
///   prose (README, CHANGELOG, docs) and code (Sources/MLXEngine), plus the model's own greedy answers
///   (thinking off) to the first eight of `SieveImporter.seedPrompts`.
/// - `MLXR_SIEVE_STRIDE_VARIANTS` ("post:e:0,pre:e:0"): head wirings to score, `hidden:order:window`.
///   hidden: `pre` (the backbone's residual before its final norm) or `post` (after it). order: `e`
///   (fc input is [embedding | hidden]) or `h`. window: positions the head's attention sees, 0 for all.
///
/// Head, as vLLM, llama.cpp and mlx-lm PR 990 describe it:
///   z_t = fc([pre_fc_norm_embedding(embed(x_{t+1})) | pre_fc_norm_hidden(h_t)]), one full-attention
///   decoder block over the z of every position, `mtp.norm`, the tied output projection → x̂_{t+2}.
/// Acceptance at t: x̂_{t+2} equals the backbone's own greedy token after x_{t+1}.
@Suite("Sieve H3: MTP draft acceptance", .serialized, .enabled(if: ProcessInfo.processInfo.environment["MLXR_SIEVE_STRIDE"] != nil))
struct SieveStrideTests {
    typealias Matrix = SieveImporter.Matrix
    static let env = ProcessInfo.processInfo.environment
    static let T = 512, skip = 16   // positions per chunk; the first `skip` have almost no context

    // MARK: Weights straight from the file

    /// One safetensors file, memory-mapped. Every range is checked against the file before use.
    struct Weights {
        let data: Data
        private(set) var infos: [String: (dtype: String, shape: [Int], start: Int, end: Int)] = [:]

        init(_ url: URL) throws {
            data = try Data(contentsOf: url, options: .alwaysMapped)
            guard data.count >= 8 else { throw EngineError.invalidFile("safetensors: too short") }
            let n = data.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
            guard n <= UInt64(data.count - 8) else { throw EngineError.invalidFile("safetensors: header length") }
            let hdr = Int(n), body = data.count - 8 - hdr
            guard let obj = try JSONSerialization.jsonObject(with: data.subdata(in: 8..<8 + hdr)) as? [String: Any] else {
                throw EngineError.invalidFile("safetensors: header")
            }
            for (name, v) in obj {
                guard let e = v as? [String: Any], let dtype = e["dtype"] as? String,
                      let shape = (e["shape"] as? [NSNumber])?.map(\.intValue), shape.count <= 4,
                      shape.allSatisfy({ $0 > 0 && $0 <= 1 << 24 }),
                      let offs = (e["data_offsets"] as? [NSNumber])?.map(\.intValue), offs.count == 2,
                      offs[0] >= 0, offs[0] <= offs[1], offs[1] <= body else { continue }
                infos[name] = (dtype, shape, 8 + hdr + offs[0], 8 + hdr + offs[1])
            }
        }

        func name(endingWith suffix: String, excluding: String? = nil) -> String? {
            infos.keys.filter { k in k.hasSuffix(suffix) && (excluding.map { !k.contains($0) } ?? true) }.sorted().first
        }

        func floats(_ name: String, count expected: Int? = nil) throws -> [Float] {
            guard let i = infos[name] else { throw EngineError.invalidFile("missing tensor \(name)") }
            let size = ["BF16": 2, "F16": 2, "F32": 4][i.dtype] ?? 0
            guard size > 0, (i.end - i.start) % size == 0 else { throw EngineError.unsupported("\(name): \(i.dtype)") }
            let count = (i.end - i.start) / size
            guard i.shape.reduce(1.0, { $0 * Double($1) }) == Double(count), expected == nil || expected == count else {
                throw EngineError.invalidFile("\(name): size \(count), shape \(i.shape)")
            }
            return data.withUnsafeBytes { raw in
                (0..<count).map { k in
                    switch i.dtype {
                    case "BF16": return Float(bitPattern: UInt32(raw.loadUnaligned(fromByteOffset: i.start + k * 2, as: UInt16.self)) << 16)
                    case "F16": return Tensor.halfToFloat(raw.loadUnaligned(fromByteOffset: i.start + k * 2, as: UInt16.self))
                    default: return raw.loadUnaligned(fromByteOffset: i.start + k * 4, as: Float.self)
                    }
                }
            }
        }

        /// A projection [out × in] as f32: float weights as they are, MLX affine-quantized ones
        /// (little-endian bit stream per row, 1 to 8 bits) as scale · code + bias per group.
        func linear(_ p: String, inDim: Int, outDim: Int) throws -> Matrix {
            guard let w = infos[p + ".weight"], w.shape.count == 2, w.shape[0] == outDim, inDim > 0 else {
                throw EngineError.invalidFile("\(p).weight: missing or not [\(outDim) × …]")
            }
            let m = Matrix(outDim, inDim)
            if w.dtype != "U32" {
                guard w.shape[1] == inDim else { throw EngineError.invalidFile("\(p).weight: shape \(w.shape)") }
                let f = try floats(p + ".weight", count: outDim * inDim)
                f.withUnsafeBufferPointer { m.p.update(from: $0.baseAddress!, count: f.count) }
                return m
            }
            let words = w.shape[1]
            guard let s = infos[p + ".scales"], s.shape.count == 2, s.shape[0] == outDim, inDim % s.shape[1] == 0,
                  (words * 32) % inDim == 0, (1...8).contains(words * 32 / inDim), w.end - w.start == outDim * words * 4 else {
                throw EngineError.invalidFile("\(p): quantized shapes")
            }
            let bits = words * 32 / inDim, groups = s.shape[1], g = inDim / groups, rowBytes = words * 4
            let scales = try floats(p + ".scales", count: outDim * groups), biases = try floats(p + ".biases", count: outDim * groups)
            let mask = UInt16((1 << bits) - 1)
            data.withUnsafeBytes { raw in
                for r in 0..<outDim {
                    let base = w.start + r * rowBytes, out = m.p + r * inDim
                    for j in 0..<inDim {
                        let bit = j * bits, at = base + bit / 8, sh = bit % 8
                        var v = UInt16(raw[at])
                        if sh + bits > 8 { v |= UInt16(raw[at + 1]) << 8 }   // inside the row: bits · inDim = 8 · rowBytes
                        out[j] = Float((v >> UInt16(sh)) & mask) * scales[r * groups + j / g] + biases[r * groups + j / g]
                    }
                }
            }
            return m
        }
    }

    // MARK: A Qwen 3.5 full-attention decoder block on the CPU

    struct Geometry {
        var H: Int, nH: Int, nKV: Int, D: Int, I: Int
        var rotDim: Int, invFreq: [Float], eps: Float, scale: Float
    }

    struct Block {
        let inNorm, postNorm, qNorm, kNorm: [Float]
        let q, k, v, o, gate, up, down: Matrix

        init(_ w: Weights, _ p: String, _ g: Geometry) throws {
            inNorm = try w.floats(p + ".input_layernorm.weight", count: g.H)
            postNorm = try w.floats(p + ".post_attention_layernorm.weight", count: g.H)
            qNorm = try w.floats(p + ".self_attn.q_norm.weight", count: g.D)
            kNorm = try w.floats(p + ".self_attn.k_norm.weight", count: g.D)
            q = try w.linear(p + ".self_attn.q_proj", inDim: g.H, outDim: 2 * g.nH * g.D)   // per head [q | gate]
            k = try w.linear(p + ".self_attn.k_proj", inDim: g.H, outDim: g.nKV * g.D)
            v = try w.linear(p + ".self_attn.v_proj", inDim: g.H, outDim: g.nKV * g.D)
            o = try w.linear(p + ".self_attn.o_proj", inDim: g.nH * g.D, outDim: g.H)
            gate = try w.linear(p + ".mlp.gate_proj", inDim: g.H, outDim: g.I)
            up = try w.linear(p + ".mlp.up_proj", inDim: g.H, outDim: g.I)
            down = try w.linear(p + ".mlp.down_proj", inDim: g.I, outDim: g.H)
        }
    }

    /// y [n × W.rows] = x [n × W.cols] · Wᵀ
    static func project(_ x: Matrix, _ n: Int, _ W: Matrix) -> Matrix {
        let y = Matrix(n, W.rows)
        BLAS.sgemm(BLAS.rowMajor, BLAS.noTrans, BLAS.trans, Int32(n), Int32(W.rows), Int32(W.cols), 1, x.p, Int32(x.cols), W.p, Int32(W.cols), 0, y.p, Int32(W.rows))
        return y
    }

    /// In place: v · rsqrt(mean(v²) + eps) · w over `dim` values.
    static func rmsNorm(_ v: UnsafeMutablePointer<Float>, _ w: [Float], dim: Int, eps: Float) {
        var ss: Float = 0
        for i in 0..<dim { ss += v[i] * v[i] }
        let inv = 1 / (ss / Float(dim) + eps).squareRoot()
        for i in 0..<dim { v[i] = v[i] * inv * w[i] }
    }

    static func normed(_ x: Matrix, _ n: Int, _ w: [Float], eps: Float) -> Matrix {
        let y = Matrix(n, x.cols)
        y.p.update(from: x.p, count: n * x.cols)
        for t in 0..<n { rmsNorm(y.p + t * x.cols, w, dim: x.cols, eps: eps) }
        return y
    }

    /// The block over rows 0..<n of `x` at positions 0..<n: pre-norm gated attention (per-head q/k
    /// RMSNorm, partial RoPE on the first rotDim dims in halves, grouped KV heads, causal,
    /// attention · sigmoid(gate)), then the SwiGLU MLP. `window` > 0 limits attention to that many
    /// most recent positions (the engine has no such limit: an ablation for the head).
    static func run(_ b: Block, _ g: Geometry, _ x: Matrix, _ n: Int, window: Int = 0) -> Matrix {
        let H = g.H, D = g.D, nH = g.nH, nKV = g.nKV, half = g.rotDim / 2
        let xn = normed(x, n, b.inNorm, eps: g.eps)
        let Q = project(xn, n, b.q), K = project(xn, n, b.k), V = project(xn, n, b.v)
        func post(_ v: UnsafeMutablePointer<Float>, _ w: [Float], pos: Int) {
            rmsNorm(v, w, dim: D, eps: g.eps)
            for i in 0..<half {
                let ang = Float(pos) * g.invFreq[i], c = cos(ang), s = sin(ang)
                let a = v[i], bb = v[i + half]
                v[i] = a * c - bb * s
                v[i + half] = bb * c + a * s
            }
        }
        for t in 0..<n {
            for h in 0..<nH { post(Q.p + t * Q.cols + h * 2 * D, b.qNorm, pos: t) }
            for h in 0..<nKV { post(K.p + t * K.cols + h * D, b.kNorm, pos: t) }
        }
        let A = Matrix(n, nH * D), S = Matrix(n, n)
        for h in 0..<nH {
            let kh = h / (nH / nKV)
            BLAS.sgemm(BLAS.rowMajor, BLAS.noTrans, BLAS.trans, Int32(n), Int32(n), Int32(D), g.scale,
                       Q.p + h * 2 * D, Int32(Q.cols), K.p + kh * D, Int32(K.cols), 0, S.p, Int32(n))
            for i in 0..<n {
                let row = S.p + i * n, lo = window > 0 ? max(0, i - window + 1) : 0
                var m = -Float.infinity
                for j in lo...i { m = max(m, row[j]) }
                var sum: Float = 0
                for j in lo...i { let e = exp(row[j] - m); row[j] = e; sum += e }
                for j in 0..<n { row[j] = (j >= lo && j <= i) ? row[j] / sum : 0 }
            }
            BLAS.sgemm(BLAS.rowMajor, BLAS.noTrans, BLAS.noTrans, Int32(n), Int32(D), Int32(n), 1,
                       S.p, Int32(n), V.p + kh * D, Int32(V.cols), 0, A.p + h * D, Int32(A.cols))
            for t in 0..<n {
                let a = A.p + t * A.cols + h * D, gate = Q.p + t * Q.cols + h * 2 * D + D
                for i in 0..<D { a[i] *= 1 / (1 + exp(-gate[i])) }
            }
        }
        let out = project(A, n, b.o)
        for i in 0..<(n * H) { out.p[i] += x.p[i] }
        let xn2 = normed(out, n, b.postNorm, eps: g.eps)
        let G = project(xn2, n, b.gate), U = project(xn2, n, b.up)
        for i in 0..<(n * g.I) { let v = G.p[i]; G.p[i] = v / (1 + exp(-v)) * U.p[i] }
        let Dn = project(G, n, b.down)
        for i in 0..<(n * H) { out.p[i] += Dn.p[i] }
        return out
    }

    /// Per row of `x` [n × d] (already normalised): the argmax of x · Wᵀ and its softmax probability.
    static func headStats(_ x: Matrix, _ n: Int, _ W: Matrix) -> (top: [Int32], prob: [Float]) {
        let V = W.rows, d = W.cols, B = 64
        let top = UnsafeMutablePointer<Int32>.allocate(capacity: n), prob = UnsafeMutablePointer<Float>.allocate(capacity: n)
        defer { top.deallocate(); prob.deallocate() }
        let Z = Matrix(B, V)
        var s0 = 0
        while s0 < n {
            let b = min(B, n - s0)
            BLAS.sgemm(BLAS.rowMajor, BLAS.noTrans, BLAS.trans, Int32(b), Int32(V), Int32(d), 1, x.p + s0 * d, Int32(d), W.p, Int32(d), 0, Z.p, Int32(V))
            let zp = Z.p, base = s0
            DispatchQueue.concurrentPerform(iterations: b) { i in
                let z = zp + i * V
                var m = z[0], am = 0
                for j in 1..<V where z[j] > m { m = z[j]; am = j }
                var sum: Float = 0   // terms below e^-25 of the best: under 1e-5 in total
                for j in 0..<V { let dz = z[j] - m; if dz > -25 { sum += exp(dz) } }
                top[base + i] = Int32(am); prob[base + i] = 1 / sum
            }
            s0 += b
        }
        return (Array(UnsafeBufferPointer(start: top, count: n)), Array(UnsafeBufferPointer(start: prob, count: n)))
    }

    static func copy(_ b: MTLBuffer, rows: Int, cols: Int) -> Matrix {
        let m = Matrix(rows, cols)
        m.p.update(from: b.contents().bindMemory(to: Float.self, capacity: rows * cols), count: rows * cols)
        return m
    }

    // MARK: Setup shared by the tests

    struct Setup {
        let t: Transformer, tok: Tokenizer, probe: SieveProbe, w: Weights, g: Geometry
        let backbone: String   // "….model." prefix of the backbone's tensors
        let mtp: String        // "….mtp." prefix of the head's tensors
        let fullLayers: [Int]
    }

    static func setup() throws -> Setup? {
        guard let repo = env["MLXR_SIEVE_STRIDE"], let dir = installedModel(repo) else { print("H3: model not installed"); return nil }
        let t = try Transformer(directory: dir, contextLength: 1024, sparse: nil)
        let c = t.config
        let full = t.layers.indices.filter { t.layers[$0].gdn == nil }
        guard c.attnOutputGate, let l0 = full.first, c.embedScale == nil, c.finalLogitSoftcap == 0, !c.ropeTraditional else {
            print("H3: not a Qwen 3.5 model"); return nil
        }
        let files = try WeightStore.shardFiles(in: dir)
        var found: (Weights, String, String)?
        for f in files {
            let w = try Weights(dir.appendingPathComponent(f))
            guard let fc = w.name(endingWith: "mtp.fc.weight"),
                  let l = w.name(endingWith: "layers.\(l0).self_attn.q_proj.weight", excluding: "mtp.") else { continue }
            found = (w, String(l.dropLast("layers.\(l0).self_attn.q_proj.weight".count)), String(fc.dropLast("fc.weight".count)))
        }
        guard let (w, backbone, mtp) = found else { print("H3: no mtp. tensors in \(files) (the export dropped the head)"); return nil }
        let spec = c.layers[l0]
        let g = Geometry(H: c.hiddenSize, nH: c.numHeads, nKV: spec.numKVHeads, D: spec.headDim, I: c.intermediateSize,
                         rotDim: spec.rotDim, invFreq: spec.invFreq, eps: c.rmsNormEps,
                         scale: c.ropeAttentionFactor * c.ropeAttentionFactor * c.queryScale)
        guard g.nH % g.nKV == 0, g.rotDim <= g.D, g.invFreq.count >= g.rotDim / 2 else { throw EngineError.invalidFile("attention geometry") }
        let probe = try SieveProbe(t)
        t.probe = probe
        return Setup(t: t, tok: try Tokenizer(directory: dir), probe: probe, w: w, g: g, backbone: backbone, mtp: mtp, fullLayers: full)
    }

    static let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    /// The repository's own text, by domain: nothing the model can have memorised.
    static func domainText(_ domain: String) throws -> String {
        let fm = FileManager.default
        func list(_ sub: String, _ ext: String) throws -> [URL] {
            try fm.contentsOfDirectory(atPath: root.appendingPathComponent(sub).path).filter { $0.hasSuffix(ext) }.sorted()
                .map { root.appendingPathComponent(sub).appendingPathComponent($0) }
        }
        let files = domain == "prose"
            ? [root.appendingPathComponent("README.md"), root.appendingPathComponent("CHANGELOG.md")] + (try list("docs", ".md"))
            : try list("Sources/MLXEngine", ".swift")
        return try files.map { try String(contentsOf: $0, encoding: .utf8) }.joined(separator: "\n\n")
    }

    static func chunks(_ ids: [Int32], _ n: Int) -> [[Int32]] {
        let n = min(n, ids.count / T)
        guard n > 0 else { return [] }
        let stride = (ids.count - T) / max(n - 1, 1)
        return (0..<n).map { Array(ids[($0 * stride)..<($0 * stride + T)]) }
    }

    /// One teacher-forced pass from position 0: the residual entering the final norm, and its output.
    static func states(_ s: Setup, _ ids: [Int32]) throws -> (pre: Matrix, post: Matrix, lastArgmax: Int32) {
        s.probe.captureLayer = s.t.layers.count
        let logits = try s.t.forward(ids, offset: 0)
        var am = 0
        for j in 1..<s.t.vocabSize where logits[j] > logits[am] { am = j }
        return (copy(s.probe.capture, rows: ids.count, cols: s.g.H), copy(s.t.xn, rows: ids.count, cols: s.g.H), Int32(am))
    }

    static func capture(_ s: Setup, _ ids: [Int32], layer: Int) throws -> Matrix {
        s.probe.captureLayer = layer
        _ = try s.t.forward(ids, offset: 0)
        return copy(s.probe.capture, rows: ids.count, cols: s.g.H)
    }

    /// Largest per-row ‖a − b‖ / ‖ref‖ where ref is b, or b − base when `base` is given.
    static func rowError(_ a: Matrix, _ b: Matrix, n: Int, base: Matrix? = nil) -> Double {
        var worst = 0.0
        for t in 0..<n {
            var e = 0.0, r = 0.0
            for i in 0..<a.cols {
                let k = t * a.cols + i
                let d = Double(a.p[k] - b.p[k]), ref = Double(b.p[k] - (base?.p[k] ?? 0))
                e += d * d; r += ref * ref
            }
            worst = max(worst, (e / max(r, 1e-30)).squareRoot())
        }
        return worst
    }

    // MARK: Validation

    /// The CPU block with a backbone full-attention layer's weights (read by the same file reader
    /// and dequantizer the head uses) against the engine: residual entering layer i → entering i + 1.
    @Test func blockMatchesEngine() throws {
        guard let s = try Self.setup() else { return }
        let ids = Array(s.tok.encode(try Self.domainText("prose")).map(Int32.init).prefix(Self.T))
        guard ids.count == Self.T else { print("H3: not enough text"); return }
        let n = ids.count
        // Embeddings: the tied output projection's rows are what enters layer 0.
        let W = try SieveImporter.dequantize(s.t.lmHead)
        let h0 = try Self.capture(s, ids, layer: 0)
        let e = Matrix(n, s.g.H)
        for t in 0..<n { (e.p + t * s.g.H).update(from: W.p + Int(ids[t]) * s.g.H, count: s.g.H) }
        let embedErr = Self.rowError(e, h0, n: n)
        print(String(format: "H3 validation: embedding rows from the dequantized output projection vs the residual entering layer 0: max relative row error %.2e", embedErr))
        #expect(embedErr < 1e-3)
        var worstDelta = 0.0
        for li in s.fullLayers {
            let block = try Block(s.w, "\(s.backbone)layers.\(li)", s.g)
            let x = try Self.capture(s, ids, layer: li), want = try Self.capture(s, ids, layer: li + 1)
            let got = Self.run(block, s.g, x, n)
            let rel = Self.rowError(got, want, n: n), relDelta = Self.rowError(got, want, n: n, base: x)
            worstDelta = max(worstDelta, relDelta)
            print(String(format: "H3 validation: CPU block as layer %d over %d positions: max row error %.2e of the output residual, %.2e of the layer's own update",
                         li, n, rel, relDelta))
        }
        // Final norm: the capture at the layer count is what the norm reads.
        let st = try Self.states(s, ids)
        let normW = try s.w.floats("\(s.backbone)norm.weight", count: s.g.H)
        let normErr = Self.rowError(Self.normed(st.pre, n, normW, eps: s.g.eps), st.post, n: n)
        print(String(format: "H3 validation: final norm of the captured pre-norm residual vs the engine's: max relative row error %.2e", normErr))
        #expect(normErr < 1e-3)
        // K and V rows are f16 in the engine and f32 here.
        #expect(worstDelta < 2e-2, "the CPU block does not reproduce the engine's full-attention layers")
    }

    // MARK: Measurement

    struct Variant {
        var preNorm: Bool, embeddingFirst: Bool, window: Int
        var label: String { "\(preNorm ? "pre-norm" : "post-norm") h, [\(embeddingFirst ? "emb | h" : "h | emb")], \(window == 0 ? "full context" : "window \(window)")" }
        static func parse(_ s: String) -> [Variant] {
            s.split(separator: ",").compactMap { v in
                let p = v.split(separator: ":").map(String.init)
                guard p.count == 3, let w = Int(p[2]), w >= 0 else { return nil }
                return Variant(preNorm: p[0] == "pre", embeddingFirst: p[1] == "e", window: w)
            }
        }
    }

    /// One drafting position t: the head saw h_t and x_{t+1}, the target is the backbone's token after x_{t+1}.
    struct Rec {
        var domain: Int
        var onPolicy: Bool      // x_{t+1} is the backbone's own greedy token at t
        var p1: Float           // backbone's top-1 probability at t (the token it would emit as x_{t+1})
        var p2: Float           // backbone's top-1 probability at t+1 (the target's own confidence)
        var backboneHitsText: Bool
        var accept: [Bool]      // per variant: draft == target
        var pm: [Float]         // per variant: the head's top-1 probability
        var draftHitsText: [Bool]
    }

    /// Greedy decode (argmax, no penalty) of up to `limit` tokens: prompt + answer, and the prompt's length.
    static func generate(_ s: Setup, prompt: String, limit: Int) throws -> (ids: [Int32], promptLen: Int) {
        let t = s.t, tok = s.tok
        let tpl = ChatTemplate(template: tok.chatTemplate, modelType: t.config.modelType, bosToken: tok.bosToken, toolTemplate: tok.toolUseTemplate)
        var stops = Set(t.config.eosTokenIds)
        if let e = tok.eosToken, let id = tok.tokenId(e) { stops.insert(id) }
        for st in tpl.stopTokens { if let id = tok.tokenId(st) { stops.insert(id) } }
        let r = tpl.renderMarked([ChatMessage(role: .user, content: prompt)], enableThinking: false)
        var ids = tok.encode(r.text, plainRanges: r.content).map(Int32.init)
        let promptLen = ids.count
        guard promptLen > 0, promptLen + 8 < T else { throw EngineError.invalidOutput("prompt length") }
        var logits = try t.forward(ids, offset: 0)
        while ids.count < min(T, promptLen + limit) {
            var am = 0
            for j in 1..<t.vocabSize where logits[j] > logits[am] { am = j }
            ids.append(Int32(am))
            if stops.contains(am) || ids.count == T { break }
            logits = try t.forward([Int32(am)], offset: ids.count - 1)
        }
        return (ids, promptLen)
    }

    @Test func draftAcceptance() throws {
        guard let s = try Self.setup() else { return }
        let env = Self.env, g = s.g, H = g.H
        let nChunks = Int(env["MLXR_SIEVE_STRIDE_CHUNKS"] ?? "") ?? 16
        let variants = Variant.parse(env["MLXR_SIEVE_STRIDE_VARIANTS"] ?? "post:e:0,pre:e:0")
        guard !variants.isEmpty else { print("H3: no variants"); return }
        let W = try SieveImporter.dequantize(s.t.lmHead)   // tied: also the embedding table
        guard W.cols == H, W.rows == s.t.vocabSize else { throw EngineError.invalidFile("head shape") }
        let block = try Block(s.w, "\(s.mtp)layers.0", g)
        let fc = try s.w.linear("\(s.mtp)fc", inDim: 2 * H, outDim: H)
        let normE = try s.w.floats("\(s.mtp)pre_fc_norm_embedding.weight", count: H)
        let normH = try s.w.floats("\(s.mtp)pre_fc_norm_hidden.weight", count: H)
        let normOut = try s.w.floats("\(s.mtp)norm.weight", count: H)
        func mean(_ v: [Float]) -> Float { v.reduce(0, +) / Float(v.count) }
        print(String(format: "H3 %@: head tensors under %@; norm weight means: pre_fc_norm_embedding %.3f, pre_fc_norm_hidden %.3f, mtp.norm %.3f, backbone final norm %.3f",
                     env["MLXR_SIEVE_STRIDE"]! as NSString, s.mtp as NSString, mean(normE), mean(normH), mean(normOut),
                     mean(try s.w.floats("\(s.backbone)norm.weight", count: H))))

        // Sequences: (domain, tokens, first drafting position, whether the targets are the tokens themselves).
        let domains = ["prose", "code", "generated"]
        var seqs: [(domain: Int, ids: [Int32], first: Int, own: Bool)] = []
        for (di, d) in domains.prefix(2).enumerated() {
            let ids = s.tok.encode(try Self.domainText(d)).map(Int32.init)
            let cs = Self.chunks(ids, nChunks)
            print("H3 \(d): \(ids.count) tokens, \(cs.count) chunks of \(Self.T)")
            seqs += cs.map { (di, $0, Self.skip, false) }
        }
        for p in SieveImporter.seedPrompts.prefix(8) {
            let r = try Self.generate(s, prompt: p, limit: 240)
            let text = s.tok.decode(r.ids[r.promptLen...].map(Int.init)).replacingOccurrences(of: "\n", with: " ")
            print("H3 generated \(r.ids.count - r.promptLen) tokens after a \(r.promptLen)-token prompt: \(text.prefix(100))…")
            // x_{t+1} must be a generated token: t + 1 >= promptLen.
            seqs.append((2, r.ids, r.promptLen - 1, true))
        }

        var recs: [Rec] = []
        var lastMismatch = 0, replayMismatch = 0, replayCount = 0
        for (si, seq) in seqs.enumerated() {
            let ids = seq.ids, n = ids.count
            let st = try Self.states(s, ids)
            let bb = Self.headStats(st.post, n, W)   // bb.top[t]: the backbone's greedy token for position t + 1
            if bb.top[n - 1] != st.lastArgmax { lastMismatch += 1 }
            var drafts: [(top: [Int32], prob: [Float])] = []
            let m = n - 1   // head positions 0..<n-1: each needs the next token
            for v in variants {
                let hidden = v.preNorm ? st.pre : st.post
                let cat = Matrix(m, 2 * H)
                for t in 0..<m {
                    let eAt = cat.p + t * 2 * H + (v.embeddingFirst ? 0 : H), hAt = cat.p + t * 2 * H + (v.embeddingFirst ? H : 0)
                    eAt.update(from: W.p + Int(ids[t + 1]) * H, count: H)
                    hAt.update(from: hidden.p + t * H, count: H)
                    Self.rmsNorm(eAt, normE, dim: H, eps: g.eps)
                    Self.rmsNorm(hAt, normH, dim: H, eps: g.eps)
                }
                let z = Self.project(cat, m, fc)
                let out = Self.run(block, g, z, m, window: v.window)
                drafts.append(Self.headStats(Self.normed(out, m, normOut, eps: g.eps), m, W))
            }
            // t drafts x_{t+2}; the backbone's own choice for it is bb.top[t + 1].
            let before = recs.count
            if seq.first <= n - 3 {
                for t in seq.first...(n - 3) {
                    if seq.own {
                        replayCount += 1
                        if bb.top[t + 1] != ids[t + 2] { replayMismatch += 1 }
                    }
                    let target = seq.own ? ids[t + 2] : bb.top[t + 1]
                    recs.append(Rec(domain: seq.domain, onPolicy: bb.top[t] == ids[t + 1], p1: bb.prob[t], p2: bb.prob[t + 1],
                                    backboneHitsText: bb.top[t + 1] == ids[t + 2],
                                    accept: drafts.map { $0.top[t] == target }, pm: drafts.map { $0.prob[t] },
                                    draftHitsText: drafts.map { $0.top[t] == ids[t + 2] }))
                }
            }
            if seq.own, recs.count > before {
                let mine = recs[before...]
                print(String(format: "H3 generated sequence %d: %d drafts, acceptance %.3f (%@)", si, mine.count,
                             Double(mine.filter { $0.accept[0] }.count) / Double(mine.count), variants[0].label as NSString))
            }
            if si % 8 == 7 { print("H3 … \(si + 1) of \(seqs.count) sequences") }
        }
        print("H3 checks: CPU head argmax differs from the engine's logits argmax on the last row of \(lastMismatch) of \(seqs.count) sequences; "
              + "teacher-forced replay disagrees with the greedy decode on \(replayMismatch) of \(replayCount) generated tokens")

        func rate(_ r: [Rec], _ f: (Rec) -> Bool) -> String {
            r.isEmpty ? "   n/a" : String(format: "%.3f", Double(r.filter(f).count) / Double(r.count))
        }
        let groups: [(String, (Rec) -> Bool)] = [
            ("prose", { $0.domain == 0 }), ("code", { $0.domain == 1 }), ("prose+code", { $0.domain < 2 }),
            ("generated", { $0.domain == 2 }),
        ]
        let bins: [(Float, Float)] = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
        for (vi, v) in variants.enumerated() {
            print("H3 ===== \(v.label) =====")
            for (name, pick) in groups {
                let r = recs.filter(pick)
                guard !r.isEmpty else { continue }
                let on = r.filter(\.onPolicy)
                print("H3 \(name): n \(r.count); acceptance \(rate(r) { $0.accept[vi] }); where x_{t+1} is the backbone's own greedy token (n \(on.count)) \(rate(on) { $0.accept[vi] }); "
                      + "draft == the text's x_{t+2} \(rate(r) { $0.draftHitsText[vi] }) (backbone's own next-token accuracy there \(rate(r) { $0.backboneHitsText }))")
                func table(_ title: String, _ key: (Rec) -> Float) {
                    let cells = bins.map { b -> String in
                        let sub = r.filter { key($0) >= b.0 && key($0) < b.1 }
                        return String(format: "[%.1f,%.1f) share %.3f acc %@", b.0, min(b.1, 1), Double(sub.count) / Double(r.count), rate(sub) { $0.accept[vi] } as NSString)
                    }
                    print("H3   by \(title): " + cells.joined(separator: " | "))
                }
                table("backbone top-1 probability at t (the token it emits as x_{t+1})", \.p1)
                table("backbone top-1 probability at t+1 (the target)", \.p2)
                let cov = [Float(0), 0.3, 0.5, 0.7, 0.9].map { tau -> String in
                    let sub = r.filter { $0.pm[vi] >= tau }
                    return String(format: "p >= %.1f: coverage %.3f acc %@", tau, Double(sub.count) / Double(r.count), rate(sub) { $0.accept[vi] } as NSString)
                }
                print("H3   drafting only when the head's top-1 probability p clears a threshold: " + cov.joined(separator: " | "))
            }
        }
        let main = recs.filter { $0.domain == 2 }
        if !main.isEmpty {
            let a = Double(main.filter { $0.accept[0] }.count) / Double(main.count)
            print(String(format: "H3 gate (acceptance >= 0.6 on the model's own greedy text, %@): %.3f over %d drafts -> %@",
                         variants[0].label as NSString, a, main.count, a >= 0.6 ? "PASSED" : "FAILED"))
        }
    }
}
