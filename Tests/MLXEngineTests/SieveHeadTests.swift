import Foundation
import Metal
import Testing
@testable import MLXEngine

/// Phase 0 of docs/sieve-architecture.md, hypothesis H1: is Head's certificate tight on real hidden
/// states? It measures and reports; the one thing it asserts is soundness (no dense top-k row
/// outside the certified set). Opt-in, and slow without optimisation:
///
///     MLXR_SIEVE_H1=mlx-community/Llama-3.2-1B-Instruct-4bit MLXR_SIEVE_TEXT=prose.txt,code.txt \
///         swift test -c release -Xswiftc -enable-testing --filter SieveHead
///
/// - `MLXR_SIEVE_TEXT`: text files, one per domain. Chunks of 512 tokens are spread over each file;
///   every fifth chunk is held out. Use text the model cannot have memorised.
/// - `MLXR_SIEVE_CHUNKS` (50): chunks in total. `MLXR_SIEVE_FIT=first` fits on the first file only,
///   so the other files measure a domain the fit never saw.
/// - `MLXR_SIEVE_EPS` (0): slack added to every bound, in logits.
/// - `MLXR_SIEVE_WHITEN=1`: quantize the sketch in coordinates scaled by each direction's standard
///   deviation (the bound's first term becomes q_i · ‖x_r / σ‖ with q_i measured in those units).
///
/// The hidden states are the engine's own (the final norm's output in `xn` after a forward pass).
/// Everything else is CPU arithmetic: the head dequantized to f32, the fit in double precision.
@Suite("Sieve H1: head certificate", .serialized, .enabled(if: ProcessInfo.processInfo.environment["MLXR_SIEVE_H1"] != nil))
struct SieveHeadTests {
    static let env = ProcessInfo.processInfo.environment
    static let T = 512, skip = 16   // positions per chunk; the first `skip` have almost no context

    /// A row-major f32 matrix in its own allocation (the head is over a gigabyte).
    final class Matrix {
        let rows: Int, cols: Int
        let p: UnsafeMutablePointer<Float>
        init(_ rows: Int, _ cols: Int) {
            self.rows = rows; self.cols = cols
            p = .allocate(capacity: max(rows * cols, 1))
            p.initialize(repeating: 0, count: max(rows * cols, 1))
        }
        deinit { p.deallocate() }
        func row(_ r: Int) -> UnsafeMutablePointer<Float> { p + r * cols }
    }

    struct Stats {
        var v: [Double] = []
        mutating func add(_ x: Double) { v.append(x) }
        var mean: Double { v.isEmpty ? 0 : v.reduce(0, +) / Double(v.count) }
        func pct(_ q: Double) -> Double {
            guard !v.isEmpty else { return 0 }
            let s = v.sorted()
            return s[min(s.count - 1, Int((q * Double(s.count)).rounded(.down)))]
        }
        var median: Double { pct(0.5) }
        var max: Double { v.max() ?? 0 }
    }

    /// The output projection as f32: w = scale · q + bias per quantization group (MLX affine, 4-bit
    /// codes packed low nibble first into little-endian words).
    static func dequantize(_ l: Linear) throws -> Matrix {
        guard l.bits == 4, let s = l.scales, let b = l.biases, l.weight.dtype == .u32 else {
            throw EngineError.unsupported("H1 reads 4-bit heads only (\(l.description))")
        }
        let V = l.outDim, d = l.inDim, g = l.groupSize, groups = d / g
        let scales = try s.floats(), biases = try b.floats()
        guard scales.count == V * groups, biases.count == V * groups, l.weight.nbytes == V * d / 2,
              l.weight.nbytes <= l.weight.buffer.length - l.weight.offset else { throw EngineError.invalidFile("head shapes") }
        let w = Matrix(V, d)
        let src = UnsafeRawPointer(l.weight.buffer.contents() + l.weight.offset)
        for r in 0..<V {
            let out = w.row(r), base = r * d / 2
            for j in 0..<d {
                let byte = src.load(fromByteOffset: base + j / 2, as: UInt8.self)
                let q = j & 1 == 0 ? byte & 0xF : byte >> 4
                out[j] = Float(q) * scales[r * groups + j / g] + biases[r * groups + j / g]
            }
        }
        return w
    }

    /// Final normalised hidden states of the chunks of one file: (states [n × d], held-out flags per state).
    /// Also the largest |CPU logit − GPU logit| over each chunk's last position.
    static func collect(_ t: Transformer, _ tok: Tokenizer, file: String, chunks: Int, head: Matrix) throws -> (x: [Float], heldOut: [Bool], gap: Float) {
        let text = try String(contentsOfFile: file, encoding: .utf8)
        let ids = tok.encode(text).map(Int32.init)
        let d = t.config.hiddenSize, V = head.rows
        let n = min(chunks, ids.count / T)
        guard n > 0 else { throw EngineError.invalidOutput("\(file): fewer than \(T) tokens") }
        let stride = (ids.count - T) / max(n - 1, 1)
        var x: [Float] = [], held: [Bool] = []
        x.reserveCapacity(n * (T - skip) * d)
        var gap: Float = 0
        var z = [Float](repeating: 0, count: V)
        for c in 0..<n {
            let start = c * stride
            let logits = try t.forward(Array(ids[start..<start + T]), offset: 0)
            let rows = t.xn.contents().bindMemory(to: Float.self, capacity: T * d)
            x.append(contentsOf: UnsafeBufferPointer(start: rows + skip * d, count: (T - skip) * d))
            held.append(contentsOf: [Bool](repeating: c % 5 == 4, count: T - skip))
            // The dequantized head against the engine's own logits for the last position.
            BLAS.sgemv(BLAS.rowMajor, BLAS.noTrans, Int32(V), Int32(d), 1, head.p, Int32(d), rows + (T - 1) * d, 1, 0, &z, 1)
            for i in 0..<V { gap = max(gap, abs(z[i] - logits[i])) }
        }
        return (x, held, gap)
    }

    /// Mean and the eigenvectors of the covariance of `x` [n × d], by descending eigenvalue (double precision).
    static func fit(_ x: UnsafePointer<Float>, n: Int, d: Int) -> (mean: [Double], values: [Double], vectors: [Double]) {
        var mean = [Double](repeating: 0, count: d)
        for i in 0..<n { for j in 0..<d { mean[j] += Double(x[i * d + j]) } }
        for j in 0..<d { mean[j] /= Double(n) }
        var c = [Double](repeating: 0, count: n * d)
        for i in 0..<n { for j in 0..<d { c[i * d + j] = Double(x[i * d + j]) - mean[j] } }
        var cov = [Double](repeating: 0, count: d * d)
        BLAS.dsyrk(BLAS.rowMajor, BLAS.upper, BLAS.trans, Int32(d), Int32(n), 1 / Double(n), c, Int32(d), 0, &cov, Int32(d))
        c = []
        // Column-major for LAPACK: the row-major upper triangle is its lower triangle.
        var jobz = Int8(UInt8(ascii: "V")), uplo = Int8(UInt8(ascii: "L"))
        var nn = Int32(d), lda = Int32(d), info = Int32(0)
        var w = [Double](repeating: 0, count: d)
        var lwork = Int32(-1), liwork = Int32(-1)
        var wq = [Double](repeating: 0, count: 1), iwq = [Int32](repeating: 0, count: 1)
        _ = BLAS.dsyevd(&jobz, &uplo, &nn, &cov, &lda, &w, &wq, &lwork, &iwq, &liwork, &info)
        lwork = Int32(wq[0]); liwork = iwq[0]
        var work = [Double](repeating: 0, count: Int(lwork)), iwork = [Int32](repeating: 0, count: Int(liwork))
        _ = BLAS.dsyevd(&jobz, &uplo, &nn, &cov, &lda, &w, &work, &lwork, &iwork, &liwork, &info)
        precondition(info == 0, "dsyevd failed")
        // Ascending eigenvalues, eigenvector k in column k (column-major): reorder to descending rows.
        var vec = [Double](repeating: 0, count: d * d)
        for k in 0..<d { for j in 0..<d { vec[k * d + j] = cov[(d - 1 - k) * d + j] } }
        return (mean, w.reversed(), vec)
    }

    @Test func certificateTightness() throws {
        let env = Self.env
        guard let repo = env["MLXR_SIEVE_H1"], let dir = installedModel(repo) else { print("H1: model not installed"); return }
        guard let files = env["MLXR_SIEVE_TEXT"]?.split(separator: ",").map(String.init), !files.isEmpty else {
            print("H1: set MLXR_SIEVE_TEXT to one or more text files"); return
        }
        let chunks = Int(env["MLXR_SIEVE_CHUNKS"] ?? "") ?? 50
        let fitFirst = env["MLXR_SIEVE_FIT"] == "first"
        let eps = Float(env["MLXR_SIEVE_EPS"] ?? "") ?? 0
        let whiten = env["MLXR_SIEVE_WHITEN"] == "1"
        let t = try Transformer(directory: dir, contextLength: 1024, sparse: nil)
        let tok = try Tokenizer(directory: dir)
        let d = t.config.hiddenSize, V = t.vocabSize
        guard t.config.finalLogitSoftcap == 0 else { print("H1: final logit soft-cap not handled here"); return }
        let W = try Self.dequantize(t.lmHead)
        let rowBytes = Double(d) * 0.5 + Double(d / t.lmHead.groupSize) * Double(2 * (t.lmHead.scales?.dtype.size ?? 2))
        let denseBytes = Double(V) * rowBytes
        print("H1 \(repo): d \(d), V \(V), dense head \(String(format: "%.1f", denseBytes / 1e6)) MB, eps \(eps)\(whiten ? ", whitened sketch" : "")")

        // 1. Hidden states.
        var fitX: [Float] = [], testX: [Float] = [], testDomain: [Int] = []
        var gap: Float = 0
        for (fi, f) in files.enumerated() {
            let c = try Self.collect(t, tok, file: f, chunks: chunks / files.count, head: W)
            gap = max(gap, c.gap)
            let n = c.heldOut.count
            var nFit = 0, nTest = 0
            c.x.withUnsafeBufferPointer { p in
                for i in 0..<n {
                    let row = UnsafeBufferPointer(rebasing: p[(i * d)..<((i + 1) * d)])
                    if c.heldOut[i] || (fitFirst && fi > 0) { testX.append(contentsOf: row); testDomain.append(fi); nTest += 1 }
                    else { fitX.append(contentsOf: row); nFit += 1 }
                }
            }
            print("H1 states from \((f as NSString).lastPathComponent): \(n) (\(nFit) fitted, \(nTest) held out)")
        }
        let nFit = fitX.count / d, nTest = testX.count / d
        print(String(format: "H1 max |dequantized head logit - engine logit| over the chunks' last positions: %.4f", gap))

        // Dense reference per held-out state: the top 64 logits and their rows.
        let B = 32, K = 64
        var denseTop = [Int32](repeating: 0, count: nTest * K)
        var denseGap = Stats(), denseGap64 = Stats()
        var Z = [Float](repeating: 0, count: B * V)
        testX.withUnsafeBufferPointer { xp in
            var s0 = 0
            while s0 < nTest {
                let b = min(B, nTest - s0)
                BLAS.sgemm(BLAS.rowMajor, BLAS.noTrans, BLAS.trans, Int32(b), Int32(V), Int32(d), 1, xp.baseAddress! + s0 * d, Int32(d), W.p, Int32(d), 0, &Z, Int32(V))
                Z.withUnsafeBufferPointer { zp in
                    for i in 0..<b {
                        let top = Self.topK(zp.baseAddress! + i * V, V, K)
                        for k in 0..<K { denseTop[(s0 + i) * K + k] = top[k].row }
                        denseGap.add(Double(top[0].value - top[1].value))
                        denseGap64.add(Double(top[0].value - top[K - 1].value))
                    }
                }
                s0 += b
            }
        }
        print(String(format: "H1 dense logits on held-out states: top-1 minus top-2 median %.2f (5th pct %.3f); top-1 minus 64th median %.2f",
                     denseGap.median, denseGap.pct(0.05), denseGap64.median))

        // 2. Fit.
        let (mu, lambda, vec) = fitX.withUnsafeBufferPointer { Self.fit($0.baseAddress!, n: nFit, d: d) }
        let totalVar = lambda.reduce(0, +)
        let muF = mu.map(Float.init)
        // Prior p = W μ, and centred held-out states.
        var prior = [Float](repeating: 0, count: V)
        BLAS.sgemv(BLAS.rowMajor, BLAS.noTrans, Int32(V), Int32(d), 1, W.p, Int32(d), muF, 1, 0, &prior, 1)
        var xc = testX
        for i in 0..<nTest { for j in 0..<d { xc[i * d + j] -= muF[j] } }
        var wNorm = Stats()
        for r in stride(from: 0, to: V, by: 97) { wNorm.add(Double(BLAS.snrm2(Int32(d), W.row(r), 1))) }
        var xNorm = Stats()
        for i in stride(from: 0, to: nTest, by: 7) { xc.withUnsafeBufferPointer { xNorm.add(Double(BLAS.snrm2(Int32(d), $0.baseAddress! + i * d, 1))) } }
        print(String(format: "H1 fit on %d states: |mu| %.2f, mean |x - mu| %.2f, mean |w| %.3f", nFit, sqrt(mu.reduce(0) { $0 + $1 * $1 }), xNorm.mean, wNorm.mean))

        var gateBest = (frac: Double.infinity, label: "")
        for r in [d / 16, d / 8, d / 4] {
            let Q = Matrix(r, d)   // rows orthonormal
            for k in 0..<r { for j in 0..<d { Q.row(k)[j] = Float(vec[k * d + j]) } }
            let captured = lambda[0..<r].reduce(0, +) / totalVar
            // Sketch S = W Qᵀ [V × r] and the unseen norm n_i = |w_i − Q s_i|, in row blocks.
            let S = Matrix(V, r)
            var unseen = [Float](repeating: 0, count: V)
            let blockRows = 4096
            let R = Matrix(blockRows, d)
            var r0 = 0
            while r0 < V {
                let nb = min(blockRows, V - r0)
                BLAS.sgemm(BLAS.rowMajor, BLAS.noTrans, BLAS.trans, Int32(nb), Int32(r), Int32(d), 1, W.row(r0), Int32(d), Q.p, Int32(d), 0, S.row(r0), Int32(r))
                memcpy(R.p, W.row(r0), nb * d * 4)
                BLAS.sgemm(BLAS.rowMajor, BLAS.noTrans, BLAS.noTrans, Int32(nb), Int32(d), Int32(r), -1, S.row(r0), Int32(r), Q.p, Int32(d), 1, R.p, Int32(d))
                for i in 0..<nb { unseen[r0 + i] = BLAS.snrm2(Int32(d), R.row(i), 1) * (1 + 1e-5) }
                r0 += nb
            }
            // Held-out states: x_r = Q x̃, ν = |x_r|, ρ = |x̃ − Qᵀ x_r|.
            let Xr = Matrix(nTest, r)
            var nu = [Float](repeating: 0, count: nTest), rho = [Float](repeating: 0, count: nTest)
            xc.withUnsafeBufferPointer { xp in
                BLAS.sgemm(BLAS.rowMajor, BLAS.noTrans, BLAS.trans, Int32(nTest), Int32(r), Int32(d), 1, xp.baseAddress!, Int32(d), Q.p, Int32(d), 0, Xr.p, Int32(r))
                var res = [Float](repeating: 0, count: d)
                for i in 0..<nTest {
                    memcpy(&res, xp.baseAddress! + i * d, d * 4)
                    BLAS.sgemv(BLAS.rowMajor, BLAS.trans, Int32(r), Int32(d), -1, Q.p, Int32(d), Xr.row(i), 1, 1, &res, 1)
                    rho[i] = BLAS.snrm2(Int32(d), res, 1) * (1 + 1e-5)
                    nu[i] = BLAS.snrm2(Int32(r), Xr.row(i), 1) * (1 + 1e-5)
                }
            }
            // Whitening: the sketch stores s · σ and the state is divided by σ, direction by direction.
            let sigma: [Float] = (0..<r).map { whiten ? Float(sqrt(max(lambda[$0], 1e-12))) : 1 }
            var nuW = nu
            if whiten {
                for i in 0..<nTest {
                    var a: Float = 0
                    for k in 0..<r { let v = Xr.row(i)[k] / sigma[k]; a += v * v }
                    nuW[i] = a.squareRoot() * (1 + 1e-5)
                }
            }
            let meanNu = Double(nu.reduce(0, +)) / Double(nTest), meanRho = Double(rho.reduce(0, +)) / Double(nTest)
            let meanUnseen = Double(unseen.reduce(0, +)) / Double(V)
            print(String(format: "H1 r %d (d/%d): variance captured %.1f%%, mean nu %.2f, mean rho %.2f, mean unseen norm n %.3f, so n*rho is about %.2f logits",
                         r, d / r, captured * 100, meanNu, meanRho, meanUnseen, meanUnseen * meanRho))

            for bits in [0, 8, 4] {
                // Quantized sketch ŝ (dequantized values) and its error norm q_i. bits 0: unquantized.
                let group = r % 64 == 0 ? 64 : 32
                var qerr = [Float](repeating: 0, count: V)
                let Sq: Matrix
                if bits == 0 { Sq = S } else {
                    Sq = Matrix(V, r)
                    let levels = Float((1 << bits) - 1)
                    for i in 0..<V {
                        let s = S.row(i), o = Sq.row(i)
                        var e2: Double = 0
                        var g0 = 0
                        while g0 < r {
                            let g1 = min(g0 + group, r)
                            var lo = Float.infinity, hi = -Float.infinity
                            for k in g0..<g1 { let v = s[k] * sigma[k]; lo = min(lo, v); hi = max(hi, v) }
                            let scale = hi > lo ? (hi - lo) / levels : 1
                            for k in g0..<g1 {
                                let v = s[k] * sigma[k]
                                let dq = lo + scale * ((v - lo) / scale).rounded()
                                o[k] = dq / sigma[k]
                                e2 += Double(dq - v) * Double(dq - v)   // in whitened units when whitening
                            }
                            g0 = g1
                        }
                        qerr[i] = Float(e2.squareRoot()) * (1 + 1e-5)
                    }
                }
                // Bytes every evaluation reads: the sketch (codes, an f16 scale and bias per group), and
                // per row the prior (f32) and the two error norms (f16).
                let sketchRow = bits == 0 ? Double(4 * r) : Double(r * bits) / 8 + Double((r + group - 1) / group * 4)
                let screenBytes = Double(V) * (sketchRow + 4 + 4)

                var rowsStat = [Stats](repeating: Stats(), count: 3), blocksStat = [Stats](repeating: Stats(), count: 3)
                var byDomain = [[Stats]](repeating: [Stats](repeating: Stats(), count: 3), count: files.count)
                var violations = [Int](repeating: 0, count: 3)
                var deficit: Float = 0
                var qTerm = Stats(), nTerm = Stats(), rank = Stats(), rowRank = Stats()
                let ks = [1, 2, 64]
                var Zh = [Float](repeating: 0, count: B * V)
                var u = [Float](repeating: 0, count: V)
                var s0 = 0
                while s0 < nTest {
                    let b = min(B, nTest - s0)
                    BLAS.sgemm(BLAS.rowMajor, BLAS.noTrans, BLAS.trans, Int32(b), Int32(V), Int32(r), 1, Xr.row(s0), Int32(r), Sq.p, Int32(r), 0, &Zh, Int32(V))
                    for i in 0..<b {
                        let si = s0 + i
                        let a = (bits == 0 ? 0 : nuW[si]), c = rho[si]
                        // Lower and upper bounds; the 64 largest lower bounds give θ for k = 1, 2, 64.
                        var top = [Float](repeating: -.infinity, count: K)   // ascending
                        var blockBest = [Float](repeating: -.infinity, count: (V + 63) / 64)
                        Zh.withUnsafeMutableBufferPointer { zp in
                            let z = zp.baseAddress! + i * V
                            for row in 0..<V {
                                let zh = z[row] + prior[row]
                                let bound = qerr[row] * a + unseen[row] * c + eps
                                let l = zh - bound
                                u[row] = zh + bound
                                z[row] = zh
                                if zh > blockBest[row >> 6] { blockBest[row >> 6] = zh }
                                if l > top[0] {
                                    var j = 0
                                    while j + 1 < K, top[j + 1] < l { top[j] = top[j + 1]; j += 1 }
                                    top[j] = l
                                }
                            }
                        }
                        let theta = [top[K - 1], top[K - 2], top[0]]
                        var rows = [0, 0, 0], blocks = [0, 0, 0]
                        var row = 0
                        while row < V {
                            let end = min(row + 64, V)
                            var m = -Float.infinity
                            var c0 = 0, c1 = 0, c2 = 0
                            for j in row..<end {
                                let v = u[j]
                                if v > m { m = v }
                                if v >= theta[0] { c0 += 1 }
                                if v >= theta[1] { c1 += 1 }
                                if v >= theta[2] { c2 += 1 }
                            }
                            rows[0] += c0; rows[1] += c1; rows[2] += c2
                            if c0 > 0 { blocks[0] += 1 }
                            if c1 > 0 { blocks[1] += 1 }
                            if c2 > 0 { blocks[2] += 1 }
                            row = end
                        }
                        for (j, k) in ks.enumerated() {
                            rowsStat[j].add(Double(rows[j])); blocksStat[j].add(Double(blocks[j]))
                            byDomain[testDomain[si]][j].add(Double(rows[j]))
                            // Soundness: every dense top-k row is certified.
                            for m in 0..<k {
                                let dr = Int(denseTop[si * K + m])
                                if u[dr] < theta[j] { violations[j] += 1; deficit = max(deficit, theta[j] - u[dr]) }
                            }
                        }
                        let am = Int(denseTop[si * K])
                        qTerm.add(Double(qerr[am] * a)); nTerm.add(Double(unseen[am] * c))
                        // Uncertified variant: blocks ordered by their best screen logit; where is the dense argmax's block?
                        let mine = blockBest[am >> 6]
                        var ahead = 0
                        for v in blockBest where v > mine { ahead += 1 }
                        rank.add(Double(ahead + 1))
                        // The same by row: how many rows the screen puts at or above the dense argmax.
                        var rowsAhead = 0
                        Zh.withUnsafeBufferPointer { zp in
                            let z = zp.baseAddress! + i * V, mineRow = z[am]
                            for row in 0..<V where z[row] > mineRow { rowsAhead += 1 }
                        }
                        rowRank.add(Double(rowsAhead + 1))
                    }
                    s0 += b
                }
                let label = bits == 0 ? "unquantized" : "\(bits)-bit g\(group)"
                for (j, k) in ks.enumerated() {
                    let expected = screenBytes + blocksStat[j].mean * 64 * rowBytes
                    let frac = expected / denseBytes
                    print(String(format: "H1 r %d %@ k %d: certified rows median %.0f (%.2f%% of V) mean %.0f p95 %.0f; blocks median %.0f mean %.0f p95 %.0f of %d; bytes %.1f MB screen + %.1f MB blocks = %.1f%% of the dense head; violations %d",
                                 r, label as NSString, k, rowsStat[j].median, rowsStat[j].median / Double(V) * 100, rowsStat[j].mean, rowsStat[j].pct(0.95),
                                 blocksStat[j].median, blocksStat[j].mean, blocksStat[j].pct(0.95), (V + 63) / 64,
                                 screenBytes / 1e6, blocksStat[j].mean * 64 * rowBytes / 1e6, frac * 100, violations[j]))
                    if files.count > 1 {
                        print("H1   by file: " + files.indices.map { String(format: "%@ median %.0f mean %.0f (n %d)", (files[$0] as NSString).lastPathComponent, byDomain[$0][j].median, byDomain[$0][j].mean, byDomain[$0][j].v.count) }.joined(separator: "; "))
                    }
                    if bits != 0, k == 1, rowsStat[j].median <= 0.02 * Double(V), frac < gateBest.frac { gateBest = (frac, "r \(r) \(label)") }
                }
                print(String(format: "H1 r %d %@: bound on the dense argmax row, median: sketch quantization %.3f + unseen %.3f logits; largest deficit of a violated row %.5f",
                             r, label as NSString, qTerm.median, nTerm.median, deficit))
                print(String(format: "H1 r %d %@ uncertified: rank of the dense argmax's block by screen logit: median %.0f, p99 %.0f, p99.9 %.0f, p99.99 %.0f, max %.0f (of %d blocks; %d states); screen bytes %.1f%% of the dense head, each block %.4f%%",
                             r, label as NSString, rank.median, rank.pct(0.99), rank.pct(0.999), rank.pct(0.9999), rank.max, (V + 63) / 64, nTest,
                             screenBytes / denseBytes * 100, 64 * rowBytes / denseBytes * 100))
                print(String(format: "H1 r %d %@ uncertified, by row: rank of the dense argmax row by screen logit: median %.0f, p99 %.0f, p99.9 %.0f, p99.99 %.0f, max %.0f; the screen's own argmax is the dense argmax in %.1f%% of states; each row %.5f%% of the dense head",
                             r, label as NSString, rowRank.median, rowRank.pct(0.99), rowRank.pct(0.999), rowRank.pct(0.9999), rowRank.max,
                             Double(rowRank.v.filter { $0 == 1 }.count) / Double(max(nTest, 1)) * 100, rowBytes / denseBytes * 100))
                // The mathematics allows no violation beyond rounding (CPU f32 sums against each other).
                #expect(deficit <= max(4 * gap, 1e-2), "a dense top-k row fell outside the certified set by \(deficit) logits")
            }
        }
        if gateBest.frac.isFinite {
            print(String(format: "H1 gate (k = 1: median certified rows <= 2%% of V and expected bytes <= 40%% of the dense head): best %@ at %.1f%% -> %@",
                         gateBest.label as NSString, gateBest.frac * 100, gateBest.frac <= 0.4 ? "PASSED" : "FAILED on bytes"))
        } else {
            print("H1 gate: FAILED, no quantized configuration has a median of certified rows within 2% of V")
        }
    }

    /// The K largest values of `z` and their rows, largest first.
    static func topK(_ z: UnsafePointer<Float>, _ n: Int, _ K: Int) -> [(value: Float, row: Int32)] {
        var top = [(value: Float, row: Int32)](repeating: (-.infinity, -1), count: K)   // ascending
        for i in 0..<n where z[i] > top[0].value {
            var j = 0
            while j + 1 < K, top[j + 1].value < z[i] { top[j] = top[j + 1]; j += 1 }
            top[j] = (z[i], Int32(i))
        }
        return top.reversed()
    }
}

/// BLAS and LAPACK from Accelerate, looked up by name: the Swift overlay marks the classic interfaces
/// deprecated, and their replacements need a compiler definition this package does not set.
enum BLAS {
    static let rowMajor: Int32 = 101, noTrans: Int32 = 111, trans: Int32 = 112, upper: Int32 = 121
    typealias FP = UnsafePointer<Float>
    typealias FM = UnsafeMutablePointer<Float>
    typealias IM = UnsafeMutablePointer<Int32>
    typealias DM = UnsafeMutablePointer<Double>
    private static let handle = dlopen("/System/Library/Frameworks/Accelerate.framework/Accelerate", RTLD_NOW)
    private static func symbol<T>(_ name: String, as: T.Type) -> T {
        guard let s = dlsym(handle, name) else { fatalError("\(name) not found in Accelerate") }
        return unsafeBitCast(s, to: T.self)
    }
    /// C [M × N] = alpha · op(A) · op(B) + beta · C
    static let sgemm = symbol("cblas_sgemm", as: (@convention(c) (Int32, Int32, Int32, Int32, Int32, Int32, Float, FP, Int32, FP, Int32, Float, FM, Int32) -> Void).self)
    /// y = alpha · op(A) · x + beta · y, A [M × N]
    static let sgemv = symbol("cblas_sgemv", as: (@convention(c) (Int32, Int32, Int32, Int32, Float, FP, Int32, FP, Int32, Float, FM, Int32) -> Void).self)
    static let snrm2 = symbol("cblas_snrm2", as: (@convention(c) (Int32, FP, Int32) -> Float).self)
    /// C [N × N] = alpha · Aᵀ A + beta · C (with `trans`, A [K × N])
    static let dsyrk = symbol("cblas_dsyrk", as: (@convention(c) (Int32, Int32, Int32, Int32, Int32, Double, UnsafePointer<Double>, Int32, Double, DM, Int32) -> Void).self)
    /// Symmetric eigendecomposition (divide and conquer), column-major.
    static let dsyevd = symbol("dsyevd_", as: (@convention(c) (UnsafeMutablePointer<Int8>, UnsafeMutablePointer<Int8>, IM, DM, IM, DM, DM, IM, IM, IM, IM) -> Int32).self)
}
