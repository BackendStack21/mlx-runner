import Foundation
import Metal
import Testing
@testable import MLXEngine

/// Phase 0 of docs/sieve-architecture.md, hypotheses H5 (prompt tokens need few layers) and H4
/// (tokens can leave early), measured with `SieveProbe` on teacher-forced chunks of 512 tokens.
/// Quality only: text NLL and top-1 agreement against the dense run. No timing. Opt-in:
///
///     MLXR_SIEVE_PYRAMID=mlx-community/Qwen3-0.6B-4bit \
///         swift test -c release -Xswiftc -enable-testing --filter SievePyramid
///
/// - Text: this repository, two domains: prose (README, CHANGELOG, docs) and code (Sources/MLXEngine).
/// - `MLXR_SIEVE_CHUNKS` (16): scored chunks per domain. `MLXR_SIEVE_PART` (`h5,h4`): what to run.
/// - `MLXR_SIEVE_SETS`: candidate exit sets for H4, e.g. `14,21;21` (default: from the depth).
/// - `MLXR_SIEVE_P`, `MLXR_SIEVE_MARGIN`: thresholds of the two exit rules (comma lists; `none` skips a rule).
/// - `MLXR_SIEVE_LENS=fit`: H4 reads exits through an affine map of the frozen final-norm output onto
///   the dense one, fitted by ridge regression on `MLXR_SIEVE_FIT_CHUNKS` (48) other chunks per
///   domain. Default `raw`: the logit lens itself (final norm and head on the frozen residual).
///
/// What the probe gives and what it does not: an exited row's residual stays frozen, and the layers
/// above compute their K/V rows (or recurrent state) from that frozen residual. That is the design's
/// frozen fill with β = 0; the extrapolated fill (β fitted towards the lens) is not measured here.
@Suite("Sieve Pyramid: H4 and H5", .serialized, .enabled(if: ProcessInfo.processInfo.environment["MLXR_SIEVE_PYRAMID"] != nil))
struct SievePyramidTests {
    static let env = ProcessInfo.processInfo.environment
    static let T = 512, skip = 16, prompt = 384
    typealias Matrix = SieveImporter.Matrix

    /// What one row's logits say: NLL of the true next token, the argmax, its probability, the logit margin to the runner-up.
    struct Out { var nll: Float = 0; var top: Int32 = -1; var p: Float = 0; var margin: Float = 0 }

    struct Acc {
        var n = 0, agree = 0, exited = 0, exitedAgree = 0
        var d = 0.0, d2 = 0.0, depth = 0.0
        var exits: [Int]
        init(_ candidates: Int = 0) { exits = [Int](repeating: 0, count: candidates) }
        var mean: Double { n == 0 ? 0 : d / Double(n) }
        /// Standard error of the mean paired difference, taking tokens as independent.
        var se: Double { n < 2 ? 0 : ((d2 / Double(n) - mean * mean) / Double(n)).squareRoot() }
        var agreement: Double { n == 0 ? 0 : Double(agree) / Double(n) }
        mutating func add(_ o: Out, dense: Out, depth f: Double = 1) {
            let x = Double(o.nll - dense.nll)
            n += 1; d += x; d2 += x * x; depth += f
            if o.top == dense.top { agree += 1 }
        }
        mutating func merge(_ o: Acc) {
            n += o.n; agree += o.agree; exited += o.exited; exitedAgree += o.exitedAgree
            d += o.d; d2 += o.d2; depth += o.depth
            for i in exits.indices where i < o.exits.count { exits[i] += o.exits[i] }
        }
    }

    enum Rule: Hashable, CustomStringConvertible {
        case prob(Float), margin(Float)
        func passes(_ o: Out) -> Bool {
            switch self { case .prob(let p): return o.p >= p; case .margin(let m): return o.margin >= m }
        }
        var description: String {
            switch self { case .prob(let p): return String(format: "p>=%.2f", p); case .margin(let m): return String(format: "margin>=%.1f", m) }
        }
    }

    /// x ↦ (x − μx) · A + μy, fitted on post-norm states (`at` holds Aᵀ row-major).
    struct Lens { let at: [Float], mx: [Float], my: [Float] }

    final class Bench {
        let t: Transformer, probe: SieveProbe, W: Matrix
        let d: Int, V: Int, L: Int
        var lenses: [Int: Lens] = [:]
        init(_ dir: URL) throws {
            t = try Transformer(directory: dir, contextLength: 1024, sparse: nil)
            probe = try SieveProbe(t)
            W = try SieveImporter.dequantize(t.lmHead)
            d = t.config.hiddenSize; V = t.vocabSize; L = t.layers.count
        }

        /// One teacher-forced pass from offset 0; the final norm's output for every row.
        @discardableResult
        func pass(_ tokens: ArraySlice<Int32>, exits: [Int32]? = nil) throws -> UnsafePointer<Float> {
            probe.setExits(exits ?? [])
            _ = try t.forward(Array(tokens.prefix(T)), offset: 0)
            return UnsafePointer(t.xn.contents().bindMemory(to: Float.self, capacity: T * d))
        }

        func states(_ xn: UnsafePointer<Float>) -> [Float] { Array(UnsafeBufferPointer(start: xn, count: T * d)) }

        /// Logits of `rows` (CPU head on their final-norm output, through `lens[i]`'s map when it is a
        /// fitted exit layer) scored against the true next tokens. `chunk` holds T + 1 tokens.
        func score(_ xn: UnsafePointer<Float>, rows: [Int], lens: [Int]? = nil, chunk: ArraySlice<Int32>) -> [Out] {
            let n = rows.count
            guard n > 0 else { return [] }
            var X = [Float](repeating: 0, count: n * d)
            var tmp = [Float](repeating: 0, count: d)
            for (i, r) in rows.enumerated() {
                if let e = lens?[i], let l = lenses[e] {
                    for j in 0..<d { tmp[j] = xn[r * d + j] - l.mx[j] }
                    X.withUnsafeMutableBufferPointer { xp in
                        BLAS.sgemv(BLAS.rowMajor, BLAS.noTrans, Int32(d), Int32(d), 1, l.at, Int32(d), tmp, 1, 0, xp.baseAddress! + i * d, 1)
                        for j in 0..<d { xp[i * d + j] += l.my[j] }
                    }
                } else {
                    X.withUnsafeMutableBufferPointer { xp in (xp.baseAddress! + i * d).update(from: xn + r * d, count: d) }
                }
            }
            let Z = UnsafeMutablePointer<Float>.allocate(capacity: n * V)
            defer { Z.deallocate() }
            BLAS.sgemm(BLAS.rowMajor, BLAS.noTrans, BLAS.trans, Int32(n), Int32(V), Int32(d), 1, X, Int32(d), W.p, Int32(d), 0, Z, Int32(V))
            var out = [Out](repeating: Out(), count: n)
            let V = self.V, base = chunk.startIndex
            out.withUnsafeMutableBufferPointer { op in
                let o = op.baseAddress!
                nonisolated(unsafe) let oo = o, zz = Z
                DispatchQueue.concurrentPerform(iterations: n) { i in
                    let z = zz + i * V
                    var m1 = -Float.infinity, m2 = -Float.infinity, arg = 0
                    for v in 0..<V {
                        let x = z[v]
                        if x > m1 { m2 = m1; m1 = x; arg = v } else if x > m2 { m2 = x }
                    }
                    var s = 0.0
                    for v in 0..<V { s += Double(expf(z[v] - m1)) }
                    let target = Int(chunk[base + rows[i] + 1])
                    oo[i] = Out(nll: Float(log(s)) - (z[target] - m1), top: Int32(arg), p: Float(1 / s), margin: m1 - m2)
                }
            }
            return out
        }
    }

    static func say(_ s: String) { print("PYR " + s); fflush(stdout) }

    /// The repository's own text: (name, text) per domain.
    static func domains() throws -> [(String, String)] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        func files(_ dir: String, _ ext: String) throws -> [URL] {
            try FileManager.default.contentsOfDirectory(at: root.appendingPathComponent(dir), includingPropertiesForKeys: nil)
                .filter { $0.pathExtension == ext }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        }
        func cat(_ urls: [URL]) throws -> String { try urls.map { try String(contentsOf: $0, encoding: .utf8) }.joined(separator: "\n") }
        let prose = [root.appendingPathComponent("README.md"), root.appendingPathComponent("CHANGELOG.md")] + (try files("docs", "md"))
        return [("prose", try cat(prose)), ("code", try cat(try files("Sources/MLXEngine", "swift")))]
    }

    /// Non-overlapping blocks of T + 1 tokens: `n` spread evenly to score, up to `fit` of the others to fit on.
    static func blocks(_ ids: [Int32], n: Int, fit: Int) -> (test: [ArraySlice<Int32>], fit: [ArraySlice<Int32>]) {
        let total = ids.count / (T + 1)
        let n = min(n, total)
        guard n > 0 else { return ([], []) }
        var isTest = [Bool](repeating: false, count: total)
        for i in 0..<n { isTest[n == 1 ? 0 : i * (total - 1) / (n - 1)] = true }
        let rest = (0..<total).filter { !isTest[$0] }
        let f = min(fit, rest.count)
        let fitIdx = (0..<f).map { rest[f == 1 ? 0 : $0 * (rest.count - 1) / (f - 1)] }
        func slice(_ b: Int) -> ArraySlice<Int32> { ids[(b * (T + 1))..<((b + 1) * (T + 1))] }
        return ((0..<total).filter { isTest[$0] }.map(slice), Set(fitIdx).sorted().map(slice))
    }

    static func floats(_ key: String, _ fallback: [Float]) -> [Float] {
        guard let s = env[key] else { return fallback }
        return s.split(separator: ",").compactMap { Float($0) }
    }

    @Test func pyramid() throws {
        let env = Self.env, T = Self.T, skip = Self.skip, say = Self.say
        guard let repo = env["MLXR_SIEVE_PYRAMID"], let dir = installedModel(repo) else { print("Pyramid: model not installed"); return }
        let b = try Bench(dir)
        let tok = try Tokenizer(directory: dir)
        let d = b.d, L = b.L
        guard b.t.config.finalLogitSoftcap == 0, b.t.maxBatch >= T else { print("Pyramid: unsupported model"); return }
        let nChunks = Int(env["MLXR_SIEVE_CHUNKS"] ?? "") ?? 16
        let parts = Set((env["MLXR_SIEVE_PART"] ?? "h5,h4").split(separator: ",").map(String.init))
        let fitLens = env["MLXR_SIEVE_LENS"] == "fit"
        let fitChunks = fitLens ? (Int(env["MLXR_SIEVE_FIT_CHUNKS"] ?? "") ?? 48) : 0
        say("\(repo): \(L) layers, d \(d), V \(b.V), hybrid \(b.t.config.isHybrid), \(nChunks) chunks per domain, parts \(parts.sorted()), lens \(fitLens ? "fit" : "raw")")

        var data: [(name: String, test: [ArraySlice<Int32>], fit: [ArraySlice<Int32>])] = []
        for (name, text) in try Self.domains() {
            let ids = tok.encode(text).map(Int32.init)
            let (test, fit) = Self.blocks(ids, n: nChunks, fit: fitChunks)
            say("domain \(name): \(ids.count) tokens, \(test.count) scored chunks, \(fit.count) fit chunks")
            #expect(test.count >= 16 || nChunks < 16, "fewer than 16 chunks in \(name)")
            data.append((name, test, fit))
        }
        guard let first = data.first?.test.first else { return }

        // ---- 0. The probe itself -------------------------------------------------------------
        do {
            let never = [Int32](repeating: Int32.max, count: T)
            func maxDiff(_ a: [Float], _ c: [Float], rows: Range<Int> = 0..<T) -> Float {
                var m: Float = 0
                for i in (rows.lowerBound * d)..<(rows.upperBound * d) {
                    if a[i].bitPattern != c[i].bitPattern { m = max(m, abs(a[i] - c[i]), .leastNonzeroMagnitude) }
                }
                return m
            }
            func capture() -> [Float] { Array(UnsafeBufferPointer(start: b.probe.capture.contents().bindMemory(to: Float.self, capacity: T * d), count: T * d)) }
            // (a) a probe with nothing set changes nothing.
            b.t.probe = nil
            let l0 = try b.t.forward(Array(first.prefix(T)), offset: 0)
            let logits0 = Array(UnsafeBufferPointer(start: l0, count: b.V))
            let dense = b.states(UnsafePointer(b.t.xn.contents().bindMemory(to: Float.self, capacity: T * d)))
            b.t.probe = b.probe
            b.probe.setExits([])
            let l1 = try b.t.forward(Array(first.prefix(T)), offset: 0)
            let logits1 = Array(UnsafeBufferPointer(start: l1, count: b.V))
            let withProbe = b.states(UnsafePointer(b.t.xn.contents().bindMemory(to: Float.self, capacity: T * d)))
            let aStates = maxDiff(dense, withProbe)
            let aLogits = zip(logits0, logits1).contains { $0.bitPattern != $1.bitPattern }
            #expect(aStates == 0 && !aLogits, "an idle probe changed the pass")
            // (b) never exiting is exiting at L (kernels encoded, nothing zeroed).
            let sNever = b.states(try b.pass(first, exits: never))
            let sTop = b.states(try b.pass(first, exits: [Int32](repeating: Int32(L), count: T)))
            let bDiff = max(maxDiff(sNever, sTop), maxDiff(dense, sTop))
            #expect(bDiff == 0, "exit at L differs from no exit")
            // (c) exit 0 keeps a row's residual at its embedding all the way up; other rows move.
            b.probe.captureLayer = 0
            try b.pass(first)
            let emb = capture()
            var e0 = never
            for r in stride(from: 5, to: T, by: 3) { e0[r] = 0 }
            b.probe.captureLayer = L - 1
            try b.pass(first, exits: e0)
            let top = capture()
            var cFrozen: Float = 0, cMoved = Float.infinity
            for r in 0..<T {
                let m = maxDiff(emb, top, rows: r..<(r + 1))
                if e0[r] == 0 { cFrozen = max(cFrozen, m) } else { cMoved = min(cMoved, m) }
            }
            #expect(cFrozen == 0 && cMoved > 0, "exit 0 does not freeze a row at its embedding")
            // (d) what enters layer E does not depend on exits at E or above (the causal sweep relies on it).
            let E = L / 2
            b.probe.captureLayer = E
            try b.pass(first)
            let hDense = capture()
            let xnAll = b.states(try b.pass(first, exits: [Int32](repeating: Int32(E), count: T)))
            let hAll = capture()
            let dDiff = maxDiff(hDense, hAll)
            #expect(dDiff == 0, "the residual entering layer \(E) changed with exits at \(E)")
            // (e) rows before the first exited row are untouched; exited rows differ from dense.
            var late = never
            for r in 256..<T { late[r] = Int32(L / 4) }
            b.probe.captureLayer = nil
            let sLate = b.states(try b.pass(first, exits: late))
            let eBefore = maxDiff(dense, sLate, rows: 0..<256), eAfter = maxDiff(dense, sLate, rows: 256..<T)
            #expect(eBefore == 0 && eAfter > 0, "exits leaked backwards")
            // The CPU head against the engine's logits (dense, last row).
            let o = b.score(try b.pass(first), rows: [T - 1], chunk: first)
            var z = [Float](repeating: 0, count: b.V)
            dense.withUnsafeBufferPointer { BLAS.sgemv(BLAS.rowMajor, BLAS.noTrans, Int32(b.V), Int32(d), 1, b.W.p, Int32(d), $0.baseAddress! + (T - 1) * d, 1, 0, &z, 1) }
            var gap: Float = 0
            for i in 0..<b.V { gap = max(gap, abs(z[i] - logits0[i])) }
            #expect(gap < 0.05, "the dequantized head disagrees with the engine")
            say(String(format: "sanity: (a) idle probe max diff %g, logits identical %@; (b) exit L vs never %g; (c) exit-0 rows vs embedding %g, other rows moved by >= %g; (d) residual entering layer %d, dense vs all exit there %g; (e) rows before the first exit %g, after %g; CPU head vs engine logits %.4f; last-row dense NLL %.3f",
                       aStates, (!aLogits).description as NSString, bDiff, cFrozen, cMoved, E, dDiff, eBefore, eAfter, gap, o[0].nll))
            _ = xnAll
        }
        b.t.probe = b.probe
        b.probe.captureLayer = nil

        // ---- H5: prefill depth ---------------------------------------------------------------
        if parts.contains("h5") {
            let P = Self.prompt
            // 0 and L/8 for reference, eight depths from L/4 to L, and the midpoints of the upper half.
            let eight = (0..<8).map { Int((Double(L) / 4 + Double($0) * Double(L) * 0.75 / 7).rounded()) }
            let depths = Array(Set([0, L / 8] + eight + (4..<7).map { (eight[$0] + eight[$0 + 1] + 1) / 2 })).sorted()
            let keeps = [0, 8, 32]
            let rows = Array(P..<T)
            var acc: [String: [[Acc]]] = [:]     // domain → [depth][keep]
            var denseNLL: [String: (Double, Int)] = [:]
            for dom in data {
                var a = [[Acc]](repeating: [Acc](repeating: Acc(), count: keeps.count), count: depths.count)
                var s = 0.0, n = 0
                for chunk in dom.test {
                    let dense = b.score(try b.pass(chunk), rows: rows, chunk: chunk)
                    for o in dense { s += Double(o.nll); n += 1 }
                    for (ei, E) in depths.enumerated() {
                        for (ki, k) in keeps.enumerated() {
                            if E == L && k > 0 { continue }
                            var exits = [Int32](repeating: Int32.max, count: T)
                            for r in 0..<(P - k) { exits[r] = Int32(E) }
                            let o = b.score(try b.pass(chunk, exits: exits), rows: rows, chunk: chunk)
                            for i in rows.indices { a[ei][ki].add(o[i], dense: dense[i]) }
                        }
                    }
                }
                acc[dom.name] = a; denseNLL[dom.name] = (s, n)
                say(String(format: "H5 %@: dense continuation NLL %.4f over %d tokens (%d-token prompt, %d-token continuation)", dom.name as NSString, s / Double(n), n, P, T - P))
            }
            say("H5 table: prompt rows exit at E (frozen fill above), the last k prompt rows stay dense; dNLL in nats/token (+- per-token paired SE), top-1 agreement with dense")
            var worst = [[Double]](repeating: [Double](repeating: -.infinity, count: keeps.count), count: depths.count)
            for (ei, E) in depths.enumerated() {
                for (ki, k) in keeps.enumerated() where !(E == L && k > 0) {
                    let depth = (Double(E) * Double(P - k) + Double(L) * Double(k)) / Double(L * P)
                    var line = String(format: "H5 E %2d (E/L %.3f) k %2d prompt mean depth %.3f:", E, Double(E) / Double(L), k, depth)
                    var pooled = Acc()
                    for dom in data {
                        let a = acc[dom.name]![ei][ki]
                        pooled.merge(a)
                        worst[ei][ki] = max(worst[ei][ki], a.mean)
                        line += String(format: " %@ dNLL %+.4f +-%.4f agree %.1f%%;", dom.name as NSString, a.mean, a.se, a.agreement * 100)
                    }
                    line += String(format: " pooled dNLL %+.4f +-%.4f agree %.1f%%", pooled.mean, pooled.se, pooled.agreement * 100)
                    say(line)
                }
            }
            for (target, name) in [(0.02, "balanced"), (0.05, "fast")] {
                for (ki, k) in keeps.enumerated() {
                    // Smallest E from which every deeper E also meets the target in both domains.
                    var best: Int?
                    for ei in depths.indices.reversed() {
                        if depths[ei] == L || worst[ei][ki] <= target { best = depths[ei] } else { break }
                    }
                    say(String(format: "H5 smallest E with dNLL <= +%.2f in both domains (%@), k %d: %@", target, name as NSString, k,
                               best.map { String(format: "E %d (E/L %.3f)", $0, Double($0) / Double(L)) } ?? "none") as String)
                }
            }
        }

        // ---- H4: decode exits ----------------------------------------------------------------
        if parts.contains("h4") {
            var sets: [[Int]]
            if let s = env["MLXR_SIEVE_SETS"] {
                sets = s.split(separator: ";").map { $0.split(separator: ",").compactMap { Int($0) }.sorted() }.filter { !$0.isEmpty }
            } else {
                sets = [[L / 2, 3 * L / 4], [3 * L / 4], [5 * L / 8, 7 * L / 8], [3 * L / 4, 7 * L / 8]]
            }
            sets = sets.filter { $0.allSatisfy { $0 > 0 && $0 < L } }
            let layers = Array(Set(sets.flatMap { $0 })).sorted()
            let ps = env["MLXR_SIEVE_P"] == "none" ? [] : Self.floats("MLXR_SIEVE_P", [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999])
            let ms = env["MLXR_SIEVE_MARGIN"] == "none" ? [] : Self.floats("MLXR_SIEVE_MARGIN", [4, 6, 8, 10])
            let rules = ps.map(Rule.prob) + ms.map(Rule.margin)
            let all = Array(0..<T)

            if fitLens {
                // Ridge regression of the dense final-norm output on the frozen one, per exit layer.
                let n0 = T - skip
                var G = [Int: [Double]](), C = [Int: [Double]](), sx = [Int: [Double]]()
                var sy = [Double](repeating: 0, count: d), n = 0
                for e in layers { G[e] = [Double](repeating: 0, count: d * d); C[e] = [Double](repeating: 0, count: d * d); sx[e] = [Double](repeating: 0, count: d) }
                var g = [Float](repeating: 0, count: d * d)
                for dom in data {
                    for chunk in dom.fit {
                        let y = b.states(try b.pass(chunk))
                        for i in skip..<T { for j in 0..<d { sy[j] += Double(y[i * d + j]) } }
                        n += n0
                        for e in layers {
                            let x = b.states(try b.pass(chunk, exits: [Int32](repeating: Int32(e), count: T)))
                            for i in skip..<T { for j in 0..<d { sx[e]![j] += Double(x[i * d + j]) } }
                            x.withUnsafeBufferPointer { xp in
                                y.withUnsafeBufferPointer { yp in
                                    let xs = xp.baseAddress! + skip * d, ys = yp.baseAddress! + skip * d
                                    BLAS.sgemm(BLAS.rowMajor, BLAS.trans, BLAS.noTrans, Int32(d), Int32(d), Int32(n0), 1, xs, Int32(d), xs, Int32(d), 0, &g, Int32(d))
                                    for i in 0..<(d * d) { G[e]![i] += Double(g[i]) }
                                    BLAS.sgemm(BLAS.rowMajor, BLAS.trans, BLAS.noTrans, Int32(d), Int32(d), Int32(n0), 1, xs, Int32(d), ys, Int32(d), 0, &g, Int32(d))
                                    for i in 0..<(d * d) { C[e]![i] += Double(g[i]) }
                                }
                            }
                        }
                    }
                }
                guard n > d else { say("H4: too few fit states (\(n))"); return }
                let ridge = Double(Self.floats("MLXR_SIEVE_RIDGE", [1e-3])[0])
                let my = sy.map { $0 / Double(n) }
                for e in layers {
                    let mx = sx[e]!.map { $0 / Double(n) }
                    var gc = G[e]!, ct = [Double](repeating: 0, count: d * d)
                    var trace = 0.0
                    for i in 0..<d {
                        for j in 0..<d {
                            gc[i * d + j] -= Double(n) * mx[i] * mx[j]
                            ct[j * d + i] = C[e]![i * d + j] - Double(n) * mx[i] * my[j]   // column-major right-hand sides
                        }
                        trace += gc[i * d + i]
                    }
                    for i in 0..<d { gc[i * d + i] += ridge * trace / Double(d) }
                    var uplo = Int8(UInt8(ascii: "U")), nn = Int32(d), nrhs = Int32(d), lda = Int32(d), ldb = Int32(d), info = Int32(0)
                    _ = Self.dposv(&uplo, &nn, &nrhs, &gc, &lda, &ct, &ldb, &info)
                    guard info == 0 else { say("H4: dposv failed at layer \(e) (\(info))"); return }
                    // ct is now A column-major, which read row-major is Aᵀ.
                    b.lenses[e] = Lens(at: ct.map(Float.init), mx: mx.map(Float.init), my: my.map(Float.init))
                }
                say("H4 fitted lenses at layers \(layers) on \(n) states (ridge \(ridge) x mean eigenvalue)")
            }

            // Static view: every row read at one layer (states as in the dense run below it).
            let grid: [Rule] = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99].map(Rule.prob) + [1, 2, 3, 4, 6, 8, 12].map(Rule.margin)
            struct Static { var all = Acc(); var pass: [Int]; var passAgree: [Int]; var passD: [Double] }
            var stat: [String: [Int: Static]] = [:]
            var sweep: [String: [[Acc]]] = [:]      // domain → [set][rule]
            for dom in data {
                var st = [Int: Static]()
                for e in layers { st[e] = Static(pass: [Int](repeating: 0, count: grid.count), passAgree: [Int](repeating: 0, count: grid.count), passD: [Double](repeating: 0, count: grid.count)) }
                var sw = sets.map { set in rules.map { _ in Acc(set.count) } }
                for (ci, chunk) in dom.test.enumerated() {
                    let dense = b.score(try b.pass(chunk), rows: all, chunk: chunk)
                    var atLayer = [Int: [Out]]()
                    for e in layers {
                        let o = b.score(try b.pass(chunk, exits: [Int32](repeating: Int32(e), count: T)), rows: all, lens: [Int](repeating: e, count: T), chunk: chunk)
                        atLayer[e] = o
                        for r in skip..<T {
                            st[e]!.all.add(o[r], dense: dense[r])
                            for (gi, rule) in grid.enumerated() where rule.passes(o[r]) {
                                st[e]!.pass[gi] += 1
                                st[e]!.passD[gi] += Double(o[r].nll - dense[r].nll)
                                if o[r].top == dense[r].top { st[e]!.passAgree[gi] += 1 }
                            }
                        }
                    }
                    // Causal sweep: decide the lowest exit from the dense states, apply, decide the next from that pass.
                    for (si, set) in sets.enumerated() {
                        for (ri, rule) in rules.enumerated() {
                            var exits = [Int32](repeating: Int32.max, count: T)
                            var final = [Out?](repeating: nil, count: T)
                            var which = [Int](repeating: -1, count: T)
                            var any = false
                            for (k, e) in set.enumerated() {
                                let open = all.filter { final[$0] == nil }
                                guard !open.isEmpty else { break }
                                let o: [Out]
                                if !any { o = open.map { atLayer[e]![$0] } } else {
                                    var trial = exits
                                    for r in open { trial[r] = Int32(e) }
                                    o = b.score(try b.pass(chunk, exits: trial), rows: open, lens: [Int](repeating: e, count: open.count), chunk: chunk)
                                }
                                for (i, r) in open.enumerated() where rule.passes(o[i]) {
                                    exits[r] = Int32(e); final[r] = o[i]; which[r] = k; any = true
                                }
                            }
                            let open = all.filter { final[$0] == nil }
                            if !any { for r in open { final[r] = dense[r] } } else if !open.isEmpty {
                                let o = b.score(try b.pass(chunk, exits: exits), rows: open, chunk: chunk)
                                for (i, r) in open.enumerated() { final[r] = o[i] }
                            }
                            for r in skip..<T {
                                let o = final[r]!
                                sw[si][ri].add(o, dense: dense[r], depth: which[r] < 0 ? 1 : Double(set[which[r]]) / Double(L))
                                if which[r] >= 0 {
                                    sw[si][ri].exits[which[r]] += 1; sw[si][ri].exited += 1
                                    if o.top == dense[r].top { sw[si][ri].exitedAgree += 1 }
                                }
                            }
                        }
                    }
                    if ci % 4 == 3 { say("H4 \(dom.name): \(ci + 1) of \(dom.test.count) chunks") }
                }
                stat[dom.name] = st; sweep[dom.name] = sw
            }

            say("H4 static table (\(fitLens ? "fitted" : "raw") lens): every row read at layer E; then, per rule, share of rows passing, their top-1 agreement with dense and their dNLL")
            for e in layers {
                for dom in data {
                    let s = stat[dom.name]![e]!
                    say(String(format: "H4 lens at E %d (E/L %.3f) %@: dNLL %+.3f, top-1 agreement with dense %.1f%% (n %d)", e, Double(e) / Double(L), dom.name as NSString, s.all.mean, s.all.agreement * 100, s.all.n))
                    var line = "H4   passing:"
                    for (gi, rule) in grid.enumerated() {
                        let n = s.pass[gi]
                        line += String(format: " %@ %.1f%% (agree %.1f%%, dNLL %+.3f);", rule.description as NSString, Double(n) / Double(max(s.all.n, 1)) * 100,
                                       Double(s.passAgree[gi]) / Double(max(n, 1)) * 100, s.passD[gi] / Double(max(n, 1)))
                    }
                    say(line)
                }
            }
            say("H4 causal table (\(fitLens ? "fitted" : "raw") lens): exits applied, every position scored with the logits it would produce; depth = mean fraction of layers run; exited agree = top-1 agreement with dense among exited rows")
            for (si, set) in sets.enumerated() {
                for (ri, rule) in rules.enumerated() {
                    var pooled = Acc(set.count)
                    var line = ""
                    for dom in data {
                        let a = sweep[dom.name]![si][ri]
                        pooled.merge(a)
                        line += String(format: " | %@ depth %.3f dNLL %+.4f +-%.4f agree %.1f%%", dom.name as NSString, a.depth / Double(max(a.n, 1)), a.mean, a.se, a.agreement * 100)
                    }
                    let shares = set.indices.map { String(format: "E%d %.1f%%", set[$0], Double(pooled.exits[$0]) / Double(max(pooled.n, 1)) * 100) }.joined(separator: " ")
                    say(String(format: "H4 exits %@ %@: exit share %@; depth %.3f; dNLL %+.4f +-%.4f; agree %.1f%%; exited agree %.1f%%%@",
                               "\(set)" as NSString, rule.description as NSString, shares as NSString, pooled.depth / Double(max(pooled.n, 1)), pooled.mean, pooled.se,
                               pooled.agreement * 100, Double(pooled.exitedAgree) / Double(max(pooled.exited, 1)) * 100, line as NSString))
                }
            }
            for (target, name) in [(0.02, "balanced"), (0.05, "fast")] {
                var best: (depth: Double, label: String)?
                for (si, set) in sets.enumerated() {
                    for (ri, rule) in rules.enumerated() {
                        let per = data.map { sweep[$0.name]![si][ri] }
                        guard per.allSatisfy({ $0.mean <= target }) else { continue }
                        var pooled = Acc(set.count)
                        per.forEach { pooled.merge($0) }
                        let depth = pooled.depth / Double(max(pooled.n, 1))
                        if best == nil || depth < best!.depth { best = (depth, String(format: "exits %@ %@: depth %.3f, pooled dNLL %+.4f", "\(set)" as NSString, rule.description as NSString, depth, pooled.mean)) }
                    }
                }
                say(String(format: "H4 lowest mean depth with dNLL <= +%.2f in both domains (%@): %@", target, name as NSString, (best?.label ?? "none") as NSString))
            }
        }
    }

    /// Cholesky solve from Accelerate's LAPACK (column-major), looked up by name like `BLAS`.
    static let dposv: @convention(c) (UnsafeMutablePointer<Int8>, UnsafeMutablePointer<Int32>, UnsafeMutablePointer<Int32>, UnsafeMutablePointer<Double>,
                                      UnsafeMutablePointer<Int32>, UnsafeMutablePointer<Double>, UnsafeMutablePointer<Int32>, UnsafeMutablePointer<Int32>) -> Int32 = {
        guard let h = dlopen("/System/Library/Frameworks/Accelerate.framework/Accelerate", RTLD_NOW), let s = dlsym(h, "dposv_") else { fatalError("dposv_ not found in Accelerate") }
        return unsafeBitCast(s, to: (@convention(c) (UnsafeMutablePointer<Int8>, UnsafeMutablePointer<Int32>, UnsafeMutablePointer<Int32>, UnsafeMutablePointer<Double>,
                                                     UnsafeMutablePointer<Int32>, UnsafeMutablePointer<Double>, UnsafeMutablePointer<Int32>, UnsafeMutablePointer<Int32>) -> Int32).self)
    }()
}
