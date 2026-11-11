import Foundation
import Metal
import Testing
@testable import MLXEngine

/// Phase 0 of docs/sieve-architecture.md, hypothesis H2: does a minority of the MLP's intermediate
/// neurons carry its output on models under 1B parameters? It measures and reports; the assertions
/// are the probe's own sanity checks. Opt-in, and slow without optimisation:
///
///     MLXR_SIEVE_EXPERTS=mlx-community/Qwen3-0.6B-4bit \
///         swift test -c release -Xswiftc -enable-testing --filter SieveExperts
///
/// - Text: the repository's own files, as two domains: prose (README.md, CHANGELOG.md, docs/*.md)
///   and code (Sources/MLXEngine/*.swift). Chunks of 512 tokens spread over each; the first 16
///   positions of a chunk are not scored.
/// - `MLXR_SIEVE_CHUNKS` (16): chunks per domain.
/// - `MLXR_SIEVE_TAUS`, `MLXR_SIEVE_TAUS64`: the thresholds of the oracle sweep for single neurons
///   and for groups of 64 consecutive neurons (comma-separated).
/// - `MLXR_SIEVE_STEPS` (`123`): which measurements run (1 oracle curve, 2 lower/upper half of the
///   layers, 3 the CPU study of selection rules and predictors, on `MLXR_SIEVE_CAPTURE` (4) chunks
///   per domain).
///
/// The oracle is `SieveProbe`'s mask: a group is kept when its best |a_n| · |down column n| reaches
/// tau times the row's best. It reads the true activations, so it bounds any screen from above.
///
/// Metric: mean NLL of the true next token (nats) and top-1 agreement with the dense run. The
/// hidden states are the engine's (`xn` after a teacher-forced pass); the logits are computed on
/// the CPU from the dequantized head.
@Suite("Sieve H2: experts", .serialized, .enabled(if: ProcessInfo.processInfo.environment["MLXR_SIEVE_EXPERTS"] != nil))
struct SieveExpertsTests {
    static let env = ProcessInfo.processInfo.environment
    static let T = 512, skip = 16
    static var scored: Int { T - skip }

    struct Domain { let name: String; let chunks: [[Int32]] }   // T + 1 tokens each: the inputs and the last target

    /// NLL of the true next token and the argmax, per scored position.
    struct Scores { var nll: [Float] = []; var top: [Int32] = [] }

    static let vvexpf: @convention(c) (UnsafeMutablePointer<Float>, UnsafePointer<Float>, UnsafePointer<Int32>) -> Void = {
        let h = dlopen("/System/Library/Frameworks/Accelerate.framework/Accelerate", RTLD_NOW)
        return unsafeBitCast(dlsym(h, "vvexpf")!, to: (@convention(c) (UnsafeMutablePointer<Float>, UnsafePointer<Float>, UnsafePointer<Int32>) -> Void).self)
    }()

    static var root: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }

    static func concatenated(_ urls: [URL]) throws -> String {
        try urls.sorted { $0.path < $1.path }.map { try String(contentsOf: $0, encoding: .utf8) }.joined(separator: "\n\n")
    }

    static func domains(_ tok: Tokenizer, chunks n: Int) throws -> [Domain] {
        let fm = FileManager.default
        let docs = try fm.contentsOfDirectory(at: root.appendingPathComponent("docs"), includingPropertiesForKeys: nil).filter { $0.pathExtension == "md" }
        let code = try fm.contentsOfDirectory(at: root.appendingPathComponent("Sources/MLXEngine"), includingPropertiesForKeys: nil).filter { $0.pathExtension == "swift" }
        let texts = [("prose", try concatenated([root.appendingPathComponent("README.md"), root.appendingPathComponent("CHANGELOG.md")]) + "\n\n" + (try concatenated(docs))),
                     ("code", try concatenated(code))]
        return try texts.map { name, text in
            let ids = tok.encode(text).map(Int32.init)
            guard ids.count >= n * (T + 1) else { throw EngineError.invalidOutput("\(name): \(ids.count) tokens, fewer than \(n) chunks") }
            let stride = (ids.count - T - 1) / max(n - 1, 1)
            print("H2 \(name): \(ids.count) tokens, \(n) chunks of \(T) every \(stride)")
            return Domain(name: name, chunks: (0..<n).map { Array(ids[($0 * stride)..<($0 * stride + T + 1)]) })
        }
    }

    /// Scores the rows `skip..<T` of the pass that just ran (`t.xn`) against `chunk`'s next tokens.
    final class Scorer {
        let W: SieveImporter.Matrix
        let V: Int, d: Int
        let block = 124
        let Z: UnsafeMutablePointer<Float>
        init(_ t: Transformer) throws {
            W = try SieveImporter.dequantize(t.lmHead)
            V = W.rows; d = W.cols
            Z = .allocate(capacity: block * V)
        }
        deinit { Z.deallocate() }

        func score(_ t: Transformer, _ chunk: [Int32], into s: inout Scores) {
            let T = SieveExpertsTests.T, skip = SieveExpertsTests.skip
            let x = t.xn.contents().bindMemory(to: Float.self, capacity: T * d)
            var nll = [Float](repeating: 0, count: T - skip), top = [Int32](repeating: 0, count: T - skip)
            var r0 = skip
            while r0 < T {
                let b = min(block, T - r0)
                BLAS.sgemm(BLAS.rowMajor, BLAS.noTrans, BLAS.trans, Int32(b), Int32(V), Int32(d), 1, x + r0 * d, Int32(d), W.p, Int32(d), 0, Z, Int32(V))
                let Z = self.Z, V = self.V
                nll.withUnsafeMutableBufferPointer { np in
                    top.withUnsafeMutableBufferPointer { tp in
                        let np = np.baseAddress!, tp = tp.baseAddress!
                        DispatchQueue.concurrentPerform(iterations: b) { i in
                            let z = Z + i * V
                            let target = z[Int(chunk[r0 + i + 1])]
                            var m = -Float.infinity, am = 0
                            for j in 0..<V where z[j] > m { m = z[j]; am = j }
                            for j in 0..<V { z[j] -= m }
                            var n = Int32(V)
                            SieveExpertsTests.vvexpf(z, z, &n)
                            var sum = 0.0
                            for j in 0..<V { sum += Double(z[j]) }
                            np[r0 - skip + i] = m + Float(log(sum)) - target
                            tp[r0 - skip + i] = Int32(am)
                        }
                    }
                }
                r0 += b
            }
            s.nll += nll; s.top += top
        }
    }

    struct Point {
        var label: String
        var group: Int, tau: Float
        var kept: Double                      // share of groups kept, over both domains
        var dNLL: [Double], se: [Double], agree: [Double]   // per domain
        var worst: Double { dNLL.max() ?? 0 }
    }

    /// One teacher-forced pass over every chunk with whatever the probe is set to.
    static func run(_ t: Transformer, _ domains: [Domain], _ scorer: Scorer) throws -> [Scores] {
        try domains.map { dom in
            var s = Scores()
            for c in dom.chunks {
                _ = try t.forward(Array(c[0..<T]), offset: 0)
                scorer.score(t, c, into: &s)
            }
            return s
        }
    }

    static func compare(_ s: [Scores], _ dense: [Scores]) -> (dNLL: [Double], se: [Double], agree: [Double]) {
        var d: [Double] = [], se: [Double] = [], ag: [Double] = []
        for (a, b) in zip(s, dense) {
            let n = Double(a.nll.count)
            var sum = 0.0, sq = 0.0, same = 0
            for i in a.nll.indices {
                let x = Double(a.nll[i]) - Double(b.nll[i])
                sum += x; sq += x * x
                if a.top[i] == b.top[i] { same += 1 }
            }
            let mean = sum / n
            d.append(mean); se.append((max(sq / n - mean * mean, 0) / n).squareRoot()); ag.append(Double(same) / n)
        }
        return (d, se, ag)
    }

    static func measure(_ t: Transformer, _ probe: SieveProbe, _ domains: [Domain], _ scorer: Scorer, dense: [Scores],
                        group: Int, tau: Float, layers: Range<Int>, label: String) throws -> Point {
        probe.mlpGroup = group; probe.mlpThreshold = tau; probe.mlpLayers = layers
        probe.resetCounters()
        let s = try run(t, domains, scorer)
        let k = probe.kept
        let c = compare(s, dense)
        let p = Point(label: label, group: group, tau: tau, kept: Double(k.groups) / Double(max(k.of, 1)), dNLL: c.dNLL, se: c.se, agree: c.agree)
        print(String(format: "H2 %@ group %d tau %.4g: kept %.1f%% | ", label as NSString, group, tau, p.kept * 100)
              + domains.indices.map { String(format: "%@ dNLL %+.4f (se %.4f) top-1 %.1f%%", domains[$0].name as NSString, p.dNLL[$0], p.se[$0], p.agree[$0] * 100) }.joined(separator: " | "))
        return p
    }

    @Test func oracle() throws {
        let env = Self.env, T = Self.T
        guard let repo = env["MLXR_SIEVE_EXPERTS"], let dir = installedModel(repo) else { print("H2: model not installed"); return }
        let nChunks = Int(env["MLXR_SIEVE_CHUNKS"] ?? "") ?? 16
        let steps = env["MLXR_SIEVE_STEPS"] ?? "123"
        func list(_ key: String, _ fallback: String) -> [Float] { (env[key] ?? fallback).split(separator: ",").compactMap { Float($0) } }
        let taus = [1: list("MLXR_SIEVE_TAUS", "0.001,0.003,0.006,0.01,0.013,0.017,0.022,0.03,0.05,0.1,0.15"),
                    64: list("MLXR_SIEVE_TAUS64", "0.01,0.03,0.05,0.07,0.1,0.14,0.2,0.3,0.45,0.6,0.75")]
        let t = try Transformer(directory: dir, contextLength: 1024, sparse: nil)
        let tok = try Tokenizer(directory: dir)
        guard t.config.finalLogitSoftcap == 0 else { print("H2: final logit soft-cap not handled here"); return }
        let L = t.layers.count, I = t.layers[0].intermediate, H = t.config.hiddenSize
        print("H2 \(repo): \(L) layers, hidden \(H), intermediate \(I), vocabulary \(t.vocabSize)")
        let domains = try Self.domains(tok, chunks: nChunks)
        let scorer = try Scorer(t)

        // 0. The probe's sanity: nothing changes at tau = 0, everything is kept at a tiny tau, and
        // the mask of the last layer (whose activations stay in `mlpBuf`) is the documented rule.
        let first = Array(domains[0].chunks[0][0..<T])
        func snapshot() throws -> (xn: [Float], logits: [Float]) {
            let lp = try t.forward(first, offset: 0)
            let xp = t.xn.contents().bindMemory(to: Float.self, capacity: T * H)
            return (Array(UnsafeBufferPointer(start: xp, count: T * H)), Array(UnsafeBufferPointer(start: lp, count: t.vocabSize)))
        }
        func maxDiff(_ a: [Float], _ b: [Float]) -> Float { var m: Float = 0; for i in a.indices { m = max(m, abs(a[i] - b[i])) }; return m }
        let ref = try snapshot()
        let refAct = Array(UnsafeBufferPointer(start: t.mlpBuf.contents().bindMemory(to: Float.self, capacity: T * I), count: T * I))
        let again = try snapshot()
        #expect(again.xn == ref.xn && again.logits == ref.logits, "the dense pass is not deterministic")
        let probe = try SieveProbe(t)
        t.probe = probe
        try probe.prepareMLP(t)
        let zero = try snapshot()
        #expect(zero.xn == ref.xn && zero.logits == ref.logits, "a probe with tau = 0 changed the pass")
        for g in [1, 64] {
            #expect(I % g == 0)
            probe.mlpGroup = g; probe.mlpThreshold = 1e-9; probe.mlpLayers = 0..<Int.max
            probe.resetCounters()
            let tiny = try snapshot()
            let k = probe.kept
            print("H2 sanity group \(g) tau 1e-9: kept \(k.groups) of \(k.of) (expected \(T * L * (I / g))); max |xn diff| \(maxDiff(tiny.xn, ref.xn)), max |logit diff| \(maxDiff(tiny.logits, ref.logits))")
            #expect(k.of == T * L * (I / g))
            #expect(Double(k.groups) >= 0.999 * Double(k.of))   // single neurons with a vanishing activation do fall under 1e-9
            #expect(maxDiff(tiny.logits, ref.logits) < 1e-3)
            // tau = 1: only the groups that tie the row's best survive.
            probe.mlpThreshold = 1
            probe.resetCounters()
            _ = try snapshot()
            print("H2 sanity group \(g) tau 1: kept \(probe.kept.groups) of \(probe.kept.of) (at least one per row and layer: \(T * L))")
            #expect(probe.kept.groups >= T * L && probe.kept.groups <= T * L + T * L / 100)
            // The last layer alone: its unmasked activations are the dense run's.
            let tau: Float = 0.1
            probe.mlpThreshold = tau; probe.mlpLayers = (L - 1)..<L
            probe.resetCounters()
            let last = try snapshot()
            #expect(maxDiff(last.xn, ref.xn) > 0)
            let masked = t.mlpBuf.contents().bindMemory(to: Float.self, capacity: T * I)
            let norms = probe.columnNorms[L - 1].contents().bindMemory(to: Float.self, capacity: I)
            var wrong = 0, expectKept = 0
            var perRow: [Double] = []
            for r in 0..<T {
                var best: Float = 0
                for n in 0..<I { best = max(best, abs(refAct[r * I + n]) * norms[n]) }
                var rowKept = 0
                for g0 in stride(from: 0, to: I, by: g) {
                    var gm: Float = 0
                    for n in g0..<(g0 + g) { gm = max(gm, abs(refAct[r * I + n]) * norms[n]) }
                    let keep = gm >= tau * best
                    if keep { rowKept += 1 }
                    for n in g0..<(g0 + g) where masked[r * I + n] != (keep ? refAct[r * I + n] : 0) { wrong += 1 }
                }
                expectKept += rowKept
                perRow.append(Double(rowKept) / Double(I / g))
            }
            perRow.sort()
            print(String(format: "H2 sanity group %d last layer tau 0.1: kept %d (CPU rule %d) of %d, %d activations differ from the CPU rule; kept share per row: p5 %.3f median %.3f p95 %.3f",
                         g, probe.kept.groups, expectKept, probe.kept.of, wrong, perRow[T / 20], perRow[T / 2], perRow[T - 1 - T / 20]))
            #expect(probe.kept.of == T * (I / g))
            #expect(probe.kept.groups == expectKept && wrong == 0)
        }
        probe.mlpThreshold = 0; probe.mlpLayers = 0..<Int.max

        // Dense reference.
        let dense = try Self.run(t, domains, scorer)
        for (i, dom) in domains.enumerated() {
            print(String(format: "H2 dense %@: NLL %.4f nats/token over %d positions", dom.name as NSString, dense[i].nll.reduce(0) { $0 + Double($1) } / Double(dense[i].nll.count), dense[i].nll.count))
        }
        let rerun = Self.compare(try Self.run(t, domains, scorer), dense)
        print("H2 dense rerun against itself: dNLL \(rerun.dNLL), agreement \(rerun.agree)")
        #expect(rerun.dNLL.allSatisfy { $0 == 0 })

        // 1. Oracle curve, every layer masked.
        var curve: [Int: [Point]] = [:]
        if steps.contains("1") {
            for g in [1, 64] {
                curve[g] = try taus[g]!.map { try Self.measure(t, probe, domains, scorer, dense: dense, group: g, tau: $0, layers: 0..<L, label: "oracle") }
                for (name, target) in [("balanced", 0.02), ("fast", 0.05)] {
                    if let best = curve[g]!.filter({ $0.worst <= target }).min(by: { $0.kept < $1.kept }) {
                        print(String(format: "H2 group %d: smallest kept share within +%.2f nats (%@) on both domains: %.1f%% (tau %.4g, worst dNLL %+.4f)", g, target, name, best.kept * 100, best.tau, best.worst))
                    } else {
                        print(String(format: "H2 group %d: no swept point is within +%.2f nats (%@) on both domains", g, target, name))
                    }
                }
            }
        }

        // How the relative threshold spreads over depth: one layer masked at a time (so its
        // activations are the dense run's), on the first chunk of each domain.
        if steps.contains("1") {
            for (g, tau) in [(1, Float(0.013)), (64, Float(0.1))] {
                var shares: [String] = []
                for l in 0..<L {
                    probe.mlpGroup = g; probe.mlpThreshold = tau; probe.mlpLayers = l..<(l + 1)
                    probe.resetCounters()
                    for dom in domains { _ = try t.forward(Array(dom.chunks[0][0..<T]), offset: 0) }
                    shares.append(String(format: "%.0f", Double(probe.kept.groups) / Double(max(probe.kept.of, 1)) * 100))
                }
                print("H2 group \(g) tau \(tau): kept % by layer (that layer alone masked): " + shares.joined(separator: " "))
            }
        }

        // 2. Depth: the same thresholds with only the lower or the upper half of the layers masked.
        if steps.contains("2") {
            for g in [1, 64] {
                let pts = curve[g] ?? []
                var picks: [Float] = []
                // The point nearest half kept, and the cheapest one whose damage is still moderate.
                if let p = pts.min(by: { abs($0.kept - 0.5) < abs($1.kept - 0.5) }) { picks.append(p.tau) }
                if let p = pts.filter({ $0.worst <= 0.3 }).min(by: { $0.kept < $1.kept }), !picks.contains(p.tau) { picks.append(p.tau) }
                if picks.isEmpty { picks = [0.05] }
                for tau in picks {
                    _ = try Self.measure(t, probe, domains, scorer, dense: dense, group: g, tau: tau, layers: 0..<(L / 2), label: "lower half")
                    _ = try Self.measure(t, probe, domains, scorer, dense: dense, group: g, tau: tau, layers: (L / 2)..<L, label: "upper half")
                }
            }
        }
        probe.mlpThreshold = 0

        // 3. On the CPU, from the dense run's MLP inputs and activations, layer by layer.
        if steps.contains("3") { try Self.predictability(t, probe, domains, chunks: Int(env["MLXR_SIEVE_CAPTURE"] ?? "") ?? 4) }
    }

    /// The `k` best units of every row of `score` [rows × units] (ties at the k-th value are all kept).
    static func top(_ score: [Float], rows: Int, units: Int, k: Int) -> [Bool] {
        var mask = [Bool](repeating: false, count: rows * units)
        guard k > 0 else { return mask }
        score.withUnsafeBufferPointer { sp in
            mask.withUnsafeMutableBufferPointer { mp in
                let sp = sp.baseAddress!, mp = mp.baseAddress!
                DispatchQueue.concurrentPerform(iterations: rows) { r in
                    let row = UnsafeBufferPointer(start: sp + r * units, count: units)
                    let cut = row.sorted(by: >)[min(k, units) - 1]
                    for u in 0..<units where row[u] >= cut { mp[r * units + u] = true }
                }
            }
        }
        return mask
    }

    /// Share of the `truth` units that `predicted` holds, over `rows`.
    static func recall(_ predicted: [Bool], _ truth: [Bool], units: Int, rows: [Int]) -> Double {
        var hit = 0, all = 0
        for r in rows { for u in (r * units)..<((r + 1) * units) where truth[u] { all += 1; if predicted[u] { hit += 1 } } }
        return Double(hit) / Double(max(all, 1))
    }

    /// H2's second half, and how fair the relative threshold is. For every layer, with the dense
    /// run's MLP input x and activations a (captured), the down projection D dequantized:
    /// - the MLP output's relative error |D(a − â)| / |D a| when only a set of neurons is kept:
    ///   the probe's rule, a fixed top-k per row at the same share, and fixed shares, for single
    ///   neurons (score |a_n| · |d_n|) and for groups of 64 consecutive ones (score: the sum of squares);
    /// - how much of the top-k set three cheap predictors find, and the error of keeping what they
    ///   predict: the previous token's set, the neurons most often in the set (fitted on the even
    ///   chunks), and the exact gate and up rows applied to a rank-r projection of x (PCA of the
    ///   even chunks' inputs). Predictors are scored on the odd chunks.
    static func predictability(_ t: Transformer, _ probe: SieveProbe, _ domains: [Domain], chunks: Int) throws {
        let T = Self.T, skip = Self.skip, S = T - skip
        let L = t.layers.count, H = t.config.hiddenSize
        try probe.prepareMLPCapture(t)
        let picked = domains.flatMap { $0.chunks.prefix(chunks).enumerated().map { ($0.offset, $0.element) } }
        let R = picked.count * S
        let fitRows = (0..<R).filter { picked[$0 / S].0 % 2 == 0 }, testRows = (0..<R).filter { picked[$0 / S].0 % 2 == 1 }
        guard !fitRows.isEmpty, !testRows.isEmpty else { print("H2 step 3 needs at least two chunks per domain"); return }
        print("H2 step 3: \(R) rows per layer (\(fitRows.count) to fit, \(testRows.count) to score the predictors); errors are the mean over rows of |D(a - a_kept)| / |D a|")
        var order: [String] = [], totals: [String: [Double]] = [:]
        func note(_ key: String, _ v: Double) { if totals[key] == nil { order.append(key) }; totals[key, default: []].append(v) }

        for l in 0..<L {
            let layer = t.layers[l], I = layer.intermediate
            guard I % 64 == 0 else { continue }
            // Capture.
            probe.captureMLPLayer = l
            var X = [Float](repeating: 0, count: R * H), A = [Float](repeating: 0, count: R * I)
            for (ci, c) in picked.enumerated() {
                _ = try t.forward(Array(c.1[0..<T]), offset: 0)
                memcpy(&X[ci * S * H], probe.mlpInput!.contents() + skip * H * 4, S * H * 4)
                memcpy(&A[ci * S * I], probe.mlpActivation!.contents() + skip * I * 4, S * I * 4)
            }
            probe.captureMLPLayer = nil
            let D = try SieveImporter.dequantize(layer.down)       // [H × I]
            let norms = UnsafeBufferPointer(start: probe.columnNorms[l].contents().bindMemory(to: Float.self, capacity: I), count: I)
            var Y = [Float](repeating: 0, count: R * H)
            BLAS.sgemm(BLAS.rowMajor, BLAS.noTrans, BLAS.trans, Int32(R), Int32(H), Int32(I), 1, A, Int32(I), D.p, Int32(I), 0, &Y, Int32(H))
            let yNorm = (0..<R).map { r in Y.withUnsafeBufferPointer { BLAS.snrm2(Int32(H), $0.baseAddress! + r * H, 1) } }
            var E = [Float](repeating: 0, count: R * H), dropped = [Float](repeating: 0, count: R * I)
            /// Mean relative output error over `rows` when the neurons of `keep` [R × I] are kept.
            func error(_ keep: [Bool], _ rows: [Int]? = nil) -> Double {
                for i in 0..<(R * I) { dropped[i] = keep[i] ? 0 : A[i] }
                BLAS.sgemm(BLAS.rowMajor, BLAS.noTrans, BLAS.trans, Int32(R), Int32(H), Int32(I), 1, dropped, Int32(I), D.p, Int32(I), 0, &E, Int32(H))
                var sum = 0.0
                let rows = rows ?? Array(0..<R)
                for r in rows { sum += Double(E.withUnsafeBufferPointer { BLAS.snrm2(Int32(H), $0.baseAddress! + r * H, 1) }) / Double(max(yNorm[r], 1e-20)) }
                return sum / Double(rows.count)
            }
            func expand(_ m: [Bool], _ g: Int) -> [Bool] { g == 1 ? m : (0..<(R * I)).map { m[($0 / I) * (I / g) + ($0 % I) / g] } }
            // Scores: single neurons, and groups of 64 (best member for the probe's rule, energy for top-k).
            var score = [Float](repeating: 0, count: R * I)
            for r in 0..<R { for n in 0..<I { score[r * I + n] = abs(A[r * I + n]) * norms[n] } }
            let G = 64, NG = I / G
            var groupBest = [Float](repeating: 0, count: R * NG), groupEnergy = groupBest
            for r in 0..<R { for g in 0..<NG {
                var m: Float = 0, e: Float = 0
                for n in (g * G)..<((g + 1) * G) { let v = score[r * I + n]; m = max(m, v); e += v * v }
                groupBest[r * NG + g] = m; groupEnergy[r * NG + g] = e
            } }
            var line = "H2 layer \(l):"
            func add(_ key: String, _ v: Double, percent: Bool = false) {
                note(key, v)
                line += String(format: percent ? " | %@ %.1f%%" : " | %@ %.3f", key as NSString, percent ? v * 100 : v)
            }

            // The probe's rule against a fixed top-k per row at the same share of the layer.
            for (g, tau, sc, units) in [(1, Float(0.013), score, I), (G, Float(0.1), groupBest, NG)] {
                var keep = [Bool](repeating: false, count: R * units), n = 0
                for r in 0..<R {
                    var best: Float = 0
                    for u in 0..<units { best = max(best, sc[r * units + u]) }
                    for u in 0..<units where sc[r * units + u] >= tau * best { keep[r * units + u] = true; n += 1 }
                }
                let k = Int((Double(n) / Double(R)).rounded())
                add("g\(g) rule tau \(tau) kept", Double(n) / Double(R * units), percent: true)
                add("g\(g) rule err", error(expand(keep, g)))
                add("g\(g) top-k same share err", error(expand(Self.top(g == 1 ? score : groupEnergy, rows: R, units: units, k: k), g)))
            }
            // Fixed shares.
            var oracle: [Int: [Double: [Bool]]] = [1: [:], G: [:]]
            for share in [0.5, 0.25, 0.1] {
                for (g, sc, units) in [(1, score, I), (G, groupEnergy, NG)] {
                    let m = Self.top(sc, rows: R, units: units, k: Int((share * Double(units)).rounded()))
                    oracle[g]![share] = m
                    add("g\(g) top \(Int(share * 100))% err", error(expand(m, g)))
                }
            }
            print(line)

            // Predictors of the top-k set.
            line = "H2 layer \(l) predictors:"
            var fx = [Float](repeating: 0, count: fitRows.count * H)
            for (i, r) in fitRows.enumerated() { for j in 0..<H { fx[i * H + j] = X[r * H + j] } }
            let (mu, _, vec) = SieveHeadTests.fit(fx, n: fitRows.count, d: H)
            let gu = try layer.gateUp.map { try SieveImporter.dequantize($0) }
            /// act(G x̂) · (U x̂) with x̂ the rank-r projection of x (r = H: x itself).
            func screen(rank r: Int) -> [Float] {
                var xh = X
                if r < H {
                    let Q = (0..<(r * H)).map { Float(vec[$0]) }, m = mu.map { Float($0) }
                    var xc = X
                    for i in 0..<R { for j in 0..<H { xc[i * H + j] -= m[j] } }
                    var C = [Float](repeating: 0, count: R * r)
                    BLAS.sgemm(BLAS.rowMajor, BLAS.noTrans, BLAS.trans, Int32(R), Int32(r), Int32(H), 1, xc, Int32(H), Q, Int32(H), 0, &C, Int32(r))
                    BLAS.sgemm(BLAS.rowMajor, BLAS.noTrans, BLAS.noTrans, Int32(R), Int32(H), Int32(r), 1, C, Int32(r), Q, Int32(H), 0, &xh, Int32(H))
                    for i in 0..<R { for j in 0..<H { xh[i * H + j] += m[j] } }
                }
                var g = [Float](repeating: 0, count: R * I), u = g
                let gateP = gu[0].p, upP = gu.count == 1 ? gu[0].p + I * H : gu[1].p   // gate rows, then up rows
                BLAS.sgemm(BLAS.rowMajor, BLAS.noTrans, BLAS.trans, Int32(R), Int32(I), Int32(H), 1, xh, Int32(H), gateP, Int32(H), 0, &g, Int32(I))
                BLAS.sgemm(BLAS.rowMajor, BLAS.noTrans, BLAS.trans, Int32(R), Int32(I), Int32(H), 1, xh, Int32(H), upP, Int32(H), 0, &u, Int32(I))
                for i in 0..<(R * I) { g[i] = g[i] / (1 + exp(-g[i])) * u[i] }
                return g
            }
            if l == 0 || l == L - 1 {
                // The CPU's activations from the captured input against the captured ones (SiLU models).
                let a = screen(rank: H)
                var worst: Float = 0, big: Float = 0
                for i in 0..<(R * I) { worst = max(worst, abs(a[i] - A[i])); big = max(big, abs(A[i])) }
                print("H2 layer \(l) capture check: max |CPU activation - captured| \(worst) (largest |activation| \(big))")
                #expect(worst <= 0.02 * max(big, 1))
            }
            func groupEnergies(_ a: [Float]) -> [Float] {
                var e = [Float](repeating: 0, count: R * NG)
                for r in 0..<R { for n in 0..<I { let v = abs(a[r * I + n]) * norms[n]; e[r * NG + n / G] += v * v } }
                return e
            }
            let lowRank = [H / 8, H / 4].map { ($0, screen(rank: $0)) }
            for (g, units, share) in [(1, I, 0.5), (1, I, 0.25), (G, NG, 0.5)] {
                let truth = oracle[g]![share]!, k = Int((share * Double(units)).rounded())
                let tag = "g\(g) top \(Int(share * 100))%"
                add("\(tag) oracle err (odd chunks)", error(expand(truth, g), testRows))
                // The previous token's set (the first scored row of a chunk has none).
                var prev = truth
                for r in 0..<R where r % S != 0 { for u in 0..<units { prev[r * units + u] = truth[(r - 1) * units + u] } }
                let prevRows = testRows.filter { $0 % S != 0 }
                add("\(tag) previous-token recall", Self.recall(prev, truth, units: units, rows: prevRows), percent: true)
                add("\(tag) previous-token err", error(expand(prev, g), prevRows))
                // The units most often in the set on the even chunks.
                var freq = [Float](repeating: 0, count: units)
                for r in fitRows { for u in 0..<units where truth[r * units + u] { freq[u] += 1 } }
                let cut = freq.sorted(by: >)[k - 1]
                var fixed = [Bool](repeating: false, count: R * units), taken = 0
                for u in 0..<units where freq[u] >= cut && taken < k { taken += 1; for r in 0..<R { fixed[r * units + u] = true } }
                add("\(tag) static recall", Self.recall(fixed, truth, units: units, rows: testRows), percent: true)
                add("\(tag) static err", error(expand(fixed, g), testRows))
                // The low-rank screen, keeping as many units as the oracle and half as many again.
                for (r, a) in lowRank {
                    let sc: [Float] = g == 1 ? (0..<(R * I)).map { abs(a[$0]) * norms[$0 % I] } : groupEnergies(a)
                    for extra in [1.0, 1.5] where Double(k) * extra <= Double(units) {
                        let m = Self.top(sc, rows: R, units: units, k: Int(Double(k) * extra))
                        let name = "\(tag) rank-\(r) screen keeping \(Int(share * extra * 100))%"
                        add("\(name) recall", Self.recall(m, truth, units: units, rows: testRows), percent: true)
                        add("\(name) err", error(expand(m, g), testRows))
                    }
                }
            }
            print(line)
        }
        print("H2 step 3, mean over the \(totals.values.first?.count ?? 0) layers:")
        for key in order {
            let v = totals[key]!, mean = v.reduce(0, +) / Double(v.count)
            let pct = key.hasSuffix("recall") || key.hasSuffix("kept")
            print(String(format: pct ? "H2   %@: %.1f%% (layers %.1f%% to %.1f%%)" : "H2   %@: %.3f (layers %.3f to %.3f)", key as NSString,
                         pct ? mean * 100 : mean, pct ? v.min()! * 100 : v.min()!, pct ? v.max()! * 100 : v.max()!))
        }
    }
}
