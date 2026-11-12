import Foundation
import Metal

/// Accelerate's BLAS and LAPACK through `dlsym`: the importer's linear algebra (a system
/// framework, and the classic interfaces without their deprecation warnings).
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
    /// C [N × N] = alpha · Aᵀ A + beta · C (with `trans`, A [K × N])
    static let dsyrk = symbol("cblas_dsyrk", as: (@convention(c) (Int32, Int32, Int32, Int32, Int32, Double, UnsafePointer<Double>, Int32, Double, DM, Int32) -> Void).self)
    /// Symmetric eigendecomposition (divide and conquer), column-major.
    static let dsyevd = symbol("dsyevd_", as: (@convention(c) (UnsafeMutablePointer<Int8>, UnsafeMutablePointer<Int8>, IM, DM, IM, DM, DM, IM, IM, IM, IM) -> Int32).self)
}

public struct SieveImportOptions: Sendable {
    /// Calibration text files. Empty: the model writes its own (answers to built-in prompts).
    public var textFiles: [URL] = []
    /// Hidden states to collect (about).
    public var tokens = 12_000
    /// Rank of the sketch; nil = an eighth of the hidden size.
    public var rank: Int? = nil
    /// Row budget; nil = 16,384, or less for a small vocabulary.
    public var rows: Int? = nil
    public var seed: UInt64 = 42
    public init() {}
}

public struct SieveImportReport: Sendable {
    public var info = SieveHeadInfo()
    /// Per row budget tried: its margin, and held-out states whose dense argmax the sketch ranked below it.
    public var budgets: [(rows: Int, delta: Float, missed: Int)] = []
    public var rowsMax = 0
    /// Every rank tried, in order; the last one is the sidecar's.
    public var tried: [SieveHeadInfo] = []
    public var sidecarBytes = 0
    public var seconds: Double = 0
    public var url: URL?
}

/// Fits the Head sidecar of a model (see `SieveProfile`) and measures it on held-out states.
public enum SieveImporter {
    /// A row-major f32 matrix in its own allocation (a dequantized head is gigabytes).
    final class Matrix {
        let rows: Int, cols: Int
        let p: UnsafeMutablePointer<Float>
        init(_ rows: Int, _ cols: Int) {
            self.rows = rows; self.cols = cols
            p = .allocate(capacity: max(rows * cols, 1))
            p.initialize(repeating: 0, count: max(rows * cols, 1))
        }
        deinit { p.deallocate() }
    }

    /// IEEE float → half bits, round to nearest even (`Float16` is unavailable on x86_64 macOS).
    static func halfBits(_ f: Float) -> UInt16 {
        let x = f.bitPattern
        let sign = UInt16((x >> 16) & 0x8000)
        let e = Int((x >> 23) & 0xFF)
        var man = x & 0x7FFFFF
        if e == 0xFF { return sign | 0x7C00 | (man != 0 ? 0x200 : 0) }
        let exp = e - 127 + 15
        if exp >= 31 { return sign | 0x7C00 }
        if exp <= 0 {
            if exp < -10 { return sign }
            man |= 0x800000
            let shift = UInt32(14 - exp)
            var h = man >> shift
            let rem = man & ((UInt32(1) << shift) - 1), half = UInt32(1) << (shift - 1)
            if rem > half || (rem == half && h & 1 == 1) { h += 1 }
            return sign | UInt16(h)
        }
        var h = UInt32(exp) << 10 | (man >> 13)
        let rem = man & 0x1FFF
        if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) { h += 1 }
        return sign | UInt16(h)
    }

    /// A quantized projection as f32: w = scale · q + bias per group (MLX affine; codes are a
    /// little-endian bit stream per row).
    static func dequantize(_ l: Linear) throws -> Matrix {
        guard l.bits > 0, l.bits <= 8, let s = l.scales, let b = l.biases else {
            throw EngineError.unsupported("Sieve Head needs a quantized output projection (\(l.description))")
        }
        let V = l.outDim, d = l.inDim, g = l.groupSize, groups = d / g, bits = l.bits
        let rowBytes = d * bits / 8
        let scales = try s.floats(), biases = try b.floats()
        guard scales.count == V * groups, biases.count == V * groups, l.weight.nbytes == V * rowBytes, l.weight.offset >= 0,
              l.weight.nbytes <= l.weight.buffer.length - l.weight.offset else { throw EngineError.invalidFile("head shapes") }
        let w = Matrix(V, d)
        let src = UnsafeRawPointer(l.weight.buffer.contents() + l.weight.offset)
        let mask = UInt16((1 << bits) - 1)
        for r in 0..<V {
            let out = w.p + r * d, base = r * rowBytes
            for j in 0..<d {
                let bit = j * bits, at = base + bit / 8, sh = bit % 8
                var v = UInt16(src.load(fromByteOffset: at, as: UInt8.self))
                if sh + bits > 8 { v |= UInt16(src.load(fromByteOffset: at + 1, as: UInt8.self)) << 8 }
                out[j] = Float((v >> UInt16(sh)) & mask) * scales[r * groups + j / g] + biases[r * groups + j / g]
            }
        }
        return w
    }

    /// Prompts the model answers to produce its own calibration text.
    static let seedPrompts = [
        "Explain how a hash table works and when to use one.",
        "Write a short story about a lighthouse keeper who finds a message in a bottle.",
        "Write a Python function that merges two sorted lists, with tests.",
        "What were the main causes of the First World War?",
        "Give me a recipe for a vegetarian lasagna, with quantities.",
        "Explain the difference between TCP and UDP to a beginner.",
        "Write a Swift struct for a 2D vector with the usual operators.",
        "Summarise the plot of Romeo and Juliet in five sentences.",
        "Traduce al español: The weather was terrible, so we stayed home and played cards all evening.",
        "List ten tips for a job interview, as a numbered list.",
        "What is 17 times 23? Show the steps.",
        "Write a SQL query that returns the three customers with the highest total orders, and explain it.",
        "Describe the water cycle for a ten-year-old.",
        "Write a polite email asking a landlord to repair a broken heater.",
        "Explique en français pourquoi le ciel est bleu.",
        "Write a JavaScript function that debounces another function, with comments.",
        "Compare electric cars and petrol cars in a table with five rows.",
        "Return a JSON object describing a book: title, author, year and three tags.",
        "What is the theory of evolution by natural selection?",
        "Write a poem about autumn in four stanzas.",
        "Explain what a mortgage is and how interest is calculated.",
        "Write a bash script that renames every .txt file in a directory to .md.",
        "Erkläre auf Deutsch, wie ein Kühlschrank funktioniert.",
        "Give me a seven-day beginner running plan.",
        "What is the difference between a virus and a bacterium?",
        "Write a Rust function that reverses the words of a sentence.",
        "Tell me about the history of the Roman Empire in three paragraphs.",
        "Solve for x: 3x + 7 = 2x - 5, and check the answer.",
        "Write a product description for a stainless steel water bottle.",
        "Explain recursion with an example in C.",
        "What should I pack for a week of hiking in the mountains?",
        "用中文介绍一下长城的历史。",
        "Write a dialogue between a customer and a waiter in a restaurant.",
        "Explain how vaccines train the immune system.",
        "Write a regular expression that matches an ISO date and explain each part.",
        "Describe the rules of chess briefly.",
        "What are black holes and how do they form?",
        "Write a commit message and a short changelog entry for a bug fix in a date parser.",
        "Give three arguments for and three against remote work.",
        "Explain the Pythagorean theorem and prove it.",
    ]

    /// What a calibration state is used for. The directions are fitted on `fit`, the margin and the
    /// rank are chosen on `tune`, and the figures a sidecar reports are measured on `test`, which
    /// nothing was chosen on.
    enum Split: UInt8 { case fit, tune, test }

    /// The split of seed prompt `i`: ten of forty to `test` (two of them not in English), five to `tune`.
    static func split(prompt i: Int) -> Split { i % 4 == 2 ? .test : i % 8 == 3 ? .tune : .fit }
    /// The split of chunk `c` of a text file: one in five to `test`, one in five to `tune`.
    static func split(chunk c: Int) -> Split { c % 5 == 4 ? .test : c % 5 == 3 ? .tune : .fit }

    /// Final normalised hidden states [n × d] with their split: the model's answers to `seedPrompts`,
    /// or 512-token chunks of the given files.
    static func collect(_ t: Transformer, _ tok: Tokenizer, options o: SieveImportOptions,
                        progress: (String) -> Void) throws -> (x: [Float], held: [Split], source: String) {
        let d = t.config.hiddenSize
        var x: [Float] = [], held: [Split] = []
        if !o.textFiles.isEmpty {
            let T = 512, skip = 16
            let per = max(1, o.tokens / (T - skip) / o.textFiles.count)
            for f in o.textFiles {
                let ids = tok.encode(try String(contentsOf: f, encoding: .utf8)).map(Int32.init)
                let n = min(per, ids.count / T)
                guard n > 0 else { throw EngineError.invalidOutput("\(f.lastPathComponent): fewer than \(T) tokens") }
                let stride = (ids.count - T) / max(n - 1, 1)
                for c in 0..<n {
                    _ = try t.forward(Array(ids[(c * stride)..<(c * stride + T)]), offset: 0)
                    let rows = t.xn.contents().bindMemory(to: Float.self, capacity: T * d)
                    x.append(contentsOf: UnsafeBufferPointer(start: rows + skip * d, count: (T - skip) * d))
                    held.append(contentsOf: [Split](repeating: split(chunk: c), count: T - skip))
                }
                progress("read \(f.lastPathComponent): \(n) chunks")
            }
            return (x, held, "text: " + o.textFiles.map(\.lastPathComponent).joined(separator: ", "))
        }
        let tpl = ChatTemplate(template: tok.chatTemplate, modelType: t.config.modelType, bosToken: tok.bosToken, toolTemplate: tok.toolUseTemplate)
        var stops = Set(t.config.eosTokenIds)
        if let e = tok.eosToken, let id = tok.tokenId(e) { stops.insert(id) }
        for s in tpl.stopTokens { if let id = tok.tokenId(s) { stops.insert(id) } }
        var sp = SamplingParameters()
        sp.temperature = 0.8; sp.topK = 50; sp.topP = 0.95; sp.repetitionPenalty = 1.0
        let sample = SampleParams(sp, vocab: min(tok.vocabularySize, t.vocabSize))
        let per = max(64, o.tokens / Self.seedPrompts.count)
        for (i, prompt) in Self.seedPrompts.enumerated() {
            let r = tpl.renderMarked([ChatMessage(role: .user, content: prompt)], enableThinking: false)
            let ids = tok.encode(r.text, plainRanges: r.content).map(Int32.init)
            guard !ids.isEmpty, ids.count + per + 2 < t.maxSeq else { continue }
            t.seedSampler(o.seed &+ UInt64(i) &* 7919)
            var token = try t.prefill(ids, offset: 0, sample: sample)
            var pos = ids.count, n = 0
            while n < per, !stops.contains(token), pos + 2 < t.maxSeq {
                token = try t.finish(try t.beginDecodeStep(offset: pos, sample: sample))
                x.append(contentsOf: UnsafeBufferPointer(start: t.xn.contents().bindMemory(to: Float.self, capacity: d), count: d))
                held.append(split(prompt: i))
                pos += 1; n += 1
            }
            if i % 8 == 7 { progress("calibration: \(held.count) states from \(i + 1) of \(Self.seedPrompts.count) prompts") }
        }
        return (x, held, "self-generated: \(Self.seedPrompts.count) prompts, temperature 0.8, seed \(o.seed)")
    }

    /// The `r` leading eigenvectors of the second moment of `x` [n × d] as rows [r × d], in double precision.
    static func basis(_ x: [Float], n: Int, d: Int, r: Int) throws -> [Float] {
        var c = [Double](repeating: 0, count: n * d)
        for i in 0..<(n * d) { c[i] = Double(x[i]) }
        var m = [Double](repeating: 0, count: d * d)
        BLAS.dsyrk(BLAS.rowMajor, BLAS.upper, BLAS.trans, Int32(d), Int32(n), 1 / Double(n), c, Int32(d), 0, &m, Int32(d))
        c = []
        // Column-major for LAPACK: the row-major upper triangle is its lower triangle.
        var jobz = Int8(UInt8(ascii: "V")), uplo = Int8(UInt8(ascii: "L"))
        var nn = Int32(d), lda = Int32(d), info = Int32(0)
        var w = [Double](repeating: 0, count: d)
        var lwork = Int32(-1), liwork = Int32(-1)
        var wq = [Double](repeating: 0, count: 1), iwq = [Int32](repeating: 0, count: 1)
        _ = BLAS.dsyevd(&jobz, &uplo, &nn, &m, &lda, &w, &wq, &lwork, &iwq, &liwork, &info)
        lwork = Int32(wq[0]); liwork = iwq[0]
        var work = [Double](repeating: 0, count: Int(lwork)), iwork = [Int32](repeating: 0, count: Int(liwork))
        _ = BLAS.dsyevd(&jobz, &uplo, &nn, &m, &lda, &w, &work, &lwork, &iwork, &liwork, &info)
        guard info == 0 else { throw EngineError.invalidOutput("eigendecomposition failed (\(info))") }
        // Ascending eigenvalues, eigenvector k in column k (column-major): the last r, as rows.
        var q = [Float](repeating: 0, count: r * d)
        for k in 0..<r { for j in 0..<d { q[k * d + j] = Float(m[(d - 1 - k) * d + j]) } }
        return q
    }

    /// Quantizes `s` [V × r] in place to MLX 4-bit affine, groups of 64 (it then holds the values the
    /// kernel will compute with). Returns the packed codes and the f16 scales and biases.
    static func quantize4(_ s: Matrix) -> (codes: Data, scales: Data, biases: Data) {
        let V = s.rows, r = s.cols, groups = r / 64
        var codes = [UInt32](repeating: 0, count: V * r / 8)
        var scales = [UInt16](repeating: 0, count: V * groups), biases = scales
        for row in 0..<V {
            let p = s.p + row * r
            for g in 0..<groups {
                var lo = Float.infinity, hi = -Float.infinity
                for j in (g * 64)..<(g * 64 + 64) { lo = min(lo, p[j]); hi = max(hi, p[j]) }
                let sb = halfBits((hi - lo) / 15), bb = halfBits(lo)
                let sc = Tensor.halfToFloat(sb), bi = Tensor.halfToFloat(bb)
                scales[row * groups + g] = sb; biases[row * groups + g] = bb
                for j in (g * 64)..<(g * 64 + 64) {
                    let q = sc > 0 ? min(15, max(0, ((p[j] - bi) / sc).rounded())) : 0
                    codes[(row * r + j) / 8] |= UInt32(q) << UInt32(4 * (j % 8))
                    p[j] = q * sc + bi
                }
            }
        }
        return (codes.withUnsafeBufferPointer { Data(buffer: $0) }, scales.withUnsafeBufferPointer { Data(buffer: $0) },
                biases.withUnsafeBufferPointer { Data(buffer: $0) })
    }

    static let budgets = [512, 1024, 2048, 4096, 8192, 16384]

    /// Fits and writes `<directory>/sieve/head.safetensors`. Nothing else may use the GPU.
    public static func importHead(directory: URL, options o: SieveImportOptions = SieveImportOptions(),
                                  progress: (String) -> Void = { _ in }) throws -> SieveImportReport {
        let start = Date()
        if let rows = o.rows, !budgets.contains(rows) {
            throw EngineError.unsupported("rows \(rows): expected one of \(budgets.map(String.init).joined(separator: ", "))")
        }
        // The fit holds the model and, on the CPU, the dequantized head and the sketch as floats.
        let cfg = try ModelConfig(directory: directory)
        let model = try InferenceEngine.memoryNeed(directory: directory, contextLength: 2048, sparse: nil).bytes
        let need = UInt64(model) + 2 * UInt64(cfg.vocabSize) * UInt64(cfg.hiddenSize) * 4
        let budget = GPU.shared.device.recommendedMaxWorkingSetSize
        guard need < budget else {
            let f = ByteCountFormatter()
            throw EngineError.unsupported("fitting needs about \(f.string(fromByteCount: Int64(need))) but about \(f.string(fromByteCount: Int64(budget))) can be used")
        }
        let t = try Transformer(directory: directory, contextLength: 2048, sparse: nil)
        let tok = try Tokenizer(directory: directory)
        let head = t.lmHead
        guard SieveHead.supports(head) else { throw EngineError.unsupported("Sieve Head needs a quantized output projection (\(head.description))") }
        let d = head.inDim, rowsV = head.outDim
        // The sampler never sees head rows the tokenizer has no token for, and neither does the screen.
        let V = min(tok.vocabularySize, rowsV)
        // Ranks to try, cheapest first: the first that meets the target is kept.
        func round64(_ x: Int) -> Int { min(d, max(64, (x + 63) / 64 * 64)) }
        var ranks = o.rank.map { [$0] } ?? [round64(d / 8), round64(d / 4), round64(d / 2)]
        ranks = ranks.enumerated().filter { ranks.firstIndex(of: $0.element) == $0.offset }.map(\.element)
        guard ranks.allSatisfy({ $0 >= 64 && $0 <= d && $0 % 64 == 0 }) else {
            throw EngineError.unsupported("sieve rank \(ranks[0]) (a multiple of 64, at most \(d))")
        }

        // 1. Hidden states.
        let c = try collect(t, tok, options: o, progress: progress)
        let n = c.held.count
        /// The states of one split, at most `limit` of them, taken evenly so every prompt contributes.
        func states(_ which: Split, limit: Int) -> [Float] {
            let rows = (0..<n).filter { c.held[$0] == which }
            let keep = rows.count <= limit ? rows : (0..<limit).map { rows[$0 * rows.count / limit] }
            var out = [Float]()
            out.reserveCapacity(keep.count * d)
            c.x.withUnsafeBufferPointer { p in for i in keep { out.append(contentsOf: p[(i * d)..<((i + 1) * d)]) } }
            return out
        }
        var fitX = states(.fit, limit: .max)
        let tuneX = states(.tune, limit: 1024), testX = states(.test, limit: 2048)
        let nFit = fitX.count / d, nTune = tuneX.count / d, nTest = testX.count / d
        guard nFit >= 2 * ranks.max()!, nTune >= 128, nTest >= 256 else {
            throw EngineError.invalidOutput("too few calibration states (\(nFit) to fit, \(nTune) to tune, \(nTest) to test) for rank \(ranks.max()!)")
        }
        progress("\(nFit) states to fit, \(nTune) to tune, \(nTest) to test")
        let allDirections = try basis(fitX, n: nFit, d: d, r: ranks.max()!)
        fitX = []
        let W = try dequantize(head)
        let groups = d / head.groupSize
        let headRow = d * head.bits / 8 + groups * 2 * (head.scales?.dtype.size ?? 2)
        typealias Packed = (codes: Data, scales: Data, biases: Data)

        func attempt(_ r: Int) -> (report: SieveImportReport, tuned: Bool, basis: Data, packed: Packed) {
        progress("fitting rank \(r)")
        // 2. The basis (rounded to f16, which is what the kernel reads) and the sketch S = W Qᵀ.
        var q = Array(allDirections[0..<(r * d)])
        let qBits = q.map(halfBits)
        for i in q.indices { q[i] = Tensor.halfToFloat(qBits[i]) }
        let S = Matrix(rowsV, r)
        BLAS.sgemm(BLAS.rowMajor, BLAS.noTrans, BLAS.trans, Int32(rowsV), Int32(r), Int32(d), 1, W.p, Int32(d), q, Int32(d), 0, S.p, Int32(r))
        let packed = quantize4(S)

        // 3. The tuning states: the rank of the dense argmax among the sketch values (rows above it),
        //    and how far below the best sketch value the `budget`-th row sits.
        let B = 32
        var Z = [Float](repeating: 0, count: B * V), Zh = Z, XR = [Float](repeating: 0, count: B * r)
        var rank = [Int](repeating: 0, count: nTune)
        var gk = [[Float]](repeating: [], count: budgets.count)
        let bins = SieveHead.bins
        func logits(_ x: UnsafePointer<Float>, _ b: Int) {
            BLAS.sgemm(BLAS.rowMajor, BLAS.noTrans, BLAS.trans, Int32(b), Int32(r), Int32(d), 1, x, Int32(d), q, Int32(d), 0, &XR, Int32(r))
            BLAS.sgemm(BLAS.rowMajor, BLAS.noTrans, BLAS.trans, Int32(b), Int32(V), Int32(r), 1, XR, Int32(r), S.p, Int32(r), 0, &Zh, Int32(V))
            BLAS.sgemm(BLAS.rowMajor, BLAS.noTrans, BLAS.trans, Int32(b), Int32(V), Int32(d), 1, x, Int32(d), W.p, Int32(d), 0, &Z, Int32(V))
        }
        var hist = [Int](repeating: 0, count: bins)
        tuneX.withUnsafeBufferPointer { xp in
            let width: Float = 1.0 / 16
            var s0 = 0
            while s0 < nTune {
                let b = min(B, nTune - s0)
                logits(xp.baseAddress! + s0 * d, b)
                for s in 0..<b {
                    var arg = 0, best = -Float.infinity, top = -Float.infinity
                    for i in 0..<V {
                        if Z[s * V + i] > best { best = Z[s * V + i]; arg = i }
                        top = max(top, Zh[s * V + i])
                    }
                    let own = Zh[s * V + arg]
                    var above = 0
                    for i in 0..<bins { hist[i] = 0 }
                    for i in 0..<V {
                        let v = Zh[s * V + i]
                        if v > own { above += 1 }
                        // A non-finite logit has no bin (and no defined integer value under -Ounchecked).
                        let b = (top - v) / width
                        if b >= 0 { hist[b < Float(bins - 1) ? Int(b) : bins - 1] += 1 }
                    }
                    rank[s0 + s] = above
                    var cum = 0, bi = 0
                    for k in budgets.indices {
                        while bi < bins - 1, cum + hist[bi] < budgets[k] { cum += hist[bi]; bi += 1 }
                        gk[k].append(Float(bi + 1) * width)
                    }
                }
                s0 += b
            }
        }
        var report = SieveImportReport()
        var chosen = budgets.count - 1
        var firstClean: Int?
        for k in budgets.indices {
            let missed = rank.filter { $0 >= budgets[k] }.count
            // The margin: where the budget-th row sits in nine of ten states. Past it, fewer rows pass.
            let delta = gk[k].sorted()[nTune * 9 / 10]
            report.budgets.append((budgets[k], delta, missed))
            if firstClean == nil, missed == 0 { firstClean = k }
        }
        _ = firstClean
        // Exact rows are cheap next to the sketch (16,384 rows of a 1B-class head are 9 to 19 MB), so
        // the budget is the largest that is still a small part of the vocabulary.
        chosen = budgets.lastIndex { $0 <= V / 4 } ?? 0
        let rowBudget = min(o.rows ?? budgets[chosen], V)
        let delta = report.budgets[budgets.firstIndex(of: rowBudget) ?? chosen].delta
        progress("row budget \(rowBudget), margin \(delta); measuring")

        // 4. The screen as the kernels run it (`head_hist`, `head_cut2`, `head_select`): on the tuning
        //    states to choose the rank, and on the test states for the figures the sidecar reports.
        func measure(_ testX: [Float], _ nTest: Int) -> (info: SieveHeadInfo, rowsMax: Int) {
        var rowsSum = 0, rowsMax = 0, tvSum = 0.0
        let K = 40
        var poolD = [Int](), poolS = [Int]()
        /// The indices of the K largest of `v(i)`, sorted (ties by lower index).
        func topK(_ v: (Int) -> Float, into out: inout [Int]) {
            var best = [(Float, Int)]()
            best.reserveCapacity(K + 1)
            var floor = -Float.infinity
            for i in 0..<V {
                let x = v(i)
                if best.count == K, x <= floor { continue }
                var j = best.count
                best.append((x, i))
                while j > 0, best[j - 1].0 < x { best[j] = best[j - 1]; j -= 1 }
                best[j] = (x, i)
                if best.count > K { best.removeLast() }
                if best.count == K { floor = best[K - 1].0 }
            }
            out = best.map(\.1).sorted()
        }
        // Per state, relative to the dense maximum: the probability mass of the exact rows (E), of the
        // other rows under the dense head (T) and under the sketch (Th), Σ p (z − ẑ) over those rows (X),
        // whether the dense argmax is an exact row, and how far the best exact row is above the best other.
        var stat: [(E: Double, T: Double, Th: Double, X: Double, kept: Bool, lead: Float)] = []
        testX.withUnsafeBufferPointer { xp in
            var s0 = 0
            while s0 < nTest {
                let b = min(B, nTest - s0)
                logits(xp.baseAddress! + s0 * d, b)
                for s in 0..<b {
                    var top = -Float.infinity
                    for i in 0..<V { top = max(top, Zh[s * V + i]) }
                    for i in 0..<bins { hist[i] = 0 }
                    let scale = Float(bins) / delta
                    for i in 0..<V {
                        let x = (top - Zh[s * V + i]) * scale
                        if x >= 0, x < Float(bins) { hist[Int(x)] += 1 }
                    }
                    var cum = 0, nb = 0
                    while nb < bins, cum + hist[nb] <= rowBudget { cum += hist[nb]; nb += 1 }
                    var rows = 0, arg = 0, best = -Float.infinity
                    for i in 0..<V where Z[s * V + i] > best { best = Z[s * V + i]; arg = i }
                    var E = 0.0, T = 0.0, Th = 0.0, X = 0.0
                    var kept = false, bestExact = -Float.infinity, bestOther = -Float.infinity
                    for i in 0..<V {
                        let zi = Z[s * V + i], v = Zh[s * V + i]
                        let x = (top - v) * scale
                        let p = Double(exp(zi - best))
                        if x >= 0, x < Float(nb) {
                            rows += 1; E += p
                            bestExact = max(bestExact, zi)
                            if i == arg { kept = true }
                        } else {
                            T += p; Th += Double(exp(v - best)); X += p * Double(zi - v)
                            bestOther = max(bestOther, v)
                        }
                    }
                    rowsSum += rows; rowsMax = max(rowsMax, rows)
                    topK({ Z[s * V + $0] }, into: &poolD)
                    topK({ i in let v = Zh[s * V + i], x = (top - v) * scale; return x >= 0 && x < Float(nb) ? Z[s * V + i] : v }, into: &poolS)
                    // Total variation between softmax over the dense pool and softmax over the screened pool.
                    func screened(_ i: Int) -> Float { let v = Zh[s * V + i], x = (top - v) * scale; return x >= 0 && x < Float(nb) ? Z[s * V + i] : v }
                    let mD = poolD.map { Z[s * V + $0] }.max()!, mS = poolS.map(screened).max()!
                    let sumD = poolD.reduce(0.0) { $0 + Double(exp(Z[s * V + $1] - mD)) }
                    let sumS = poolS.reduce(0.0) { $0 + Double(exp(screened($1) - mS)) }
                    var tv = 0.0
                    for i in Set(poolD).union(poolS) {
                        let p = poolD.contains(i) ? Double(exp(Z[s * V + i] - mD)) / sumD : 0
                        let q = poolS.contains(i) ? Double(exp(screened(i) - mS)) / sumS : 0
                        tv += abs(p - q)
                    }
                    tvSum += tv / 2
                    stat.append((E, T, Th, X, kept, bestExact - bestOther))
                }
                s0 += b
            }
        }
        // KL(dense ‖ screened) = X / (E + T) − log(E + T) + log(E + Th), per state.
        let kls = stat.map { $0.X / ($0.E + $0.T) - log($0.E + $0.T) + log($0.E + $0.Th) }.sorted()
        let agree = stat.filter { $0.kept && $0.lead > 0 }.count

        var info = SieveHeadInfo()
        info.rank = r; info.delta = delta; info.rows = rowBudget
        info.rowsMean = Double(rowsSum) / Double(nTest)
        info.agreement = Double(agree) / Double(nTest)
        info.kl = kls.reduce(0, +) / Double(nTest)
        info.klMedian = kls[nTest / 2]
        info.sampledChange = tvSum / Double(nTest)
        info.heldOutStates = nTest; info.fittedStates = nFit
        info.denseBytes = rowsV * headRow
        info.touchedBytes = r * d * 2 + V * (r / 2 + r / 64 * 4) + V * 4 + Int(info.rowsMean.rounded()) * headRow
        info.calibration = c.source
        info.meetsTarget = info.agreement >= SieveHeadInfo.targetAgreement && info.sampledChange <= SieveHeadInfo.targetSampledChange
        return (info, rowsMax)
        }
        let tuned = measure(tuneX, nTune), tested = measure(testX, nTest)
        report.info = tested.info
        report.rowsMax = tested.rowsMax
        return (report, tuned.info.meetsTarget, qBits.withUnsafeBufferPointer { Data(buffer: $0) }, packed)
        }

        // The cheapest rank that meets the target on the tuning states and then on the test states, or
        // the largest tried. A rank is only ever replaced by a higher one, and the figures reported
        // are the test states' at the rank kept.
        var best = attempt(ranks[0])
        var tried = [best.report.info]
        for r in ranks.dropFirst() where !(best.tuned && best.report.info.meetsTarget) {
            best = attempt(r)
            tried.append(best.report.info)
        }
        var report = best.report
        report.tried = tried
        let info = report.info, r = info.rank

        // 5. The sidecar.
        let meta: [String: String] = [
            "format": SieveHead.format, "fingerprint": SieveHead.fingerprint(of: head),
            "rank": "\(r)", "delta": "\(info.delta)", "rows": "\(info.rows)", "rows_mean": "\(info.rowsMean)",
            "agreement": "\(info.agreement)", "kl": "\(info.kl)", "kl_median": "\(info.klMedian)", "sampled_change": "\(info.sampledChange)", "meets_target": info.meetsTarget ? "1" : "0",
            "held_out_states": "\(nTest)", "fitted_states": "\(nFit)", "tuning_states": "\(nTune)", "touched_bytes": "\(info.touchedBytes)",
            "dense_bytes": "\(info.denseBytes)", "calibration": c.source,
        ]
        let url = SieveHead.url(in: directory)
        try SafeTensorWriter.write(to: url, tensors: [
            ("basis", .f16, [r, d], best.basis),
            ("sketch.weight", .u32, [rowsV, r / 8], best.packed.codes),
            ("sketch.scales", .f16, [rowsV, r / 64], best.packed.scales),
            ("sketch.biases", .f16, [rowsV, r / 64], best.packed.biases),
        ], metadata: meta)
        report.url = url
        report.sidecarBytes = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
        report.seconds = Date().timeIntervalSince(start)
        return report
    }
}

extension SieveImporter {
    /// The measurements stored in a model's sidecar, or nil when it has none that can be read.
    public static func sidecarInfo(directory: URL) -> SieveHeadInfo? {
        guard let m = try? SieveHead.metadata(of: SieveHead.url(in: directory)), m["format"] == SieveHead.format else { return nil }
        return SieveHead.info(from: m)
    }
}
