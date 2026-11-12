import Foundation
import Metal
import Testing
@testable import MLXEngine

/// Sieve Head (Sources/MLXEngine/Sieve.swift): the kernels against a CPU reference on a synthetic
/// head, the sidecar's validation, and, for an installed model that has been fitted, that the dense
/// paths are untouched and the screened path is deterministic.
@Suite("Sieve", .serialized)
struct SieveTests {
    struct RNG {
        var s: UInt64
        mutating func next() -> UInt64 { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s }
        mutating func unit() -> Float { Float(next() >> 40) / Float(1 << 24) }           // [0, 1)
        mutating func signed() -> Float { unit() * 2 - 1 }
    }

    /// A 4-bit, group-64 projection [rows × cols] with random codes, as the engine loads one.
    static func randomLinear(rows: Int, cols: Int, scale: Float, rng: inout RNG, zero: Bool = false) throws -> (Linear, [Data]) {
        let gpu = GPU.shared
        var codes = [UInt32](repeating: 0, count: rows * cols / 8)
        var scales = [UInt16](repeating: 0, count: rows * cols / 64), biases = scales
        if !zero {
            for i in codes.indices { codes[i] = UInt32(truncatingIfNeeded: rng.next()) }
            for i in scales.indices {
                scales[i] = SieveImporter.halfBits(scale * (0.5 + rng.unit()))
                biases[i] = SieveImporter.halfBits(-7.5 * scale * (0.5 + rng.unit()))
            }
        }
        func tensor<T>(_ a: [T], _ dtype: DType, _ shape: [Int]) throws -> (Tensor, Data) {
            let data = a.withUnsafeBufferPointer { Data(buffer: $0) }
            let b = try gpu.buffer(data.count)
            data.withUnsafeBytes { b.contents().copyMemory(from: $0.baseAddress!, byteCount: data.count) }
            return (Tensor(buffer: b, offset: 0, dtype: dtype, shape: shape), data)
        }
        let (w, wd) = try tensor(codes, .u32, [rows, cols / 8])
        let (s, sd) = try tensor(scales, .f16, [rows, cols / 64])
        let (b, bd) = try tensor(biases, .f16, [rows, cols / 64])
        return (try Linear(weight: w, scales: s, biases: b, bias: nil, inDim: cols, outDim: rows, name: "test"), [wd, sd, bd])
    }

    struct Fixture {
        let directory: URL
        let head: Linear
        let W: SieveImporter.Matrix      // the head, dequantized [V × d]
        let S: SieveImporter.Matrix      // the sketch, dequantized [V × r]
        let q: [Float]                   // the basis [r × d]
        let V: Int, d: Int, r: Int
    }

    /// A synthetic head with a sidecar on disk. `meta` edits the sidecar's metadata before it is written.
    static func fixture(V: Int, d: Int = 128, r: Int = 64, rows: Int, delta: Float, seed: UInt64 = 7, zeroSketch: Bool = false,
                        meta edit: (inout [String: String]) -> Void = { _ in }) throws -> Fixture {
        var rng = RNG(s: seed)
        let (head, _) = try randomLinear(rows: V, cols: d, scale: 0.02, rng: &rng)
        let (sketch, sk) = try randomLinear(rows: V, cols: r, scale: 0.05, rng: &rng, zero: zeroSketch)
        let qBits = (0..<(r * d)).map { _ in SieveImporter.halfBits(rng.signed() * 0.2) }
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mlxr-sieve-test-\(UUID().uuidString)")
        var meta: [String: String] = [
            "format": SieveHead.format, "fingerprint": SieveHead.fingerprint(of: head),
            "rank": "\(r)", "delta": "\(delta)", "rows": "\(rows)",
        ]
        edit(&meta)
        try SafeTensorWriter.write(to: SieveHead.url(in: dir), tensors: [
            ("basis", .f16, [r, d], qBits.withUnsafeBufferPointer { Data(buffer: $0) }),
            ("sketch.weight", .u32, [V, r / 8], sk[0]),
            ("sketch.scales", .f16, [V, r / 64], sk[1]),
            ("sketch.biases", .f16, [V, r / 64], sk[2]),
        ], metadata: meta)
        return Fixture(directory: dir, head: head, W: try SieveImporter.dequantize(head), S: try SieveImporter.dequantize(sketch),
                       q: qBits.map(Tensor.halfToFloat), V: V, d: d, r: r)
    }

    /// Runs the screen on the GPU for one state: the logits and the number of exact rows.
    static func run(_ f: Fixture, _ sh: SieveHead, x: [Float], vocab: Int? = nil, fill: Float = 0) throws -> (logits: [Float], rows: Int) {
        let gpu = GPU.shared
        let xb = try gpu.buffer(f.d * 4), lb = try gpu.buffer(f.V * 4)
        x.withUnsafeBytes { xb.contents().copyMemory(from: $0.baseAddress!, byteCount: f.d * 4) }
        let lp = lb.contents().bindMemory(to: Float.self, capacity: f.V)
        for i in 0..<f.V { lp[i] = fill }
        let cb = gpu.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        sh.encode(enc, head: f.head, x: xb, xOffset: 0, logits: lb, vocab: vocab ?? f.V)
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        if let e = cb.error { throw e }
        return (Array(UnsafeBufferPointer(start: lb.contents().bindMemory(to: Float.self, capacity: f.V), count: f.V)), sh.lastRowCount)
    }

    /// CPU reference: the sketch logits and the exact logits of every row.
    static func reference(_ f: Fixture, x: [Float]) -> (sketch: [Float], exact: [Float]) {
        var xr = [Float](repeating: 0, count: f.r)
        for k in 0..<f.r { var a: Float = 0; for j in 0..<f.d { a += f.q[k * f.d + j] * x[j] }; xr[k] = a }
        var zh = [Float](repeating: 0, count: f.V), z = zh
        for i in 0..<f.V {
            var a: Float = 0, b: Float = 0
            for k in 0..<f.r { a += f.S.p[i * f.r + k] * xr[k] }
            for j in 0..<f.d { b += f.W.p[i * f.d + j] * x[j] }
            zh[i] = a; z[i] = b
        }
        return (zh, z)
    }

    static func close(_ a: Float, _ b: Float) -> Bool { abs(a - b) <= 2e-3 * max(1, abs(a), abs(b)) }

    @Test func profileParsing() throws {
        #expect(try SieveProfile.parse(nil) == nil)
        for off in ["", "0", "off", "OFF", " no "] { #expect(try SieveProfile.parse(off) == nil) }
        for on in ["balanced", "1", "on", "Balanced"] { #expect(try SieveProfile.parse(on) == .balanced) }
        for bad in ["exact", "fast", "max", "turbo"] { #expect(throws: EngineError.self) { try SieveProfile.parse(bad) } }
    }

    @Test func halfBitsRoundTrip() {
        var rng = RNG(s: 3)
        for v: Float in [0, -0.0, 1, -1, 0.1, 65504, 1e-5, 6e-8, 1e9, -1e9, .infinity] {
            let h = Tensor.halfToFloat(SieveImporter.halfBits(v))
            if abs(v) > 65520 { #expect(h.isInfinite && (h > 0) == (v > 0)) } else { #expect(abs(h - v) <= max(abs(v) * 0.0005, 3e-8)) }
        }
        for _ in 0..<10_000 {
            // Every half value survives float → half exactly.
            let bits = UInt16(truncatingIfNeeded: rng.next())
            let f = Tensor.halfToFloat(bits)
            if f.isNaN { continue }
            #expect(SieveImporter.halfBits(f) == bits)
        }
    }

    /// The rows the screen marks exact carry the head's own logit, the others the sketch's, the list
    /// stays within the budget, and the split sits where the histogram puts it. V is not a multiple of
    /// the block size.
    @Test func screenMatchesReference() throws {
        let rows = 100
        let f = try Self.fixture(V: 1000, rows: rows, delta: 4)
        defer { try? FileManager.default.removeItem(at: f.directory) }
        let sh = try SieveHead(directory: f.directory, head: f.head)
        var rng = RNG(s: 11)
        for _ in 0..<8 {
            let x = (0..<f.d).map { _ in rng.signed() }
            let (got, count) = try Self.run(f, sh, x: x)
            let ref = Self.reference(f, x: x)
            let top = ref.sketch.max()!
            // The reference's own cut, in bins.
            var hist = [Int](repeating: 0, count: SieveHead.bins)
            let scale = Float(SieveHead.bins) / 4
            for v in ref.sketch { let b = (top - v) * scale; if b >= 0, b < Float(SieveHead.bins) { hist[Int(b)] += 1 } }
            var cum = 0, n = 0
            while n < SieveHead.bins, cum + hist[n] <= rows { cum += hist[n]; n += 1 }
            #expect(count <= rows)
            #expect(count > 0)
            var exact = 0
            for i in 0..<f.V {
                let b = (top - ref.sketch[i]) * scale
                let isExact = Self.close(got[i], ref.exact[i]), isSketch = Self.close(got[i], ref.sketch[i])
                #expect(isExact || isSketch)
                // Away from the cut (float rounding can move a row across one bin edge) the side is fixed.
                if b < Float(n) - 2 { #expect(isExact) }
                if b > Float(n) + 2 { #expect(isSketch) }
                if isExact && !isSketch { exact += 1 }
            }
            #expect(exact <= count)
        }
    }

    /// With a budget of every row and a margin that covers them all, the screen is the dense head.
    @Test func denseLimitIsTheHead() throws {
        let f = try Self.fixture(V: 777, rows: 777, delta: 1000)
        defer { try? FileManager.default.removeItem(at: f.directory) }
        let sh = try SieveHead(directory: f.directory, head: f.head)
        var rng = RNG(s: 5)
        let x = (0..<f.d).map { _ in rng.signed() }
        let (got, count) = try Self.run(f, sh, x: x)
        let ref = Self.reference(f, x: x)
        #expect(count == f.V)
        for i in 0..<f.V { #expect(Self.close(got[i], ref.exact[i])) }
    }

    /// Every row tied at the maximum: the first bin alone exceeds the budget, so no row is listed
    /// (which rows would be is not deterministic) and every logit is the sketch's.
    @Test func tieAtTheMaximumListsNothing() throws {
        let f = try Self.fixture(V: 600, rows: 50, delta: 4, zeroSketch: true)
        defer { try? FileManager.default.removeItem(at: f.directory) }
        let sh = try SieveHead(directory: f.directory, head: f.head)
        let (got, count) = try Self.run(f, sh, x: [Float](repeating: 0.5, count: f.d))
        #expect(count == 0)
        #expect(got.allSatisfy { $0 == 0 })
    }

    /// A sidecar that does not belong to the head, or is malformed, never reaches a kernel.
    @Test func badSidecarsAreRefused() throws {
        let fm = FileManager.default
        func refused(_ what: String, _ edit: (inout [String: String]) -> Void) throws {
            let f = try Self.fixture(V: 512, rows: 64, delta: 4, meta: edit)
            defer { try? fm.removeItem(at: f.directory) }
            #expect(throws: EngineError.self, "\(what)") { _ = try SieveHead(directory: f.directory, head: f.head) }
        }
        try refused("another model's fingerprint") { $0["fingerprint"] = String(repeating: "0", count: 64) }
        try refused("unknown format") { $0["format"] = "something-else" }
        try refused("rank that is not the basis's") { $0["rank"] = "128" }
        try refused("row budget above the vocabulary") { $0["rows"] = "100000" }
        try refused("zero rows") { $0["rows"] = "0" }
        try refused("negative margin") { $0["delta"] = "-1" }
        try refused("non-finite margin") { $0["delta"] = "nan" }

        // Truncated, and missing.
        let f = try Self.fixture(V: 512, rows: 64, delta: 4)
        defer { try? fm.removeItem(at: f.directory) }
        let url = SieveHead.url(in: f.directory)
        let data = try Data(contentsOf: url)
        try data.prefix(data.count / 2).write(to: url)
        #expect(throws: EngineError.self) { _ = try SieveHead(directory: f.directory, head: f.head) }
        try data.prefix(4).write(to: url)
        #expect(throws: EngineError.self) { _ = try SieveHead(directory: f.directory, head: f.head) }
        try fm.removeItem(at: url)
        #expect(throws: EngineError.self) { _ = try SieveHead(directory: f.directory, head: f.head) }
        // A sidecar fitted to a head of another shape.
        let other = try Self.fixture(V: 256, rows: 64, delta: 4)
        defer { try? fm.removeItem(at: other.directory) }
        #expect(throws: EngineError.self) { _ = try SieveHead(directory: other.directory, head: f.head) }
    }

    /// An installed, fitted model: passes that return logits to the CPU are the dense head bit for bit
    /// with Sieve on, and the screened, GPU-sampled path gives the same tokens twice and stays in budget.
    @Test func fittedModelKeepsDensePathsAndIsDeterministic() throws {
        guard let dir = installedModel("mlx-community/Qwen3-0.6B-4bit"), SieveImporter.sidecarInfo(directory: dir) != nil else { return }
        let dense = try Transformer(directory: dir, contextLength: 512, sparse: nil)
        let sieve = try Transformer(directory: dir, contextLength: 512, sparse: nil, sieve: .balanced)
        guard let info = sieve.sieveInfo else { Issue.record("sidecar present but not loaded: \(sieve.sieveOffReason ?? "?")"); return }
        let prompt: [Int32] = (0..<24).map { Int32(1000 + $0 * 37) }
        let V = dense.vocabSize
        let a = Array(UnsafeBufferPointer(start: try dense.forward(prompt, offset: 0), count: V))
        let b = Array(UnsafeBufferPointer(start: try sieve.forward(prompt, offset: 0), count: V))
        #expect(a == b)

        var sp = SamplingParameters()
        sp.temperature = 0; sp.repetitionPenalty = 1
        let sample = SampleParams(sp, vocab: V)
        func generate(_ t: Transformer) throws -> [Int] {
            t.seedSampler(42)
            var out = [try t.prefill(prompt, offset: 0, sample: sample)]
            for i in 0..<48 {
                out.append(try t.finish(try t.beginDecodeStep(offset: prompt.count + i, sample: sample)))
                if t === sieve { #expect(t.sieveLastRows > 0 && t.sieveLastRows <= info.rows) }
            }
            return out
        }
        let first = try generate(sieve)
        #expect(try generate(sieve) == first)
        let ref = try generate(dense)
        let same = zip(first, ref).prefix { $0 == $1 }.count
        print("SIEVE: greedy tokens identical to dense for the first \(same) of \(ref.count)")
    }

    // MARK: The fit

    /// A projection from packed 4-bit codes and f16 scales and biases, as the engine loads one.
    static func linear(codes: Data, scales: Data, biases: Data, rows: Int, cols: Int) throws -> Linear {
        func tensor(_ data: Data, _ dtype: DType, _ shape: [Int]) throws -> Tensor {
            let b = try GPU.shared.buffer(data.count)
            data.withUnsafeBytes { b.contents().copyMemory(from: $0.baseAddress!, byteCount: data.count) }
            return Tensor(buffer: b, offset: 0, dtype: dtype, shape: shape)
        }
        return try Linear(weight: tensor(codes, .u32, [rows, cols / 8]), scales: tensor(scales, .f16, [rows, cols / 64]),
                          biases: tensor(biases, .f16, [rows, cols / 64]), bias: nil, inDim: cols, outDim: rows, name: "test")
    }

    /// The directions come out strongest first: states built from two known directions give those two back.
    @Test func basisFindsTheDominantDirectionsInOrder() throws {
        let d = 64, n = 4000
        var rng = RNG(s: 21)
        var u = [Float](repeating: 0, count: d), v = u
        u[3] = 0.6; u[10] = 0.8                       // unit, orthogonal to v
        v[20] = 1
        var x = [Float](repeating: 0, count: n * d)
        for i in 0..<n {
            let a = rng.signed() * 5, b = rng.signed() * 2
            for j in 0..<d { x[i * d + j] = a * u[j] + b * v[j] + rng.signed() * 0.05 }
        }
        let q = try SieveImporter.basis(x, n: n, d: d, r: 3)
        func dot(_ k: Int, _ w: [Float]) -> Float { (0..<d).reduce(0) { $0 + q[k * d + $1] * w[$1] } }
        #expect(abs(dot(0, u)) > 0.999)
        #expect(abs(dot(1, v)) > 0.999)
        #expect(abs(dot(2, u)) < 0.05 && abs(dot(2, v)) < 0.05)
        for k in 0..<3 { #expect(abs((0..<d).reduce(0) { $0 + q[k * d + $1] * q[k * d + $1] } - 1) < 1e-4) }
    }

    /// The packed sketch decodes, by the engine's own reading of a 4-bit projection, to the values the
    /// fit measured with, and those lie within half a quantization step of the input.
    @Test func quantizedSketchIsWhatTheEngineReads() throws {
        let V = 300, r = 128
        var rng = RNG(s: 9)
        let s = SieveImporter.Matrix(V, r)
        var original = [Float](repeating: 0, count: V * r)
        for i in 0..<(V * r) { original[i] = rng.signed() * (1 + Float(i % 7)); s.p[i] = original[i] }
        let packed = SieveImporter.quantize4(s)
        let read = try SieveImporter.dequantize(try Self.linear(codes: packed.codes, scales: packed.scales, biases: packed.biases, rows: V, cols: r))
        for row in 0..<V {
            for g in 0..<(r / 64) {
                let range = (row * r + g * 64)..<(row * r + g * 64 + 64)
                let step = (original[range].max()! - original[range].min()!) / 15
                for i in range {
                    #expect(abs(read.p[i] - s.p[i]) <= 1e-5 * max(1, abs(s.p[i])))
                    #expect(abs(s.p[i] - original[i]) <= step * 0.51 + 0.002 * max(1, abs(original[i])))
                }
            }
        }
    }

    /// Fit, tune and test never share a prompt or a chunk, and each gets its share.
    @Test func calibrationSplits() {
        let prompts = (0..<SieveImporter.seedPrompts.count).map(SieveImporter.split(prompt:))
        #expect(SieveImporter.seedPrompts.count == 40)
        #expect(prompts.filter { $0 == .test }.count == 10)
        #expect(prompts.filter { $0 == .tune }.count == 5)
        #expect(prompts.filter { $0 == .fit }.count == 25)
        let chunks = (0..<10).map(SieveImporter.split(chunk:))
        #expect(chunks.filter { $0 == .test }.count == 2 && chunks.filter { $0 == .tune }.count == 2)
    }

    /// The fingerprint changes with any scale, any bias and a weight in the middle of the matrix.
    @Test func fingerprintCoversScalesBiasesAndWeights() throws {
        var rng = RNG(s: 4)
        let (_, parts) = try Self.randomLinear(rows: 512, cols: 128, scale: 0.02, rng: &rng)
        func print(_ edit: (inout [Data]) -> Void) throws -> String {
            var p = parts
            edit(&p)
            return SieveHead.fingerprint(of: try Self.linear(codes: p[0], scales: p[1], biases: p[2], rows: 512, cols: 128))
        }
        let base = try print { _ in }
        #expect(try print { _ in } == base)
        #expect(try print { $0[0][$0[0].count / 2] ^= 1 } != base)
        #expect(try print { $0[1][$0[1].count / 2] ^= 1 } != base)
        #expect(try print { $0[2][$0[2].count - 1] ^= 1 } != base)
    }

    /// Text and flags read from a sidecar that may have come with a downloaded model.
    @Test func sidecarMetadataIsSanitized() {
        let hostile = "ok\u{1b}[31m\u{07}\n\u{9b}" + String(repeating: "x", count: 5000)
        let info = SieveHead.info(from: ["calibration": hostile, "meets_target": "0", "rank": "abc", "rows": "64"])
        #expect(info.calibration.hasPrefix("ok[31m"))
        #expect(info.calibration.count == 200)
        #expect(info.calibration.unicodeScalars.allSatisfy { $0.value >= 0x20 && $0.value != 0x7f && !(0x80...0x9f).contains($0.value) })
        #expect(!info.meetsTarget)
        #expect(info.rank == 0 && info.rows == 64)
        #expect(SieveHead.info(from: ["meets_target": "1"]).meetsTarget)
    }

    /// The sidecar is never written through a link planted in the model directory, and a failed
    /// write leaves nothing behind.
    @Test func writerRefusesLinks() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("mlxr-sieve-link-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: root) }
        let outside = root.appendingPathComponent("outside"), model = root.appendingPathComponent("model")
        try fm.createDirectory(at: outside, withIntermediateDirectories: true)
        try fm.createDirectory(at: model, withIntermediateDirectories: true)
        let tensors: [(name: String, dtype: DType, shape: [Int], data: Data)] = [("a", .f16, [2], Data([1, 2, 3, 4]))]

        // `sieve` is a link to another directory.
        try fm.createSymbolicLink(at: model.appendingPathComponent("sieve"), withDestinationURL: outside)
        #expect(throws: EngineError.self) { try SafeTensorWriter.write(to: SieveHead.url(in: model), tensors: tensors, metadata: [:]) }
        #expect(try fm.contentsOfDirectory(atPath: outside.path).isEmpty)

        // `head.safetensors` is a link to a file elsewhere.
        try fm.removeItem(at: model.appendingPathComponent("sieve"))
        try fm.createDirectory(at: model.appendingPathComponent("sieve"), withIntermediateDirectories: true)
        let victim = outside.appendingPathComponent("victim")
        try Data("keep".utf8).write(to: victim)
        try fm.createSymbolicLink(at: SieveHead.url(in: model), withDestinationURL: victim)
        #expect(throws: EngineError.self) { try SafeTensorWriter.write(to: SieveHead.url(in: model), tensors: tensors, metadata: [:]) }
        #expect(try Data(contentsOf: victim) == Data("keep".utf8))

        // A regular file is replaced, and nothing else is left in the directory.
        try fm.removeItem(at: SieveHead.url(in: model))
        try Data("old".utf8).write(to: SieveHead.url(in: model))
        try SafeTensorWriter.write(to: SieveHead.url(in: model), tensors: tensors, metadata: ["k": "v"])
        #expect(try SieveHead.metadata(of: SieveHead.url(in: model))["k"] == "v")
        #expect(try fm.contentsOfDirectory(atPath: model.appendingPathComponent("sieve").path) == ["head.safetensors"])
    }

    /// Options the fit cannot honour are refused before anything is loaded.
    @Test func fitRefusesARowBudgetItHasNoMarginFor() {
        var o = SieveImportOptions()
        o.rows = 700
        #expect(throws: EngineError.self) { _ = try SieveImporter.importHead(directory: URL(fileURLWithPath: "/nonexistent-model"), options: o) }
    }

    /// The screen scores and lists only the rows the sampler considers: head rows without a token
    /// are neither written nor counted.
    @Test func rowsBeyondTheSamplerVocabularyAreLeftAlone() throws {
        let f = try Self.fixture(V: 1000, rows: 100, delta: 1000)
        defer { try? FileManager.default.removeItem(at: f.directory) }
        let sh = try SieveHead(directory: f.directory, head: f.head)
        var rng = RNG(s: 13)
        let x = (0..<f.d).map { _ in rng.signed() }
        let vocab = 700
        let (got, count) = try Self.run(f, sh, x: x, vocab: vocab, fill: 12345)
        let ref = Self.reference(f, x: x)
        #expect(count > 0 && count <= 100)
        for i in vocab..<f.V { #expect(got[i] == 12345) }
        for i in 0..<vocab { #expect(Self.close(got[i], ref.exact[i]) || Self.close(got[i], ref.sketch[i])) }
        // The best row by sketch value among the sampler's rows is exact, whatever the rows beyond it hold.
        let best = (0..<vocab).max { ref.sketch[$0] < ref.sketch[$1] }!
        #expect(Self.close(got[best], ref.exact[best]))
    }

    /// The whole fit on an installed model, into a scratch copy of its directory (the model's own
    /// sidecar is not touched): the sidecar loads, the screen keeps the dense head's likeliest token
    /// on states nothing was fitted or tuned on, and generation runs within the row budget.
    @Test func fitOfAnInstalledModelLoadsAndAgrees() throws {
        guard let dir = installedModel("mlx-community/Qwen3-0.6B-4bit") else { return }
        let fm = FileManager.default
        let copy = fm.temporaryDirectory.appendingPathComponent("mlxr-sieve-fit-\(UUID().uuidString)")
        try fm.createDirectory(at: copy, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: copy) }
        for name in try fm.contentsOfDirectory(atPath: dir.path) where name != "sieve" {
            try fm.createSymbolicLink(at: copy.appendingPathComponent(name), withDestinationURL: dir.appendingPathComponent(name))
        }
        var o = SieveImportOptions()
        o.tokens = 3000
        let report = try SieveImporter.importHead(directory: copy, options: o)
        let info = report.info
        #expect(info.heldOutStates >= 256)
        #expect(info.agreement >= 0.98)
        #expect(info.sampledChange <= 0.05)
        #expect(info.rowsMean > 0 && info.rowsMean <= Double(info.rows))
        #expect(report.rowsMax <= info.rows)
        #expect(report.tried.last == info)
        #expect(SieveImporter.sidecarInfo(directory: copy) == info)

        let t = try Transformer(directory: copy, contextLength: 512, sparse: nil, sieve: .balanced)
        #expect(t.sieveInfo == info, "the fit's own sidecar must load: \(t.sieveOffReason ?? "")")
        var sp = SamplingParameters()
        sp.temperature = 0; sp.repetitionPenalty = 1
        let sample = SampleParams(sp, vocab: t.vocabSize)
        let prompt: [Int32] = (0..<16).map { Int32(2000 + $0 * 53) }
        _ = try t.prefill(prompt, offset: 0, sample: sample)
        for i in 0..<16 {
            _ = try t.finish(try t.beginDecodeStep(offset: prompt.count + i, sample: sample))
            #expect(t.sieveLastRows > 0 && t.sieveLastRows <= info.rows)
        }
    }
}

/// What Sieve changes in real generations, observed, for the model named by `MLXR_SIEVE_OBSERVE`
/// (opt-in: it needs the GPU to itself and prints its result):
///
///     MLXR_SIEVE_OBSERVE=mlx-community/Qwen3-0.6B-4bit swift test -c release -Xswiftc -enable-testing --filter SieveObserved
///
/// Each test-split prompt is answered dense and with Sieve from the same seeds, greedy and with the
/// app's default sampler. A pair of answers is compared up to its first different token: that token
/// counts as one change among the tokens generated so far, and nothing after it is compared.
@Suite("SieveObserved", .serialized, .enabled(if: ProcessInfo.processInfo.environment["MLXR_SIEVE_OBSERVE"] != nil))
struct SieveObservedTests {
    @Test func generationsAgainstDense() throws {
        guard let repo = ProcessInfo.processInfo.environment["MLXR_SIEVE_OBSERVE"], let dir = installedModel(repo) else {
            print("SIEVE-OBSERVED: model not installed"); return
        }
        let length = 96, seeds: [UInt64] = [1, 2, 3]
        let tok = try Tokenizer(directory: dir)
        var greedy = SamplingParameters()
        greedy.temperature = 0; greedy.repetitionPenalty = 1
        let modes = [("greedy", greedy, [UInt64(1)]), ("sampled", SamplingParameters(), seeds)]

        /// Every answer of one transformer, by mode: [mode][prompt × seed] → tokens.
        func answers(sieve: SieveProfile?) throws -> [[[Int]]] {
            let t = try Transformer(directory: dir, contextLength: 1024, sparse: nil, sieve: sieve)
            if sieve != nil, t.sieveInfo == nil { throw EngineError.unsupported(t.sieveOffReason ?? "no sidecar") }
            let tpl = ChatTemplate(template: tok.chatTemplate, modelType: t.config.modelType, bosToken: tok.bosToken, toolTemplate: tok.toolUseTemplate)
            var stops = Set(t.config.eosTokenIds)
            if let e = tok.eosToken, let id = tok.tokenId(e) { stops.insert(id) }
            for s in tpl.stopTokens { if let id = tok.tokenId(s) { stops.insert(id) } }
            var out: [[[Int]]] = []
            for (_, params, seeds) in modes {
                let sample = SampleParams(params, vocab: min(tok.vocabularySize, t.vocabSize))
                var runs: [[Int]] = []
                for (i, prompt) in SieveImporter.seedPrompts.enumerated() where SieveImporter.split(prompt: i) == .test {
                    let r = tpl.renderMarked([ChatMessage(role: .user, content: prompt)], enableThinking: false)
                    let ids = tok.encode(r.text, plainRanges: r.content).map(Int32.init)
                    for seed in seeds {
                        t.seedSampler(seed &+ UInt64(i) &* 104729)
                        var tokens = [try t.prefill(ids, offset: 0, sample: sample)]
                        while tokens.count < length, !stops.contains(tokens.last!) {
                            tokens.append(try t.finish(try t.beginDecodeStep(offset: ids.count + tokens.count - 1, sample: sample)))
                        }
                        runs.append(tokens)
                    }
                }
                out.append(runs)
            }
            return out
        }
        let dense = try answers(sieve: nil)
        let again = try answers(sieve: nil)
        let sieve = try answers(sieve: .balanced)
        for (m, mode) in modes.enumerated() {
            func compare(_ a: [[Int]], _ b: [[Int]]) -> (same: Int, changed: Int, compared: Int) {
                var same = 0, changed = 0, compared = 0
                for (x, y) in zip(a, b) {
                    let common = zip(x, y).prefix { $0 == $1 }.count
                    if common == x.count, common == y.count { same += 1; compared += common } else { changed += 1; compared += common + 1 }
                }
                return (same, changed, compared)
            }
            let s = compare(dense[m], sieve[m]), control = compare(dense[m], again[m])
            print(String(format: "SIEVE-OBSERVED %@ %@: %d of %d answers identical to dense over up to %d tokens; first change after %d of %d compared tokens (%.2f%% per token); dense against itself: %d of %d identical",
                         repo, mode.0, s.same, dense[m].count, length, s.changed, s.compared, 100 * Double(s.changed) / Double(max(s.compared, 1)),
                         control.same, dense[m].count))
            #expect(control.changed == 0, "dense generation is not reproducible, so the comparison means nothing")
        }
    }
}
