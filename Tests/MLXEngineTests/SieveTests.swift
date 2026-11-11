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
    static func run(_ f: Fixture, _ sh: SieveHead, x: [Float]) throws -> (logits: [Float], rows: Int) {
        let gpu = GPU.shared
        let xb = try gpu.buffer(f.d * 4), lb = try gpu.buffer(f.V * 4)
        x.withUnsafeBytes { xb.contents().copyMemory(from: $0.baseAddress!, byteCount: f.d * 4) }
        let cb = gpu.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        sh.encode(enc, head: f.head, x: xb, xOffset: 0, logits: lb)
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
}
