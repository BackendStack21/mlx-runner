import CryptoKit
import Foundation
import Metal

/// Sieve Head (docs/sieve-head.md): an opt-in mode that evaluates the output projection by rows,
/// reads less of it per token and changes the output. Off unless asked for; off means no Sieve
/// buffer and no Sieve dispatch.
///
/// A low-rank, 4-bit sketch of the head scores every vocabulary row, and only the best-scoring
/// rows (a fixed budget, within a fitted margin of the best score) get the head's own logit. The
/// others keep the sketch's value. It is not exact: `mlxr sieve MODEL` fits it per model and
/// reports how often the dense argmax survives.
public enum SieveProfile: String, Sendable, Hashable, Codable, CaseIterable {
    case balanced

    /// Parses a `--sieve` / `MLXR_SIEVE` value: nil for off. Throws for a profile that does not exist.
    public static func parse(_ value: String?) throws -> SieveProfile? {
        let s = (value ?? "").trimmingCharacters(in: .whitespaces).lowercased()
        switch s {
        case "", "0", "off", "false", "no": return nil
        case "1", "on", "true", "yes", "balanced": return .balanced
        default:
            throw EngineError.unsupported("sieve profile '\(s)' (expected off or balanced)")
        }
    }

    /// `MLXR_SIEVE`; unset or unreadable means off.
    public static var fromEnvironment: SieveProfile? {
        (try? parse(ProcessInfo.processInfo.environment["MLXR_SIEVE"])) ?? nil
    }
}

/// What `mlxr sieve MODEL` measured, stored in the sidecar and shown by `mlxr sieve MODEL --info`.
public struct SieveHeadInfo: Sendable, Hashable {
    /// Why a model without a sidecar loads with Sieve off.
    public static let notFittedReason = "this model has not been fitted yet (mlxr sieve MODEL)"
    public var rank = 0
    public var delta: Float = 0
    /// Row budget: about this many rows get the head's own logit per step (fewer when the sketch is peaked).
    public var rows = 0
    /// Mean rows that get the head's own logit per step, on held-out states.
    public var rowsMean: Double = 0
    /// Share of held-out states whose screened argmax is the dense argmax.
    public var agreement: Double = 0
    /// KL(dense ‖ screened) over the held-out states, in nats: the mean, and the median. The mean is
    /// carried by flat states, where the dense head spreads probability over more rows than the budget.
    public var kl: Double = 0
    public var klMedian: Double = 0
    /// Mean total variation between the dense and the screened sampling distributions at top-k 40,
    /// temperature 1, over the held-out states: the share of sampled tokens that have to differ.
    public var sampledChange: Double = 0
    public var heldOutStates = 0
    public var fittedStates = 0
    /// Bytes the screen reads per step against the dense head's.
    public var touchedBytes = 0
    public var denseBytes = 0
    public var calibration = ""
    /// The fit met the target below on the held-out states.
    public var meetsTarget = false
    /// What `balanced` asks of a fit, on held-out states: the dense argmax kept in 99.5% of them
    /// (greedy decoding) and at most 1% of sampled tokens changed (top-k 40, temperature 1).
    public static let targetAgreement = 0.995
    public static let targetSampledChange = 0.01
}

struct HeadParams { var V: Int32; var delta: Float; var rows: Int32; var K: Int32 }

/// The Head sidecar of a model, loaded and validated: `<model>/sieve/head.safetensors`.
final class SieveHead {
    static let directoryName = "sieve"
    static let fileName = "head.safetensors"
    static let format = "mlxr-sieve-head-1"
    static let bins = 1024   // HEAD_BINS
    static func url(in directory: URL) -> URL {
        directory.appendingPathComponent(directoryName).appendingPathComponent(fileName)
    }

    let basis: Linear       // [r × d]: x_r = Q x
    let info: SieveHeadInfo
    let rank: Int
    private let file: SafeTensorFile
    private let xr, bmax, state, hist, list: MTLBuffer
    private let sketchWeight, sketchScales, sketchBiases: Tensor
    private let sketchPSO, cutPSO, histPSO, cut2PSO, selectPSO, exactPSO: MTLComputePipelineState
    private let blocks: Int
    /// Bytes held in Metal buffers.
    let byteCount: Int

    /// Identifies the head a sidecar was fitted to: its shape and quantization, all of its scales and
    /// biases, and 4 KB of its weights at each of 1,024 evenly spaced places (all of them when the
    /// matrix is smaller). A sidecar from another model never reaches a kernel.
    static func fingerprint(of head: Linear) -> String {
        var h = SHA256()
        let meta = "v2 \(head.outDim)x\(head.inDim) \(head.description) \(head.weight.nbytes)"
        h.update(data: Data(meta.utf8))
        func whole(_ t: Tensor?) {
            guard let t, t.offset >= 0, t.nbytes <= t.buffer.length - t.offset else { return }
            h.update(bufferPointer: UnsafeRawBufferPointer(start: t.buffer.contents() + t.offset, count: t.nbytes))
        }
        whole(head.scales)
        whole(head.biases)
        let w = head.weight, n = w.nbytes, part = 4 << 10, places = 1024
        if w.offset >= 0, n <= w.buffer.length - w.offset {
            let p = w.buffer.contents() + w.offset
            if n <= part * places {
                h.update(bufferPointer: UnsafeRawBufferPointer(start: p, count: n))
            } else {
                for i in 0..<places {
                    let at = Int(UInt64(i) * UInt64(n - part) / UInt64(places - 1))
                    h.update(bufferPointer: UnsafeRawBufferPointer(start: p + at, count: part))
                }
            }
        }
        return h.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Whether Head can serve this head at all (a quantized projection with scales the row kernel reads).
    static func supports(_ head: Linear) -> Bool {
        // `head_exact` reads the scales and the biases as one type.
        head.bits > 0 && head.scales?.dtype.suffix != nil && head.biases?.dtype == head.scales?.dtype
            && head.bias == nil && head.inDim % 64 == 0
    }

    /// The `__metadata__` strings of a safetensors file (untrusted: sizes are checked).
    static func metadata(of url: URL) throws -> [String: String] {
        guard let fh = try? FileHandle(forReadingFrom: url) else { throw EngineError.invalidFile("cannot open \(url.lastPathComponent)") }
        defer { try? fh.close() }
        guard let lenData = try fh.read(upToCount: 8), lenData.count == 8 else { throw EngineError.invalidFile("\(url.lastPathComponent) is truncated") }
        let len = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        guard len <= 1 << 20, let json = try fh.read(upToCount: Int(len)), json.count == Int(len),
              let obj = try? JSONSerialization.jsonObject(with: json) as? [String: Any] else {
            throw EngineError.invalidFile("bad header in \(url.lastPathComponent)")
        }
        return obj["__metadata__"] as? [String: String] ?? [:]
    }

    static func info(from m: [String: String]) -> SieveHeadInfo {
        var i = SieveHeadInfo()
        i.rank = Int(m["rank"] ?? "") ?? 0
        i.delta = Float(m["delta"] ?? "") ?? 0
        i.rows = Int(m["rows"] ?? "") ?? 0
        i.rowsMean = Double(m["rows_mean"] ?? "") ?? 0
        i.agreement = Double(m["agreement"] ?? "") ?? 0
        i.kl = Double(m["kl"] ?? "") ?? 0
        i.klMedian = Double(m["kl_median"] ?? "") ?? 0
        i.sampledChange = Double(m["sampled_change"] ?? "") ?? 0
        i.heldOutStates = Int(m["held_out_states"] ?? "") ?? 0
        i.fittedStates = Int(m["fitted_states"] ?? "") ?? 0
        i.touchedBytes = Int(m["touched_bytes"] ?? "") ?? 0
        i.denseBytes = Int(m["dense_bytes"] ?? "") ?? 0
        // Free text from a file that may come with a downloaded model: printable, and short.
        i.calibration = String((m["calibration"] ?? "").unicodeScalars.filter { $0.value >= 0x20 && $0.value != 0x7f && !(0x80...0x9f).contains($0.value) }.prefix(200).map(Character.init))
        i.meetsTarget = m["meets_target"] == "1"
        return i
    }

    /// Loads the sidecar for `head`. Throws when it is missing, malformed or fitted to another head;
    /// the caller then runs with Sieve off.
    init(directory: URL, head: Linear) throws {
        let url = Self.url(in: directory)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw EngineError.unsupported(SieveHeadInfo.notFittedReason)
        }
        guard Self.supports(head) else { throw EngineError.unsupported("Sieve Head needs a quantized output projection") }
        let meta = try Self.metadata(of: url)
        guard meta["format"] == Self.format else { throw EngineError.invalidFile("Sieve sidecar: unknown format") }
        guard meta["fingerprint"] == Self.fingerprint(of: head) else {
            throw EngineError.invalidFile("Sieve sidecar was fitted to other weights (fit it again: mlxr sieve MODEL)")
        }
        let info = Self.info(from: meta)
        guard info.delta.isFinite, info.delta > 0, info.delta <= 1000, info.rows >= 1, info.rows <= 1 << 20 else {
            throw EngineError.invalidFile("Sieve sidecar: delta or row budget out of range")
        }
        let gpu = GPU.shared
        let f = try SafeTensorFile(url: url, device: gpu.device)
        file = f
        func need(_ n: String) throws -> Tensor {
            guard let t = f.tensor(n) else { throw EngineError.invalidFile("Sieve sidecar: missing \(n)") }
            return t
        }
        let q = try need("basis"), w = try need("sketch.weight"), s = try need("sketch.scales"), b = try need("sketch.biases")
        let d = head.inDim, V = head.outDim
        guard q.shape.count == 2, q.shape[1] == d, q.dtype.suffix != nil else { throw EngineError.invalidFile("Sieve sidecar: basis shape \(q.shape)") }
        let r = q.shape[0]
        guard r >= 64, r <= d, r % 64 == 0, r == info.rank else { throw EngineError.invalidFile("Sieve sidecar: rank \(r)") }
        guard w.dtype == .u32, w.shape == [V, r / 8], s.shape == [V, r / 64], b.shape == [V, r / 64],
              s.dtype == .f16, b.dtype == .f16 else {
            throw EngineError.invalidFile("Sieve sidecar: sketch shapes \(w.shape) \(s.shape) \(b.shape)")
        }
        basis = try Linear(weight: q, scales: nil, biases: nil, bias: nil, inDim: d, outDim: r, name: "sieve.basis")
        sketchWeight = w; sketchScales = s; sketchBiases = b
        // The list holds at most `rows` entries (see head_cut2); the kernel reads `rows` slots.
        guard info.rows <= V else { throw EngineError.invalidFile("Sieve sidecar: row budget above the vocabulary") }
        rank = r
        self.info = info
        blocks = (V + 255) / 256
        xr = try gpu.buffer(r * 4)
        bmax = try gpu.buffer(blocks * 4)
        state = try gpu.buffer(16)
        hist = try gpu.buffer(Self.bins * 4)
        list = try gpu.buffer(V * 4)
        sketchPSO = try gpu.pipeline("head_sketch")
        cutPSO = try gpu.pipeline("head_cut")
        histPSO = try gpu.pipeline("head_hist")
        cut2PSO = try gpu.pipeline("head_cut2")
        selectPSO = try gpu.pipeline("head_select")
        exactPSO = try gpu.pipeline("head_exact_\(head.bits)_\(head.scales!.dtype.suffix!)")
        byteCount = f.byteCount + (r + blocks + V + Self.bins) * 4 + 32
    }

    /// The last row's logits into `logits` ([V] floats): sketch values everywhere, the head's own for
    /// the `rows` best rows by sketch value (fewer when they lie further than `delta` below the best).
    /// `x` at `xOffset` is the final normalised state.
    /// `vocab` is how many rows the sampler considers (the tokenizer may have fewer tokens than the
    /// head has rows): the screen scores and ranks those only, and leaves the others unwritten.
    func encode(_ enc: MTLComputeCommandEncoder, head: Linear, x: MTLBuffer, xOffset: Int, logits: MTLBuffer, vocab: Int) {
        let d = head.inDim
        let V = max(1, min(vocab, head.outDim))
        let one = MTLSize(width: 1, height: 1, depth: 1)
        let perBlock = MTLSize(width: (V + 255) / 256, height: 1, depth: 1), block = MTLSize(width: 256, height: 1, depth: 1)
        basis.encode(enc, x: x, xOffset: xOffset, xStride: d, y: xr, yStride: rank, T: 1)
        let hp = HeadParams(V: Int32(V), delta: info.delta, rows: Int32(min(info.rows, V)), K: Int32(rank))
        enc.setComputePipelineState(sketchPSO)
        enc.set(sketchWeight, 0)
        enc.set(sketchScales, 1)
        enc.set(sketchBiases, 2)
        enc.set(xr, 0, 3)
        enc.set(logits, 0, 4)
        enc.params(hp, 5)
        enc.set(bmax, 0, 6)
        enc.dispatchThreadgroups(perBlock, threadsPerThreadgroup: block)
        enc.setComputePipelineState(cutPSO)
        enc.set(bmax, 0, 0)
        enc.params(hp, 1)
        enc.set(state, 0, 2)
        enc.set(hist, 0, 3)
        enc.dispatchThreadgroups(one, threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.setComputePipelineState(histPSO)
        enc.set(logits, 0, 0)
        enc.params(hp, 1)
        enc.set(bmax, 0, 2)
        enc.set(state, 0, 3)
        enc.set(hist, 0, 4)
        enc.dispatchThreadgroups(perBlock, threadsPerThreadgroup: block)
        enc.setComputePipelineState(cut2PSO)
        enc.params(hp, 1)
        enc.set(state, 0, 2)
        enc.set(hist, 0, 3)
        enc.dispatchThreadgroups(one, threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.setComputePipelineState(selectPSO)
        enc.set(logits, 0, 0)
        enc.params(hp, 1)
        enc.set(bmax, 0, 2)
        enc.set(state, 0, 3)
        enc.set(state, 8, 4)
        enc.set(list, 0, 5)
        enc.dispatchThreadgroups(perBlock, threadsPerThreadgroup: block)
        enc.setComputePipelineState(exactPSO)
        enc.set(head.weight, 0)
        enc.set(head.scales!, 1)
        enc.set(head.biases!, 2)
        enc.set(x, xOffset, 3)
        enc.set(logits, 0, 4)
        enc.params(MMParams(K: Int32(d), O: Int32(V), T: 1, gs: Int32(head.groupSize), xStride: Int32(d), yStride: Int32(head.outDim)), 5)
        enc.set(state, 8, 6)
        enc.set(list, 0, 7)
        enc.dispatchThreadgroups(MTLSize(width: (info.rows + 1) / 2, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    /// Rows that got the head's own logit in the last step (diagnostics; read after the step completed).
    var lastRowCount: Int { Int(state.contents().load(fromByteOffset: 8, as: UInt32.self)) }
}

/// A minimal safetensors writer: the header padded so the data starts on a 16-byte boundary, the
/// tensors in the given order.
enum SafeTensorWriter {
    static func write(to url: URL, tensors: [(name: String, dtype: DType, shape: [Int], data: Data)], metadata: [String: String]) throws {
        var header: [String: Any] = ["__metadata__": metadata]
        var offset = 0
        for t in tensors {
            header[t.name] = ["dtype": t.dtype.rawValue, "shape": t.shape, "data_offsets": [offset, offset + t.data.count]]
            offset += t.data.count
        }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while (8 + json.count) % 16 != 0 { json.append(0x20) }
        var out = Data()
        var len = UInt64(json.count)
        withUnsafeBytes(of: &len) { out.append(contentsOf: $0) }
        out.append(json)
        for t in tensors { out.append(t.data) }
        // Into a real directory and over a regular file only: a link planted in a model directory
        // must not redirect the write. The write itself goes to a temporary file and is renamed.
        let fm = FileManager.default, dir = url.deletingLastPathComponent()
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        guard (try fm.attributesOfItem(atPath: dir.path))[.type] as? FileAttributeType == .typeDirectory else {
            throw EngineError.invalidFile("\(dir.lastPathComponent) is not a directory")
        }
        if let existing = try? fm.attributesOfItem(atPath: url.path), existing[.type] as? FileAttributeType != .typeRegular {
            throw EngineError.invalidFile("\(url.lastPathComponent) exists and is not a regular file")
        }
        try out.write(to: url, options: .atomic)
    }
}
