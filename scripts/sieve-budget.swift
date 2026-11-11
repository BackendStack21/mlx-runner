// The arithmetic behind the decode and prefill tables of docs/sieve-architecture.md (section 6),
// as docs/sieve-handoff.md, Appendix A, defines it. Every figure it prints is a projection: a
// measured dense rate times the ratio of a cost model at an assumed operating point. Nothing here
// measures anything.
//
//   swift scripts/sieve-budget.swift              # both constant sets, one after the other
//   swift scripts/sieve-budget.swift --spec       # the specification's first constants and assumed shapes
//   swift scripts/sieve-budget.swift --measured   # constants and shapes measured on the M2 Pro (phase 0)
//
// `--spec` must reproduce the tables the specification was first written with (within rounding).
// `--measured` uses docs/sieve-phase0-results.md: per-model overheads from `mlxr bench`, real shapes
// from the safetensors headers (scripts/sieve-shapes.swift), Head's measured touched share.
import Foundation

let MB = 1e6

struct Model {
    var name: String
    var H: Double            // head bytes
    var A: Double            // attention and mixer bytes of the body
    var M: Double            // MLP bytes of the body
    var L: Double            // layers
    var fill: Double         // share of A a frozen position still pays (its k and v, or a mixer's input projections)
    var stride: Double       // tokens per pass (1 without a multi-token head)
    var denseDecode: Double? // measured dense decode at 16 tokens of context, tok/s (nil: the model's own figure)
    var densePrefill: Double? // measured dense prefill, tok/s
    /// Measured per-model overheads (nil: the constant set's c0 and c1 · L).
    var fixed: Double? = nil      // ms per step outside the layers and the head read
    var perLayer: Double? = nil   // ms per layer outside its weight reads
    var hf: Double? = nil         // Head's touched share of the dense head, per evaluation
    var step: Double? = nil       // measured dense decode step at 16 tokens of context, ms (sets perLayer)
    var bwHead: Double? = nil     // measured rate of the head matvec, GB/s
    var g: Double? = nil          // measured GEMM share of dense prefill time
    var body: Double { A + M }
    var size: Double { H + A + M }
}

struct Constants {
    var label: String
    var eta: Double      // share of the bandwidth the matvecs reach
    var c0: Double       // ms per step
    var c1: Double       // ms per layer
    var cx: Double       // ms per exit test (readback between two command buffers)
    var hf: Double       // Head's touched share of the dense head per evaluation
    var bw: Double       // GB/s
    var g: Double        // GEMM share of dense prefill time
    var tile: Double     // efficiency of GEMM tiles over token subsets
    var headFixed = 0.0  // ms per Head evaluation outside its reads (its extra dispatches)
}

struct Profile { var name: String; var k: Double; var d1: Double; var p1: Double; var d2: Double; var p2: Double; var rho: Double }

let profiles = [
    Profile(name: "balanced", k: 0.50, d1: 0.50, p1: 0.25, d2: 0.75, p2: 0.25, rho: 0.70),
    Profile(name: "fast", k: 0.40, d1: 0.375, p1: 0.45, d2: 0.625, p2: 0.25, rho: 0.40),
    Profile(name: "max", k: 0.35, d1: 0.25, p1: 0.60, d2: 0.50, p2: 0.25, rho: 0.35),
]

let BPW = 0.5625   // bytes per weight: 4 bits + 32 bits of scale and bias per 64 weights

/// A plain transformer from its shape.
func plain(_ name: String, d: Double, I: Double, L: Double, nH: Double, nKV: Double, hd: Double, V: Double,
           decode: Double?, prefill: Double?) -> Model {
    let attn = nH * hd * d + 2 * nKV * hd * d + d * nH * hd
    return Model(name: name, H: V * d * BPW, A: L * attn * BPW, M: L * 3 * d * I * BPW, L: L,
                 fill: 2 * nKV * hd * d / attn, stride: 1, denseDecode: decode, densePrefill: prefill)
}

/// Qwen 3.5 as the specification assumed it: a 248,000-row head, a body split 35 / 65.
func assumedQwen35(_ name: String, sizeMB: Double, hidden: Double, L: Double, decode: Double?, prefill: Double?) -> Model {
    let H = 248000 * hidden * BPW, body = sizeMB * MB - H
    return Model(name: name, H: H, A: 0.35 * body, M: 0.65 * body, L: L, fill: 0.7, stride: 1.25, denseDecode: decode, densePrefill: prefill)
}

struct Decode { var tDense: Double; var tPass: Double; var ratio: Double; var touched: Double; var depth: Double }

/// Seconds-per-byte terms: the head at its own measured rate when there is one, scaled with the bandwidth.
func headMs(_ m: Model, _ c: Constants, bw: Double, bytes: Double) -> Double {
    bytes / ((m.bwHead.map { $0 * bw / c.bw } ?? c.eta * bw) * 1e9) * 1e3
}
func bodyMs(_ c: Constants, bw: Double, bytes: Double) -> Double { bytes / (c.eta * bw * 1e9) * 1e3 }

/// Per-layer cost outside the weight reads: the constant, a model's own, or what a measured step leaves
/// after the head, the body's bytes and the fixed cost.
func perLayer(_ m: Model, _ c: Constants) -> Double {
    if let p = m.perLayer { return p }
    guard let step = m.step else { return c.c1 }
    return (step - headMs(m, c, bw: c.bw, bytes: m.H) - bodyMs(c, bw: c.bw, bytes: m.body) - (m.fixed ?? c.c0)) / m.L
}

func tDense(_ m: Model, _ c: Constants, bw: Double) -> Double {
    headMs(m, c, bw: bw, bytes: m.H) + bodyMs(c, bw: bw, bytes: m.body) + (m.fixed ?? c.c0) + perLayer(m, c) * m.L
}

func decode(_ m: Model, _ p: Profile, _ c: Constants, bw: Double) -> Decode {
    let fd = p.d1 * p.p1 + p.d2 * p.p2 + (1 - p.p1 - p.p2)   // mean depth
    let nx = 1 + (1 - p.p1)                                  // exit tests per token
    let nh = 1 + (1 - p.p1) + (1 - p.p1 - p.p2)              // Head evaluations per token
    let fW = 0.963 * p.k + 0.037 + 0.02                      // share of the MLP's bytes touched
    let headBytes = m.H * (m.hf ?? c.hf) * nh, bodyBytes = fd * (m.A + fW * m.M)
    let bPass = headBytes + bodyBytes
    let tPass = headMs(m, c, bw: bw, bytes: headBytes) + bodyMs(c, bw: bw, bytes: bodyBytes) + c.headFixed * nh
        + (m.fixed ?? c.c0) + c.cx * nx + perLayer(m, c) * m.L * fd
    let td = tDense(m, c, bw: bw)
    return Decode(tDense: td, tPass: tPass, ratio: td / tPass * m.stride, touched: bPass / m.stride, depth: fd)
}

func prefillFraction(_ m: Model, _ p: Profile, _ c: Constants) -> Double {
    let a = m.A / m.body, mm = 1 - a
    let g = m.g ?? c.g
    return p.rho * (g * (a + mm * p.k) + (1 - g)) + (1 - p.rho) * g * a * m.fill
}

/// Rounds like the specification's tables: two significant digits below 100, three (to tens) above.
func r(_ v: Double) -> String {
    if v >= 1000 { return String(Int((v / 10).rounded()) * 10) }
    if v >= 100 { return String(Int((v / 10).rounded()) * 10) }
    return String(Int(v.rounded()))
}
func pad(_ s: String, _ n: Int) -> String { s.count >= n ? s : s + String(repeating: " ", count: n - s.count) }

func tables(_ c: Constants, _ models: [Model], otherBW: Double) {
    print("=== \(c.label)")
    print(String(format: "eta %.2f, c0 %.3f ms, c1 %.3f ms/layer (unless a model has its own), cx %.3f ms, Head share %.2f, BW %.0f GB/s, GEMM share %.2f, tile efficiency %.2f",
                 c.eta, c.c0, c.c1, c.cx, c.hf, c.bw, c.g, c.tile))
    print("\nDecode, tok/s (dense measured unless marked; the rest projected)")
    print(pad("model", 26) + pad("dense", 12) + pad("model t_dense", 16) + profiles.map { pad($0.name, 10) }.joined() + "max touches")
    for bw in [c.bw, otherBW] {
        print("-- \(Int(bw)) GB/s" + (bw == c.bw ? "" : " (assumed; the cost model alone)"))
        for m in models {
            let td = tDense(m, c, bw: bw)
            let dense = bw == c.bw ? m.denseDecode : nil
            let base = dense ?? 1000 / td
            var line = pad(m.name, 26) + pad(dense == nil ? "\(r(base)) (model)" : String(format: "%.1f", base), 12) + pad(String(format: "%.2f ms", td), 16)
            var touched = ""
            for p in profiles {
                let d = decode(m, p, c, bw: bw)
                line += pad(r(base * d.ratio), 10)
                if p.name == "max" { touched = String(format: "%.0f MB, A%.1f", d.touched / MB, d.touched / (562.5 * MB)) }
            }
            print(line + touched)
        }
    }
    print("\nPrefill, tok/s on the M2 Pro (dense measured; the rest projected)")
    print(pad("model", 26) + pad("dense", 12) + profiles.map { pad($0.name, 10) }.joined())
    for m in models {
        guard let dense = m.densePrefill else { continue }
        print(pad(m.name, 26) + pad(r(dense), 12) + profiles.map { pad(r(dense / prefillFraction(m, $0, c) * c.tile), 10) }.joined())
    }
    print("\nMean decode depth: " + profiles.map { p in String(format: "%@ %.2f", p.name, decode(models[0], p, c, bw: c.bw).depth) }.joined(separator: ", "))
    print("")
}

// MARK: The specification's first constants (fitted to the README, shapes assumed)

let specConstants = Constants(label: "Specification as first written: constants fitted to the README, Qwen 3.5 shapes assumed",
                              eta: 0.91, c0: 0.25, c1: 0.046, cx: 0.05, hf: 0.26, bw: 197, g: 0.85, tile: 0.8)
let q4 = assumedQwen35("Qwen3.5 4B", sizeMB: 3030, hidden: 2560, L: 32, decode: 62.3, prefill: 403)
let specModels: [Model] = [
    plain("Qwen3 0.6B", d: 1024, I: 3072, L: 28, nH: 16, nKV: 8, hd: 128, V: 151936, decode: 286, prefill: 3020),
    assumedQwen35("Qwen3.5 0.8B", sizeMB: 424, hidden: 1024, L: 24, decode: 271, prefill: 2780),
    plain("Llama 3.2 1B", d: 2048, I: 8192, L: 16, nH: 32, nKV: 8, hd: 64, V: 128256, decode: 202, prefill: 1720),
    q4,
    // The 9B: size, head and layer count assumed; its dense prefill is the 4B's scaled by the bodies.
    Model(name: "9B (assumed shape)", H: 500 * MB, A: 0.35 * 5100 * MB, M: 0.65 * 5100 * MB, L: 40, fill: 0.7, stride: 1.25,
          denseDecode: nil, densePrefill: 403 * q4.body / (5100 * MB)),
]

let args = Set(CommandLine.arguments.dropFirst())
if args.isEmpty || args.contains("--spec") { tables(specConstants, specModels, otherBW: 500) }

// MARK: Measured on the M2 Pro (16-core GPU, 16 GB, macOS 27.2), 2026-11-11: docs/sieve-phase0-results.md

/// Bytes from the safetensors headers (scripts/sieve-shapes.swift); rates are medians of three
/// alternating `mlxr bench` rounds. `fill` is the share of the mixer and attention bytes a frozen
/// position still pays: k and v of a full-attention layer, every input projection of a linear one.
func measured(_ name: String, H: Double, A: Double, M: Double, L: Double, fill: Double, step: Double, decode: Double,
              prefill: Double, bwHead: Double, g: Double) -> Model {
    var m = Model(name: name, H: H * MB, A: A * MB, M: M * MB, L: L, fill: fill, stride: 1, denseDecode: decode, densePrefill: prefill)
    m.step = step; m.bwHead = bwHead; m.g = g
    return m
}

// Body bytes are read at 183 GB/s on top of a fixed cost per dispatch (the slope of matvec time against
// matvec bytes across the five models: 175 to 191 GB/s); what a measured step spends beyond its bytes
// is charged per layer. A readback between two command buffers costs 0.18 ms (SieveTimingTests).
// Head: the certificate failed (H1), so the share is the uncertified screen's: a 4-bit sketch of rank
// d/4 (25.7% of the head) and the rows that hold the dense argmax in 99.9% of states (about 1%), plus
// three extra dispatches per evaluation. Stride is 1: no installed export has the multi-token head.
// Still assumed, not measured: the operating points (profiles) and the tile efficiency.
let measuredConstants = Constants(label: "Measured constants and shapes, M2 Pro, 2026-11-11 (operating points still assumed; Head uncertified)",
                                  eta: 183.0 / 197, c0: 0.05, c1: 0, cx: 0.18, hf: 0.27, bw: 197, g: 0.85, tile: 0.8, headFixed: 0.04)
let measuredModels: [Model] = [
    measured("Qwen3 0.6B", H: 87.5, A: 99.1, M: 148.6, L: 28, fill: 0.333, step: 3.47, decode: 288.0, prefill: 3025, bwHead: 171, g: 0.768),
    measured("Qwen3.5 0.8B", H: 143.0, A: 132.2, M: 148.6, L: 24, fill: 0.68, step: 3.94, decode: 253.1, prefill: 2779, bwHead: 183, g: 0.781),
    measured("Llama 3.2 1B", H: 147.8, A: 94.4, M: 453.0, L: 16, fill: 0.20, step: 4.93, decode: 202.8, prefill: 1724, bwHead: 186, g: 0.936),
    measured("Qwen3.5 4B", H: 357.6, A: 735.2, M: 1274.0, L: 32, fill: 0.615, step: 16.18, decode: 61.8, prefill: 453.7, bwHead: 185, g: 0.915),
    measured("Qwen3.5 9B", H: 572.1, A: 1175.3, M: 2717.9, L: 32, fill: 0.615, step: 27.30, decode: 36.6, prefill: 243.8, bwHead: 188, g: 0.949),
]
if args.isEmpty || args.contains("--measured") {
    tables(measuredConstants, measuredModels, otherBW: 500)
    print("Per-layer cost outside the weight reads, ms: " + measuredModels.map { String(format: "%@ %.3f", $0.name, perLayer($0, measuredConstants)) }.joined(separator: ", "))
    // The lossless part alone: nothing (Head is not exact, Stride has no weights). The same table with
    // the dense head, to show what the Bounded head contributes.
    var noHead = measuredConstants
    noHead.label = "The same with the dense head (Head off): what Experts and Pyramid give alone"
    noHead.hf = 1; noHead.headFixed = 0
    tables(noHead, measuredModels, otherBW: 500)
}
