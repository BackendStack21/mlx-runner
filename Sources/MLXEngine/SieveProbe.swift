import Foundation
import Metal

struct ProbeParams { var cols: Int32; var layer: Int32; var group: Int32; var threshold: Float }

/// Experiment hooks for the Sieve hypotheses (docs/sieve-architecture.md, section 13). Nothing in
/// the app or the CLI sets `Transformer.probe`: with it nil the forward pass encodes exactly what
/// it did before. The tests that measure Pyramid and Experts use it on teacher-forced batches.
///
/// - **Exits** (Pyramid): row t of the pass contributes nothing from layer `exitLayer[t]` on, so its
///   residual stays what it was after the layers below. The layers above still run on that frozen
///   state, which writes their K/V rows and advances a recurrent state from it: the design's
///   "frozen fill". The final norm and the head then see the frozen state (the logit lens).
/// - **Capture**: the residual stream [T × hidden] as it enters layer `captureLayer`.
/// - **MLP mask** (Experts): after the activation, groups of `mlpGroup` intermediate neurons are
///   kept when their best score |h_n| · ‖down column n‖ reaches `mlpThreshold` times the row's
///   best, and zeroed otherwise. `counters` accumulates groups kept and groups seen.
final class SieveProbe {
    let exitLayer: MTLBuffer        // [maxBatch] Int32
    var exitsActive = false
    var captureLayer: Int?
    let capture: MTLBuffer          // [maxBatch × hidden] Float
    var mlpThreshold: Float = 0     // 0: no mask
    var mlpGroup = 1
    var mlpLayers: Range<Int> = 0..<Int.max
    private(set) var columnNorms: [MTLBuffer] = []   // per layer [intermediate] Float
    let counters: MTLBuffer         // [2] UInt32: groups kept, groups seen
    let exitPSO, copyPSO, mlpPSO: MTLComputePipelineState
    private let hidden: Int

    init(_ t: Transformer) throws {
        let gpu = GPU.shared
        hidden = t.config.hiddenSize
        exitLayer = try gpu.buffer(t.maxBatch * 4)
        capture = try gpu.buffer(t.maxBatch * hidden * 4)
        counters = try gpu.buffer(16)
        exitPSO = try gpu.pipeline("probe_exit")
        copyPSO = try gpu.pipeline("probe_copy")
        mlpPSO = try gpu.pipeline("probe_mlp")
        setExits([Int32](repeating: Int32.max, count: t.maxBatch))
    }

    /// Exit layer per row of the next passes (rows beyond the array never exit).
    func setExits(_ e: [Int32]) {
        let p = exitLayer.contents().bindMemory(to: Int32.self, capacity: exitLayer.length / 4)
        for i in 0..<(exitLayer.length / 4) { p[i] = i < e.count ? e[i] : Int32.max }
        exitsActive = e.contains { $0 != Int32.max }
    }

    /// ‖column n‖ of every layer's down projection (dequantized on the CPU): the MLP mask's weights.
    func prepareMLP(_ t: Transformer) throws {
        columnNorms = try t.layers.map { layer in
            let w = try SieveImporter.dequantize(layer.down)     // [hidden × intermediate]
            var n = [Float](repeating: 0, count: w.cols)
            for r in 0..<w.rows { for c in 0..<w.cols { let v = w.p[r * w.cols + c]; n[c] += v * v } }
            let b = try GPU.shared.buffer(w.cols * 4)
            let p = b.contents().bindMemory(to: Float.self, capacity: w.cols)
            for c in 0..<w.cols { p[c] = n[c].squareRoot() }
            return b
        }
    }

    func resetCounters() { counters.contents().initializeMemory(as: UInt32.self, repeating: 0, count: 4) }
    var kept: (groups: Int, of: Int) {
        let p = counters.contents().bindMemory(to: UInt32.self, capacity: 2)
        return (Int(p[0]), Int(p[1]))
    }

    /// Zeroes the rows of `block` ([T × hidden], a layer's attention or MLP output) that exited at or before `layer`.
    func encodeExit(_ enc: MTLComputeCommandEncoder, block: MTLBuffer, layer: Int, T: Int) {
        guard exitsActive else { return }
        enc.setComputePipelineState(exitPSO)
        enc.set(block, 0, 0)
        enc.set(exitLayer, 0, 1)
        enc.params(ProbeParams(cols: Int32(hidden), layer: Int32(layer), group: 1, threshold: 0), 2)
        enc.dispatchThreads(MTLSize(width: hidden, height: T, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func encodeCapture(_ enc: MTLComputeCommandEncoder, h: MTLBuffer, layer: Int, T: Int) {
        guard captureLayer == layer else { return }
        enc.setComputePipelineState(copyPSO)
        enc.set(h, 0, 0)
        enc.set(capture, 0, 1)
        enc.params(ProbeParams(cols: Int32(hidden), layer: 0, group: 1, threshold: 0), 2)
        enc.dispatchThreads(MTLSize(width: hidden, height: T, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Masks `act` ([T × intermediate], the activation's output) in place.
    func encodeMLP(_ enc: MTLComputeCommandEncoder, act: MTLBuffer, layer: Int, intermediate: Int, T: Int) {
        guard mlpThreshold > 0, mlpLayers.contains(layer), layer < columnNorms.count, intermediate % mlpGroup == 0 else { return }
        enc.setComputePipelineState(mlpPSO)
        enc.set(act, 0, 0)
        enc.set(columnNorms[layer], 0, 1)
        enc.params(ProbeParams(cols: Int32(intermediate), layer: Int32(layer), group: Int32(mlpGroup), threshold: mlpThreshold), 2)
        enc.set(counters, 0, 3)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: T, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}
