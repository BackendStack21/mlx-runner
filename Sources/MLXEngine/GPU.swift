import Foundation
import Metal

public enum EngineError: LocalizedError {
    case noMetalDevice
    case kernelCompile(String)
    case missingPipeline(String)
    case invalidFile(String)
    case missingTensor(String)
    case unsupported(String)
    case contextFull
    case outOfMemory(String)
    case cancelled
    case invalidOutput(String)

    public var errorDescription: String? {
        switch self {
        case .noMetalDevice: return "No Metal GPU available."
        case .outOfMemory(let s): return "Not enough memory: \(s)"
        case .cancelled: return "Cancelled."
        case .invalidOutput(let s): return "The model produced invalid output: \(s)"
        case .kernelCompile(let s): return "Metal kernel compilation failed: \(s)"
        case .missingPipeline(let s): return "Missing Metal kernel: \(s)"
        case .invalidFile(let s): return "Invalid model file: \(s)"
        case .missingTensor(let s): return "Missing tensor: \(s)"
        case .unsupported(let s): return "Unsupported model: \(s)"
        case .contextFull: return "The context window is full. Start a new chat or raise the context length."
        }
    }
}

// Parameter blocks mirrored in Kernels.swift.
struct MMParams {
    var K: Int32; var O: Int32; var T: Int32; var gs: Int32; var xStride: Int32; var yStride: Int32
    var o1: Int32 = 0; var o2: Int32 = 0; var hasBias: Int32 = 0; var pad: Int32 = 0
}
struct QKVPostParams { var D: Int32; var nH: Int32; var nKV: Int32; var rowStride: Int32; var offset: Int32; var maxSeq: Int32; var traditional: Int32; var rotDim: Int32; var eps: Float; var hasNorm: Int32 }
struct NormParams { var dim: Int32; var headsPerRow: Int32; var rowStride: Int32; var eps: Float }
struct AttnParams { var D: Int32; var nH: Int32; var nKV: Int32; var qStride: Int32; var maxSeq: Int32; var offset: Int32; var scale: Float; var outStride: Int32; var T: Int32 = 1 }
struct EWParams { var n: Int32; var stride: Int32; var cols: Int32; var pad: Int32 = 0 }

/// Shared Metal state: device, queue, and the compiled kernel library.
public final class GPU: @unchecked Sendable {
    public static let shared: GPU = {
        do { return try GPU() } catch { fatalError("\(error)") }
    }()

    public let device: MTLDevice
    let queue: MTLCommandQueue
    private let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private let lock = NSLock()

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError.noMetalDevice
        }
        self.device = device
        self.queue = queue
        let options = MTLCompileOptions()
        options.mathMode = .fast
        options.languageVersion = .version3_1
        do {
            library = try device.makeLibrary(source: KernelSource.source, options: options)
        } catch {
            throw EngineError.kernelCompile("\(error)")
        }
    }

    /// Warm up: forces the shared instance (and kernel compilation) to exist.
    public static func prepare() { _ = shared }

    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        lock.lock(); defer { lock.unlock() }
        if let p = pipelines[name] { return p }
        guard let fn = library.makeFunction(name: name) else { throw EngineError.missingPipeline(name) }
        let p = try device.makeComputePipelineState(function: fn)
        pipelines[name] = p
        return p
    }

    func buffer(_ length: Int) throws -> MTLBuffer {
        guard length >= 0, length <= device.maxBufferLength,
              let b = device.makeBuffer(length: max(length, 16), options: .storageModeShared) else {
            throw EngineError.outOfMemory("cannot allocate a \(length)-byte GPU buffer")
        }
        return b
    }

    public var recommendedMaxWorkingSetSize: UInt64 { device.recommendedMaxWorkingSetSize }
}

// MARK: - Tensors

public enum DType: String {
    case f32 = "F32", f16 = "F16", bf16 = "BF16", u32 = "U32", i32 = "I32", u8 = "U8", i8 = "I8", i64 = "I64", u16 = "U16", i16 = "I16", f64 = "F64", bool = "BOOL"

    var size: Int {
        switch self {
        case .f32, .u32, .i32: return 4
        case .f16, .bf16, .u16, .i16: return 2
        case .u8, .i8, .bool: return 1
        case .i64, .f64: return 8
        }
    }

    /// Kernel name suffix for floating types.
    var suffix: String? {
        switch self {
        case .f32: return "f32"
        case .f16: return "f16"
        case .bf16: return "bf16"
        default: return nil
        }
    }
}

/// A view into a Metal buffer (usually a memory-mapped safetensors file).
public struct Tensor {
    init(buffer: MTLBuffer, offset: Int, dtype: DType, shape: [Int]) {
        self.buffer = buffer; self.offset = offset; self.dtype = dtype; self.shape = shape
    }
    let buffer: MTLBuffer
    let offset: Int
    let dtype: DType
    let shape: [Int]
    var count: Int { shape.reduce(1, *) }
    var nbytes: Int { count * dtype.size }
}

extension MTLComputeCommandEncoder {
    func set(_ t: Tensor, _ index: Int) { setBuffer(t.buffer, offset: t.offset, index: index) }
    func set(_ b: MTLBuffer, _ offset: Int = 0, _ index: Int) { setBuffer(b, offset: offset, index: index) }
    func params<T>(_ v: T, _ index: Int) {
        var v = v
        setBytes(&v, length: MemoryLayout<T>.stride, index: index)
    }
}

// MARK: - Safetensors (zero-copy via mmap)

final class SafeTensorFile {
    let buffer: MTLBuffer
    private(set) var infos: [String: (dtype: DType, shape: [Int], start: Int, end: Int)] = [:]
    let url: URL

    init(url: URL, device: MTLDevice) throws {
        self.url = url
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw EngineError.invalidFile("cannot open \(url.lastPathComponent)") }
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0 else { throw EngineError.invalidFile(url.lastPathComponent) }
        let size = Int(st.st_size)
        let page = Int(getpagesize())
        let mapped = (size + page - 1) / page * page
        guard let ptr = mmap(nil, mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0), ptr != MAP_FAILED else {
            throw EngineError.invalidFile("mmap failed for \(url.lastPathComponent)")
        }
        // Ask the kernel to start paging weights in now.
        madvise(ptr, mapped, MADV_WILLNEED)
        guard let buf = device.makeBuffer(bytesNoCopy: ptr, length: mapped, options: .storageModeShared, deallocator: { p, len in
            munmap(p, len)
        }) else {
            munmap(ptr, mapped)
            throw EngineError.invalidFile("cannot wrap \(url.lastPathComponent) in a Metal buffer")
        }
        buffer = buf
        let base = ptr.assumingMemoryBound(to: UInt8.self)
        let file = url.lastPathComponent
        guard size >= 8 else { throw EngineError.invalidFile("\(file) is truncated") }
        var headerLen: UInt64 = 0
        memcpy(&headerLen, base, 8)
        // Untrusted input: every size and offset is validated before use (the engine
        // is built with -Ounchecked, so arithmetic must not be allowed to overflow).
        guard headerLen <= UInt64(min(100 << 20, size - 8)) else { throw EngineError.invalidFile("bad header length in \(file)") }
        let hLen = Int(headerLen)
        let headerData = Data(bytes: base + 8, count: hLen)
        guard let header = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw EngineError.invalidFile("bad header JSON in \(file)")
        }
        let dataStart = 8 + hLen
        let dataSize = size - dataStart
        for (name, value) in header where name != "__metadata__" {
            guard let d = value as? [String: Any], let dt = d["dtype"] as? String else {
                throw EngineError.invalidFile("bad entry '\(name)' in \(file)")
            }
            guard let dtype = DType(rawValue: dt) else { continue }  // unused dtypes (e.g. I64 buffers)
            guard                   let shapeN = d["shape"] as? [NSNumber], shapeN.count <= 8,
                  let offsN = d["data_offsets"] as? [NSNumber], offsN.count == 2 else {
                throw EngineError.invalidFile("bad entry '\(name)' in \(file)")
            }
            let shape = shapeN.map { $0.int64Value }
            let offs = offsN.map { $0.int64Value }
            guard shape.allSatisfy({ $0 >= 0 && $0 <= Int64(Int32.max) }),
                  offs[0] >= 0, offs[0] <= offs[1], offs[1] <= Int64(dataSize) else {
                throw EngineError.invalidFile("tensor '\(name)' is out of bounds in \(file)")
            }
            var count = 1
            for s in shape {
                let (m, o) = count.multipliedReportingOverflow(by: Int(s))
                guard !o else { throw EngineError.invalidFile("tensor '\(name)' is too large in \(file)") }
                count = m
            }
            let (bytes, o) = count.multipliedReportingOverflow(by: dtype.size)
            guard !o, bytes == Int(offs[1] - offs[0]) else {
                throw EngineError.invalidFile("tensor '\(name)' size does not match its shape in \(file)")
            }
            infos[name] = (dtype, shape.map { Int($0) }, dataStart + Int(offs[0]), dataStart + Int(offs[1]))
        }
    }

    func tensor(_ name: String) -> Tensor? {
        guard let i = infos[name] else { return nil }
        // Kernels use vector loads; copy the rare misaligned tensor into its own buffer.
        if i.start % 16 != 0 {
            let len = i.end - i.start
            let copy = buffer.device.makeBuffer(bytes: buffer.contents() + i.start, length: max(len, 16), options: .storageModeShared)!
            return Tensor(buffer: copy, offset: 0, dtype: i.dtype, shape: i.shape)
        }
        return Tensor(buffer: buffer, offset: i.start, dtype: i.dtype, shape: i.shape)
    }
}

/// All safetensors shards of a model directory.
final class WeightStore {
    private var files: [SafeTensorFile] = []
    private var index: [String: Int] = [:]

    init(directory: URL, device: MTLDevice) throws {
        let fm = FileManager.default
        let names = try fm.contentsOfDirectory(atPath: directory.path)
            .filter { $0.hasSuffix(".safetensors") }
            .sorted()
        guard !names.isEmpty else { throw EngineError.invalidFile("no .safetensors files in \(directory.lastPathComponent)") }
        for (i, n) in names.enumerated() {
            let f = try SafeTensorFile(url: directory.appendingPathComponent(n), device: device)
            files.append(f)
            for k in f.infos.keys { index[k] = i }
        }
    }

    var names: Dictionary<String, Int>.Keys { index.keys }

    func has(_ name: String) -> Bool { index[name] != nil }

    func get(_ name: String) -> Tensor? {
        guard let i = index[name] else { return nil }
        return files[i].tensor(name)
    }

    func require(_ name: String) throws -> Tensor {
        guard let t = get(name) else { throw EngineError.missingTensor(name) }
        return t
    }

    var totalBytes: Int { files.reduce(0) { $0 + $1.buffer.length } }
}
