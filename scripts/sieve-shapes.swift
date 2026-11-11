// Shapes and weight bytes of MLX model directories, read from config.json and the safetensors
// headers only (no weights are loaded). Phase 0 of docs/sieve-architecture.md, step 2: what the
// head, each layer's mixer or attention projections and each layer's MLP weigh, and whether the
// export holds multi-token-prediction (`mtp.`) tensors.
//
//   swift scripts/sieve-shapes.swift <model directory>...
//   swift scripts/sieve-shapes.swift --tensors <model directory>     # also every tensor group
//
// Bytes are tensor data bytes (weights, quantization scales and biases), in MB of 1e6 bytes.
import Foundation

struct TensorInfo { let name: String; let dtype: String; let shape: [Int]; let bytes: Int }

enum ShapeError: Error, CustomStringConvertible {
    case bad(String)
    var description: String { if case .bad(let s) = self { return s }; return "" }
}

/// Tensors of one safetensors file: an 8-byte little-endian header length, then a JSON header.
func readHeader(_ url: URL) throws -> [TensorInfo] {
    let fh = try FileHandle(forReadingFrom: url)
    defer { try? fh.close() }
    guard let lenData = try fh.read(upToCount: 8), lenData.count == 8 else { throw ShapeError.bad("\(url.lastPathComponent): no header") }
    let len = lenData.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) }
    let size = (try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? UInt64) ?? 0
    guard len > 0, len <= 256 << 20, len + 8 <= size else { throw ShapeError.bad("\(url.lastPathComponent): header length \(len)") }
    guard let json = try fh.read(upToCount: Int(len)), json.count == Int(len),
          let dict = try JSONSerialization.jsonObject(with: json) as? [String: Any] else { throw ShapeError.bad("\(url.lastPathComponent): header JSON") }
    var out: [TensorInfo] = []
    for (name, v) in dict where name != "__metadata__" {
        guard let t = v as? [String: Any], let dtype = t["dtype"] as? String, let shape = t["shape"] as? [Int],
              let off = t["data_offsets"] as? [Int], off.count == 2, off[1] >= off[0] else { throw ShapeError.bad("\(name): entry") }
        out.append(TensorInfo(name: name, dtype: dtype, shape: shape, bytes: off[1] - off[0]))
    }
    return out
}

func mb(_ b: Int) -> String { String(format: "%8.1f MB", Double(b) / 1e6) }

/// `a.b.layers.7.mlp.gate_proj.scales` → (`mlp.gate_proj`, 7); a tensor outside the layers → (its module, nil).
func split(_ name: String) -> (module: String, layer: Int?) {
    var parts = name.split(separator: ".").map(String.init)
    if let last = parts.last, ["weight", "scales", "biases", "bias"].contains(last) { parts.removeLast() }
    if let i = parts.firstIndex(of: "layers"), i + 1 < parts.count, let n = Int(parts[i + 1]) {
        return (parts[(i + 2)...].joined(separator: "."), n)
    }
    return (parts.joined(separator: "."), nil)
}

func report(_ dir: URL, tensors showTensors: Bool) throws {
    let fm = FileManager.default
    let cfgData = try Data(contentsOf: dir.appendingPathComponent("config.json"))
    guard let root = try JSONSerialization.jsonObject(with: cfgData) as? [String: Any] else { throw ShapeError.bad("config.json") }
    let text = root["text_config"] as? [String: Any] ?? [:]
    func cfg(_ k: String) -> Any? { text[k] ?? root[k] }
    func int(_ k: String) -> Int? { cfg(k) as? Int }
    let files = try fm.contentsOfDirectory(atPath: dir.path).filter { $0.hasSuffix(".safetensors") }.sorted()
    var all: [TensorInfo] = []
    var fileBytes = 0
    for f in files {
        let u = dir.appendingPathComponent(f)
        all += try readHeader(u)
        fileBytes += (try fm.attributesOfItem(atPath: u.path)[.size] as? Int) ?? 0
    }
    let mtpPrefixes = ["mtp.", "model.mtp.", "language_model.mtp."]
    let visionPrefixes = ["vision_tower.", "model.visual.", "visual."]
    func isMTP(_ n: String) -> Bool { mtpPrefixes.contains { n.hasPrefix($0) } }
    func isVision(_ n: String) -> Bool { visionPrefixes.contains { n.hasPrefix($0) } }

    let layerTypes = cfg("layer_types") as? [String]
    let L = int("num_hidden_layers") ?? 0
    print("== \(dir.lastPathComponent)")
    print("model_type \(cfg("model_type") ?? "?"), hidden \(int("hidden_size") ?? 0), MLP \(int("intermediate_size") ?? 0), layers \(L), " +
          "heads \(int("num_attention_heads") ?? 0), KV heads \(int("num_key_value_heads") ?? 0), head_dim \(int("head_dim").map(String.init) ?? "-"), " +
          "vocabulary \(int("vocab_size") ?? 0), tied \(cfg("tie_word_embeddings") ?? "?")")
    if let lt = layerTypes {
        let lin = lt.filter { $0 == "linear_attention" }.count
        print("layer types: \(lin) linear_attention, \(lt.count - lin) full_attention (full at \(lt.enumerated().filter { $0.element != "linear_attention" }.map { String($0.offset) }.joined(separator: ",")))")
        print("linear attention: key heads \(int("linear_num_key_heads") ?? 0) x \(int("linear_key_head_dim") ?? 0), value heads \(int("linear_num_value_heads") ?? 0) x \(int("linear_value_head_dim") ?? 0), conv kernel \(int("linear_conv_kernel_dim") ?? 0)")
    }
    if let q = root["quantization"] as? [String: Any] { print("quantization: \(q["bits"] ?? "?") bits, group \(q["group_size"] ?? "?"), mode \(q["mode"] ?? "affine")") }
    print("mtp_num_hidden_layers: \(int("mtp_num_hidden_layers").map(String.init) ?? "absent")")

    // Totals.
    var head = 0, embed = 0, mtp = 0, vision = 0, other = 0
    var perLayer: [Int: [String: Int]] = [:]     // layer → module → bytes
    var groups: [String: (bytes: Int, count: Int, shape: String)] = [:]
    for t in all {
        let (module, layer) = split(t.name)
        let key = (layer == nil ? "" : "layers.*.") + module
        let g = groups[key] ?? (0, 0, "")
        groups[key] = (g.bytes + t.bytes, g.count + 1, t.name.hasSuffix(".weight") ? "\(t.dtype) \(t.shape)" : g.shape)
        if isMTP(t.name) { mtp += t.bytes }
        else if isVision(t.name) { vision += t.bytes }
        else if let layer { perLayer[layer, default: [:]][module, default: 0] += t.bytes }
        else if module.hasSuffix("lm_head") { head += t.bytes }
        else if module.hasSuffix("embed_tokens") { embed += t.bytes }
        else { other += t.bytes }
    }
    let tied = head == 0
    if tied { head = embed; embed = 0 }
    func kind(_ m: String) -> String {
        if m.hasPrefix("mlp.") { return "mlp" }
        if m.hasPrefix("self_attn.") { return "attn" }
        if m.hasPrefix("linear_attn.") { return "linear" }
        return "norms"
    }
    var sums: [String: Int] = [:]
    var fill = 0   // what a frozen position still needs: k and v projections
    for (_, mods) in perLayer { for (m, b) in mods {
        sums[kind(m), default: 0] += b
        if m == "self_attn.k_proj" || m == "self_attn.v_proj" { fill += b }
    } }
    let body = sums.values.reduce(0, +) + other
    let textTotal = head + embed + body
    print("files \(mb(fileBytes))   text model \(mb(textTotal))   head \(mb(head))\(tied ? " (tied to the embedding)" : "   separate embedding \(mb(embed))")")
    print("body \(mb(body)):  attention \(mb(sums["attn"] ?? 0))   linear attention \(mb(sums["linear"] ?? 0))   MLP \(mb(sums["mlp"] ?? 0))   norms and other \(mb((sums["norms"] ?? 0) + other))")
    let mixer = (sums["attn"] ?? 0) + (sums["linear"] ?? 0)
    print(String(format: "shares of the body: mixer and attention %.1f%%, MLP %.1f%%;  head / text model %.1f%%;  k+v share of full attention %.1f%%",
                 100 * Double(mixer) / Double(max(body, 1)), 100 * Double(sums["mlp"] ?? 0) / Double(max(body, 1)),
                 100 * Double(head) / Double(max(textTotal, 1)), 100 * Double(fill) / Double(max(sums["attn"] ?? 0, 1))))
    print("mtp tensors \(mb(mtp)) (\(all.filter { isMTP($0.name) }.count) tensors)   vision tensors \(mb(vision))")
    // One layer of each type.
    var seen = Set<String>()
    for l in perLayer.keys.sorted() {
        let type = layerTypes.map { l < $0.count ? $0[l] : "?" } ?? "full_attention"
        guard seen.insert(type).inserted, let mods = perLayer[l] else { continue }
        let line = mods.sorted { $0.key < $1.key }.filter { $0.value > 100_000 }.map { "\($0.key) \(String(format: "%.2f", Double($0.value) / 1e6))" }.joined(separator: ", ")
        print("layer \(l) (\(type)), MB: \(line)")
    }
    if showTensors {
        for (k, g) in groups.sorted(by: { $0.value.bytes > $1.value.bytes }) {
            print("  \(mb(g.bytes))  x\(g.count)  \(k)  \(g.shape)")
        }
    }
    print("")
}

var args = Array(CommandLine.arguments.dropFirst())
let showTensors = args.contains("--tensors")
args.removeAll { $0 == "--tensors" }
guard !args.isEmpty else {
    FileHandle.standardError.write(Data("usage: swift scripts/sieve-shapes.swift [--tensors] <model directory>...\n".utf8))
    exit(2)
}
for a in args {
    do { try report(URL(fileURLWithPath: (a as NSString).expandingTildeInPath), tensors: showTensors) }
    catch { FileHandle.standardError.write(Data("\(a): \(error)\n".utf8)); exit(1) }
}
