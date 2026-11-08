// Compares generation speed of MLX Runner (mlxr), Ollama and mlx-lm on one model and one prompt:
// prefill and decode tokens per second, median of N runs after one warm-up run per tool.
//
//   swift build -c release --product mlxr
//   ollama serve &                      # Ollama must be running, with the model pulled
//   MLX_LM_PYTHON=~/venvs/mlx/bin/python swift scripts/bench-compare.swift
//
// Options (defaults in brackets):
//   --mlx-model ID|PATH   MLX model for mlxr and mlx-lm, same weights for both [mlx-community/Qwen3.5-4B-4bit]
//   --ollama-model NAME   an Ollama model to use instead of an import of the MLX weights (e.g. qwen3.5:4b)
//   --runs N              measured runs per tool [3]
//   --max-tokens N        tokens to generate [256]
//   --prompt-words N      prompt length in words, about 1.35 tokens per word [400]
//   --tools LIST          comma-separated subset of mlxr,ollama,mlx-lm [all three]
//   --mlxr PATH           mlxr binary [.build/release/mlxr, else mlxr on PATH]
//   --python PATH         a Python with mlx-lm installed [$MLX_LM_PYTHON]
//   --ollama-host URL     [$OLLAMA_HOST, else http://127.0.0.1:11434]
//   --cooldown SECONDS    pause between runs [2]
//   --json PATH           also write every run as JSON
//
// Fairness: greedy decoding with no repetition penalty, the same token budget and the same prompt for
// every tool, and each tool's own chat template (thinking follows the model's default everywhere). Runs
// go round-robin across tools, so thermal drift spreads evenly. Each prompt starts with a different run
// number, so no tool can reuse a cached prefix. All three run the same MLX weights: unless --ollama-model
// is given, the script imports the MLX model directory into Ollama once (`ollama create`, which keeps the
// MLX quantization and runs it on Ollama's MLX runner). With --ollama-model, Ollama runs its own build,
// whose format, quantization and size are printed next to the results.
// Nothing else should use the GPU while this runs (AGENTS.md); the script warns if it looks busy.
import Foundation

// MARK: options

struct Options {
    var mlxModel = "mlx-community/Qwen3.5-4B-4bit"
    var ollamaModel: String?
    var runs = 3
    var maxTokens = 256
    var promptWords = 400
    var tools = ["mlxr", "ollama", "mlx-lm"]
    var mlxr: String?
    var python = ProcessInfo.processInfo.environment["MLX_LM_PYTHON"]
    var ollamaHost = ProcessInfo.processInfo.environment["OLLAMA_HOST"].map { $0.hasPrefix("http") ? $0 : "http://\($0)" }
        ?? "http://127.0.0.1:11434"
    var cooldown = 2.0
    var json: String?
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("error: \(message)\n".utf8))
    exit(1)
}

func note(_ message: String) {
    FileHandle.standardError.write(Data("\(message)\n".utf8))
}

func parseOptions() -> Options {
    var o = Options()
    var args = Array(CommandLine.arguments.dropFirst())
    func value(_ flag: String) -> String {
        guard !args.isEmpty else { fail("\(flag) needs a value") }
        return args.removeFirst()
    }
    func positive(_ flag: String) -> Int {
        guard let n = Int(value(flag)), n > 0 else { fail("\(flag) takes a positive integer") }
        return n
    }
    while !args.isEmpty {
        let flag = args.removeFirst()
        switch flag {
        case "--mlx-model": o.mlxModel = value(flag)
        case "--ollama-model": o.ollamaModel = value(flag)
        case "--runs": o.runs = positive(flag)
        case "--max-tokens": o.maxTokens = positive(flag)
        case "--prompt-words": o.promptWords = positive(flag)
        case "--tools":
            o.tools = value(flag).split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) }
            for t in o.tools where !["mlxr", "ollama", "mlx-lm"].contains(t) { fail("unknown tool \(t)") }
        case "--mlxr": o.mlxr = value(flag)
        case "--python": o.python = value(flag)
        case "--ollama-host": o.ollamaHost = value(flag)
        case "--cooldown":
            guard let s = Double(value(flag)), s >= 0, s.isFinite else { fail("--cooldown takes seconds") }
            o.cooldown = s
        case "--json": o.json = value(flag)
        case "-h", "--help":
            let header = (try? String(contentsOfFile: #filePath, encoding: .utf8))?
                .split(separator: "\n", omittingEmptySubsequences: false)
                .prefix { $0.hasPrefix("//") }.map { $0.dropFirst(min(3, $0.count)) }.joined(separator: "\n")
            print(header ?? "see the header of scripts/bench-compare.swift")
            exit(0)
        default: fail("unknown option \(flag) (see --help)")
        }
    }
    return o
}

// MARK: processes and HTTP

/// Runs a program with stdin closed, returning its exit status and combined stdout and stderr.
func run(_ argv: [String], timeout: TimeInterval = 900) -> (status: Int32, output: String) {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: argv[0])
    p.arguments = Array(argv.dropFirst())
    let pipe = Pipe()
    p.standardOutput = pipe
    p.standardError = pipe
    p.standardInput = FileHandle.nullDevice
    do { try p.run() } catch { return (127, "cannot run \(argv[0]): \(error.localizedDescription)") }
    let timer = DispatchWorkItem { if p.isRunning { p.terminate() } }
    DispatchQueue.global().asyncAfter(deadline: .now() + timeout, execute: timer)
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    p.waitUntilExit()
    timer.cancel()
    return (p.terminationStatus, String(decoding: data, as: UTF8.self))
}

/// The executable for a command name: an explicit path, or the first match on PATH.
func which(_ name: String) -> String? {
    if name.contains("/") { return FileManager.default.isExecutableFile(atPath: name) ? name : nil }
    for dir in (ProcessInfo.processInfo.environment["PATH"] ?? "").split(separator: ":") {
        let candidate = "\(dir)/\(name)"
        if FileManager.default.isExecutableFile(atPath: candidate) { return candidate }
    }
    return nil
}

func request(_ url: String, body: [String: Any]? = nil, timeout: TimeInterval = 900) -> [String: Any]? {
    guard let u = URL(string: url) else { return nil }
    var req = URLRequest(url: u, timeoutInterval: timeout)
    if let body {
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try? JSONSerialization.data(withJSONObject: body)
    }
    let done = DispatchSemaphore(value: 0)
    var result: [String: Any]?
    URLSession.shared.dataTask(with: req) { data, response, _ in
        if let data, (response as? HTTPURLResponse)?.statusCode == 200 {
            result = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        }
        done.signal()
    }.resume()
    done.wait()
    return result
}

func firstMatch(_ pattern: String, in text: String) -> [String]? {
    guard let re = try? NSRegularExpression(pattern: pattern),
          let m = re.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) else { return nil }
    return (1..<m.numberOfRanges).map { i in Range(m.range(at: i), in: text).map { String(text[$0]) } ?? "" }
}

// MARK: environment

func sysctl(_ name: String) -> String {
    run(["/usr/sbin/sysctl", "-n", name], timeout: 10).output.trimmingCharacters(in: .whitespacesAndNewlines)
}

/// GPU busy percentage from the IOAccelerator statistics (no root needed), or nil if unreadable.
func gpuUtilization() -> Int? {
    let out = run(["/usr/sbin/ioreg", "-r", "-c", "IOAccelerator", "-d", "1"], timeout: 10).output
    return firstMatch(#""Device Utilization %"=(\d+)"#, in: out).flatMap { Int($0[0]) }
}

/// A local MLX model directory: the path itself, MLX Runner's library, or the Hugging Face cache.
func mlxModelDirectory(_ idOrPath: String) -> URL? {
    let fm = FileManager.default
    var isDir: ObjCBool = false
    if fm.fileExists(atPath: idOrPath, isDirectory: &isDir), isDir.boolValue { return URL(fileURLWithPath: idOrPath) }
    let parts = idOrPath.split(separator: "/")
    guard parts.count == 2, parts.allSatisfy({ !$0.isEmpty && $0 != ".." && $0 != "." }) else { return nil }
    let home = fm.homeDirectoryForCurrentUser
    let library = home.appendingPathComponent("Library/Application Support/MLX Runner/models/\(idOrPath)")
    if fm.fileExists(atPath: library.appendingPathComponent("config.json").path) { return library }
    let env = ProcessInfo.processInfo.environment
    let hub = env["HF_HUB_CACHE"].map { URL(fileURLWithPath: $0) }
        ?? env["HF_HOME"].map { URL(fileURLWithPath: $0).appendingPathComponent("hub") }
        ?? home.appendingPathComponent(".cache/huggingface/hub")
    let snapshots = hub.appendingPathComponent("models--\(parts[0])--\(parts[1])/snapshots")
    let newest = (try? fm.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: [.contentModificationDateKey]))?
        .filter { fm.fileExists(atPath: $0.appendingPathComponent("config.json").path) }
        .max { a, b in
            let da = (try? a.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let db = (try? b.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return da < db
        }
    return newest
}

func weightBytes(_ dir: URL) -> Int64 {
    let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
    return files.filter { $0.pathExtension == "safetensors" }.reduce(0) { sum, f in
        let real = f.resolvingSymlinksInPath()
        let size = (try? FileManager.default.attributesOfItem(atPath: real.path))?[.size] as? NSNumber
        return sum + (size?.int64Value ?? 0)
    }
}

func gigabytes(_ bytes: Int64) -> String { String(format: "%.2f GB", Double(bytes) / 1e9) }

// MARK: the prompt

/// A deterministic prompt of about `words` words. The run number goes first, so no tool can reuse a
/// prefix cached from the previous run.
func prompt(run: Int, words: Int) -> String {
    let source = """
    The first electronic computers filled whole rooms and were programmed by rewiring panels by hand. \
    Stored programs changed that: instructions lived in memory next to the data they worked on, so a \
    machine could be given a new task without touching a single cable. Transistors replaced vacuum tubes, \
    integrated circuits replaced boards of discrete parts, and each step made machines smaller, cheaper \
    and faster. Operating systems grew from simple loaders into layers that share a processor among many \
    programs, protect memory, and manage files and devices. Networks then linked machines across \
    buildings and continents, and the web turned those links into a medium anyone could publish to. \
    Graphics processors, first built to draw triangles for games, turned out to be very good at the dense \
    arithmetic of neural networks, and machine learning moved from research labs into everyday tools.
    """.split(separator: " ")
    let body = (0..<words).map { source[$0 % source.count] }.joined(separator: " ")
    return "Run \(run). Read these notes, then write a long, detailed essay that expands on every point.\n\n\(body)"
}

// MARK: measurements

struct Sample: Codable {
    let tool: String
    let run: Int
    let promptTokens: Int
    let prefillTPS: Double
    let generatedTokens: Int
    let decodeTPS: Double
}

struct Tool {
    let name: String
    let version: String
    let weights: String
    let measure: (_ run: Int, _ prompt: String) -> Sample?
    var finish: () -> Void = {}
}

func mlxrTool(_ o: Options, model: URL) -> Tool? {
    let candidates = [o.mlxr, ".build/release/mlxr", "mlxr"].compactMap { $0 }
    guard let bin = candidates.lazy.compactMap(which).first else {
        note("skipping mlxr: no binary (build it with `swift build -c release --product mlxr` or pass --mlxr)")
        return nil
    }
    let versionLine = run([bin, "-v"], timeout: 30).output
    let version = firstMatch(#"version is ([\w.\-]+)"#, in: versionLine)?[0] ?? "1.x"
    return Tool(name: "mlxr", version: version, weights: "MLX \(gigabytes(weightBytes(model)))") { i, text in
        // --verbose prints the stats lines on 2.x; 1.x prints them anyway and ignores unknown flags.
        let r = run([bin, "run", model.path, "-p", text, "--temp", "0", "--repeat-penalty", "1", "--seed", "42",
                     "--max", String(o.maxTokens), "--verbose"])
        guard r.status == 0,
              let p = firstMatch(#"\[prompt: (\d+) tokens, ([\d.]+) tok/s"#, in: r.output),
              let g = firstMatch(#"\[generated (\d+) tokens at ([\d.]+) tok/s"#, in: r.output) else {
            note("mlxr run \(i) failed (exit \(r.status)):\n\(r.output.suffix(600))")
            return nil
        }
        return Sample(tool: "mlxr", run: i, promptTokens: Int(p[0]) ?? 0, prefillTPS: Double(p[1]) ?? 0,
                      generatedTokens: Int(g[0]) ?? 0, decodeTPS: Double(g[1]) ?? 0)
    }
}

/// The Ollama name for an import of an MLX model directory, e.g. `mlx-bench-qwen3.5-4b-4bit`.
func importName(_ dir: URL) -> String {
    let base = dir.lastPathComponent.lowercased().map { $0.isLetter || $0.isNumber || "._-".contains($0) ? $0 : "-" }
    return "mlx-bench-" + String(base)
}

func ollamaTool(_ o: Options, mlxModel: URL?) -> Tool? {
    guard let v = request("\(o.ollamaHost)/api/version", timeout: 5)?["version"] as? String else {
        note("skipping ollama: no server at \(o.ollamaHost) (start it with `ollama serve`)")
        return nil
    }
    func tags() -> [[String: Any]] { (request("\(o.ollamaHost)/api/tags", timeout: 10)?["models"] as? [[String: Any]]) ?? [] }
    let model: String
    if let given = o.ollamaModel {
        model = given
    } else {
        // Same weights as mlxr and mlx-lm: import the MLX directory once, keeping its quantization.
        guard let dir = mlxModel else { return nil }
        model = importName(dir)
        if !tags().contains(where: { ($0["name"] as? String) == model + ":latest" }) {
            guard let ollama = which("ollama") else {
                note("skipping ollama: the `ollama` command is needed to import \(dir.path)")
                return nil
            }
            let work = FileManager.default.temporaryDirectory.appendingPathComponent("bench-compare-\(getpid())")
            try? FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: work) }
            let modelfile = work.appendingPathComponent("Modelfile")
            do { try "FROM \(dir.path)\n".write(to: modelfile, atomically: true, encoding: .utf8) } catch { return nil }
            note("importing \(dir.lastPathComponent) into Ollama as \(model) (once)")
            let r = run([ollama, "create", model, "-f", modelfile.path], timeout: 1800)
            guard r.status == 0 else {
                note("skipping ollama: `ollama create` failed:\n\(r.output.suffix(600))")
                return nil
            }
        }
    }
    let wanted = model.contains(":") ? model : model + ":latest"
    guard let entry = tags().first(where: { ($0["name"] as? String) == wanted }) else {
        note("skipping ollama: \(model) is not pulled (run `ollama pull \(model)`)")
        return nil
    }
    let details = entry["details"] as? [String: Any] ?? [:]
    let format = [details["quantization_level"], details["format"]].compactMap { $0 as? String }.joined(separator: " ")
    let runner = (details["runner"] as? String).map { ", \($0) runner" } ?? ""
    let size = (entry["size"] as? NSNumber)?.int64Value ?? 0
    var tool = Tool(name: "ollama", version: v, weights: "\(format)\(runner), \(gigabytes(size))") { i, text in
        let options: [String: Any] = ["temperature": 0, "seed": 42, "repeat_penalty": 1.0,
                                      "num_predict": o.maxTokens, "num_ctx": 4096]
        guard let r = request("\(o.ollamaHost)/api/generate",
                              body: ["model": model, "prompt": text, "stream": false,
                                     "options": options, "keep_alive": "10m"]),
              let pc = (r["prompt_eval_count"] as? NSNumber)?.intValue,
              let pd = (r["prompt_eval_duration"] as? NSNumber)?.doubleValue,
              let ec = (r["eval_count"] as? NSNumber)?.intValue,
              let ed = (r["eval_duration"] as? NSNumber)?.doubleValue, pd > 0, ed > 0 else {
            note("ollama run \(i) failed")
            return nil
        }
        return Sample(tool: "ollama", run: i, promptTokens: pc, prefillTPS: Double(pc) / pd * 1e9,
                      generatedTokens: ec, decodeTPS: Double(ec) / ed * 1e9)
    }
    // Unload the model, so it does not hold memory after the benchmark.
    tool.finish = { _ = request("\(o.ollamaHost)/api/generate", body: ["model": model, "keep_alive": 0]) }
    return tool
}

func mlxLMTool(_ o: Options, model: URL) -> Tool? {
    guard let py = o.python.flatMap(which) else {
        note("skipping mlx-lm: pass --python or set MLX_LM_PYTHON to a Python with mlx-lm installed")
        return nil
    }
    let show = run([py, "-I", "-m", "pip", "show", "mlx-lm", "mlx"], timeout: 60).output
    let versions = show.components(separatedBy: "\n").filter { $0.hasPrefix("Version:") }
        .map { $0.dropFirst(8).trimmingCharacters(in: .whitespaces) }
    guard !versions.isEmpty else {
        note("skipping mlx-lm: it is not installed for \(py)")
        return nil
    }
    let version = versions.count == 2 ? "\(versions[0]) (mlx \(versions[1]))" : versions[0]
    return Tool(name: "mlx-lm", version: version, weights: "MLX \(gigabytes(weightBytes(model)))") { i, text in
        let r = run([py, "-I", "-m", "mlx_lm.generate", "--model", model.path, "--prompt", text,
                     "--max-tokens", String(o.maxTokens), "--temp", "0.0", "--seed", "42"])
        guard r.status == 0,
              let p = firstMatch(#"Prompt: (\d+) tokens, ([\d.]+) tokens-per-sec"#, in: r.output),
              let g = firstMatch(#"Generation: (\d+) tokens, ([\d.]+) tokens-per-sec"#, in: r.output) else {
            note("mlx-lm run \(i) failed (exit \(r.status)):\n\(r.output.suffix(600))")
            return nil
        }
        return Sample(tool: "mlx-lm", run: i, promptTokens: Int(p[0]) ?? 0, prefillTPS: Double(p[1]) ?? 0,
                      generatedTokens: Int(g[0]) ?? 0, decodeTPS: Double(g[1]) ?? 0)
    }
}

func median(_ xs: [Double]) -> Double {
    let s = xs.sorted()
    return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
}

func rate(_ x: Double) -> String { x >= 100 ? String(format: "%.0f", x) : String(format: "%.1f", x) }

// MARK: main

let o = parseOptions()
let needsMLX = o.tools.contains("mlxr") || o.tools.contains("mlx-lm") || (o.tools.contains("ollama") && o.ollamaModel == nil)
let modelDir = needsMLX ? mlxModelDirectory(o.mlxModel) : nil
if needsMLX && modelDir == nil { fail("MLX model \(o.mlxModel) not found (download it with `mlxr pull \(o.mlxModel)`)") }

var tools: [Tool] = []
for name in o.tools {
    switch name {
    case "mlxr": if let t = mlxrTool(o, model: modelDir!) { tools.append(t) }
    case "ollama": if let t = ollamaTool(o, mlxModel: modelDir) { tools.append(t) }
    default: if let t = mlxLMTool(o, model: modelDir!) { tools.append(t) }
    }
}
guard !tools.isEmpty else { fail("no tool is available to benchmark") }

if let busy = gpuUtilization(), busy > 20 {
    note("warning: the GPU is \(busy)% busy before starting; close other GPU work for clean numbers")
}

note("warm-up: one run per tool (loads weights, compiles kernels)")
for t in tools where t.measure(0, prompt(run: 0, words: o.promptWords)) == nil { fail("\(t.name) failed its warm-up run") }

var samples: [Sample] = []
for i in 1...o.runs {
    for t in tools {
        Thread.sleep(forTimeInterval: o.cooldown)
        note("run \(i)/\(o.runs): \(t.name)")
        if let s = t.measure(i, prompt(run: i, words: o.promptWords)) { samples.append(s) }
    }
}
for t in tools { t.finish() }

// MARK: report

let chip = sysctl("machdep.cpu.brand_string")
let memory = (Int64(sysctl("hw.memsize")) ?? 0) / 1_073_741_824
let date = ISO8601DateFormatter().string(from: Date())
print("""
## Generation speed: \(o.mlxModel)\(o.ollamaModel.map { " / Ollama \($0)" } ?? "")

\(chip), \(memory) GB, macOS \(ProcessInfo.processInfo.operatingSystemVersionString), \(date)
Prompt: \(o.promptWords) words; \(o.maxTokens) generated tokens; greedy, no repetition penalty; \
median of \(o.runs) runs after one warm-up run per tool.

| Tool | Version | Weights | Prompt tokens | Prefill tok/s | Generated | Decode tok/s | Decode vs mlxr |
|---|---|---|---:|---:|---:|---:|---:|
""")
let base = samples.filter { $0.tool == "mlxr" }.map(\.decodeTPS)
for t in tools {
    let s = samples.filter { $0.tool == t.name }
    guard !s.isEmpty else {
        print("| \(t.name) | \(t.version) | \(t.weights) | – | – | – | failed | – |")
        continue
    }
    let decode = median(s.map(\.decodeTPS))
    let spread = s.count > 1 ? " (\(rate(s.map(\.decodeTPS).min()!))–\(rate(s.map(\.decodeTPS).max()!)))" : ""
    let versus = base.isEmpty || t.name == "mlxr" ? "–" : String(format: "%+.1f%%", (decode / median(base) - 1) * 100)
    print("| \(t.name) | \(t.version) | \(t.weights) | \(Int(median(s.map { Double($0.promptTokens) }))) | "
          + "\(rate(median(s.map(\.prefillTPS)))) | \(Int(median(s.map { Double($0.generatedTokens) }))) | "
          + "\(rate(decode))\(spread) | \(versus) |")
}
print("""

Each tool reports its own numbers: mlxr its `[prompt]`/`[generated]` stats, Ollama the API's eval counts \
and durations, mlx-lm its `Prompt:`/`Generation:` lines (mlx-lm's prefill rate tends to read high: it \
excludes work the others count). \(o.ollamaModel == nil
    ? "All three load the same MLX weights (Ollama through an import of the model directory, on its MLX runner)."
    : "Ollama runs its own build of the model, so its format and size differ from the MLX weights mlxr and mlx-lm share.") \
Prompt token counts can differ slightly because each tool applies its own chat template.
""")
if samples.count < tools.count * o.runs { note("warning: \(tools.count * o.runs - samples.count) run(s) failed") }

if let path = o.json {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    do { try encoder.encode(samples).write(to: URL(fileURLWithPath: path)) } catch { fail("cannot write \(path): \(error)") }
}
