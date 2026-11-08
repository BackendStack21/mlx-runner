import Foundation
import Testing
import MLXEngine

@Suite("generation_config.json sanitizing (AC-7)")
struct GenerationDefaultsTests {
    func parse(_ json: String) -> GenerationDefaults { GenerationDefaults.parse(Data(json.utf8)) }

    @Test func validValuesAreKept() {
        let g = parse(#"{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.05,"repetition_penalty":1.1,"eos_token_id":1}"#)
        #expect(g.temperature == 0.6)
        #expect(g.topP == 0.95)
        #expect(g.topK == 20)
        #expect(g.minP == 0.05)
        #expect(g.repetitionPenalty == 1.1)
        #expect(g.warnings.isEmpty)
    }

    @Test func badTemperatureIsDroppedAndTheRestKept() {
        let g = parse(#"{"temperature":99,"top_p":0.8,"top_k":20}"#)
        #expect(g.temperature == nil)
        #expect(g.topP == 0.8)
        #expect(g.topK == 20)
        #expect(g.warnings.count == 1)
        #expect(g.warnings.first?.contains("generation_config.json") == true)
        #expect(g.warnings.first?.contains("temperature") == true)
    }

    @Test func everyOutOfRangeValueIsDropped() {
        let cases: [(String, KeyPath<GenerationDefaults, Bool>)] = [
            (#"{"temperature":-0.1}"#, \.temperatureNil),
            (#"{"top_p":0}"#, \.topPNil),
            (#"{"top_p":1.5}"#, \.topPNil),
            (#"{"top_k":100001}"#, \.topKNil),
            (#"{"top_k":-1}"#, \.topKNil),
            (#"{"min_p":2}"#, \.minPNil),
            (#"{"repetition_penalty":0}"#, \.repetitionNil),
            (#"{"repetition_penalty":500}"#, \.repetitionNil),
            (#"{"temperature":"hot"}"#, \.temperatureNil),
            (#"{"top_k":1.5}"#, \.topKNil),
        ]
        for (json, isNil) in cases {
            let g = parse(json)
            #expect(g[keyPath: isNil], "\(json)")
            #expect(g.warnings.count == 1, "\(json): \(g.warnings)")
        }
    }

    @Test func boundaryValuesAreAccepted() {
        let g = parse(#"{"temperature":2,"top_p":1,"top_k":100000,"min_p":1,"repetition_penalty":100}"#)
        #expect(g.temperature == 2 && g.topP == 1 && g.topK == 100_000 && g.minP == 1 && g.repetitionPenalty == 100)
        #expect(g.warnings.isEmpty)
        let z = parse(#"{"temperature":0,"top_k":0,"min_p":0}"#)
        #expect(z.temperature == 0 && z.topK == 0 && z.minP == 0)
        #expect(z.warnings.isEmpty)
    }

    @Test func garbageGivesEmptyDefaults() {
        for bad in ["", "not json", "[1,2]", "null"] {
            let g = parse(bad)
            #expect(g.temperature == nil && g.topP == nil && g.topK == nil, "\(bad)")
        }
    }

    @Test func loadReadsTheFileAndToleratesItsAbsence() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("gen-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let none = GenerationDefaults.load(directory: dir)
        #expect(none.temperature == nil && none.warnings.isEmpty)
        try Data(#"{"temperature":0.7,"top_k":99999999}"#.utf8).write(to: dir.appendingPathComponent("generation_config.json"))
        let g = GenerationDefaults.load(directory: dir)
        #expect(g.temperature == 0.7)
        #expect(g.topK == nil)
        #expect(g.warnings.count == 1)
    }
}

private extension GenerationDefaults {
    var temperatureNil: Bool { temperature == nil }
    var topPNil: Bool { topP == nil }
    var topKNil: Bool { topK == nil }
    var minPNil: Bool { minP == nil }
    var repetitionNil: Bool { repetitionPenalty == nil }
}
