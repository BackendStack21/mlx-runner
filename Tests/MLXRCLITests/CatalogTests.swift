import Foundation
import Testing
import MLXEngine

@Suite("Model catalog (AC-10)")
struct CatalogTests {
    @Test func ollamaStyleAliasesNameMLXRepos() {
        let expected: [String: String] = [
            "qwen3:0.6b": "mlx-community/Qwen3-0.6B-4bit",
            "qwen3:4b": "mlx-community/Qwen3-4B-4bit",
            "qwen3:8b": "mlx-community/Qwen3-8B-4bit",
            "qwen3.5:0.8b": "mlx-community/Qwen3.5-0.8B-4bit",
            "llama3.2:3b": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "phi3.5": "mlx-community/Phi-3.5-mini-instruct-4bit",
            "mistral:7b": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        ]
        for (alias, repo) in expected { #expect(Catalog.standard.repo(forAlias: alias) == repo, "\(alias)") }
    }

    @Test func bareFamilyAndLatestMeanTheFamilyDefault() {
        let c = Catalog.standard
        #expect(c.repo(forAlias: "qwen3") == "mlx-community/Qwen3-8B-4bit")
        #expect(c.repo(forAlias: "qwen3:latest") == c.repo(forAlias: "qwen3"))
        #expect(c.repo(forAlias: "phi3.5:latest") == c.repo(forAlias: "phi3.5"))
        #expect(c.repo(forAlias: "qwen3.5") == "mlx-community/Qwen3.5-0.8B-4bit")
    }

    @Test func aliasesAreCaseInsensitiveAndUnknownOnesMiss() {
        #expect(Catalog.standard.repo(forAlias: "Qwen3:8B") == "mlx-community/Qwen3-8B-4bit")
        #expect(Catalog.standard.repo(forAlias: "nope:1b") == nil)
        #expect(Catalog.standard.repo(forAlias: "") == nil)
    }

    @Test func everyRepoIsAValidMLXCommunityIdAndAliasesAreUnique() {
        let c = Catalog.standard
        #expect(!c.models.isEmpty)
        for m in c.models {
            #expect(RepoID.isValid(m.id), "\(m.id)")
            #expect(m.id.hasPrefix("mlx-community/"), "\(m.id)")
            #expect(!m.aliases.isEmpty, "\(m.id) has no alias")
            #expect(m.aliases.allSatisfy { $0 == $0.lowercased() && !$0.isEmpty }, "\(m.id)")
            #expect(m.gigabytes > 0)
        }
        let all = c.models.flatMap(\.aliases)
        #expect(Set(all).count == all.count, "an alias names two models")
        #expect(Set(c.aliases) == Set(all))
    }

    @Test func keepsEveryModelTheAppRecommendedBefore() {
        let previous = [
            "mlx-community/Qwen3-0.6B-4bit", "mlx-community/Qwen3.5-0.8B-4bit", "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "mlx-community/Qwen3-4B-4bit", "mlx-community/Phi-3.5-mini-instruct-4bit",
            "mlx-community/Mistral-7B-Instruct-v0.3-4bit", "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
            "mlx-community/Qwen3-8B-4bit", "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
            "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
        ]
        let ids = Set(Catalog.standard.models.map(\.id))
        for p in previous { #expect(ids.contains(p), "\(p) dropped from the catalog") }
    }

    @Test func aCustomCatalogIsInjectable() {
        let tiny = CatalogModel(id: "acme/Tiny-4bit", title: "Tiny", blurb: "b", gigabytes: 0.1, aliases: ["tiny", "tiny:1b", "tiny:latest"])
        let c = Catalog(models: [tiny])
        #expect(c.repo(forAlias: "tiny") == "acme/Tiny-4bit")
        #expect(c.repo(forAlias: "TINY:1B") == "acme/Tiny-4bit")
        #expect(c.repo(forAlias: "qwen3") == nil)
        #expect(c.aliases.sorted() == ["tiny", "tiny:1b", "tiny:latest"])
    }
}
