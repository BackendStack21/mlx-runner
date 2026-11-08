import Foundation

/// What `run` does with the words, the `-p` text and the piped input.
enum RunInput: Equatable {
    /// Interactive chat.
    case chat
    /// One generation for this prompt; an empty prompt only loads the model.
    case oneShot(String)
}

extension CLI {
    /// Ollama's input rules: non-empty piped `stdin` goes before the prompt words (space-joined);
    /// `-p` text is another way to give the words (both together are an error); without a prompt and
    /// with stdin and stdout on a terminal the answer is chat, otherwise one generation.
    /// `stdin` is "" when stdin is a terminal (nothing is read from it).
    static func assembleRunInput(stdin: String, words: [String], promptFlag: String?,
                                 stdinIsTTY: Bool, stdoutIsTTY: Bool) throws -> RunInput {
        if promptFlag != nil && !words.isEmpty {
            throw CLIError.failure("give the prompt either as arguments or with -p, not both")
        }
        let given = promptFlag.map { [$0] } ?? words
        let parts = (stdin.isEmpty ? [] : [stdin]) + given
        if parts.isEmpty && stdinIsTTY && stdoutIsTTY { return .chat }
        return .oneShot(parts.joined(separator: " "))
    }
}
