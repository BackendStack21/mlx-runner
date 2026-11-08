import Foundation
import MLXEngine

/// What a running download reports: bytes done, bytes in total, bytes per second, and status lines.
public struct DownloadHooks: Sendable {
    public var progress: @Sendable (_ done: Int64, _ total: Int64, _ bytesPerSecond: Double) -> Void
    public var event: @Sendable (String) -> Void

    public init(progress: @escaping @Sendable (Int64, Int64, Double) -> Void, event: @escaping @Sendable (String) -> Void) {
        self.progress = progress; self.event = event
    }
}

/// Hugging Face access for pull, show and auto-pull, injected so tests never touch the network or
/// the real model library.
public struct HubServices: Sendable {
    public var details: @Sendable (_ repo: String) async throws -> HubModelDetails
    public var destination: @Sendable (_ repo: String) throws -> URL
    public var download: @Sendable (_ details: HubModelDetails, _ destination: URL, _ hooks: DownloadHooks) async throws -> URL

    public init(details: @escaping @Sendable (String) async throws -> HubModelDetails,
                destination: @escaping @Sendable (String) throws -> URL,
                download: @escaping @Sendable (HubModelDetails, URL, DownloadHooks) async throws -> URL) {
        self.details = details; self.destination = destination; self.download = download
    }

    public static let live = HubServices(
        details: { try await HubClient.details($0) },
        destination: { try ModelStore.directory(for: $0) },
        download: { details, destination, hooks in
            let dl = ModelDownload(details: details, destination: destination)
            dl.onProgress = { _, done, total, speed in hooks.progress(done, total, speed) }
            dl.onEvent = hooks.event
            return try await withTaskCancellationHandler { try await dl.start() } onCancel: { dl.cancel() }
        })
}
