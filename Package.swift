// swift-tools-version: 6.0
// MLX Runner — a native macOS app for running MLX models from Hugging Face.
// Zero third-party dependencies: only the Swift standard library and Apple system frameworks.
import PackageDescription

let package = Package(
    name: "MLXRunner",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "MLXRunner", targets: ["MLXRunner"]),
        .executable(name: "mlxr", targets: ["mlxr"]),
        .library(name: "MLXEngine", targets: ["MLXEngine"]),
    ],
    targets: [
        .target(
            name: "MLXEngine",
            swiftSettings: [.unsafeFlags(["-Ounchecked"], .when(configuration: .release))]
        ),
        .executableTarget(name: "MLXRunner", dependencies: ["MLXEngine"]),
        .executableTarget(name: "mlxr", dependencies: ["MLXEngine"]),
        .testTarget(name: "MLXEngineTests", dependencies: ["MLXEngine"]),
    ],
    swiftLanguageModes: [.v5]
)
