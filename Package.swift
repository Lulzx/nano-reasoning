// swift-tools-version: 6.0
// Nano-Reasoning: FastRL Adaptive Drafter for Apple Silicon
// Requires: macOS 14+ (Sonoma, Sequoia, Tahoe)

import PackageDescription

let package = Package(
    name: "NanoReasoning",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(
            name: "nano-reasoning",
            targets: ["NanoReasoning"]
        ),
        .library(
            name: "NanoReasoningCore",
            targets: ["NanoReasoningCore"]
        )
    ],
    dependencies: [
        // MLX Swift - Apple's ML framework for Apple Silicon
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0"),
        // ArgumentParser for CLI
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    ],
    targets: [
        .executableTarget(
            name: "NanoReasoning",
            dependencies: [
                "NanoReasoningCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .target(
            name: "NanoReasoningCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "NanoReasoningTests",
            dependencies: ["NanoReasoningCore"]
        ),
    ],
    swiftLanguageModes: [.v6]
)
