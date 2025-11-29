// SPDX-License-Identifier: MIT
// Nano-Reasoning: Configuration Management
// Handles model selection, runtime configuration, and persistence

import Foundation

/// Main configuration for Nano-Reasoning
public struct NanoReasoningConfig: Codable, Sendable {
    /// Hardware configuration
    public var hardware: HardwareConfig
    /// Model configuration
    public var models: ModelsConfig
    /// Generation configuration
    public var generation: GenerationSettings
    /// Training configuration
    public var training: TrainingSettings
    /// Runtime configuration
    public var runtime: RuntimeSettings
    
    public static let `default` = NanoReasoningConfig(
        hardware: .auto,
        models: .auto,
        generation: .default,
        training: .default,
        runtime: .default
    )
    
    /// Load configuration from file
    public static func load(from url: URL) throws -> NanoReasoningConfig {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(NanoReasoningConfig.self, from: data)
    }
    
    /// Save configuration to file
    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        try data.write(to: url)
    }
}

/// Hardware configuration
public struct HardwareConfig: Codable, Sendable {
    /// Force a specific tier (nil = auto-detect)
    public var forceTier: String?
    /// Override memory detection (GB)
    public var overrideMemoryGB: Int?
    /// GPU utilization target (0.0 - 1.0)
    public var gpuUtilizationTarget: Float
    /// Enable NPU offload on supported hardware
    public var enableNPUOffload: Bool
    
    public static let auto = HardwareConfig(
        forceTier: nil,
        overrideMemoryGB: nil,
        gpuUtilizationTarget: 0.85,
        enableNPUOffload: true
    )
}

/// Model configuration
public struct ModelsConfig: Codable, Sendable {
    /// Target model ID (HuggingFace or local path)
    public var targetModelId: String?
    /// Drafter model ID
    public var drafterModelId: String?
    /// Target model quantization
    public var targetQuantization: String
    /// Drafter model quantization
    public var drafterQuantization: String
    /// Local model cache directory
    public var cacheDirectory: String?
    
    public static let auto = ModelsConfig(
        targetModelId: nil,
        drafterModelId: nil,
        targetQuantization: "int4",
        drafterQuantization: "fp16",
        cacheDirectory: nil
    )
    
    /// Get model configuration for a specific tier
    public static func forTier(_ tier: HardwareTier) -> ModelsConfig {
        let config = ModelConfiguration.forTier(tier)
        return ModelsConfig(
            targetModelId: config.targetModelId,
            drafterModelId: config.drafterModelId,
            targetQuantization: config.targetQuantization.rawValue,
            drafterQuantization: config.drafterQuantization.rawValue,
            cacheDirectory: nil
        )
    }
}

/// Generation settings
public struct GenerationSettings: Codable, Sendable {
    /// Maximum tokens to generate
    public var maxTokens: Int
    /// Sampling temperature
    public var temperature: Float
    /// Top-p (nucleus) sampling parameter
    public var topP: Float
    /// Top-k sampling parameter
    public var topK: Int
    /// Repetition penalty
    public var repetitionPenalty: Float
    /// Number of draft tokens (k)
    public var draftCount: Int
    /// Enable speculative decoding
    public var speculativeDecoding: Bool
    
    public static let `default` = GenerationSettings(
        maxTokens: 2048,
        temperature: 0.7,
        topP: 0.9,
        topK: 40,
        repetitionPenalty: 1.1,
        draftCount: 5,
        speculativeDecoding: true
    )
    
    /// Convert to GenerationConfig
    public func toGenerationConfig() -> GenerationConfig {
        GenerationConfig(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            topK: topK,
            repetitionPenalty: repetitionPenalty,
            stopTokens: []
        )
    }
}

/// Training settings
public struct TrainingSettings: Codable, Sendable {
    /// Enable training
    public var enabled: Bool
    /// Learning rate
    public var learningRate: Float
    /// Batch size
    public var batchSize: Int
    /// Gradient accumulation steps
    public var gradientAccumulationSteps: Int
    /// Training buffer capacity
    public var bufferCapacity: Int
    /// Checkpoint save interval (steps)
    public var checkpointInterval: Int
    /// Checkpoint directory
    public var checkpointDirectory: String?
    /// Enable LoRA adapters
    public var enableLoRA: Bool
    /// LoRA rank
    public var loraRank: Int
    /// Enable EAGLE head training
    public var enableEAGLE: Bool
    
    public static let `default` = TrainingSettings(
        enabled: true,
        learningRate: 1e-4,
        batchSize: 4,
        gradientAccumulationSteps: 4,
        bufferCapacity: 50,
        checkpointInterval: 500,
        checkpointDirectory: nil,
        enableLoRA: true,
        loraRank: 16,
        enableEAGLE: true
    )
    
    /// Convert to TrainingConfig
    public func toTrainingConfig() -> TrainingConfig {
        TrainingConfig(
            learningRate: learningRate,
            batchSize: batchSize,
            gradientAccumulationSteps: gradientAccumulationSteps,
            maxGradNorm: 1.0,
            warmupSteps: 100,
            logInterval: 10,
            checkpointInterval: checkpointInterval,
            minSamplesBeforeTraining: 16
        )
    }
}

/// Runtime settings
public struct RuntimeSettings: Codable, Sendable {
    /// Enable verbose logging
    public var verbose: Bool
    /// Enable metrics collection
    public var metricsEnabled: Bool
    /// Metrics reporting interval (seconds)
    public var metricsInterval: Int
    /// Enable profiling
    public var profilingEnabled: Bool
    /// Thread pool size (0 = auto)
    public var threadPoolSize: Int
    
    public static let `default` = RuntimeSettings(
        verbose: false,
        metricsEnabled: true,
        metricsInterval: 10,
        profilingEnabled: false,
        threadPoolSize: 0
    )
}

// Model Registry

/// Registry of supported models
public struct ModelRegistry {
    /// All known target models
    public static let targetModels: [ModelInfo] = [
        // Entry tier (16GB)
        ModelInfo(
            id: "mlx-community/Qwen3-4B-Instruct-4bit",
            name: "Qwen3 4B",
            size: "4B",
            quantization: "4-bit",
            memoryRequired: 3.0,
            tier: .entry
        ),
        // Pro tier (24GB+)
        ModelInfo(
            id: "mlx-community/Qwen3-7B-Instruct-4bit",
            name: "Qwen3 7B",
            size: "7B",
            quantization: "4-bit",
            memoryRequired: 5.5,
            tier: .pro
        ),
        ModelInfo(
            id: "mlx-community/Qwen3-14B-Instruct-4bit",
            name: "Qwen3 14B",
            size: "14B",
            quantization: "4-bit",
            memoryRequired: 9.5,
            tier: .pro
        ),
        // Elite tier (36GB+)
        ModelInfo(
            id: "mlx-community/Qwen3-32B-Instruct-4bit",
            name: "Qwen3 32B",
            size: "32B",
            quantization: "4-bit",
            memoryRequired: 21.0,
            tier: .elite
        ),
        ModelInfo(
            id: "mlx-community/Qwen3-72B-Instruct-4bit",
            name: "Qwen3 72B",
            size: "72B",
            quantization: "4-bit",
            memoryRequired: 45.0,
            tier: .elite
        ),
    ]
    
    /// All known drafter models
    public static let drafterModels: [ModelInfo] = [
        ModelInfo(
            id: "mlx-community/Qwen3-0.6B-Instruct-4bit",
            name: "Qwen3 0.6B (4-bit)",
            size: "0.6B",
            quantization: "4-bit",
            memoryRequired: 0.6,
            tier: .entry
        ),
        ModelInfo(
            id: "mlx-community/Qwen3-0.6B-Instruct",
            name: "Qwen3 0.6B (FP16)",
            size: "0.6B",
            quantization: "fp16",
            memoryRequired: 1.2,
            tier: .pro
        ),
        ModelInfo(
            id: "mlx-community/Qwen3-1.8B-Instruct",
            name: "Qwen3 1.8B (FP16)",
            size: "1.8B",
            quantization: "fp16",
            memoryRequired: 3.5,
            tier: .elite
        ),
    ]
    
    /// Get recommended target model for memory size
    public static func recommendedTarget(forMemoryGB memory: Int) -> ModelInfo? {
        targetModels
            .filter { $0.memoryRequired * 1.5 <= Double(memory) }
            .sorted { $0.memoryRequired > $1.memoryRequired }
            .first
    }
    
    /// Get recommended drafter model for tier
    public static func recommendedDrafter(forTier tier: HardwareTier) -> ModelInfo? {
        drafterModels.first { $0.tier == tier } ?? drafterModels.first
    }
}

/// Model information
public struct ModelInfo: Sendable {
    public let id: String
    public let name: String
    public let size: String
    public let quantization: String
    public let memoryRequired: Double  // GB
    public let tier: HardwareTier
}

// Environment Variables

/// Environment configuration
public struct EnvironmentConfig {
    /// Get model cache directory from environment or default
    public static var modelCacheDirectory: URL {
        if let envPath = ProcessInfo.processInfo.environment["NANO_REASONING_CACHE"] {
            return URL(fileURLWithPath: envPath)
        }
        
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        return homeDir.appendingPathComponent(".cache/nano-reasoning/models")
    }
    
    /// Get checkpoint directory from environment or default
    public static var checkpointDirectory: URL {
        if let envPath = ProcessInfo.processInfo.environment["NANO_REASONING_CHECKPOINTS"] {
            return URL(fileURLWithPath: envPath)
        }
        
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        return homeDir.appendingPathComponent(".cache/nano-reasoning/checkpoints")
    }
    
    /// Get config file path from environment or default
    public static var configFilePath: URL {
        if let envPath = ProcessInfo.processInfo.environment["NANO_REASONING_CONFIG"] {
            return URL(fileURLWithPath: envPath)
        }
        
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        return homeDir.appendingPathComponent(".config/nano-reasoning/config.json")
    }
    
    /// Check if debug mode is enabled
    public static var debugEnabled: Bool {
        ProcessInfo.processInfo.environment["NANO_REASONING_DEBUG"] == "1"
    }
    
    /// Get log level from environment
    public static var logLevel: String {
        ProcessInfo.processInfo.environment["NANO_REASONING_LOG_LEVEL"] ?? "info"
    }
    
    /// Optional override for pre-downloaded model weights (safetensors directory or file)
    public static var modelWeightsPath: URL? {
        guard let envPath = ProcessInfo.processInfo.environment["NANO_REASONING_WEIGHTS"] else {
            return nil
        }
        return URL(fileURLWithPath: envPath)
    }
}

// Configuration Builder

/// Builder for creating configurations
public struct ConfigurationBuilder {
    private var config: NanoReasoningConfig
    
    public init() {
        self.config = .default
    }
    
    public func withTier(_ tier: String) -> ConfigurationBuilder {
        var builder = self
        builder.config.hardware.forceTier = tier
        return builder
    }
    
    public func withTargetModel(_ modelId: String) -> ConfigurationBuilder {
        var builder = self
        builder.config.models.targetModelId = modelId
        return builder
    }
    
    public func withDrafterModel(_ modelId: String) -> ConfigurationBuilder {
        var builder = self
        builder.config.models.drafterModelId = modelId
        return builder
    }
    
    public func withTemperature(_ temp: Float) -> ConfigurationBuilder {
        var builder = self
        builder.config.generation.temperature = temp
        return builder
    }
    
    public func withMaxTokens(_ tokens: Int) -> ConfigurationBuilder {
        var builder = self
        builder.config.generation.maxTokens = tokens
        return builder
    }
    
    public func withTrainingEnabled(_ enabled: Bool) -> ConfigurationBuilder {
        var builder = self
        builder.config.training.enabled = enabled
        return builder
    }
    
    public func withLearningRate(_ lr: Float) -> ConfigurationBuilder {
        var builder = self
        builder.config.training.learningRate = lr
        return builder
    }
    
    public func withVerbose(_ verbose: Bool) -> ConfigurationBuilder {
        var builder = self
        builder.config.runtime.verbose = verbose
        return builder
    }
    
    public func build() -> NanoReasoningConfig {
        config
    }
}

// Validation

extension NanoReasoningConfig {
    /// Validate configuration
    public func validate() throws {
        // Validate generation settings
        guard generation.maxTokens > 0 else {
            throw ConfigurationError.invalidValue("maxTokens must be positive")
        }
        
        guard generation.temperature >= 0 else {
            throw ConfigurationError.invalidValue("temperature must be non-negative")
        }
        
        guard generation.topP > 0 && generation.topP <= 1 else {
            throw ConfigurationError.invalidValue("topP must be between 0 and 1")
        }
        
        guard generation.draftCount > 0 && generation.draftCount <= 20 else {
            throw ConfigurationError.invalidValue("draftCount must be between 1 and 20")
        }
        
        // Validate training settings
        if training.enabled {
            guard training.learningRate > 0 else {
                throw ConfigurationError.invalidValue("learningRate must be positive when training is enabled")
            }
            
            guard training.batchSize > 0 else {
                throw ConfigurationError.invalidValue("batchSize must be positive")
            }
        }
        
        // Validate hardware settings
        guard hardware.gpuUtilizationTarget > 0 && hardware.gpuUtilizationTarget <= 1 else {
            throw ConfigurationError.invalidValue("gpuUtilizationTarget must be between 0 and 1")
        }
    }
}

public enum ConfigurationError: Error, LocalizedError {
    case invalidValue(String)
    case missingRequired(String)
    case incompatible(String)
    
    public var errorDescription: String? {
        switch self {
        case .invalidValue(let msg):
            return "Invalid configuration value: \(msg)"
        case .missingRequired(let msg):
            return "Missing required configuration: \(msg)"
        case .incompatible(let msg):
            return "Incompatible configuration: \(msg)"
        }
    }
}
