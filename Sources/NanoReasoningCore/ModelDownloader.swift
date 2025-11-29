// SPDX-License-Identifier: MIT
// Nano-Reasoning: HuggingFace Model Downloader
// Automatic weight downloading and caching for Qwen3 models

import Foundation
import Hub
@preconcurrency import MLX

/// Model download progress
public struct DownloadProgress: Sendable {
    public let bytesDownloaded: Int64
    public let totalBytes: Int64
    public let currentFile: String
    public let filesCompleted: Int
    public let totalFiles: Int
    
    public var fractionCompleted: Double {
        guard totalBytes > 0 else { return 0 }
        return Double(bytesDownloaded) / Double(totalBytes)
    }
    
    public var description: String {
        let mb = Double(bytesDownloaded) / 1_000_000
        let totalMB = Double(totalBytes) / 1_000_000
        return String(format: "%.1f/%.1f MB (%d/%d files)", mb, totalMB, filesCompleted, totalFiles)
    }
}

/// Model download configuration
public struct ModelDownloadConfig: Sendable {
    public let modelId: String
    public let revision: String
    public let cacheDirectory: URL
    public let forceRedownload: Bool
    
    public init(
        modelId: String,
        revision: String = "main",
        cacheDirectory: URL? = nil,
        forceRedownload: Bool = false
    ) {
        self.modelId = modelId
        self.revision = revision
        self.cacheDirectory = cacheDirectory ?? EnvironmentConfig.modelCacheDirectory
        self.forceRedownload = forceRedownload
    }
}

/// Downloaded model information
public struct DownloadedModel: Sendable {
    public let modelId: String
    public let localPath: URL
    public let configPath: URL?
    public let weightsPath: URL
    public let tokenizerPath: URL?
    public let quantization: ModelConfiguration.Quantization?
    
    /// Check if all required files exist
    public var isComplete: Bool {
        FileManager.default.fileExists(atPath: weightsPath.path)
    }
}

/// Model file types to download
public enum ModelFileType: String, CaseIterable, Sendable {
    case weights = "model.safetensors"
    case weightsShard = "model-00001-of-"  // For sharded models
    case config = "config.json"
    case tokenizer = "tokenizer.json"
    case tokenizerConfig = "tokenizer_config.json"
    case specialTokens = "special_tokens_map.json"
    case generationConfig = "generation_config.json"
}

/// Errors during model downloading
public enum ModelDownloadError: Error, LocalizedError {
    case networkError(String)
    case fileNotFound(String)
    case invalidModelId(String)
    case downloadFailed(String)
    case checksumMismatch(String)
    case insufficientDiskSpace(required: Int64, available: Int64)
    case weightLoadingFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .networkError(let msg):
            return "Network error: \(msg)"
        case .fileNotFound(let file):
            return "File not found: \(file)"
        case .invalidModelId(let id):
            return "Invalid model ID: \(id)"
        case .downloadFailed(let msg):
            return "Download failed: \(msg)"
        case .checksumMismatch(let file):
            return "Checksum mismatch for: \(file)"
        case .insufficientDiskSpace(let required, let available):
            let reqGB = Double(required) / 1_000_000_000
            let availGB = Double(available) / 1_000_000_000
            return String(format: "Insufficient disk space: need %.1f GB, have %.1f GB", reqGB, availGB)
        case .weightLoadingFailed(let msg):
            return "Weight loading failed: \(msg)"
        }
    }
}

/// Model downloader with HuggingFace Hub integration
public actor ModelDownloader {
    
    /// Shared instance
    public static let shared = ModelDownloader()
    
    private let fileManager = FileManager.default
    private var downloadTasks: [String: Task<DownloadedModel, Error>] = [:]
    
    public init() {}
    
    /// Create a new Hub API instance (nonisolated to avoid data race)
    private nonisolated func createHubApi() -> HubApi {
        HubApi()
    }
    
    /// Download or retrieve cached model
    public func downloadModel(
        config: ModelDownloadConfig,
        progressHandler: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws -> DownloadedModel {
        // Check if already downloading
        if let existingTask = downloadTasks[config.modelId] {
            return try await existingTask.value
        }
        
        // Check cache first
        if !config.forceRedownload {
            if let cached = try? getCachedModel(config: config) {
                if cached.isComplete {
                    return cached
                }
            }
        }
        
        // Start download
        let task = Task {
            try await performDownload(config: config, progressHandler: progressHandler)
        }
        
        downloadTasks[config.modelId] = task
        
        defer {
            downloadTasks.removeValue(forKey: config.modelId)
        }
        
        return try await task.value
    }
    
    /// Get cached model if available
    public func getCachedModel(config: ModelDownloadConfig) throws -> DownloadedModel? {
        let modelDir = getModelDirectory(for: config)
        
        guard fileManager.fileExists(atPath: modelDir.path) else {
            return nil
        }
        
        // Check for weights file
        let weightsPath = findWeightsFile(in: modelDir)
        guard let weights = weightsPath else {
            return nil
        }
        
        return DownloadedModel(
            modelId: config.modelId,
            localPath: modelDir,
            configPath: modelDir.appendingPathComponent("config.json"),
            weightsPath: weights,
            tokenizerPath: modelDir.appendingPathComponent("tokenizer.json"),
            quantization: detectQuantization(from: config.modelId)
        )
    }
    
    /// Perform the actual download
    private func performDownload(
        config: ModelDownloadConfig,
        progressHandler: (@Sendable (DownloadProgress) -> Void)?
    ) async throws -> DownloadedModel {
        let modelDir = getModelDirectory(for: config)
        
        // Create directory
        try fileManager.createDirectory(at: modelDir, withIntermediateDirectories: true)
        
        // Parse model ID
        let repo = Hub.Repo(id: config.modelId)
        
        // Create Hub API (nonisolated to avoid data races)
        let hubApi = createHubApi()
        
        // Get file list from hub
        let snapshot = try await hubApi.snapshot(from: repo, matching: ["*.safetensors", "*.json"])
        
        // Download using Hub API
        var filesCompleted = 0
        let totalFiles = 4  // Approximate: weights, config, tokenizer, tokenizer_config
        
        // Report initial progress
        progressHandler?(DownloadProgress(
            bytesDownloaded: 0,
            totalBytes: 1,
            currentFile: "Preparing download...",
            filesCompleted: 0,
            totalFiles: totalFiles
        ))
        
        // The snapshot is already downloaded to Hub's cache
        // Copy or link files to our cache directory
        for file in try fileManager.contentsOfDirectory(at: snapshot, includingPropertiesForKeys: nil) {
            let destPath = modelDir.appendingPathComponent(file.lastPathComponent)
            
            if !fileManager.fileExists(atPath: destPath.path) {
                try fileManager.copyItem(at: file, to: destPath)
            }
            
            filesCompleted += 1
            progressHandler?(DownloadProgress(
                bytesDownloaded: Int64(filesCompleted),
                totalBytes: Int64(totalFiles),
                currentFile: file.lastPathComponent,
                filesCompleted: filesCompleted,
                totalFiles: totalFiles
            ))
        }
        
        // Find weights file
        guard let weightsPath = findWeightsFile(in: modelDir) else {
            throw ModelDownloadError.fileNotFound("model.safetensors")
        }
        
        progressHandler?(DownloadProgress(
            bytesDownloaded: Int64(totalFiles),
            totalBytes: Int64(totalFiles),
            currentFile: "Complete",
            filesCompleted: totalFiles,
            totalFiles: totalFiles
        ))
        
        return DownloadedModel(
            modelId: config.modelId,
            localPath: modelDir,
            configPath: modelDir.appendingPathComponent("config.json"),
            weightsPath: weightsPath,
            tokenizerPath: modelDir.appendingPathComponent("tokenizer.json"),
            quantization: detectQuantization(from: config.modelId)
        )
    }
    
    /// Get model directory path
    private func getModelDirectory(for config: ModelDownloadConfig) -> URL {
        let safeId = config.modelId.replacingOccurrences(of: "/", with: "_")
        return config.cacheDirectory.appendingPathComponent(safeId)
    }
    
    /// Find weights file in directory (handles sharded models)
    private func findWeightsFile(in directory: URL) -> URL? {
        // Try single weights file first
        let singleWeights = directory.appendingPathComponent("model.safetensors")
        if fileManager.fileExists(atPath: singleWeights.path) {
            return singleWeights
        }
        
        // Try consolidated weights
        let consolidated = directory.appendingPathComponent("weights.safetensors")
        if fileManager.fileExists(atPath: consolidated.path) {
            return consolidated
        }
        
        // Look for sharded weights
        do {
            let contents = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            let shards = contents.filter { $0.lastPathComponent.contains("model-") && $0.pathExtension == "safetensors" }
            if let firstShard = shards.sorted(by: { $0.path < $1.path }).first {
                return firstShard
            }
        } catch {
            return nil
        }
        
        return nil
    }
    
    /// Detect quantization from model ID
    private func detectQuantization(from modelId: String) -> ModelConfiguration.Quantization? {
        let lowered = modelId.lowercased()
        if lowered.contains("4bit") || lowered.contains("int4") || lowered.contains("q4") {
            return .int4
        } else if lowered.contains("8bit") || lowered.contains("int8") || lowered.contains("q8") {
            return .int8
        } else if lowered.contains("fp16") || lowered.contains("float16") {
            return .fp16
        }
        return nil
    }
    
    /// Check available disk space
    public func checkDiskSpace(required: Int64) throws {
        let cacheDir = EnvironmentConfig.modelCacheDirectory
        
        do {
            let attrs = try fileManager.attributesOfFileSystem(forPath: cacheDir.path)
            if let available = attrs[.systemFreeSize] as? Int64 {
                if available < required {
                    throw ModelDownloadError.insufficientDiskSpace(required: required, available: available)
                }
            }
        } catch let error as ModelDownloadError {
            throw error
        } catch {
            // If we can't check, proceed anyway
        }
    }
    
    /// Clear cached model
    public func clearCache(for modelId: String) throws {
        let safeId = modelId.replacingOccurrences(of: "/", with: "_")
        let modelDir = EnvironmentConfig.modelCacheDirectory.appendingPathComponent(safeId)
        
        if fileManager.fileExists(atPath: modelDir.path) {
            try fileManager.removeItem(at: modelDir)
        }
    }
    
    /// Clear all cached models
    public func clearAllCache() throws {
        let cacheDir = EnvironmentConfig.modelCacheDirectory
        
        if fileManager.fileExists(atPath: cacheDir.path) {
            try fileManager.removeItem(at: cacheDir)
            try fileManager.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        }
    }
    
    /// Get total cache size
    public func getCacheSize() -> Int64 {
        let cacheDir = EnvironmentConfig.modelCacheDirectory
        var totalSize: Int64 = 0
        
        if let enumerator = fileManager.enumerator(at: cacheDir, includingPropertiesForKeys: [.fileSizeKey]) {
            for case let fileURL as URL in enumerator {
                if let attrs = try? fileURL.resourceValues(forKeys: [.fileSizeKey]),
                   let size = attrs.fileSize {
                    totalSize += Int64(size)
                }
            }
        }
        
        return totalSize
    }
    
    /// List all cached models
    public func listCachedModels() -> [String] {
        let cacheDir = EnvironmentConfig.modelCacheDirectory
        
        guard let contents = try? fileManager.contentsOfDirectory(at: cacheDir, includingPropertiesForKeys: nil) else {
            return []
        }
        
        return contents
            .filter { $0.hasDirectoryPath }
            .map { $0.lastPathComponent.replacingOccurrences(of: "_", with: "/") }
    }
}

// MARK: - Weight Loading Utilities

/// Utilities for loading and validating model weights
public struct WeightLoader {
    
    /// Load weights from safetensors file
    public static func loadWeights(from url: URL) throws -> [String: MLXArray] {
        try loadArrays(url: url)
    }
    
    /// Load sharded weights from multiple files
    public static func loadShardedWeights(from directory: URL) throws -> [String: MLXArray] {
        let fm = FileManager.default
        let contents = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        
        let shards = contents
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.path < $1.path }
        
        var allWeights: [String: MLXArray] = [:]
        
        for shard in shards {
            let shardWeights = try loadArrays(url: shard)
            for (key, value) in shardWeights {
                allWeights[key] = value
            }
        }
        
        return allWeights
    }
    
    /// Validate weight shapes against expected configuration
    public static func validateWeights(
        _ weights: [String: MLXArray],
        config: ModelWeightConfig
    ) throws {
        // Validate embedding
        if let embedWeight = weights["model.embed_tokens.weight"] ?? weights["embed_tokens.weight"] {
            let expectedShape = [config.vocabSize, config.hiddenSize]
            guard embedWeight.shape == expectedShape else {
                throw ModelDownloadError.weightLoadingFailed(
                    "Embedding shape mismatch: expected \(expectedShape), got \(embedWeight.shape)"
                )
            }
        }
        
        // Validate LM head
        if let lmHead = weights["lm_head.weight"] {
            let expectedShape = [config.vocabSize, config.hiddenSize]
            guard lmHead.shape == expectedShape else {
                throw ModelDownloadError.weightLoadingFailed(
                    "LM head shape mismatch: expected \(expectedShape), got \(lmHead.shape)"
                )
            }
        }
        
        // Validate layer count by checking for layer weights
        var maxLayerIndex = -1
        for key in weights.keys {
            if let range = key.range(of: "layers.") {
                let afterLayers = key[range.upperBound...]
                if let dotIndex = afterLayers.firstIndex(of: ".") {
                    let layerStr = String(afterLayers[..<dotIndex])
                    if let layerIdx = Int(layerStr) {
                        maxLayerIndex = max(maxLayerIndex, layerIdx)
                    }
                }
            }
        }
        
        if maxLayerIndex >= 0 {
            let numLayers = maxLayerIndex + 1
            guard numLayers == config.numLayers else {
                throw ModelDownloadError.weightLoadingFailed(
                    "Layer count mismatch: expected \(config.numLayers), found \(numLayers)"
                )
            }
        }
    }
    
    /// Map weight keys from HuggingFace format to internal format
    public static func mapWeightKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var mapped: [String: MLXArray] = [:]
        
        for (key, value) in weights {
            // Remove "model." prefix if present
            var newKey = key
            if newKey.hasPrefix("model.") {
                newKey = String(newKey.dropFirst(6))
            }
            
            // Map common key patterns
            newKey = newKey
                .replacingOccurrences(of: "self_attn", with: "selfAttn")
                .replacingOccurrences(of: "input_layernorm", with: "inputLayerNorm")
                .replacingOccurrences(of: "post_attention_layernorm", with: "postAttnLayerNorm")
                .replacingOccurrences(of: "q_proj", with: "qProj")
                .replacingOccurrences(of: "k_proj", with: "kProj")
                .replacingOccurrences(of: "v_proj", with: "vProj")
                .replacingOccurrences(of: "o_proj", with: "oProj")
                .replacingOccurrences(of: "gate_proj", with: "gateProj")
                .replacingOccurrences(of: "up_proj", with: "upProj")
                .replacingOccurrences(of: "down_proj", with: "downProj")
            
            mapped[newKey] = value
        }
        
        return mapped
    }
}

/// Expected weight configuration for validation
public struct ModelWeightConfig: Sendable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let numLayers: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let intermediateSize: Int
    
    public static func forQwen3(tier: HardwareTier) -> ModelWeightConfig {
        switch tier {
        case .entry:
            // Qwen3-4B
            return ModelWeightConfig(
                vocabSize: 151936,
                hiddenSize: 2560,
                numLayers: 36,
                numHeads: 20,
                numKVHeads: 4,
                intermediateSize: 6912
            )
        case .pro:
            // Qwen3-7B
            return ModelWeightConfig(
                vocabSize: 152064,
                hiddenSize: 3584,
                numLayers: 28,
                numHeads: 28,
                numKVHeads: 4,
                intermediateSize: 18944
            )
        case .elite:
            // Qwen3-32B
            return ModelWeightConfig(
                vocabSize: 152064,
                hiddenSize: 5120,
                numLayers: 64,
                numHeads: 40,
                numKVHeads: 8,
                intermediateSize: 27648
            )
        }
    }
    
    /// Qwen3 0.6B drafter config
    public static let qwen3_0_6B = ModelWeightConfig(
        vocabSize: 151936,
        hiddenSize: 1024,
        numLayers: 28,
        numHeads: 16,
        numKVHeads: 2,
        intermediateSize: 3072
    )
}

// MARK: - Model Configuration Loading

/// Loads and parses model config.json
public struct ModelConfigLoader {
    
    /// Parsed model configuration
    public struct ParsedConfig: Sendable {
        public let vocabSize: Int
        public let hiddenSize: Int
        public let numLayers: Int
        public let numHeads: Int
        public let numKVHeads: Int
        public let intermediateSize: Int
        public let maxPositionEmbeddings: Int
        public let ropeTheta: Float
        public let tieWordEmbeddings: Bool
    }
    
    /// Load configuration from config.json
    public static func load(from url: URL) throws -> ParsedConfig {
        let data = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        
        return ParsedConfig(
            vocabSize: json["vocab_size"] as? Int ?? 151936,
            hiddenSize: json["hidden_size"] as? Int ?? 4096,
            numLayers: json["num_hidden_layers"] as? Int ?? 32,
            numHeads: json["num_attention_heads"] as? Int ?? 32,
            numKVHeads: json["num_key_value_heads"] as? Int ?? 8,
            intermediateSize: json["intermediate_size"] as? Int ?? 14336,
            maxPositionEmbeddings: json["max_position_embeddings"] as? Int ?? 32768,
            ropeTheta: (json["rope_theta"] as? Double).map { Float($0) } ?? 1000000.0,
            tieWordEmbeddings: json["tie_word_embeddings"] as? Bool ?? false
        )
    }
}

/// Extension to convert parsed config to internal config
extension ModelConfigLoader.ParsedConfig {
    public func toModelWeightConfig() -> ModelWeightConfig {
        ModelWeightConfig(
            vocabSize: vocabSize,
            hiddenSize: hiddenSize,
            numLayers: numLayers,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            intermediateSize: intermediateSize
        )
    }
}
