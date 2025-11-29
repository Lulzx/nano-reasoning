// SPDX-License-Identifier: MIT
// Nano-Reasoning: Background Training Loop
// Runs at background priority, yields to inference on M1/M2

import Foundation
import MLX
import MLXOptimizers

/// Training metrics for monitoring
public struct TrainingMetrics: Sendable {
    public let step: Int
    public let loss: Float
    public let learningRate: Float
    public let samplesProcessed: Int
    public let acceptanceRate: Float
    public let gpuUtilization: Float
    public let timestamp: Date
    
    public init(
        step: Int,
        loss: Float,
        learningRate: Float,
        samplesProcessed: Int,
        acceptanceRate: Float,
        gpuUtilization: Float
    ) {
        self.step = step
        self.loss = loss
        self.learningRate = learningRate
        self.samplesProcessed = samplesProcessed
        self.acceptanceRate = acceptanceRate
        self.gpuUtilization = gpuUtilization
        self.timestamp = Date()
    }
}

/// Training configuration
public struct TrainingConfig: Sendable {
    public let learningRate: Float
    public let batchSize: Int
    public let gradientAccumulationSteps: Int
    public let maxGradNorm: Float
    public let warmupSteps: Int
    public let logInterval: Int
    public let checkpointInterval: Int
    public let minSamplesBeforeTraining: Int
    
    public static let `default` = TrainingConfig(
        learningRate: 1e-4,
        batchSize: 4,
        gradientAccumulationSteps: 4,
        maxGradNorm: 1.0,
        warmupSteps: 100,
        logInterval: 10,
        checkpointInterval: 500,
        minSamplesBeforeTraining: 16
    )
    
    public static func forTier(_ tier: HardwareTier) -> TrainingConfig {
        switch tier {
        case .entry:
            // No training for entry tier
            return TrainingConfig(
                learningRate: 0,
                batchSize: 0,
                gradientAccumulationSteps: 0,
                maxGradNorm: 0,
                warmupSteps: 0,
                logInterval: 0,
                checkpointInterval: 0,
                minSamplesBeforeTraining: Int.max
            )
        case .pro:
            // Conservative training for M1/M2/M3 Pro
            return TrainingConfig(
                learningRate: 5e-5,
                batchSize: 2,
                gradientAccumulationSteps: 8,
                maxGradNorm: 1.0,
                warmupSteps: 200,
                logInterval: 20,
                checkpointInterval: 1000,
                minSamplesBeforeTraining: 32
            )
        case .elite:
            // Full training for M4/M5
            return TrainingConfig(
                learningRate: 1e-4,
                batchSize: 8,
                gradientAccumulationSteps: 4,
                maxGradNorm: 1.0,
                warmupSteps: 100,
                logInterval: 10,
                checkpointInterval: 500,
                minSamplesBeforeTraining: 16
            )
        }
    }
}

/// State of the trainer task
public enum TrainerState: Sendable {
    case idle
    case running
    case paused
    case stopped
    case error(String)
}

/// Background training task for adaptive drafter improvement
public actor TrainerTask {
    private let drafter: DrafterActor
    private let buffer: TrainingBuffer
    private let loadMonitor: GPULoadMonitor
    private let config: TrainingConfig
    private let tier: HardwareTier
    
    private var state: TrainerState = .idle
    private var currentStep: Int = 0
    private var totalSamplesProcessed: Int = 0
    private var recentLosses: [Float] = []
    private var trainingTask: Task<Void, Never>?
    
    // Metrics callback
    private var metricsCallback: ((TrainingMetrics) async -> Void)?
    
    public init(
        drafter: DrafterActor,
        buffer: TrainingBuffer,
        loadMonitor: GPULoadMonitor,
        config: TrainingConfig,
        tier: HardwareTier
    ) {
        self.drafter = drafter
        self.buffer = buffer
        self.loadMonitor = loadMonitor
        self.config = config
        self.tier = tier
    }
    
    /// Set callback for training metrics
    public func setMetricsCallback(_ callback: @escaping (TrainingMetrics) async -> Void) {
        self.metricsCallback = callback
    }
    
    /// Start the background training loop
    public func start() {
        guard tier.trainingEnabled else {
            state = .idle
            return
        }
        
        guard case .idle = state else { return }
        
        state = .running
        trainingTask = Task(priority: .background) { [weak self] in
            await self?.trainingLoop()
        }
    }
    
    /// Stop the training loop
    public func stop() {
        trainingTask?.cancel()
        trainingTask = nil
        state = .stopped
    }
    
    /// Pause training
    public func pause() {
        if case .running = state {
            state = .paused
        }
    }
    
    /// Resume training
    public func resume() {
        if case .paused = state {
            state = .running
        }
    }
    
    /// Get current trainer state
    public func getState() -> TrainerState {
        state
    }
    
    /// Get current training step
    public func getCurrentStep() -> Int {
        currentStep
    }
    
    /// Get average recent loss
    public func getAverageLoss() -> Float {
        guard !recentLosses.isEmpty else { return 0 }
        return recentLosses.reduce(0, +) / Float(recentLosses.count)
    }
    

    // Training Loop
    
    private func trainingLoop() async {
        while !Task.isCancelled {
            // Check state
            guard case .running = state else {
                if case .paused = state {
                    try? await Task.sleep(for: .milliseconds(500))
                    continue
                }
                break
            }
            
            // Check GPU load - yield to inference if under load
            let isUnderLoad = await loadMonitor.checkUnderLoad()
            if isUnderLoad {
                // M1/M2 optimization: back off when GPU is saturated
                try? await Task.sleep(for: .milliseconds(200))
                continue
            }
            
            // Check if we have enough samples
            let hasEnoughSamples = await buffer.hasTrainingBatch()
            if !hasEnoughSamples {
                try? await Task.sleep(for: .milliseconds(100))
                continue
            }
            
            // Get training batch
            let batch = await buffer.popBatch(maxSize: config.batchSize)
            if batch.isEmpty {
                try? await Task.sleep(for: .milliseconds(50))
                continue
            }
            
            // Perform training step with gradient accumulation
            await performTrainingStep(batch: batch)
            
            // Yield to allow inference tasks
            await Task.yield()
            
            // Small delay to prevent GPU monopolization on M1/M2
            if tier == .pro {
                try? await Task.sleep(for: .milliseconds(10))
            }
        }
    }
    
    private func performTrainingStep(batch: [TrainingSample]) async {
        // Train on batch
        let loss = await drafter.trainStep(samples: batch)
        
        currentStep += 1
        totalSamplesProcessed += batch.count
        
        // Track losses
        recentLosses.append(loss)
        if recentLosses.count > 100 {
            recentLosses.removeFirst()
        }
        
        // Report metrics
        if currentStep % config.logInterval == 0 {
            await reportMetrics(loss: loss)
        }
        
        // Checkpoint
        if config.checkpointInterval > 0 && currentStep % config.checkpointInterval == 0 {
            await saveCheckpoint()
        }
    }
    
    private func reportMetrics(loss: Float) async {
        let gpuUtilization = await loadMonitor.getAverageLoad()
        let acceptanceRate = await drafter.getAcceptanceRate()
        let currentLR = getLearningRate()
        
        let metrics = TrainingMetrics(
            step: currentStep,
            loss: loss,
            learningRate: currentLR,
            samplesProcessed: totalSamplesProcessed,
            acceptanceRate: acceptanceRate,
            gpuUtilization: Float(gpuUtilization)
        )
        
        await metricsCallback?(metrics)
    }
    
    private func getLearningRate() -> Float {
        // Warmup + constant learning rate
        if currentStep < config.warmupSteps {
            return config.learningRate * Float(currentStep) / Float(config.warmupSteps)
        }
        return config.learningRate
    }
    
    private func saveCheckpoint() async {
        let checkpointManager = CheckpointManager.shared
        
        do {
            let metadata = CheckpointMetadata(
                step: currentStep,
                loss: getAverageLoss(),
                acceptanceRate: await drafter.getAcceptanceRate(),
                timestamp: Date(),
                tier: tier
            )
            
            try await checkpointManager.saveCheckpoint(
                drafter: drafter,
                metadata: metadata
            )
        } catch {
            // Log error but don't fail training
            print("Warning: Failed to save checkpoint at step \(currentStep): \(error)")
        }
    }
}

// Checkpoint Management

/// Metadata stored with each checkpoint
public struct CheckpointMetadata: Codable, Sendable {
    public let step: Int
    public let loss: Float
    public let acceptanceRate: Float
    public let timestamp: Date
    public let tier: String
    
    public init(
        step: Int,
        loss: Float,
        acceptanceRate: Float,
        timestamp: Date,
        tier: HardwareTier
    ) {
        self.step = step
        self.loss = loss
        self.acceptanceRate = acceptanceRate
        self.timestamp = timestamp
        self.tier = tier.description
    }
}

/// Manages checkpoint saving and loading
public actor CheckpointManager {
    public static let shared = CheckpointManager()
    
    private let fileManager = FileManager.default
    private var checkpointDirectory: URL
    private let maxCheckpoints: Int
    
    public init(maxCheckpoints: Int = 5) {
        self.maxCheckpoints = maxCheckpoints
        
        // Default checkpoint directory
        let cacheDir = fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first!
        self.checkpointDirectory = cacheDir
            .appendingPathComponent("nano-reasoning")
            .appendingPathComponent("checkpoints")
    }
    
    /// Set custom checkpoint directory
    public func setCheckpointDirectory(_ url: URL) {
        checkpointDirectory = url
    }
    
    /// Get checkpoint directory
    public func getCheckpointDirectory() -> URL {
        checkpointDirectory
    }
    
    /// Save a checkpoint
    public func saveCheckpoint(
        drafter: DrafterActor,
        metadata: CheckpointMetadata
    ) async throws {
        // Ensure directory exists
        try fileManager.createDirectory(at: checkpointDirectory, withIntermediateDirectories: true)
        
        // Create checkpoint name with timestamp
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withFullDate, .withTime, .withColonSeparatorInTime]
        let timestamp = formatter.string(from: metadata.timestamp)
            .replacingOccurrences(of: ":", with: "-")
        
        let checkpointName = "checkpoint-\(metadata.step)-\(timestamp)"
        let checkpointDir = checkpointDirectory.appendingPathComponent(checkpointName)
        try fileManager.createDirectory(at: checkpointDir, withIntermediateDirectories: true)
        
        // Save weights
        let weightsURL = checkpointDir.appendingPathComponent("weights.safetensors")
        try await drafter.saveWeights(to: weightsURL)
        
        // Save LoRA weights separately if available
        let loraURL = checkpointDir.appendingPathComponent("lora.safetensors")
        try await drafter.saveLoRAWeights(to: loraURL)
        
        // Save metadata
        let metadataURL = checkpointDir.appendingPathComponent("metadata.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let metadataData = try encoder.encode(metadata)
        try metadataData.write(to: metadataURL)
        
        // Cleanup old checkpoints
        try cleanupOldCheckpoints()
    }
    
    /// Load the latest checkpoint
    public func loadLatestCheckpoint(into drafter: DrafterActor) async throws -> CheckpointMetadata? {
        guard let latest = try getLatestCheckpoint() else {
            return nil
        }
        
        return try await loadCheckpoint(from: latest, into: drafter)
    }
    
    /// Load a specific checkpoint
    public func loadCheckpoint(
        from checkpointDir: URL,
        into drafter: DrafterActor
    ) async throws -> CheckpointMetadata {
        // Load metadata
        let metadataURL = checkpointDir.appendingPathComponent("metadata.json")
        let metadataData = try Data(contentsOf: metadataURL)
        let metadata = try JSONDecoder().decode(CheckpointMetadata.self, from: metadataData)
        
        // Load weights
        let weightsURL = checkpointDir.appendingPathComponent("weights.safetensors")
        if fileManager.fileExists(atPath: weightsURL.path) {
            try await drafter.loadWeights(from: weightsURL)
        }
        
        // Load LoRA weights if available
        let loraURL = checkpointDir.appendingPathComponent("lora.safetensors")
        if fileManager.fileExists(atPath: loraURL.path) {
            try await drafter.loadLoRAWeights(from: loraURL)
        }
        
        return metadata
    }
    
    /// Get all available checkpoints sorted by step (newest first)
    public func listCheckpoints() throws -> [(url: URL, metadata: CheckpointMetadata)] {
        guard fileManager.fileExists(atPath: checkpointDirectory.path) else {
            return []
        }
        
        let contents = try fileManager.contentsOfDirectory(
            at: checkpointDirectory,
            includingPropertiesForKeys: nil
        )
        
        var checkpoints: [(url: URL, metadata: CheckpointMetadata)] = []
        
        for dir in contents where dir.hasDirectoryPath {
            let metadataURL = dir.appendingPathComponent("metadata.json")
            guard fileManager.fileExists(atPath: metadataURL.path) else { continue }
            
            do {
                let data = try Data(contentsOf: metadataURL)
                let metadata = try JSONDecoder().decode(CheckpointMetadata.self, from: data)
                checkpoints.append((dir, metadata))
            } catch {
                continue
            }
        }
        
        return checkpoints.sorted { $0.metadata.step > $1.metadata.step }
    }
    
    /// Get the latest checkpoint
    public func getLatestCheckpoint() throws -> URL? {
        try listCheckpoints().first?.url
    }
    
    /// Delete a checkpoint
    public func deleteCheckpoint(at url: URL) throws {
        try fileManager.removeItem(at: url)
    }
    
    /// Cleanup old checkpoints, keeping only the most recent ones
    private func cleanupOldCheckpoints() throws {
        let checkpoints = try listCheckpoints()
        
        if checkpoints.count > maxCheckpoints {
            let toDelete = checkpoints.dropFirst(maxCheckpoints)
            for checkpoint in toDelete {
                try? fileManager.removeItem(at: checkpoint.url)
            }
        }
    }
    
    /// Get total size of all checkpoints
    public func getTotalCheckpointSize() throws -> Int64 {
        guard fileManager.fileExists(atPath: checkpointDirectory.path) else {
            return 0
        }
        
        var totalSize: Int64 = 0
        
        if let enumerator = fileManager.enumerator(
            at: checkpointDirectory,
            includingPropertiesForKeys: [.fileSizeKey]
        ) {
            for case let fileURL as URL in enumerator {
                let attributes = try fileURL.resourceValues(forKeys: [.fileSizeKey])
                totalSize += Int64(attributes.fileSize ?? 0)
            }
        }
        
        return totalSize
    }
}

// Adaptive Learning Rate Scheduler

public struct AdaptiveLearningRateScheduler: Sendable {
    private let baseLR: Float
    private let warmupSteps: Int
    private let decayFactor: Float
    private let minLR: Float
    
    public init(
        baseLR: Float = 1e-4,
        warmupSteps: Int = 100,
        decayFactor: Float = 0.99,
        minLR: Float = 1e-6
    ) {
        self.baseLR = baseLR
        self.warmupSteps = warmupSteps
        self.decayFactor = decayFactor
        self.minLR = minLR
    }
    
    /// Get learning rate for current step
    public func getLearningRate(step: Int, recentLoss: Float? = nil) -> Float {
        var lr: Float
        
        // Warmup phase
        if step < warmupSteps {
            lr = baseLR * Float(step + 1) / Float(warmupSteps)
        } else {
            // Exponential decay
            let decaySteps = step - warmupSteps
            lr = baseLR * pow(decayFactor, Float(decaySteps / 100))
        }
        
        return max(lr, minLR)
    }
    
    /// Adaptive adjustment based on loss
    public func adjustForLoss(currentLR: Float, loss: Float, previousLoss: Float?) -> Float {
        guard let prevLoss = previousLoss else { return currentLR }
        
        // If loss increased significantly, reduce learning rate
        if loss > prevLoss * 1.5 {
            return max(currentLR * 0.5, minLR)
        }
        
        // If loss decreased significantly, can slightly increase
        if loss < prevLoss * 0.9 {
            return min(currentLR * 1.1, baseLR)
        }
        
        return currentLR
    }
}

// Training Coordinator

/// Coordinates training across multiple drafters (for future multi-drafter support)
public actor TrainingCoordinator {
    private var trainers: [String: TrainerTask] = [:]
    private let buffer: TrainingBuffer
    private let loadMonitor: GPULoadMonitor
    
    public init(buffer: TrainingBuffer, loadMonitor: GPULoadMonitor) {
        self.buffer = buffer
        self.loadMonitor = loadMonitor
    }
    
    /// Register a drafter for training
    public func registerDrafter(
        _ drafter: DrafterActor,
        name: String,
        config: TrainingConfig,
        tier: HardwareTier
    ) {
        let trainer = TrainerTask(
            drafter: drafter,
            buffer: buffer,
            loadMonitor: loadMonitor,
            config: config,
            tier: tier
        )
        trainers[name] = trainer
    }
    
    /// Start all trainers
    public func startAll() async {
        for trainer in trainers.values {
            await trainer.start()
        }
    }
    
    /// Stop all trainers
    public func stopAll() async {
        for trainer in trainers.values {
            await trainer.stop()
        }
    }
    
    /// Get trainer for a specific drafter
    public func getTrainer(name: String) -> TrainerTask? {
        trainers[name]
    }
    
    /// Get all trainer states
    public func getAllStates() async -> [String: TrainerState] {
        var states: [String: TrainerState] = [:]
        for (name, trainer) in trainers {
            states[name] = await trainer.getState()
        }
        return states
    }
}

// Gradient Utilities

/// Gradient clipping and normalization utilities
public struct GradientUtils {
    
    /// Clip gradients by global norm
    public static func clipByNorm(
        gradients: [String: MLXArray],
        maxNorm: Float
    ) -> [String: MLXArray] {
        // Calculate global norm
        var totalNormSquared: Float = 0
        for grad in gradients.values {
            let normSquared = MLX.sum(grad * grad).item(Float.self)
            totalNormSquared += normSquared
        }
        let globalNorm = sqrt(totalNormSquared)
        
        // Clip if necessary
        if globalNorm > maxNorm {
            let scale = maxNorm / globalNorm
            var clipped: [String: MLXArray] = [:]
            for (key, grad) in gradients {
                clipped[key] = grad * scale
            }
            return clipped
        }
        
        return gradients
    }
    
    /// Apply gradient accumulation
    public static func accumulateGradients(
        accumulated: inout [String: MLXArray]?,
        newGradients: [String: MLXArray],
        scale: Float = 1.0
    ) {
        if accumulated == nil {
            accumulated = newGradients.mapValues { $0 * scale }
        } else {
            for (key, grad) in newGradients {
                if let existing = accumulated?[key] {
                    accumulated?[key] = existing + grad * scale
                } else {
                    accumulated?[key] = grad * scale
                }
            }
        }
    }
}
