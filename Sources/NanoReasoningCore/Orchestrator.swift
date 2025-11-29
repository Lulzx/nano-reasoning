// SPDX-License-Identifier: MIT
// Nano-Reasoning: Orchestrator for the Adaptive Pipeline
// Coordinates drafter, target, and training components

import Foundation
@preconcurrency import MLX
import Tokenizers

/// Generation statistics
public struct GenerationStatistics: Sendable {
    public let totalTokens: Int
    public let draftedTokens: Int
    public let acceptedTokens: Int
    public let rejectedTokens: Int
    public let verificationCalls: Int
    public let averageAcceptanceRate: Float
    public let tokensPerSecond: Float
    public let elapsedTime: TimeInterval
    
    public init(
        totalTokens: Int,
        draftedTokens: Int,
        acceptedTokens: Int,
        rejectedTokens: Int,
        verificationCalls: Int,
        elapsedTime: TimeInterval
    ) {
        self.totalTokens = totalTokens
        self.draftedTokens = draftedTokens
        self.acceptedTokens = acceptedTokens
        self.rejectedTokens = rejectedTokens
        self.verificationCalls = verificationCalls
        self.averageAcceptanceRate = draftedTokens > 0 ? Float(acceptedTokens) / Float(draftedTokens) : 0
        self.elapsedTime = elapsedTime
        self.tokensPerSecond = elapsedTime > 0 ? Float(totalTokens) / Float(elapsedTime) : 0
    }
}

/// Orchestrator state
public enum OrchestratorState: Sendable {
    case uninitialized
    case loading
    case ready
    case generating
    case error(String)
}

/// Token streaming callback
public typealias TokenCallback = (String) async -> Bool

/// Main orchestrator coordinating the adaptive pipeline
public actor Orchestrator {
    // Components
    private var targetModel: TargetModel?
    private var drafterActor: DrafterActor?
    private var trainingBuffer: TrainingBuffer?
    private var trainerTask: TrainerTask?
    private var loadMonitor: GPULoadMonitor
    private var tokenizerManager: TokenizerManager?
    private var adaptiveEngine: AdaptiveRolloutEngine?
    
    // FastRL components (lightweight single-layer drafter)
    private var fastRLDrafter: FastRLDrafterActor?
    private var fastRLDecoder: FastRLSpeculativeDecoder?
    private var useFastRLMode: Bool = false
    
    // Configuration
    private let hardwareProfile: HardwareProfile
    private let modelConfig: ModelConfiguration
    private var generationConfig: GenerationConfig
    
    // State
    private var state: OrchestratorState = .uninitialized
    private var currentContext: MLXArray?
    
    // Statistics
    private var sessionStats: GenerationStatistics?
    private var totalTokensGenerated: Int = 0
    private var totalDraftedTokens: Int = 0
    private var totalAcceptedTokens: Int = 0
    private var totalVerifications: Int = 0
    
    /// Initialize orchestrator with hardware detection
    public init(
        hardwareProfile: HardwareProfile? = nil,
        generationConfig: GenerationConfig = .default
    ) {
        let detector = HardwareDetector()
        self.hardwareProfile = hardwareProfile ?? detector.detectHardware()
        self.modelConfig = ModelConfiguration.forTier(self.hardwareProfile.tier)
        self.generationConfig = generationConfig
        self.loadMonitor = GPULoadMonitor()
    }
    
    /// Get current state
    public func getState() -> OrchestratorState {
        state
    }
    
    /// Get hardware profile
    public func getHardwareProfile() -> HardwareProfile {
        hardwareProfile
    }
    
    /// Get model configuration
    public func getModelConfig() -> ModelConfiguration {
        modelConfig
    }
    

    // Initialization
    
    /// Initialize all components
    public func initialize(progressHandler: ((String, Double) -> Void)? = nil) async throws {
        guard case .uninitialized = state else {
            if case .ready = state { return }
            throw OrchestratorError.invalidState("Cannot initialize from current state")
        }
        
        state = .loading
        prepareCacheDirectories()
        
        do {
            // Initialize target model
            progressHandler?("Loading target model...", 0.0)
            targetModel = TargetModel(configuration: modelConfig, tier: hardwareProfile.tier)
            try await targetModel?.loadModel(progressHandler: nil)
            progressHandler?("Loading target model...", 0.6)
            
            // Initialize drafter
            progressHandler?("Initializing drafter...", 0.6)
            let drafterConfig = DrafterConfig.qwen3_0_6B
            let enableEAGLE = hardwareProfile.tier != .entry
            let enableLoRA = hardwareProfile.tier == .pro
            
            drafterActor = DrafterActor(
                config: drafterConfig,
                enableEAGLE: enableEAGLE,
                enableLoRA: enableLoRA,
                learningRate: 1e-4,
                loadMonitor: loadMonitor
            )
            
            // Initialize training components (if enabled)
            if hardwareProfile.tier.trainingEnabled {
                progressHandler?("Setting up training...", 0.8)
                trainingBuffer = TrainingBuffer(tier: hardwareProfile.tier)
                
                let trainingConfig = TrainingConfig.forTier(hardwareProfile.tier)
                trainerTask = TrainerTask(
                    drafter: drafterActor!,
                    buffer: trainingBuffer!,
                    loadMonitor: loadMonitor,
                    config: trainingConfig,
                    tier: hardwareProfile.tier
                )
            }
            
            // Adaptive rollout engine with MAB scheduling
            adaptiveEngine = AdaptiveRolloutEngine(
                drafter: drafterActor!,
                target: targetModel!,
                trainingBuffer: trainingBuffer,
                loadMonitor: loadMonitor,
                enableAdaptiveTraining: hardwareProfile.tier.trainingEnabled
            )
            
            // Initialize tokenizer
            progressHandler?("Loading tokenizer...", 0.9)
            tokenizerManager = TokenizerManager(config: .qwen3)
            try await tokenizerManager?.load()
            
            progressHandler?("Ready!", 1.0)
            state = .ready
            
        } catch {
            state = .error(error.localizedDescription)
            throw error
        }
    }
    
    /// Start background training (if available)
    public func startTraining() async {
        if useFastRLMode {
            await fastRLDecoder?.startBackgroundTraining()
        } else {
            await trainerTask?.start()
            await adaptiveEngine?.startAdaptiveTraining()
        }
    }
    
    /// Stop background training
    public func stopTraining() async {
        if useFastRLMode {
            await fastRLDecoder?.stopBackgroundTraining()
        } else {
            await trainerTask?.stop()
            await adaptiveEngine?.stopAdaptiveTraining()
        }
    }
    
    /// Initialize with FastRL-style lightweight drafter
    /// This uses a single-layer EAGLE head instead of a full drafter model
    public func initializeFastRL(progressHandler: ((String, Double) -> Void)? = nil) async throws {
        guard case .uninitialized = state else {
            if case .ready = state { return }
            throw OrchestratorError.invalidState("Cannot initialize from current state")
        }
        
        state = .loading
        useFastRLMode = true
        
        do {
            // Initialize target model
            progressHandler?("Loading target model...", 0.0)
            targetModel = TargetModel(configuration: modelConfig, tier: hardwareProfile.tier)
            try await targetModel?.loadModel(progressHandler: nil)
            progressHandler?("Loading target model...", 0.5)
            
            // Initialize FastRL lightweight drafter (single-layer EAGLE head)
            progressHandler?("Initializing FastRL drafter...", 0.6)
            let drafterConfig = SingleLayerDrafterConfig.forQwen3(size: getTargetModelSize())
            fastRLDrafter = FastRLDrafterActor(
                config: drafterConfig,
                learningRate: 1e-4,
                loadMonitor: loadMonitor
            )
            
            // Initialize FastRL speculative decoder
            progressHandler?("Setting up FastRL decoder...", 0.7)
            fastRLDecoder = FastRLSpeculativeDecoder(
                drafter: fastRLDrafter!,
                target: targetModel!,
                loadMonitor: loadMonitor
            )
            
            adaptiveEngine = nil
            
            // Initialize tokenizer
            progressHandler?("Loading tokenizer...", 0.9)
            tokenizerManager = TokenizerManager(config: .qwen3)
            try await tokenizerManager?.load()
            
            progressHandler?("Ready (FastRL mode)!", 1.0)
            state = .ready
            
        } catch {
            state = .error(error.localizedDescription)
            throw error
        }
    }
    
    /// Get target model size string for config selection
    private func getTargetModelSize() -> String {
        switch hardwareProfile.tier {
        case .entry:
            return "4B"
        case .pro:
            return "7B"
        case .elite:
            return "32B"
        }
    }
    
    /// Check if running in FastRL mode
    public func isFastRLMode() -> Bool {
        useFastRLMode
    }
    
    /// Load drafter weights from the latest checkpoint
    public func loadLatestCheckpoint() async throws -> CheckpointMetadata? {
        guard let drafter = drafterActor else {
            throw OrchestratorError.componentNotInitialized
        }
        
        let manager = CheckpointManager.shared
        return try await manager.loadLatestCheckpoint(into: drafter)
    }
    
    /// Save current drafter state as checkpoint
    public func saveCheckpoint() async throws {
        guard let drafter = drafterActor else {
            throw OrchestratorError.componentNotInitialized
        }
        
        let stats = await getTrainingStatistics()
        let metadata = CheckpointMetadata(
            step: stats?.step ?? 0,
            loss: stats?.loss ?? 0,
            acceptanceRate: stats?.acceptanceRate ?? 0,
            acceptanceLength: sessionStats?.averageAcceptanceRate ?? 0,
            timestamp: Date(),
            tier: hardwareProfile.tier
        )
        
        try await CheckpointManager.shared.saveCheckpoint(
            drafter: drafter,
            metadata: metadata
        )
    }
    
    /// List available checkpoints
    public func listCheckpoints() async throws -> [(url: URL, metadata: CheckpointMetadata)] {
        try await CheckpointManager.shared.listCheckpoints()
    }
    

    // Generation
    
    /// Generate text with speculative decoding
    public func generate(
        prompt: String,
        maxTokens: Int? = nil,
        tokenCallback: TokenCallback? = nil
    ) async throws -> String {
        guard case .ready = state else {
            throw OrchestratorError.invalidState("Orchestrator not ready")
        }
        
        state = .generating
        let startTime = Date()
        
        var config = generationConfig
        if let max = maxTokens {
            config.maxTokens = max
        }
        
        // Convert prompt text to token IDs and add batch dimension for engines
        let promptTokens = try await tokenize(prompt)
        let promptArray = MLXArray(promptTokens)
        
        var generatedTokens: [Int32] = []
        var draftedCount = 0
        var acceptedCount = 0
        var verificationCalls = 0
        var elapsed: TimeInterval = 0
        
        if useFastRLMode {
            guard let decoder = fastRLDecoder else {
                state = .ready
                throw OrchestratorError.componentNotInitialized
            }
            
            let (tokensResult, draftedResult, acceptedResult, elapsedResult) = try await decoder.generate(
                context: promptArray,
                maxTokens: config.maxTokens,
                temperature: config.temperature,
                topK: config.topK
            )
            generatedTokens = tokensResult
            draftedCount = draftedResult
            acceptedCount = acceptedResult
            elapsed = elapsedResult
        } else if let engine = adaptiveEngine {
            let (tokensResult, draftedResult, acceptedResult, elapsedResult) = try await engine.generate(
                context: promptArray,
                maxTokens: config.maxTokens,
                generationConfig: config
            )
            generatedTokens = tokensResult
            draftedCount = draftedResult
            acceptedCount = acceptedResult
            elapsed = elapsedResult
        } else if let target = targetModel {
            // Fallback to standard generation if engines are unavailable
            let promptFlat = MLXArray(promptTokens)
            let tokens = try await target.generate(prompt: promptFlat, config: config, tokenCallback: nil)
            generatedTokens = tokens
            draftedCount = tokens.count
            acceptedCount = tokens.count
        } else {
            state = .ready
            throw OrchestratorError.componentNotInitialized
        }
        
        // Stream tokens if requested
        if let callback = tokenCallback {
            for token in generatedTokens {
                let tokenText = try await detokenize([token])
                let shouldContinue = await callback(tokenText)
                if !shouldContinue { break }
            }
        }
        
        // Estimate verification calls based on drafted tokens
        if draftedCount > 0 {
            let k = max(1, hardwareProfile.tier.draftCount)
            verificationCalls = max(1, (draftedCount + k - 1) / k)
        }
        
        let measuredElapsed = elapsed > 0 ? elapsed : Date().timeIntervalSince(startTime)
        
        // Update statistics
        totalTokensGenerated += generatedTokens.count
        totalDraftedTokens += draftedCount
        totalAcceptedTokens += acceptedCount
        totalVerifications += verificationCalls
        
        sessionStats = GenerationStatistics(
            totalTokens: generatedTokens.count,
            draftedTokens: draftedCount,
            acceptedTokens: acceptedCount,
            rejectedTokens: draftedCount - acceptedCount,
            verificationCalls: verificationCalls,
            elapsedTime: measuredElapsed
        )
        
        state = .ready
        
        return try await detokenize(generatedTokens)
    }
    
    /// Generate without speculative decoding (fallback)
    public func generateStandard(
        prompt: String,
        maxTokens: Int? = nil,
        tokenCallback: TokenCallback? = nil
    ) async throws -> String {
        guard case .ready = state else {
            throw OrchestratorError.invalidState("Orchestrator not ready")
        }
        
        guard let target = targetModel else {
            throw OrchestratorError.componentNotInitialized
        }
        
        state = .generating
        
        var config = generationConfig
        if let max = maxTokens {
            config.maxTokens = max
        }
        
        let promptTokens = try await tokenize(prompt)
        let promptArray = MLXArray(promptTokens)
        
        let generated = try await target.generate(prompt: promptArray, config: config, tokenCallback: nil)
        
        // Handle token callback separately if needed
        if let callback = tokenCallback {
            for token in generated {
                let text = try await detokenize([token])
                let shouldContinue = await callback(text)
                if !shouldContinue { break }
            }
        }
        
        state = .ready
        return try await detokenize(generated)
    }
    

    // Training Data
    
    private func pushTrainingData(
        hiddenStates: MLXArray,
        targetLogits: MLXArray,
        inputIds: MLXArray,
        rejectedCount: Int
    ) async {
        await trainingBuffer?.pushRejectionData(
            hiddenStates: hiddenStates,
            targetLogits: targetLogits,
            inputIds: inputIds,
            rejectedCount: rejectedCount,
            rejectionPositions: [],
            contextLength: inputIds.shape[inputIds.ndim - 1]
        )
    }
    

    // Tokenization
    // Uses HuggingFace Tokenizers via swift-transformers
    
    private func tokenize(_ text: String) async throws -> [Int32] {
        guard let tokenizer = tokenizerManager else {
            throw OrchestratorError.tokenizationFailed("Tokenizer not initialized")
        }
        return try await tokenizer.encode(text)
    }
    
    /// Public helper to tokenize text for external consumers (RL rollout, etc.)
    public func tokenizeText(_ text: String) async throws -> [Int32] {
        try await tokenize(text)
    }
    
    private func detokenize(_ tokens: [Int32]) async throws -> String {
        guard let tokenizer = tokenizerManager else {
            throw OrchestratorError.tokenizationFailed("Tokenizer not initialized")
        }
        return try await tokenizer.decode(tokens)
    }
    
    /// Tokenize a chat conversation
    public func tokenizeChat(messages: [ChatMessage], addGenerationPrompt: Bool = true) async throws -> [Int32] {
        guard let tokenizer = tokenizerManager else {
            throw OrchestratorError.tokenizationFailed("Tokenizer not initialized")
        }
        return try await tokenizer.encodeChat(messages: messages, addGenerationPrompt: addGenerationPrompt)
    }
    
    /// Get the tokenizer manager for direct access
    public func getTokenizer() -> TokenizerManager? {
        tokenizerManager
    }
    

    // Statistics
    
    /// Get session statistics
    public func getSessionStatistics() -> GenerationStatistics? {
        sessionStats
    }
    
    /// Get cumulative statistics
    public func getCumulativeStatistics() -> (
        totalTokens: Int,
        totalDrafted: Int,
        totalAccepted: Int,
        totalVerifications: Int,
        overallAcceptanceRate: Float
    ) {
        let rate = totalDraftedTokens > 0 ? Float(totalAcceptedTokens) / Float(totalDraftedTokens) : 0
        return (totalTokensGenerated, totalDraftedTokens, totalAcceptedTokens, totalVerifications, rate)
    }
    
    /// Get training statistics
    public func getTrainingStatistics() async -> (step: Int, loss: Float, acceptanceRate: Float)? {
        guard let drafter = drafterActor else { return nil }
        let step = await drafter.getTrainingStep()
        let loss = await drafter.getLastLoss()
        let rate = await drafter.getAcceptanceRate()
        return (step, loss, rate)
    }
    
    /// Get buffer statistics
    public func getBufferStatistics() async -> BufferStatistics? {
        await trainingBuffer?.getStatistics()
    }
    
    /// Reset all statistics
    public func resetStatistics() async {
        totalTokensGenerated = 0
        totalDraftedTokens = 0
        totalAcceptedTokens = 0
        totalVerifications = 0
        sessionStats = nil
        await targetModel?.resetStatistics()
        await trainingBuffer?.resetStatistics()
        await loadMonitor.reset()
    }
    

    // Configuration
    
    /// Update generation configuration
    public func setGenerationConfig(_ config: GenerationConfig) {
        generationConfig = config
    }
    
    /// Get generation configuration
    public func getGenerationConfig() -> GenerationConfig {
        generationConfig
    }
    
    /// Internal accessor for target model (read-only)
    public func getTargetModelActor() -> TargetModel? {
        targetModel
    }
    
    /// Internal accessor for tokenizer manager
    public func getTokenizerManager() -> TokenizerManager? {
        tokenizerManager
    }
    
    /// Ensure cache/checkpoint directories exist
    private func prepareCacheDirectories() {
        let fm = FileManager.default
        let cacheDirs = [
            EnvironmentConfig.modelCacheDirectory,
            EnvironmentConfig.checkpointDirectory
        ]
        for dir in cacheDirs {
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
    }
    
    /// Run a lightweight evaluation over prompts and return throughput/acceptance stats
    public func evaluate(
        prompts: [String],
        maxTokens: Int
    ) async throws -> (tokensPerSecond: Float, acceptanceRate: Float) {
        var totalTokens: Int = 0
        var totalDrafted: Int = 0
        var totalAccepted: Int = 0
        let start = Date()
        
        for prompt in prompts {
            let _ = try await generate(prompt: prompt, maxTokens: maxTokens, tokenCallback: nil)
            if let stats = getSessionStatistics() {
                totalTokens += stats.totalTokens
                totalDrafted += stats.draftedTokens
                totalAccepted += stats.acceptedTokens
            }
            await resetStatistics()
        }
        
        let elapsed = Date().timeIntervalSince(start)
        let tps = elapsed > 0 ? Float(totalTokens) / Float(elapsed) : 0
        let acceptance = totalDrafted > 0 ? Float(totalAccepted) / Float(totalDrafted) : 0
        return (tps, acceptance)
    }
}

// Errors

public enum OrchestratorError: Error, LocalizedError {
    case invalidState(String)
    case componentNotInitialized
    case generationFailed(String)
    case tokenizationFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .invalidState(let msg):
            return "Invalid orchestrator state: \(msg)"
        case .componentNotInitialized:
            return "Required component not initialized"
        case .generationFailed(let msg):
            return "Generation failed: \(msg)"
        case .tokenizationFailed(let msg):
            return "Tokenization failed: \(msg)"
        }
    }
}

// Pipeline Builder

/// Builder pattern for constructing the orchestrator
public struct OrchestratorBuilder {
    private var hardwareProfile: HardwareProfile?
    private var generationConfig: GenerationConfig = .default
    private var customTargetModel: String?
    private var customDrafterModel: String?
    
    public init() {}
    
    public func withHardwareProfile(_ profile: HardwareProfile) -> OrchestratorBuilder {
        var builder = self
        builder.hardwareProfile = profile
        return builder
    }
    
    public func withGenerationConfig(_ config: GenerationConfig) -> OrchestratorBuilder {
        var builder = self
        builder.generationConfig = config
        return builder
    }
    
    public func withTargetModel(_ modelId: String) -> OrchestratorBuilder {
        var builder = self
        builder.customTargetModel = modelId
        return builder
    }
    
    public func withDrafterModel(_ modelId: String) -> OrchestratorBuilder {
        var builder = self
        builder.customDrafterModel = modelId
        return builder
    }
    
    public func build() -> Orchestrator {
        Orchestrator(
            hardwareProfile: hardwareProfile,
            generationConfig: generationConfig
        )
    }
}
