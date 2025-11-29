// SPDX-License-Identifier: MIT
// Nano-Reasoning: Orchestrator for the Adaptive Pipeline
// Coordinates drafter, target, and training components

import Foundation
@preconcurrency import MLX

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
            
            progressHandler?("Ready!", 1.0)
            state = .ready
            
        } catch {
            state = .error(error.localizedDescription)
            throw error
        }
    }
    
    /// Start background training (if available)
    public func startTraining() async {
        await trainerTask?.start()
    }
    
    /// Stop background training
    public func stopTraining() async {
        await trainerTask?.stop()
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
        
        guard let target = targetModel, let drafter = drafterActor else {
            throw OrchestratorError.componentNotInitialized
        }
        
        state = .generating
        let startTime = Date()
        
        var config = generationConfig
        if let max = maxTokens {
            config.maxTokens = max
        }
        
        // Convert prompt text to token IDs
        let promptTokens = try await tokenize(prompt)
        var contextTokens = promptTokens
        var generatedTokens: [Int32] = []
        
        let draftCount = hardwareProfile.tier.draftCount
        var localDrafted = 0
        var localAccepted = 0
        var localVerifications = 0
        
        while generatedTokens.count < config.maxTokens {
            // Generate draft tokens - create separate arrays to avoid data race
            let contextForDraft = MLXArray(contextTokens)
            let (_, hiddenStates) = try await target.getLogitsAndHiddenStates(inputIds: contextForDraft)
            
            let contextForGenerate = MLXArray(contextTokens)
            let draftTokens = await drafter.generateDraft(
                hiddenStates: hiddenStates,
                previousTokens: contextForGenerate,
                count: draftCount,
                temperature: config.temperature
            )
            localDrafted += draftTokens.count
            
            // Verify draft tokens - create another context array
            let contextForVerify = MLXArray(contextTokens)
            let verificationResult = try await target.verifyDraftTokens(
                contextIds: contextForVerify,
                draftTokens: draftTokens,
                config: config
            )
            localVerifications += 1
            
            // Accept verified tokens
            let acceptedCount = verificationResult.acceptedCount
            localAccepted += acceptedCount
            
            for i in 0..<acceptedCount {
                let token = draftTokens[i]
                contextTokens.append(token)
                generatedTokens.append(token)
                
                // Stream token if callback provided
                if let callback = tokenCallback {
                    let tokenText = try await detokenize([token])
                    let shouldContinue = await callback(tokenText)
                    if !shouldContinue {
                        break
                    }
                }
                
                // Check for stop tokens
                if config.stopTokens.contains(token) {
                    break
                }
            }
            
            // Add corrected token if any draft was rejected
            if let corrected = verificationResult.correctedToken {
                contextTokens.append(corrected)
                generatedTokens.append(corrected)
                
                if let callback = tokenCallback {
                    let tokenText = try await detokenize([corrected])
                    _ = await callback(tokenText)
                }
            }
            
            // Push rejection data to training buffer
            if hardwareProfile.tier.trainingEnabled && acceptedCount < draftTokens.count {
                await pushTrainingData(
                    hiddenStates: verificationResult.hiddenStates,
                    targetLogits: verificationResult.logits,
                    inputIds: contextForVerify,
                    rejectedCount: draftTokens.count - acceptedCount
                )
            }
            
            // Record acceptance statistics
            await drafter.recordAcceptance(accepted: acceptedCount, total: draftTokens.count)
            
            // Check if we hit a stop condition
            if let lastToken = generatedTokens.last, config.stopTokens.contains(lastToken) {
                break
            }
        }
        
        // Update statistics
        let elapsed = Date().timeIntervalSince(startTime)
        totalTokensGenerated += generatedTokens.count
        totalDraftedTokens += localDrafted
        totalAcceptedTokens += localAccepted
        totalVerifications += localVerifications
        
        sessionStats = GenerationStatistics(
            totalTokens: generatedTokens.count,
            draftedTokens: localDrafted,
            acceptedTokens: localAccepted,
            rejectedTokens: localDrafted - localAccepted,
            verificationCalls: localVerifications,
            elapsedTime: elapsed
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
            rejectedCount: rejectedCount
        )
    }
    

    // Tokenization
    // For production, integrate with HuggingFace Tokenizers via swift-transformers
    
    private func tokenize(_ text: String) async throws -> [Int32] {
        // UTF-8 byte encoding as fallback tokenization
        // Production: Use AutoTokenizer.from(pretrained: modelConfig.targetModelId)
        return text.utf8.map { Int32($0) }
    }
    
    private func detokenize(_ tokens: [Int32]) async throws -> String {
        // UTF-8 byte decoding as fallback detokenization
        // Production: Use tokenizer.decode(tokens)
        let bytes = tokens.compactMap { UInt8(exactly: $0) }
        return String(bytes: bytes, encoding: .utf8) ?? ""
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
