// SPDX-License-Identifier: MIT
// Nano-Reasoning: Speculative Decoding Implementation
// Draft -> Verify -> Accept loop with EAGLE support

import Foundation
@preconcurrency import MLX
import MLXRandom

/// Speculative decoding configuration
public struct SpeculativeConfig: Sendable {
    /// Number of draft tokens to generate per iteration (k)
    public let draftCount: Int
    /// Temperature for draft generation
    public let draftTemperature: Float
    /// Temperature for target verification
    public let verifyTemperature: Float
    /// Whether to use tree-based speculation (future)
    public let useTreeSpeculation: Bool
    /// Maximum tree width for tree speculation
    public let maxTreeWidth: Int
    /// Acceptance threshold for soft matching
    public let acceptanceThreshold: Float
    /// EAGLE-style top-k for draft candidates
    public let eagleTopK: Int
    /// Number of tokens to verify in parallel
    public let verifyTokens: Int
    
    public init(
        draftCount: Int,
        draftTemperature: Float,
        verifyTemperature: Float,
        useTreeSpeculation: Bool,
        maxTreeWidth: Int,
        acceptanceThreshold: Float,
        eagleTopK: Int = 8,
        verifyTokens: Int = 32
    ) {
        self.draftCount = draftCount
        self.draftTemperature = draftTemperature
        self.verifyTemperature = verifyTemperature
        self.useTreeSpeculation = useTreeSpeculation
        self.maxTreeWidth = maxTreeWidth
        self.acceptanceThreshold = acceptanceThreshold
        self.eagleTopK = eagleTopK
        self.verifyTokens = verifyTokens
    }
    
    public static let `default` = SpeculativeConfig(
        draftCount: 5,
        draftTemperature: 0.7,
        verifyTemperature: 0.7,
        useTreeSpeculation: false,
        maxTreeWidth: 3,
        acceptanceThreshold: 0.9,
        eagleTopK: 8,
        verifyTokens: 32
    )
    
    public static func forTier(_ tier: HardwareTier) -> SpeculativeConfig {
        SpeculativeConfig(
            draftCount: tier.draftCount,
            draftTemperature: 0.7,
            verifyTemperature: 0.7,
            useTreeSpeculation: tier == .elite,
            maxTreeWidth: tier == .elite ? 4 : 2,
            acceptanceThreshold: 0.9,
            eagleTopK: tier == .elite ? 8 : 4,
            verifyTokens: tier == .elite ? 32 : 16
        )
    }
}

// MARK: - FastRL Adaptive Rollout Engine

/// Multi-Armed Bandit (MAB) configuration for SD strategy selection
/// Based on FastRL's adaptive speculative decoding approach
public struct MABConfig: Sendable, Hashable {
    public let topK: Int
    public let steps: Int
    public let verifyTokens: Int
    
    public var identifier: String {
        "\(topK)_\(steps)_\(verifyTokens)"
    }
    
    public init(topK: Int, steps: Int, verifyTokens: Int) {
        self.topK = topK
        self.steps = steps
        self.verifyTokens = verifyTokens
    }
    
    /// Parse from string format "topK_steps_verifyTokens"
    public static func parse(_ str: String) -> MABConfig? {
        let parts = str.split(separator: "_")
        guard parts.count == 3,
              let topK = Int(parts[0]),
              let steps = Int(parts[1]),
              let verifyTokens = Int(parts[2]) else {
            return nil
        }
        return MABConfig(topK: topK, steps: steps, verifyTokens: verifyTokens)
    }
    
    /// Default configurations from FastRL paper
    public static let defaultConfigs: [MABConfig] = [
        MABConfig(topK: 8, steps: 4, verifyTokens: 32),
        MABConfig(topK: 8, steps: 4, verifyTokens: 16),
        MABConfig(topK: 8, steps: 4, verifyTokens: 8),
        MABConfig(topK: 4, steps: 3, verifyTokens: 16),
        MABConfig(topK: 4, steps: 2, verifyTokens: 8),
    ]
}

/// Multi-Armed Bandit algorithm for selecting optimal SD configuration
/// Uses Upper Confidence Bound (UCB1) for exploration-exploitation balance
public actor MultiArmedBandit {
    /// Arm statistics for UCB calculation
    private struct ArmStats {
        var totalReward: Double = 0
        var pullCount: Int = 0
        
        var meanReward: Double {
            pullCount > 0 ? totalReward / Double(pullCount) : 0
        }
    }
    
    private var armStats: [MABConfig: ArmStats] = [:]
    private var totalPulls: Int = 0
    private let explorationConstant: Double
    private let configs: [MABConfig]
    
    public init(
        configs: [MABConfig] = MABConfig.defaultConfigs,
        explorationConstant: Double = 2.0
    ) {
        self.configs = configs
        self.explorationConstant = explorationConstant
        
        // Initialize all arms
        for config in configs {
            armStats[config] = ArmStats()
        }
    }
    
    /// Select the best arm using UCB1 algorithm
    public func selectArm() -> MABConfig {
        totalPulls += 1
        
        // First, ensure each arm is pulled at least once
        for config in configs {
            if let stats = armStats[config], stats.pullCount == 0 {
                return config
            }
        }
        
        // Calculate UCB for each arm and select the best
        var bestConfig = configs[0]
        var bestUCB = -Double.infinity
        
        for config in configs {
            guard let stats = armStats[config] else { continue }
            
            let exploitation = stats.meanReward
            let exploration = sqrt(explorationConstant * log(Double(totalPulls)) / Double(stats.pullCount))
            let ucb = exploitation + exploration
            
            if ucb > bestUCB {
                bestUCB = ucb
                bestConfig = config
            }
        }
        
        return bestConfig
    }
    
    /// Update the reward for an arm after pulling
    /// Reward is typically the acceptance rate or tokens/second
    public func updateReward(config: MABConfig, reward: Double) {
        if var stats = armStats[config] {
            stats.totalReward += reward
            stats.pullCount += 1
            armStats[config] = stats
        }
    }
    
    /// Get statistics for all arms
    public func getStatistics() -> [(config: MABConfig, mean: Double, pulls: Int)] {
        configs.compactMap { config in
            guard let stats = armStats[config] else { return nil }
            return (config, stats.meanReward, stats.pullCount)
        }
    }
    
    /// Reset all statistics
    public func reset() {
        totalPulls = 0
        for config in configs {
            armStats[config] = ArmStats()
        }
    }
}

/// Adaptive Rollout Engine following FastRL's TLT approach
/// Dynamically enables/disables speculative decoding based on batch characteristics
public actor AdaptiveRolloutEngine {
    private let drafter: DrafterActor
    private let target: TargetModel
    private let trainingBuffer: TrainingBuffer?
    private let mab: MultiArmedBandit
    
    // Hidden state collection for training
    private let hiddenStateCollector: HiddenStateCollector
    private var adaptiveTrainer: AdaptiveDrafterTrainer?
    private let loadMonitor: GPULoadMonitor?
    
    // Batch size thresholds for enabling SD (from FastRL)
    // SD is disabled for small batches where overhead isn't worth it
    private let batchSizeThresholds: [Int]
    
    // Long-tail detection
    private var recentGenerationLengths: [Int] = []
    private let longTailThreshold: Int
    private let longTailWindowSize: Int
    
    // Performance tracking
    private var sdEnabledTokensPerSecond: Float = 0
    private var sdDisabledTokensPerSecond: Float = 0
    private var totalGenerations: Int = 0
    
    // Current SD configuration
    private var currentConfig: MABConfig
    private var sdEnabled: Bool = true
    
    // Training mode
    private var adaptiveTrainingEnabled: Bool = false
    
    public init(
        drafter: DrafterActor,
        target: TargetModel,
        trainingBuffer: TrainingBuffer? = nil,
        loadMonitor: GPULoadMonitor? = nil,
        batchSizeThresholds: [Int] = [1, 2, 5, 21],
        longTailThreshold: Int = 512,
        longTailWindowSize: Int = 10,
        enableAdaptiveTraining: Bool = false
    ) {
        self.drafter = drafter
        self.target = target
        self.trainingBuffer = trainingBuffer
        self.loadMonitor = loadMonitor
        self.batchSizeThresholds = batchSizeThresholds
        self.longTailThreshold = longTailThreshold
        self.longTailWindowSize = longTailWindowSize
        self.mab = MultiArmedBandit()
        self.currentConfig = MABConfig.defaultConfigs[0]
        self.hiddenStateCollector = HiddenStateCollector(maxSamples: 500, prioritizeRejections: true)
        self.adaptiveTrainingEnabled = enableAdaptiveTraining
        
        // Initialize adaptive trainer if enabled
        if enableAdaptiveTraining, let monitor = loadMonitor {
            self.adaptiveTrainer = AdaptiveDrafterTrainer(
                drafter: drafter,
                collector: hiddenStateCollector,
                loadMonitor: monitor,
                batchSize: 4,
                trainingInterval: .milliseconds(100)
            )
        }
    }
    
    /// Start adaptive background training
    public func startAdaptiveTraining() async {
        await adaptiveTrainer?.start()
    }
    
    /// Stop adaptive background training
    public func stopAdaptiveTraining() async {
        await adaptiveTrainer?.stop()
    }
    
    /// Get hidden state collector for external access
    public func getHiddenStateCollector() -> HiddenStateCollector {
        hiddenStateCollector
    }
    
    /// Determine whether to enable SD for a given batch
    public func shouldEnableSD(batchSize: Int, contextLength: Int) -> Bool {
        // Disable SD for very small batches (overhead not worth it)
        if batchSize < batchSizeThresholds[0] {
            return false
        }
        
        // Enable SD for batches likely to have long-tail responses
        if detectLongTail() {
            return true
        }
        
        // Use batch size threshold heuristic from FastRL
        // Larger batches benefit more from SD
        let thresholdIndex = batchSizeThresholds.lastIndex { batchSize >= $0 } ?? 0
        return thresholdIndex >= 1  // Enable for batches above first threshold
    }
    
    /// Detect if we're in a long-tail generation scenario
    private func detectLongTail() -> Bool {
        guard recentGenerationLengths.count >= 3 else { return false }
        
        let recentAvg = Float(recentGenerationLengths.suffix(3).reduce(0, +)) / 3.0
        return recentAvg > Float(longTailThreshold)
    }
    
    /// Record generation length for long-tail detection
    public func recordGenerationLength(_ length: Int) {
        recentGenerationLengths.append(length)
        if recentGenerationLengths.count > longTailWindowSize {
            recentGenerationLengths.removeFirst()
        }
    }
    
    /// Select SD configuration for current batch using MAB
    public func selectConfiguration() async -> (config: MABConfig, enabled: Bool) {
        let config = await mab.selectArm()
        currentConfig = config
        return (config, sdEnabled)
    }
    
    /// Update MAB with reward based on generation performance
    public func updatePerformance(
        config: MABConfig,
        tokensGenerated: Int,
        timeElapsed: TimeInterval,
        acceptedTokens: Int,
        draftedTokens: Int
    ) async {
        totalGenerations += 1
        
        // Calculate reward as tokens per second * acceptance rate
        let tokensPerSecond = timeElapsed > 0 ? Float(tokensGenerated) / Float(timeElapsed) : 0
        let acceptanceRate = draftedTokens > 0 ? Float(acceptedTokens) / Float(draftedTokens) : 0
        
        // Reward combines throughput and accuracy
        let reward = Double(tokensPerSecond * (0.5 + 0.5 * acceptanceRate))
        
        await mab.updateReward(config: config, reward: reward)
        
        // Update running averages
        if sdEnabled {
            sdEnabledTokensPerSecond = 0.9 * sdEnabledTokensPerSecond + 0.1 * tokensPerSecond
        } else {
            sdDisabledTokensPerSecond = 0.9 * sdDisabledTokensPerSecond + 0.1 * tokensPerSecond
        }
    }
    
    /// Generate tokens with adaptive SD
    public func generate(
        context: MLXArray,
        maxTokens: Int,
        generationConfig: GenerationConfig
    ) async throws -> (
        tokens: [Int32],
        drafted: Int,
        accepted: Int,
        elapsed: TimeInterval
    ) {
        let batchSize = context.ndim > 1 ? context.shape[0] : 1
        let contextLength = context.shape[context.ndim - 1]
        
        // Decide whether to use SD
        sdEnabled = shouldEnableSD(batchSize: batchSize, contextLength: contextLength)
        
        if sdEnabled {
            // Select configuration using MAB
            let (config, _) = await selectConfiguration()
            
            // Create speculative config from MAB selection
            let specConfig = SpeculativeConfig(
                draftCount: config.steps,
                draftTemperature: generationConfig.temperature,
                verifyTemperature: generationConfig.temperature,
                useTreeSpeculation: false,
                maxTreeWidth: 2,
                acceptanceThreshold: 0.9,
                eagleTopK: config.topK,
                verifyTokens: config.verifyTokens
            )
            
            // Use speculative decoding
            let decoder = SpeculativeDecoder(
                drafter: drafter,
                target: target,
                config: specConfig,
                trainingBuffer: trainingBuffer
            )
            
            let startTime = Date()
            var allTokens: [Int32] = []
            var lastHiddenStates: MLXArray?
            var totalDrafted = 0
            var totalAccepted = 0
            var currentContext = context
            var inputTokens = Array(context.asArray(Int32.self))
            
            while allTokens.count < maxTokens {
                let result = try await decoder.speculate(
                    context: currentContext,
                    generationConfig: generationConfig
                )
                
                allTokens.append(contentsOf: result.acceptedTokens)
                if let bonus = result.bonusToken {
                    allTokens.append(bonus)
                }
                
                lastHiddenStates = result.hiddenStates
                totalDrafted += result.draftCount
                totalAccepted += result.acceptedCount
                
                // Collect hidden states for adaptive training
                if adaptiveTrainingEnabled, let hidden = result.hiddenStates, let logits = result.targetLogits {
                    let outputTokens = result.acceptedTokens + (result.bonusToken.map { [$0] } ?? [])
                    let wasFullyAccepted = result.acceptedCount == result.draftCount
                    
                    await hiddenStateCollector.collect(
                        hiddenStates: hidden,
                        targetLogits: logits,
                        inputTokens: inputTokens,
                        outputTokens: outputTokens,
                        wasAccepted: wasFullyAccepted
                    )
                }
                
                // Update context for next iteration
                let newTokens = MLXArray(result.acceptedTokens + (result.bonusToken.map { [$0] } ?? []))
                currentContext = MLX.concatenated([currentContext, newTokens], axis: -1)
                inputTokens.append(contentsOf: result.acceptedTokens)
                if let bonus = result.bonusToken {
                    inputTokens.append(bonus)
                }
                
                // Check for stop tokens
                if let lastToken = allTokens.last,
                   generationConfig.stopTokens.contains(lastToken) {
                    break
                }
            }
            
            let elapsed = Date().timeIntervalSince(startTime)
            await updatePerformance(
                config: config,
                tokensGenerated: allTokens.count,
                timeElapsed: elapsed,
                acceptedTokens: totalAccepted,
                draftedTokens: totalDrafted
            )
            
            recordGenerationLength(allTokens.count)
            
            return (
                Array(allTokens.prefix(maxTokens)),
                totalDrafted,
                totalAccepted,
                elapsed
            )
        } else {
            // Standard autoregressive generation with hidden state collection
            let startTime = Date()
            
            // Get hidden states during standard generation for training
            var tokens: [Int32] = []
            var currentContext = context
            var inputTokens = Array(context.asArray(Int32.self))
            
            for _ in 0..<maxTokens {
                let (logits, hiddenStates) = try await target.getLogitsAndHiddenStates(inputIds: currentContext)
                let lastLogits = logits[0, logits.shape[1] - 1]
                
                // Sample next token using TokenSampler
                let nextToken: Int32
                if generationConfig.temperature <= 0 {
                    nextToken = TokenSampler.greedy(lastLogits)
                } else if let topK = generationConfig.topK as Int?, topK > 0 {
                    nextToken = TokenSampler.sample(
                        lastLogits,
                        temperature: generationConfig.temperature,
                        topK: topK,
                        topP: generationConfig.topP
                    )
                } else {
                    nextToken = TokenSampler.sample(lastLogits, temperature: generationConfig.temperature)
                }
                
                // Collect hidden states for training (even in non-SD mode)
                if adaptiveTrainingEnabled {
                    await hiddenStateCollector.collect(
                        hiddenStates: hiddenStates,
                        targetLogits: logits,
                        inputTokens: inputTokens,
                        outputTokens: [nextToken],
                        wasAccepted: true  // Standard generation is always "accepted"
                    )
                }
                
                tokens.append(nextToken)
                inputTokens.append(nextToken)
                
                // Check for stop tokens
                if generationConfig.stopTokens.contains(nextToken) {
                    break
                }
                
                // Update context
                let nextTokenArray = MLXArray([nextToken])
                currentContext = MLX.concatenated([currentContext, nextTokenArray], axis: -1)
            }
            
            let elapsed = Date().timeIntervalSince(startTime)
            await updatePerformance(
                config: currentConfig,
                tokensGenerated: tokens.count,
                timeElapsed: elapsed,
                acceptedTokens: tokens.count,
                draftedTokens: tokens.count
            )
            
            recordGenerationLength(tokens.count)
            
        return (tokens, tokens.count, tokens.count, elapsed)
    }
    }
    
    /// Get current performance statistics
    public func getStatistics() async -> (
        sdEnabled: Float,
        sdDisabled: Float,
        totalGenerations: Int,
        mabStats: [(config: MABConfig, mean: Double, pulls: Int)]
    ) {
        let mabStats = await mab.getStatistics()
        return (sdEnabledTokensPerSecond, sdDisabledTokensPerSecond, totalGenerations, mabStats)
    }
    
    /// Get adaptive training statistics
    public func getAdaptiveTrainingStats() async -> (
        collectorStats: (total: Int, rejections: Int, acceptances: Int, currentSize: Int),
        trainerStats: (steps: Int, avgLoss: Float, isRunning: Bool)?
    ) {
        let collectorStats = await hiddenStateCollector.getStatistics()
        let trainerStats = await adaptiveTrainer?.getStatistics()
        return (collectorStats, trainerStats)
    }
    
    /// Check if adaptive training is enabled
    public func isAdaptiveTrainingEnabled() -> Bool {
        adaptiveTrainingEnabled
    }
    
    /// Enable/disable adaptive training mode
    public func setAdaptiveTrainingEnabled(_ enabled: Bool) async {
        adaptiveTrainingEnabled = enabled
        if !enabled {
            await adaptiveTrainer?.stop()
        }
    }
    
    /// Toggle SD enabled state manually
    public func setSDEnabled(_ enabled: Bool) {
        sdEnabled = enabled
    }
}

/// Result of a speculation round
public struct SpeculationResult: @unchecked Sendable {
    /// Tokens that were accepted from the draft
    public let acceptedTokens: [Int32]
    /// The bonus token from target model (if any draft rejected)
    public let bonusToken: Int32?
    /// Hidden states for training
    public let hiddenStates: MLXArray?
    /// Target logits for training
    public let targetLogits: MLXArray?
    /// Number of draft tokens generated
    public let draftCount: Int
    /// Number of tokens accepted
    public let acceptedCount: Int
    
    public var speedup: Float {
        guard draftCount > 0 else { return 1.0 }
        let totalAccepted = acceptedCount + (bonusToken != nil ? 1 : 0)
        // Speedup = tokens generated / verification calls
        // In speculative decoding, we verify k+1 tokens in one forward pass
        return Float(totalAccepted)
    }
}

/// Speculative decoding engine
public actor SpeculativeDecoder {
    private let drafter: DrafterActor
    private let target: TargetModel
    private let config: SpeculativeConfig
    private let trainingBuffer: TrainingBuffer?
    
    // Statistics
    private var totalRounds: Int = 0
    private var totalDrafted: Int = 0
    private var totalAccepted: Int = 0
    private var totalBonusTokens: Int = 0
    
    public init(
        drafter: DrafterActor,
        target: TargetModel,
        config: SpeculativeConfig,
        trainingBuffer: TrainingBuffer? = nil
    ) {
        self.drafter = drafter
        self.target = target
        self.config = config
        self.trainingBuffer = trainingBuffer
    }
    
    /// Perform one round of speculative decoding
    public func speculate(
        context: MLXArray,
        generationConfig: GenerationConfig
    ) async throws -> SpeculationResult {
        totalRounds += 1
        
        // Step 1: Get hidden states from target for context
        let (_, hiddenStates) = try await target.getLogitsAndHiddenStates(inputIds: context)
        
        // Step 2: Generate draft tokens
        let draft = await drafter.generateDraftWithLogProbs(
            hiddenStates: hiddenStates,
            previousTokens: context,
            count: config.draftCount,
            temperature: config.draftTemperature
        )
        let draftTokens = draft.tokens
        totalDrafted += draftTokens.count
        
        // Step 3: Verify draft tokens with target model
        let verification = try await target.verifyDraftTokens(
            contextIds: context,
            draftTokens: draftTokens,
            config: generationConfig,
            draftLogProbs: draft.logProbs
        )
        
        // Step 4: Determine accepted tokens
        var acceptedTokens: [Int32] = []
        for (i, accepted) in verification.acceptanceMask.enumerated() {
            if accepted {
                acceptedTokens.append(draftTokens[i])
            } else {
                break  // Stop at first rejection
            }
        }
        totalAccepted += acceptedTokens.count
        
        // Step 5: Get bonus token if needed
        let bonusToken = verification.correctedToken
        if bonusToken != nil {
            totalBonusTokens += 1
        }
        
        // Step 6: Push training data if we have rejections
        if let buffer = trainingBuffer, acceptedTokens.count < draftTokens.count {
            let rejectedCount = draftTokens.count - acceptedTokens.count
            let rejectionPositions = verification.acceptanceMask.enumerated().compactMap { idx, accepted in
                accepted ? nil : idx
            }
            await buffer.pushRejectionData(
                hiddenStates: verification.hiddenStates,
                targetLogits: verification.logits,
                inputIds: context,
                rejectedCount: rejectedCount,
                rejectionPositions: rejectionPositions,
                contextLength: context.shape[context.ndim - 1]
            )
        }
        
        // Record acceptance for drafter
        await drafter.recordAcceptance(accepted: acceptedTokens.count, total: draftTokens.count)
        
        return SpeculationResult(
            acceptedTokens: acceptedTokens,
            bonusToken: bonusToken,
            hiddenStates: verification.hiddenStates,
            targetLogits: verification.logits,
            draftCount: draftTokens.count,
            acceptedCount: acceptedTokens.count
        )
    }
    
    /// Get speculation statistics
    public func getStatistics() -> (
        rounds: Int,
        drafted: Int,
        accepted: Int,
        bonusTokens: Int,
        acceptanceRate: Float,
        averageSpeedup: Float
    ) {
        let rate = totalDrafted > 0 ? Float(totalAccepted) / Float(totalDrafted) : 0
        let avgTokensPerRound = totalRounds > 0 ? Float(totalAccepted + totalBonusTokens) / Float(totalRounds) : 0
        return (totalRounds, totalDrafted, totalAccepted, totalBonusTokens, rate, avgTokensPerRound)
    }
    
    /// Reset statistics
    public func resetStatistics() {
        totalRounds = 0
        totalDrafted = 0
        totalAccepted = 0
        totalBonusTokens = 0
    }
}

// Verification Strategies

/// Protocol for different verification strategies
public protocol VerificationStrategy: Sendable {
    /// Verify draft tokens against target logits
    func verify(
        draftTokens: [Int32],
        targetLogits: MLXArray,
        temperature: Float
    ) -> [Bool]
}

/// Standard greedy verification
public struct GreedyVerification: VerificationStrategy {
    public init() {}
    
    public func verify(
        draftTokens: [Int32],
        targetLogits: MLXArray,
        temperature: Float
    ) -> [Bool] {
        var results: [Bool] = []
        
        for (i, draft) in draftTokens.enumerated() {
            let logits = targetLogits[0, i]
            let predicted = Int32(MLX.argMax(logits).item(Int32.self))
            results.append(draft == predicted)
            
            if draft != predicted {
                break  // Stop at first mismatch
            }
        }
        
        return results
    }
}

/// Probabilistic verification (nucleus sampling based)
public struct ProbabilisticVerification: VerificationStrategy {
    let threshold: Float
    
    public init(threshold: Float = 0.1) {
        self.threshold = threshold
    }
    
    public func verify(
        draftTokens: [Int32],
        targetLogits: MLXArray,
        temperature: Float
    ) -> [Bool] {
        var results: [Bool] = []
        
        for (i, draft) in draftTokens.enumerated() {
            let logits = targetLogits[0, i]
            let probs = MLX.softmax(logits / temperature)
            
            // Check if draft token has sufficient probability
            let draftProb = probs[Int(draft)].item(Float.self)
            let isAccepted = draftProb >= threshold
            
            results.append(isAccepted)
            
            if !isAccepted {
                break
            }
        }
        
        return results
    }
}

/// Speculative sampling verification (rejection sampling based)
public struct SpeculativeSamplingVerification: VerificationStrategy {
    public init() {}
    
    public func verify(
        draftTokens: [Int32],
        targetLogits: MLXArray,
        temperature: Float
    ) -> [Bool] {
        var results: [Bool] = []
        
        // This implements the theoretical speculative sampling algorithm
        // where we accept with probability min(1, p_target/p_draft)
        
        for (i, draft) in draftTokens.enumerated() {
            let logits = targetLogits[0, i]
            let targetProbs = MLX.softmax(logits / temperature)
            let targetProb = targetProbs[Int(draft)].item(Float.self)
            
            // For simplicity, using uniform acceptance probability
            // Full implementation would track draft probabilities
            let r = Float.random(in: 0...1)
            let acceptanceProb = min(1.0, targetProb * 10)  // Scaled for demo
            
            results.append(r < acceptanceProb)
            
            if r >= acceptanceProb {
                break
            }
        }
        
        return results
    }
}

// Token Sampling Utilities

public struct TokenSampler: Sendable {
    
    /// Greedy sampling (argmax)
    public static func greedy(_ logits: MLXArray) -> Int32 {
        Int32(MLX.argMax(logits).item(Int32.self))
    }
    
    /// Temperature-scaled sampling
    public static func sample(_ logits: MLXArray, temperature: Float) -> Int32 {
        if temperature <= 0 {
            return greedy(logits)
        }
        
        let scaled = logits / temperature
        let probs = MLX.softmax(scaled)
        let sample = MLXRandom.categorical(probs.expandedDimensions(axis: 0))
        return sample.item(Int32.self)
    }
    
    /// Top-k sampling
    public static func topK(_ logits: MLXArray, k: Int, temperature: Float) -> Int32 {
        guard k > 0 && k < logits.shape[0] else {
            return sample(logits, temperature: temperature)
        }
        
        // Sort and take top k values
        let sortedIndices = argSort(logits)
        let topKIndices = sortedIndices[(logits.shape[0] - k)...]
        
        // Gather the top-k logits
        var topKValues: [Float] = []
        for i in 0..<k {
            let idx = Int(topKIndices[i].item(Int32.self))
            topKValues.append(logits[idx].item(Float.self))
        }
        let topKLogits = MLXArray(topKValues)
        
        let scaled = topKLogits / temperature
        let probs = softmax(scaled)
        let localIndex = MLXRandom.categorical(probs.expandedDimensions(axis: 0))
        
        return topKIndices[localIndex.item(Int.self)].item(Int32.self)
    }
    
    /// Top-p (nucleus) sampling
    public static func topP(_ logits: MLXArray, p: Float, temperature: Float) -> Int32 {
        let scaled = logits / temperature
        let probs = MLX.softmax(scaled)
        
        // Sort probabilities
        let sortedIndices = MLX.argSort(probs)
        let sortedProbs = MLX.takeAlong(probs, sortedIndices, axis: 0)
        
        // Find cumulative sum cutoff
        let cumsum = MLX.cumsum(sortedProbs, axis: 0)
        
        // Create mask for nucleus
        let mask = cumsum .<= (1.0 - p)
        let maskedProbs = MLX.where(mask, MLXArray(0.0), sortedProbs)
        
        // Renormalize and sample
        let normalizedProbs = maskedProbs / MLX.sum(maskedProbs)
        let localIndex = MLXRandom.categorical(normalizedProbs.expandedDimensions(axis: 0))
        
        return sortedIndices[localIndex.item(Int.self)].item(Int32.self)
    }
    
    /// Combined top-k and top-p sampling
    public static func sample(
        _ logits: MLXArray,
        temperature: Float,
        topK: Int? = nil,
        topP: Float? = nil
    ) -> Int32 {
        var processedLogits = logits
        
        // Apply top-k filtering using argsort
        if let k = topK, k > 0 && k < logits.shape[0] {
            let sortedIndices = argSort(logits)
            let kthIndex = sortedIndices[logits.shape[0] - k]
            let threshold = logits[Int(kthIndex.item(Int32.self))]
            processedLogits = MLX.where(
                processedLogits .< threshold,
                MLXArray(-Float.infinity),
                processedLogits
            )
        }
        
        // Apply top-p (nucleus) sampling
        if let p = topP, p < 1.0 {
            return self.topP(processedLogits, p: p, temperature: temperature)
        }
        
        return sample(processedLogits, temperature: temperature)
    }
}

// Tree-Based Speculation

/// A node in the speculation tree
public final class TreeNode: @unchecked Sendable {
    public let token: Int32
    public let logProb: Float
    public let depth: Int
    public weak var parent: TreeNode?
    public var children: [TreeNode] = []
    public var hiddenState: MLXArray?
    
    // Path probability (product of all log probs from root)
    public var pathLogProb: Float {
        (parent?.pathLogProb ?? 0) + logProb
    }
    
    public init(token: Int32, logProb: Float, depth: Int, parent: TreeNode? = nil) {
        self.token = token
        self.logProb = logProb
        self.depth = depth
        self.parent = parent
    }
    
    /// Get the token sequence from root to this node
    public func getPath() -> [Int32] {
        var path: [Int32] = []
        var current: TreeNode? = self
        while let node = current {
            path.insert(node.token, at: 0)
            current = node.parent
        }
        return path
    }
    
    /// Get all leaf nodes in subtree
    public func getLeaves() -> [TreeNode] {
        if children.isEmpty {
            return [self]
        }
        return children.flatMap { $0.getLeaves() }
    }
}

/// Result of tree speculation
public struct TreeSpeculationResult: @unchecked Sendable {
    /// The accepted token path
    public let acceptedPath: [Int32]
    /// The bonus token from target (if any)
    public let bonusToken: Int32?
    /// Hidden states for training
    public let hiddenStates: MLXArray?
    /// Target logits for training
    public let targetLogits: MLXArray?
    /// Total nodes explored
    public let nodesExplored: Int
    /// Acceptance rate for this tree
    public let acceptanceRate: Float
    
    public var totalTokens: Int {
        acceptedPath.count + (bonusToken != nil ? 1 : 0)
    }
}

/// Tree-based speculative decoder for higher acceptance rates
public actor TreeSpeculativeDecoder {
    private let drafter: DrafterActor
    private let target: TargetModel
    private let config: SpeculativeConfig
    private let trainingBuffer: TrainingBuffer?
    
    // Tree configuration
    private let maxDepth: Int
    private let maxWidth: Int
    private let beamSize: Int
    private let pruneThreshold: Float
    
    // Statistics
    private var totalTrees: Int = 0
    private var totalNodesExplored: Int = 0
    private var totalAccepted: Int = 0
    
    // Object pool for tree nodes to reduce allocations
    private var nodePool: [TreeNode] = []
    private let maxPoolSize = 1000
    
    public init(
        drafter: DrafterActor,
        target: TargetModel,
        config: SpeculativeConfig,
        trainingBuffer: TrainingBuffer? = nil,
        maxDepth: Int = 5,
        beamSize: Int = 4,
        pruneThreshold: Float = 0.01
    ) {
        self.drafter = drafter
        self.target = target
        self.config = config
        self.trainingBuffer = trainingBuffer
        self.maxDepth = maxDepth
        self.maxWidth = config.maxTreeWidth
        self.beamSize = beamSize
        self.pruneThreshold = pruneThreshold
    }
    
    /// Perform tree-based speculation
    public func speculate(
        context: MLXArray,
        generationConfig: GenerationConfig
    ) async throws -> TreeSpeculationResult {
        totalTrees += 1
        
        // Step 1: Get initial hidden states from target
        let (_, initialHidden) = try await target.getLogitsAndHiddenStates(inputIds: context)
        
        // Step 2: Build speculation tree using beam search
        let root = getNodeFromPool(token: -1, logProb: 0, depth: 0)
        root.hiddenState = initialHidden
        
        var currentLevel = [root]
        var allNodes: [TreeNode] = [root]
        
        for depth in 0..<maxDepth {
            var nextLevel: [TreeNode] = []
            
            for node in currentLevel {
                // Generate candidate tokens for this node
                let candidates = try await generateCandidates(
                    for: node,
                    context: context,
                    temperature: config.draftTemperature
                )
                
                for (token, logProb) in candidates.prefix(maxWidth) {
                    let child = getNodeFromPool(token: token, logProb: logProb, depth: depth + 1)
                    child.parent = node
                    node.children.append(child)
                    nextLevel.append(child)
                    allNodes.append(child)
                }
            }
            
            // Prune low-probability branches
            nextLevel = nextLevel.filter { $0.pathLogProb > log(pruneThreshold) }
            
            // Keep top beam candidates
            nextLevel.sort { $0.pathLogProb > $1.pathLogProb }
            currentLevel = Array(nextLevel.prefix(beamSize))
            
            if currentLevel.isEmpty {
                break
            }
        }
        
        totalNodesExplored += allNodes.count
        
        // Step 3: Batch verify all paths with target model
        let verificationResult = try await verifyTree(
            root: root,
            context: context,
            generationConfig: generationConfig
        )
        
        // Step 5: Return nodes to pool
        for node in allNodes {
            returnNodeToPool(node)
        }
        
        // Update drafter with acceptance data
        await drafter.recordAcceptance(
            accepted: verificationResult.acceptedPath.count,
            total: allNodes.count - 1  // Exclude root
        )
        
        totalAccepted += verificationResult.acceptedPath.count
        
        return verificationResult
    }
    
    /// Generate candidate tokens for a tree node
    private func generateCandidates(
        for node: TreeNode,
        context: MLXArray,
        temperature: Float
    ) async throws -> [(token: Int32, logProb: Float)] {
        guard let hiddenState = node.hiddenState else {
            return []
        }
        
        // Build token sequence from root to node
        let path = node.getPath().filter { $0 >= 0 }  // Filter out root's -1 token
        
        // Combine context with path
        let fullContext: MLXArray
        if path.isEmpty {
            fullContext = context
        } else {
            let pathArray = MLXArray(path)
            fullContext = MLX.concatenated([context, pathArray], axis: 0)
        }
        
        // Get logits from drafter
        let model = await drafter.getModel()
        let embeddings = model.getEmbeddings(fullContext)
        let lastEmbedding = embeddings[embeddings.shape[0] - 1].expandedDimensions(axis: 0)
        let lastHidden = hiddenState[hiddenState.shape[0] - 1].expandedDimensions(axis: 0)
        
        let logits = model(lastHidden, previousEmbeddings: lastEmbedding)
        let logProbs = MLX.log(MLX.softmax(logits.squeezed() / temperature))
        
        // Get top-k candidates
        let k = min(maxWidth * 2, logits.shape[logits.ndim - 1])
        let topIndices = MLX.argSort(logProbs)[(-k)...]
        
        var candidates: [(Int32, Float)] = []
        for i in 0..<k {
            let idx = Int(topIndices[i].item(Int32.self))
            let prob = logProbs[idx].item(Float.self)
            candidates.append((Int32(idx), prob))
        }
        
        // Sort by probability
        candidates.sort { $0.1 > $1.1 }
        
        return candidates
    }
    
    /// Verify the entire tree with the target model
    private func verifyTree(
        root: TreeNode,
        context: MLXArray,
        generationConfig: GenerationConfig
    ) async throws -> TreeSpeculationResult {
        // Collect all paths from leaves
        let leaves = root.getLeaves()
        
        // Find the best accepted path
        var bestPath: [Int32] = []
        var bestBonus: Int32? = nil
        var bestHidden: MLXArray? = nil
        var bestLogits: MLXArray? = nil
        
        for leaf in leaves.sorted(by: { $0.pathLogProb > $1.pathLogProb }) {
            let path = leaf.getPath().filter { $0 >= 0 }
            guard !path.isEmpty else { continue }
            
            // Verify this path
            let verificationResult = try await target.verifyDraftTokens(
                contextIds: context,
                draftTokens: path,
                config: generationConfig
            )
            
            // Count accepted tokens
            let acceptedCount = verificationResult.acceptedCount
            let acceptedTokens = Array(path.prefix(acceptedCount))
            
            // If this path is better (more accepted tokens), use it
            if acceptedTokens.count > bestPath.count {
                bestPath = acceptedTokens
                bestBonus = verificationResult.correctedToken
                bestHidden = verificationResult.hiddenStates
                bestLogits = verificationResult.logits
                
                // Push rejection data for training
                if let buffer = trainingBuffer, acceptedCount < path.count {
                    await buffer.pushRejectionData(
                        hiddenStates: verificationResult.hiddenStates,
                        targetLogits: verificationResult.logits,
                        inputIds: context,
                        rejectedCount: path.count - acceptedCount
                    )
                }
                
                // If we got a fully accepted path, we're done
                if acceptedCount == path.count {
                    break
                }
            }
        }
        
        let totalNodes = leaves.reduce(0) { $0 + $1.depth }
        let acceptanceRate = totalNodes > 0 ? Float(bestPath.count) / Float(totalNodes) : 0
        
        return TreeSpeculationResult(
            acceptedPath: bestPath,
            bonusToken: bestBonus,
            hiddenStates: bestHidden,
            targetLogits: bestLogits,
            nodesExplored: totalNodes,
            acceptanceRate: acceptanceRate
        )
    }
    
    /// Get a node from the pool or create a new one
    private func getNodeFromPool(token: Int32, logProb: Float, depth: Int) -> TreeNode {
        if let node = nodePool.popLast() {
            // Reset node
            node.children.removeAll()
            node.hiddenState = nil
            return TreeNode(token: token, logProb: logProb, depth: depth)
        }
        return TreeNode(token: token, logProb: logProb, depth: depth)
    }
    
    /// Return a node to the pool
    private func returnNodeToPool(_ node: TreeNode) {
        if nodePool.count < maxPoolSize {
            node.children.removeAll()
            node.hiddenState = nil
            nodePool.append(node)
        }
    }
    
    /// Get tree speculation statistics
    public func getStatistics() -> (
        trees: Int,
        nodesExplored: Int,
        accepted: Int,
        avgNodesPerTree: Float,
        acceptanceRate: Float
    ) {
        let avgNodes = totalTrees > 0 ? Float(totalNodesExplored) / Float(totalTrees) : 0
        let rate = totalNodesExplored > 0 ? Float(totalAccepted) / Float(totalNodesExplored) : 0
        return (totalTrees, totalNodesExplored, totalAccepted, avgNodes, rate)
    }
    
    /// Reset statistics
    public func resetStatistics() {
        totalTrees = 0
        totalNodesExplored = 0
        totalAccepted = 0
    }
}

// MARK: - Hidden State Collector for EAGLE Training

/// Collects hidden states from target model during inference for drafter training
/// Implements FastRL's approach of gathering training data during long-tail generation
public actor HiddenStateCollector {
    /// Collected sample with hidden states and target outputs
    public struct CollectedSample: @unchecked Sendable {
        public let hiddenStates: MLXArray      // From target model
        public let targetLogits: MLXArray       // Target model outputs
        public let inputTokens: [Int32]         // Input context
        public let outputTokens: [Int32]        // Target tokens
        public let timestamp: Date
        public let wasAccepted: Bool            // Whether draft was accepted
        
        public init(
            hiddenStates: MLXArray,
            targetLogits: MLXArray,
            inputTokens: [Int32],
            outputTokens: [Int32],
            wasAccepted: Bool
        ) {
            self.hiddenStates = hiddenStates
            self.targetLogits = targetLogits
            self.inputTokens = inputTokens
            self.outputTokens = outputTokens
            self.timestamp = Date()
            self.wasAccepted = wasAccepted
        }
    }
    
    private var collectedSamples: [CollectedSample] = []
    private let maxSamples: Int
    private let prioritizeRejections: Bool
    
    // Statistics
    private var totalCollected: Int = 0
    private var totalRejections: Int = 0
    private var totalAcceptances: Int = 0
    
    public init(maxSamples: Int = 1000, prioritizeRejections: Bool = true) {
        self.maxSamples = maxSamples
        self.prioritizeRejections = prioritizeRejections
    }
    
    /// Collect a sample from target model inference
    public func collect(
        hiddenStates: MLXArray,
        targetLogits: MLXArray,
        inputTokens: [Int32],
        outputTokens: [Int32],
        wasAccepted: Bool
    ) {
        let sample = CollectedSample(
            hiddenStates: hiddenStates,
            targetLogits: targetLogits,
            inputTokens: inputTokens,
            outputTokens: outputTokens,
            wasAccepted: wasAccepted
        )
        
        totalCollected += 1
        if wasAccepted {
            totalAcceptances += 1
        } else {
            totalRejections += 1
        }
        
        // If at capacity, decide what to drop
        if collectedSamples.count >= maxSamples {
            if prioritizeRejections && wasAccepted {
                // Don't add accepted samples if we're prioritizing rejections
                // and buffer is full - rejections are more valuable for learning
                if let firstAcceptedIdx = collectedSamples.firstIndex(where: { $0.wasAccepted }) {
                    collectedSamples.remove(at: firstAcceptedIdx)
                } else {
                    // All samples are rejections, drop oldest
                    collectedSamples.removeFirst()
                }
            } else {
                // Drop oldest
                collectedSamples.removeFirst()
            }
        }
        
        collectedSamples.append(sample)
    }
    
    /// Get a batch of samples for training
    public func getBatch(size: Int, preferRejections: Bool = true) -> [CollectedSample] {
        guard !collectedSamples.isEmpty else { return [] }
        
        if preferRejections {
            // Get rejections first, then fill with acceptances
            let rejections = collectedSamples.filter { !$0.wasAccepted }
            let acceptances = collectedSamples.filter { $0.wasAccepted }
            
            var batch: [CollectedSample] = []
            batch.append(contentsOf: rejections.prefix(size))
            
            if batch.count < size {
                batch.append(contentsOf: acceptances.prefix(size - batch.count))
            }
            
            return batch
        } else {
            return Array(collectedSamples.prefix(size))
        }
    }
    
    /// Pop a batch of samples (removes from collection)
    public func popBatch(size: Int, preferRejections: Bool = true) -> [CollectedSample] {
        guard !collectedSamples.isEmpty else { return [] }
        
        var indicesToRemove: [Int] = []
        var batch: [CollectedSample] = []
        
        if preferRejections {
            // First pass: collect rejections
            for (idx, sample) in collectedSamples.enumerated() {
                if !sample.wasAccepted && batch.count < size {
                    batch.append(sample)
                    indicesToRemove.append(idx)
                }
            }
            
            // Second pass: fill with acceptances if needed
            for (idx, sample) in collectedSamples.enumerated() {
                if sample.wasAccepted && batch.count < size && !indicesToRemove.contains(idx) {
                    batch.append(sample)
                    indicesToRemove.append(idx)
                }
            }
        } else {
            for (idx, sample) in collectedSamples.enumerated() {
                if batch.count < size {
                    batch.append(sample)
                    indicesToRemove.append(idx)
                }
            }
        }
        
        // Remove in reverse order to maintain indices
        for idx in indicesToRemove.sorted().reversed() {
            collectedSamples.remove(at: idx)
        }
        
        return batch
    }
    
    /// Convert collected samples to training samples for the drafter
    public func toTrainingSamples(batch: [CollectedSample]) -> [TrainingSample] {
        batch.map { sample in
            TrainingSample(
                hiddenStates: sample.hiddenStates,
                targetLogits: sample.targetLogits,
                inputIds: MLXArray(sample.inputTokens)
            )
        }
    }
    
    /// Get statistics
    public func getStatistics() -> (total: Int, rejections: Int, acceptances: Int, currentSize: Int) {
        (totalCollected, totalRejections, totalAcceptances, collectedSamples.count)
    }
    
    /// Clear all collected samples
    public func clear() {
        collectedSamples.removeAll()
    }
    
    /// Check if we have enough samples for training
    public func hasEnoughForTraining(minSize: Int = 8) -> Bool {
        collectedSamples.count >= minSize
    }
}

// MARK: - Adaptive Drafter Trainer

/// Background trainer that trains the drafter during idle GPU time
/// Implements FastRL's adaptive drafter training approach
public actor AdaptiveDrafterTrainer {
    private let drafter: DrafterActor
    private let collector: HiddenStateCollector
    private let loadMonitor: GPULoadMonitor
    
    private var isRunning: Bool = false
    private var trainingTask: Task<Void, Never>?
    private var totalTrainingSteps: Int = 0
    private var averageLoss: Float = 0
    
    // Training configuration
    private let batchSize: Int
    private let trainingInterval: Duration
    private let gpuIdleThreshold: Double
    
    public init(
        drafter: DrafterActor,
        collector: HiddenStateCollector,
        loadMonitor: GPULoadMonitor,
        batchSize: Int = 4,
        trainingInterval: Duration = .milliseconds(100),
        gpuIdleThreshold: Double = 0.5
    ) {
        self.drafter = drafter
        self.collector = collector
        self.loadMonitor = loadMonitor
        self.batchSize = batchSize
        self.trainingInterval = trainingInterval
        self.gpuIdleThreshold = gpuIdleThreshold
    }
    
    /// Start background training
    public func start() {
        guard !isRunning else { return }
        isRunning = true
        
        trainingTask = Task(priority: .background) { [weak self] in
            await self?.trainingLoop()
        }
    }
    
    /// Stop background training
    public func stop() {
        isRunning = false
        trainingTask?.cancel()
        trainingTask = nil
    }
    
    /// Main training loop - runs when GPU is idle
    private func trainingLoop() async {
        while isRunning && !Task.isCancelled {
            // Check GPU utilization
            let gpuLoad = await loadMonitor.getAverageLoad()
            
            if gpuLoad < gpuIdleThreshold {
                // GPU is idle, check if we have training data
                let hasData = await collector.hasEnoughForTraining(minSize: batchSize)
                
                if hasData {
                    await performTrainingStep()
                }
            }
            
            // Wait before next check
            try? await Task.sleep(for: trainingInterval)
        }
    }
    
    /// Perform a single training step
    private func performTrainingStep() async {
        // Get batch of samples, prioritizing rejections
        let samples = await collector.popBatch(size: batchSize, preferRejections: true)
        guard !samples.isEmpty else { return }
        
        // Convert to training samples
        let trainingSamples = await collector.toTrainingSamples(batch: samples)
        
        // Train drafter
        let loss = await drafter.trainStep(samples: trainingSamples)
        
        // Update statistics
        totalTrainingSteps += 1
        averageLoss = 0.9 * averageLoss + 0.1 * loss
    }
    
    /// Get training statistics
    public func getStatistics() -> (steps: Int, avgLoss: Float, isRunning: Bool) {
        (totalTrainingSteps, averageLoss, isRunning)
    }
    
    /// Check if training is active
    public func isTraining() -> Bool {
        isRunning
    }
}

// Batch Speculation

/// Batch speculation for improved throughput
public actor BatchSpeculativeDecoder {
    private let drafter: DrafterActor
    private let target: TargetModel
    private let config: SpeculativeConfig
    
    public init(
        drafter: DrafterActor,
        target: TargetModel,
        config: SpeculativeConfig
    ) {
        self.drafter = drafter
        self.target = target
        self.config = config
    }
    
    /// Process multiple sequences in parallel
    public func speculateBatch(
        contexts: [MLXArray],
        generationConfig: GenerationConfig
    ) async throws -> [SpeculationResult] {
        // Process in parallel using task group
        try await withThrowingTaskGroup(of: (Int, SpeculationResult).self) { group in
            for (index, context) in contexts.enumerated() {
                group.addTask {
                    let decoder = SpeculativeDecoder(
                        drafter: self.drafter,
                        target: self.target,
                        config: self.config
                    )
                    let result = try await decoder.speculate(
                        context: context,
                        generationConfig: generationConfig
                    )
                    return (index, result)
                }
            }
            
            var results: [(Int, SpeculationResult)] = []
            for try await result in group {
                results.append(result)
            }
            
            return results.sorted { $0.0 < $1.0 }.map { $0.1 }
        }
    }
    
    /// Process using tree speculation
    public func speculateBatchWithTree(
        contexts: [MLXArray],
        generationConfig: GenerationConfig
    ) async throws -> [TreeSpeculationResult] {
        try await withThrowingTaskGroup(of: (Int, TreeSpeculationResult).self) { group in
            for (index, context) in contexts.enumerated() {
                group.addTask {
                    let decoder = TreeSpeculativeDecoder(
                        drafter: self.drafter,
                        target: self.target,
                        config: self.config
                    )
                    let result = try await decoder.speculate(
                        context: context,
                        generationConfig: generationConfig
                    )
                    return (index, result)
                }
            }
            
            var results: [(Int, TreeSpeculationResult)] = []
            for try await result in group {
                results.append(result)
            }
            
            return results.sorted { $0.0 < $1.0 }.map { $0.1 }
        }
    }
}

// MARK: - FastRL Speculative Decoder

/// FastRL-style speculative decoder using the lightweight SingleLayerEAGLEDrafter
/// This implementation follows the TLT paper's approach:
/// 1. Uses a single-layer drafter head (not a full model)
/// 2. Collects hidden states from target during inference for training
/// 3. Integrates with MAB for adaptive SD configuration
/// 4. Supports background drafter training during idle GPU time
public actor FastRLSpeculativeDecoder {
    private let drafter: FastRLDrafterActor
    private let target: TargetModel
    private let mab: MultiArmedBandit
    private let loadMonitor: GPULoadMonitor
    
    // Data collection for training
    private var collectedData: [(hiddenStates: MLXArray, tokenIds: MLXArray, targetLogits: MLXArray)] = []
    private let maxCollectedData: Int = 512
    
    // Configuration
    private let batchSizeThresholds: [Int]
    private var currentConfig: MABConfig
    private var sdEnabled: Bool = true
    
    // Statistics
    private var totalGenerations: Int = 0
    private var totalTokensGenerated: Int = 0
    private var totalAccepted: Int = 0
    private var totalDrafted: Int = 0
    
    // Background training
    private var trainingTask: Task<Void, Never>?
    private var isTraining: Bool = false
    
    public init(
        drafter: FastRLDrafterActor,
        target: TargetModel,
        loadMonitor: GPULoadMonitor,
        batchSizeThresholds: [Int] = [1, 2, 5, 21]
    ) {
        self.drafter = drafter
        self.target = target
        self.loadMonitor = loadMonitor
        self.mab = MultiArmedBandit()
        self.batchSizeThresholds = batchSizeThresholds
        self.currentConfig = MABConfig.defaultConfigs[0]
    }
    
    /// Generate tokens with FastRL-style speculative decoding
    /// Automatically collects training data during inference
    public func generate(
        context: MLXArray,
        maxTokens: Int,
        temperature: Float = 1.0,
        topK: Int = 8
    ) async throws -> (
        tokens: [Int32],
        drafted: Int,
        accepted: Int,
        elapsed: TimeInterval
    ) {
        totalGenerations += 1
        
        let batchSize = context.ndim > 1 ? context.shape[0] : 1
        let contextLength = context.shape[context.ndim - 1]
        
        // Decide whether to use SD based on batch characteristics
        sdEnabled = shouldEnableSD(batchSize: batchSize, contextLength: contextLength)
        
        if sdEnabled {
            return try await generateWithSD(
                context: context,
                maxTokens: maxTokens,
                temperature: temperature,
                topK: topK
            )
        } else {
            return try await generateStandard(
                context: context,
                maxTokens: maxTokens,
                temperature: temperature,
                topK: topK
            )
        }
    }
    
    /// Generate with speculative decoding enabled
    private func generateWithSD(
        context: MLXArray,
        maxTokens: Int,
        temperature: Float,
        topK: Int
    ) async throws -> (
        tokens: [Int32],
        drafted: Int,
        accepted: Int,
        elapsed: TimeInterval
    ) {
        // Select MAB configuration
        let config = await mab.selectArm()
        currentConfig = config
        
        var generatedTokens: [Int32] = []
        var currentContext = context
        var localAccepted = 0
        var localDrafted = 0
        
        let startTime = Date()
        
        while generatedTokens.count < maxTokens {
            // Get hidden states and logits from target
            let (targetLogits, hiddenStates) = try await target.getLogitsAndHiddenStates(inputIds: currentContext)
            
            // Get last hidden state for drafter
            let lastHidden = hiddenStates[0, hiddenStates.shape[1] - 1].expandedDimensions(axis: 0)
            let lastToken = currentContext[currentContext.shape[0] - 1].expandedDimensions(axis: 0)
            
            // Generate draft tokens using lightweight drafter (with log probs for lossless check)
            let draft = await drafter.generateDraftsWithLogProbs(
                hiddenState: lastHidden,
                prevToken: lastToken,
                count: config.steps,
                temperature: temperature,
                topK: config.topK
            )
            let draftTokens = draft.tokens
            localDrafted += draftTokens.count
            
            // Verify drafts with target
            let genConfig = GenerationConfig(
                maxTokens: maxTokens,
                temperature: temperature,
                topP: 0.9,
                topK: topK,
                repetitionPenalty: 1.0,
                stopTokens: []
            )
            
            let verifyResult = try await target.verifyDraftTokens(
                contextIds: currentContext.squeezed(),
                draftTokens: draftTokens,
                config: genConfig,
                draftLogProbs: draft.logProbs
            )
            
            // Accept verified tokens
            let acceptedTokens = verifyResult.acceptedTokens
            localAccepted += acceptedTokens.count
            generatedTokens.append(contentsOf: acceptedTokens)
            
            // Add bonus token if available
            if let bonus = verifyResult.bonusToken {
                generatedTokens.append(bonus)
            }
            
            // Collect training data
            await collectTrainingData(
                hiddenStates: hiddenStates,
                tokenIds: currentContext,
                targetLogits: targetLogits
            )
            
            // Update context
            let newTokens = acceptedTokens + (verifyResult.bonusToken.map { [$0] } ?? [])
            let newTokensArray = MLXArray(newTokens)
            currentContext = MLX.concatenated([currentContext, newTokensArray], axis: -1)
            
            // Check for stop condition
            if generatedTokens.count >= maxTokens {
                break
            }
        }
        
        // Update MAB with performance (using tokens/second as reward)
        let elapsed = Date().timeIntervalSince(startTime)
        let tokensPerSecond = Double(generatedTokens.count) / max(elapsed, 0.001)
        await mab.updateReward(config: config, reward: tokensPerSecond)
        
        totalTokensGenerated += generatedTokens.count
        totalAccepted += localAccepted
        totalDrafted += localDrafted
        
        return (
            Array(generatedTokens.prefix(maxTokens)),
            localDrafted,
            localAccepted,
            elapsed
        )
        }
    
    /// Generate without speculative decoding (for small batches)
    private func generateStandard(
        context: MLXArray,
        maxTokens: Int,
        temperature: Float,
        topK: Int
    ) async throws -> (
        tokens: [Int32],
        drafted: Int,
        accepted: Int,
        elapsed: TimeInterval
    ) {
        var generatedTokens: [Int32] = []
        var currentContext = context
        let startTime = Date()
        
        for _ in 0..<maxTokens {
            let (logits, hiddenStates) = try await target.getLogitsAndHiddenStates(inputIds: currentContext)
            
            // Sample next token
            let lastLogits = logits[0, logits.shape[1] - 1]
            let nextToken = sampleToken(logits: lastLogits, temperature: temperature, topK: topK)
            generatedTokens.append(nextToken)
            
            // Collect training data (even in non-SD mode for adaptation)
            await collectTrainingData(
                hiddenStates: hiddenStates,
                tokenIds: currentContext,
                targetLogits: logits
            )
            
            // Update context
            let nextTokenArray = MLXArray([nextToken])
            currentContext = MLX.concatenated([currentContext, nextTokenArray], axis: -1)
        }
        
        totalTokensGenerated += generatedTokens.count
        totalAccepted += generatedTokens.count
        totalDrafted += generatedTokens.count
        let elapsed = Date().timeIntervalSince(startTime)
        
        return (generatedTokens, generatedTokens.count, generatedTokens.count, elapsed)
    }
    
    /// Collect training data from inference
    private func collectTrainingData(
        hiddenStates: MLXArray,
        tokenIds: MLXArray,
        targetLogits: MLXArray
    ) async {
        // Forward to drafter for collection
        await drafter.collectTrainingData(
            hiddenStates: hiddenStates,
            tokenIds: tokenIds,
            targetLogits: targetLogits
        )
        
        // Also keep local copy for potential distributed sync
        collectedData.append((hiddenStates, tokenIds, targetLogits))
        if collectedData.count > maxCollectedData {
            collectedData.removeFirst()
        }
    }
    
    /// Decide whether to enable SD based on batch characteristics
    private func shouldEnableSD(batchSize: Int, contextLength: Int) -> Bool {
        // FastRL's batch size thresholds: [1, 2, 5, 21]
        // SD disabled for very small batches where overhead isn't worth it
        if batchSize < batchSizeThresholds[0] {
            return false
        }
        
        // For larger batches, enable SD
        if batchSize >= batchSizeThresholds[batchSizeThresholds.count - 1] {
            return true
        }
        
        // For long contexts (long-tail), always enable SD
        if contextLength > 512 {
            return true
        }
        
        return true
    }
    
    /// Sample token from logits
    private func sampleToken(logits: MLXArray, temperature: Float, topK: Int) -> Int32 {
        var processedLogits = logits
        
        if temperature > 0 {
            processedLogits = processedLogits / temperature
        }
        
        if topK > 0 && topK < logits.shape[0] {
            let sorted = MLX.sorted(processedLogits)
            let threshold = sorted[sorted.shape[0] - topK]
            let mask = processedLogits .< threshold
            processedLogits = MLX.where(mask, MLXArray(-Float.infinity), processedLogits)
        }
        
        if temperature <= 0 {
            return MLX.argMax(processedLogits).item(Int32.self)
        } else {
            let probs = softmax(processedLogits)
            let sampled = MLXRandom.categorical(probs.expandedDimensions(axis: 0))
            return sampled.item(Int32.self)
        }
    }
    
    /// Start background drafter training
    public func startBackgroundTraining() {
        guard !isTraining else { return }
        isTraining = true
        
        trainingTask = Task(priority: .background) { [weak self] in
            await self?.backgroundTrainingLoop()
        }
    }
    
    /// Stop background training
    public func stopBackgroundTraining() {
        isTraining = false
        trainingTask?.cancel()
        trainingTask = nil
    }
    
    /// Background training loop
    private func backgroundTrainingLoop() async {
        while isTraining && !Task.isCancelled {
            let gpuLoad = await loadMonitor.getAverageLoad()
            
            // Only train when GPU is idle
            if gpuLoad < 0.5 {
                let loss = await drafter.trainStep(batchSize: 8)
                if loss > 0 {
                    // Training happened
                }
            }
            
            try? await Task.sleep(for: .milliseconds(100))
        }
    }
    
    /// Get statistics
    public func getStatistics() -> (
        generations: Int,
        tokensGenerated: Int,
        acceptanceRate: Float,
        sdEnabled: Bool,
        currentConfig: MABConfig
    ) {
        let rate = totalDrafted > 0 ? Float(totalAccepted) / Float(totalDrafted) : 0
        return (totalGenerations, totalTokensGenerated, rate, sdEnabled, currentConfig)
    }
    
    /// Get drafter training statistics
    public func getDrafterStats() async -> (step: Int, loss: Float, bufferSize: Int) {
        let step = await drafter.getTrainingStep()
        let loss = await drafter.getLastLoss()
        let bufferSize = await drafter.getBufferSize()
        return (step, loss, bufferSize)
    }
}
