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
    
    public static let `default` = SpeculativeConfig(
        draftCount: 5,
        draftTemperature: 0.7,
        verifyTemperature: 0.7,
        useTreeSpeculation: false,
        maxTreeWidth: 3,
        acceptanceThreshold: 0.9
    )
    
    public static func forTier(_ tier: HardwareTier) -> SpeculativeConfig {
        SpeculativeConfig(
            draftCount: tier.draftCount,
            draftTemperature: 0.7,
            verifyTemperature: 0.7,
            useTreeSpeculation: tier == .elite,
            maxTreeWidth: tier == .elite ? 4 : 2,
            acceptanceThreshold: 0.9
        )
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
        let draftTokens = await drafter.generateDraft(
            hiddenStates: hiddenStates,
            previousTokens: context,
            count: config.draftCount,
            temperature: config.draftTemperature
        )
        totalDrafted += draftTokens.count
        
        // Step 3: Verify draft tokens with target model
        let verification = try await target.verifyDraftTokens(
            contextIds: context,
            draftTokens: draftTokens,
            config: generationConfig
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
            await buffer.pushRejectionData(
                hiddenStates: verification.hiddenStates,
                targetLogits: verification.logits,
                inputIds: context,
                rejectedCount: rejectedCount
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

// Batch Speculation (Future: Tree Speculation)

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
        // For now, process sequentially
        // Future: Implement true batch processing
        var results: [SpeculationResult] = []
        
        for context in contexts {
            let decoder = SpeculativeDecoder(
                drafter: drafter,
                target: target,
                config: config
            )
            let result = try await decoder.speculate(
                context: context,
                generationConfig: generationConfig
            )
            results.append(result)
        }
        
        return results
    }
}
