// SPDX-License-Identifier: MIT
// Nano-Reasoning: RL Rollout Surface
// Provides an on-policy rollout trace with token-level logprobs for RL trainers

import Foundation
@preconcurrency import MLX

/// One step in an RL rollout
public struct RolloutStep: Sendable {
    public let token: Int32
    public let logProb: Float
    public let position: Int
}

/// Complete rollout trace
public struct RolloutTrace: Sendable {
    public let promptTokens: [Int32]
    public let generatedTokens: [Int32]
    public let steps: [RolloutStep]
    
    public init(promptTokens: [Int32], generatedTokens: [Int32], steps: [RolloutStep]) {
        self.promptTokens = promptTokens
        self.generatedTokens = generatedTokens
        self.steps = steps
    }
}

public extension Orchestrator {
    /// Run an on-policy rollout and return token-level logprobs for RL training
    func rollout(
        prompt: String,
        maxTokens: Int
    ) async throws -> RolloutTrace {
        // Generate with current speculative path (lossless acceptance)
        let promptIds = try await tokenizeText(prompt)
        let promptArray = MLXArray(promptIds)
        
        // Use standard generate to ensure pipeline stays consistent
        let generated = try await generate(prompt: prompt, maxTokens: maxTokens, tokenCallback: nil)
        let generatedIds = try await tokenizeText(generated)
        
        // Compute logprobs for generated tokens using target model
        guard let target = getTargetModelActor() else {
            throw OrchestratorError.componentNotInitialized
        }
        
        let fullSequence = MLX.concatenated([promptArray, MLXArray(generatedIds)], axis: 0)
        let (logits, _) = try await target.getLogitsAndHiddenStates(inputIds: fullSequence)
        
        var steps: [RolloutStep] = []
        for (idx, token) in generatedIds.enumerated() {
            let pos = promptIds.count + idx - 1
            guard pos >= 0 && pos < logits.shape[1] else { continue }
            let tokenLogits = logits[0, pos]
            let logProbs = log(softmax(tokenLogits))
            let lp = logProbs[Int(token)].item(Float.self)
            steps.append(RolloutStep(token: token, logProb: lp, position: pos))
        }
        
        return RolloutTrace(promptTokens: promptIds, generatedTokens: generatedIds, steps: steps)
    }
}
