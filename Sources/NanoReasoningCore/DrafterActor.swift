// SPDX-License-Identifier: MIT
// Nano-Reasoning: Drafter Actor with Atomic Weight Updates
// Swift 6.2 concurrency-safe implementation

import Foundation
@preconcurrency import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

/// Training sample for the drafter model
public struct TrainingSample: @unchecked Sendable {
    /// Hidden states from target model
    public let hiddenStates: MLXArray
    /// Target token logits for supervision
    public let targetLogits: MLXArray
    /// Input token ids
    public let inputIds: MLXArray
    /// Attention mask
    public let attentionMask: MLXArray?
    /// Optional positions where drafts were rejected (for focused training)
    public let rejectionPositions: [Int]
    /// Context length at time of collection (for long-tail detection)
    public let contextLength: Int?
    
    public init(
        hiddenStates: MLXArray,
        targetLogits: MLXArray,
        inputIds: MLXArray,
        attentionMask: MLXArray? = nil,
        rejectionPositions: [Int] = [],
        contextLength: Int? = nil
    ) {
        self.hiddenStates = hiddenStates
        self.targetLogits = targetLogits
        self.inputIds = inputIds
        self.attentionMask = attentionMask
        self.rejectionPositions = rejectionPositions
        self.contextLength = contextLength
    }
}

/// KV Cache wrapper for efficient inference
public struct KVCacheEntry: @unchecked Sendable {
    public var keys: MLXArray
    public var values: MLXArray
    
    public init(keys: MLXArray, values: MLXArray) {
        self.keys = keys
        self.values = values
    }
}

/// Drafter model configuration
public struct DrafterConfig: Sendable {
    public let hiddenSize: Int
    public let numLayers: Int
    public let numHeads: Int
    public let vocabSize: Int
    public let intermediateSize: Int
    public let maxPositionEmbeddings: Int
    public let ropeTheta: Float
    
    /// Default config for Qwen3-0.6B compatible model
    public static let qwen3_0_6B = DrafterConfig(
        hiddenSize: 1024,
        numLayers: 28,
        numHeads: 16,
        vocabSize: 151936,
        intermediateSize: 2816,
        maxPositionEmbeddings: 32768,
        ropeTheta: 1000000.0
    )
    
    /// Smaller config for testing
    public static let test = DrafterConfig(
        hiddenSize: 256,
        numLayers: 4,
        numHeads: 4,
        vocabSize: 32000,
        intermediateSize: 512,
        maxPositionEmbeddings: 2048,
        ropeTheta: 10000.0
    )
}

/// EAGLE-style speculative head that predicts next hidden states
/// Based on the EAGLE paper: https://arxiv.org/abs/2401.15077
/// The key insight is that we can train a small model to predict the hidden states
/// of the target model, allowing for accurate draft generation.
public class EAGLEHead: Module, @unchecked Sendable {
    let inputProjection: Linear
    let fusionLayer: Linear
    let outputProjection: Linear
    let layerNorm: LayerNorm
    let hiddenSize: Int
    
    // EAGLE-2 improvements: tree attention support
    let attentionHead: Linear?
    let attentionOut: Linear?
    let useTreeAttention: Bool
    
    public init(hiddenSize: Int, useTreeAttention: Bool = false) {
        self.hiddenSize = hiddenSize
        self.useTreeAttention = useTreeAttention
        
        // Project concatenated [hidden_state, embedding] to hidden_size
        self.inputProjection = Linear(hiddenSize * 2, hiddenSize)
        self.fusionLayer = Linear(hiddenSize, hiddenSize)
        self.outputProjection = Linear(hiddenSize, hiddenSize)
        self.layerNorm = LayerNorm(dimensions: hiddenSize)
        
        // Tree attention for EAGLE-2 (optional)
        if useTreeAttention {
            self.attentionHead = Linear(hiddenSize, hiddenSize)
            self.attentionOut = Linear(hiddenSize, hiddenSize)
        } else {
            self.attentionHead = nil
            self.attentionOut = nil
        }
        
        super.init()
    }
    
    /// Standard forward pass for single-step prediction
    public func callAsFunction(
        _ hiddenStates: MLXArray,
        embeddings: MLXArray
    ) -> MLXArray {
        // Concatenate hidden states and token embeddings
        let concat = MLX.concatenated([hiddenStates, embeddings], axis: -1)
        
        // Project and fuse
        var x = inputProjection(concat)
        x = silu(x)
        x = fusionLayer(x)
        x = silu(x)
        x = outputProjection(x)
        
        // Residual connection with layer norm
        x = layerNorm(x + hiddenStates)
        
        return x
    }
    
    /// Multi-step forward pass for tree speculation (EAGLE-2)
    /// Predicts hidden states for multiple draft tokens at once
    public func multiStepForward(
        hiddenStates: MLXArray,           // [batch, seq_len, hidden]
        embeddings: MLXArray,             // [batch, draft_len, hidden]
        treeMask: MLXArray? = nil         // [batch, draft_len, draft_len] attention mask
    ) -> MLXArray {
        guard useTreeAttention, let attnHead = attentionHead, let attnOut = attentionOut else {
            // Fall back to sequential processing
            return callAsFunction(hiddenStates, embeddings: embeddings)
        }
        
        let draftLen = embeddings.shape[1]
        
        // Get the last hidden state from context
        let lastHidden = hiddenStates[0..., (hiddenStates.shape[1] - 1)..., 0...]
        
        // Expand last hidden to match draft length
        let expandedHidden = MLX.repeated(lastHidden, count: draftLen, axis: 1)
        
        // Concatenate for fusion
        let concat = MLX.concatenated([expandedHidden, embeddings], axis: -1)
        
        // Project through fusion layers
        var x = inputProjection(concat)
        x = silu(x)
        x = fusionLayer(x)
        x = silu(x)
        
        // Apply tree attention if mask provided
        if let mask = treeMask {
            // Simple self-attention with tree mask
            let attn = attnHead(x)
            let scores = MLX.matmul(attn, attn.transposed(0, 2, 1)) / sqrt(Float(hiddenSize))
            let maskedScores = scores + mask  // mask should have -inf for blocked positions
            let weights = MLX.softmax(maskedScores, axis: -1)
            x = attnOut(MLX.matmul(weights, x))
        }
        
        x = outputProjection(x)
        x = layerNorm(x + expandedHidden)
        
        return x
    }
    
    /// Compute EAGLE training loss
    /// Trains to predict next hidden state given current hidden + embedding
    public static func computeEAGLELoss(
        predictedHidden: MLXArray,
        targetHidden: MLXArray,
        targetLogits: MLXArray,
        lmHead: Linear
    ) -> MLXArray {
        // Primary loss: hidden state prediction (MSE)
        let hiddenLoss = MLX.mean(MLX.pow(predictedHidden - targetHidden, 2))
        
        // Secondary loss: next token prediction (cross-entropy)
        // This ensures the predicted hidden states produce correct token distributions
        let predictedLogits = lmHead(predictedHidden)
        let targetProbs = MLX.softmax(targetLogits)
        let logPredictedProbs = MLX.log(MLX.softmax(predictedLogits) + 1e-10)
        let ceLoss = -MLX.mean(MLX.sum(targetProbs * logPredictedProbs, axis: -1))
        
        // Combined loss with weighting
        // Hidden loss helps with alignment, CE loss ensures correct outputs
        return 0.5 * hiddenLoss + 0.5 * ceLoss
    }
}

// MARK: - FastRL Single-Layer EAGLE Drafter

/// Configuration for the lightweight single-layer EAGLE drafter
/// Following FastRL's design: a minimal head that predicts next tokens from target hidden states
public struct SingleLayerDrafterConfig: Sendable {
    public let hiddenSize: Int       // Must match target model's hidden size
    public let vocabSize: Int        // Must match target model's vocab size
    public let intermediateSize: Int // MLP intermediate dimension
    
    /// Default config for Qwen3 models
    public static func forQwen3(size: String) -> SingleLayerDrafterConfig {
        switch size {
        case "7B":
            return SingleLayerDrafterConfig(hiddenSize: 4096, vocabSize: 152064, intermediateSize: 4608)
        case "32B":
            return SingleLayerDrafterConfig(hiddenSize: 5120, vocabSize: 152064, intermediateSize: 6144)
        case "4B":
            return SingleLayerDrafterConfig(hiddenSize: 3072, vocabSize: 152064, intermediateSize: 3584)
        case "0.6B":
            return SingleLayerDrafterConfig(hiddenSize: 1024, vocabSize: 151936, intermediateSize: 1536)
        default:
            // Default to 7B config
            return SingleLayerDrafterConfig(hiddenSize: 4096, vocabSize: 152064, intermediateSize: 4608)
        }
    }
    
    public init(hiddenSize: Int, vocabSize: Int, intermediateSize: Int) {
        self.hiddenSize = hiddenSize
        self.vocabSize = vocabSize
        self.intermediateSize = intermediateSize
    }
}

/// FastRL-style Single-Layer EAGLE Drafter
/// This is a VERY lightweight head that sits on top of the target model's hidden states.
/// Unlike a full drafter model, this only has:
/// 1. A fusion layer (hidden_state + embedding -> fused)
/// 2. An MLP layer for transformation
/// 3. A LM head for token prediction
///
/// Key insight from FastRL: We don't need a full model - just a thin layer that learns
/// to predict the next token distribution given the target's hidden states.
public class SingleLayerEAGLEDrafter: Module, @unchecked Sendable {
    let config: SingleLayerDrafterConfig
    
    // Embedding layer (shared with target or separate)
    let embedTokens: Embedding
    
    // EAGLE fusion: concatenate hidden_state + embedding, project back
    let fusionIn: Linear      // [hidden*2] -> [hidden]
    let fusionNorm: LayerNorm
    
    // Single MLP layer for transformation
    let mlpGate: Linear       // [hidden] -> [intermediate]
    let mlpUp: Linear         // [hidden] -> [intermediate]
    let mlpDown: Linear       // [intermediate] -> [hidden]
    let mlpNorm: LayerNorm
    
    // LM head for token prediction
    let lmHead: Linear        // [hidden] -> [vocab]
    
    public init(config: SingleLayerDrafterConfig) {
        self.config = config
        
        // Initialize embedding layer
        self.embedTokens = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        
        // Fusion layer: combines target hidden state with token embedding
        self.fusionIn = Linear(config.hiddenSize * 2, config.hiddenSize)
        self.fusionNorm = LayerNorm(dimensions: config.hiddenSize)
        
        // Single MLP layer (SwiGLU style like Qwen/Llama)
        self.mlpGate = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self.mlpUp = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self.mlpDown = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        self.mlpNorm = LayerNorm(dimensions: config.hiddenSize)
        
        // LM head
        self.lmHead = Linear(config.hiddenSize, config.vocabSize, bias: false)
        
        super.init()
    }
    
    /// Forward pass: predict next token logits from target hidden state and previous token
    /// - Parameters:
    ///   - hiddenState: Last hidden state from target model [batch, hidden] or [batch, seq, hidden]
    ///   - prevTokenId: Previous token ID for embedding lookup [batch] or [batch, seq]
    /// - Returns: Logits for next token prediction [batch, vocab] or [batch, seq, vocab]
    public func callAsFunction(hiddenState: MLXArray, prevTokenId: MLXArray) -> MLXArray {
        // Get embedding for previous token
        let embedding = embedTokens(prevTokenId)
        
        // Ensure dimensions match for concatenation
        var h = hiddenState
        var e = embedding
        
        // Handle different input shapes
        if h.ndim == 2 && e.ndim == 2 {
            // [batch, hidden] - single position
            // Nothing to do
        } else if h.ndim == 3 && e.ndim == 2 {
            // h is [batch, seq, hidden], e is [batch, hidden]
            e = e.expandedDimensions(axis: 1)
        } else if h.ndim == 2 && e.ndim == 3 {
            // h is [batch, hidden], e is [batch, seq, hidden]
            h = h.expandedDimensions(axis: 1)
        }
        
        // EAGLE fusion: concatenate hidden state and embedding
        let concat = MLX.concatenated([h, e], axis: -1)
        var x = fusionIn(concat)
        x = silu(x)
        x = fusionNorm(x)
        
        // Add residual from hidden state
        x = x + h
        
        // MLP layer (SwiGLU)
        let gate = silu(mlpGate(x))
        let up = mlpUp(x)
        let mlpOut = mlpDown(gate * up)
        x = mlpNorm(x + mlpOut)
        
        // LM head
        let logits = lmHead(x)
        
        return logits
    }
    
    /// Generate multiple draft tokens autoregressively
    /// - Parameters:
    ///   - initialHidden: Initial hidden state from target [batch, hidden]
    ///   - prevTokenId: Starting token ID [batch]
    ///   - numDrafts: Number of draft tokens to generate
    ///   - temperature: Sampling temperature
    ///   - topK: Top-k filtering (0 = disabled)
    /// - Returns: Array of draft token IDs and their predicted hidden states
    public func generateDrafts(
        initialHidden: MLXArray,
        prevTokenId: MLXArray,
        numDrafts: Int,
        temperature: Float = 1.0,
        topK: Int = 0
    ) -> (tokens: [Int32], hiddenStates: [MLXArray]) {
        var draftTokens: [Int32] = []
        var draftHiddens: [MLXArray] = []
        var currentHidden = initialHidden
        var currentToken = prevTokenId
        
        for _ in 0..<numDrafts {
            // Get logits
            let logits = self.callAsFunction(hiddenState: currentHidden, prevTokenId: currentToken)
            
            // Sample next token
            let nextToken = sampleFromLogits(logits.squeezed(axis: 0), temperature: temperature, topK: topK)
            draftTokens.append(nextToken)
            
            // Update hidden state prediction using the fusion output
            // For single-layer drafter, we approximate next hidden as the MLP output
            let embedding = embedTokens(MLXArray([nextToken]))
            let concat = MLX.concatenated([currentHidden, embedding], axis: -1)
            var x = fusionIn(concat)
            x = silu(x)
            x = fusionNorm(x)
            x = x + currentHidden
            
            let gate = silu(mlpGate(x))
            let up = mlpUp(x)
            let mlpOut = mlpDown(gate * up)
            currentHidden = mlpNorm(x + mlpOut)
            
            draftHiddens.append(currentHidden)
            currentToken = MLXArray([nextToken])
        }
        
        return (draftTokens, draftHiddens)
    }
    
    /// Sample a token from logits
    private func sampleFromLogits(_ logits: MLXArray, temperature: Float, topK: Int) -> Int32 {
        var sampledLogits = logits
        
        // Apply temperature
        if temperature > 0 && temperature != 1.0 {
            sampledLogits = sampledLogits / temperature
        }
        
        // Apply top-k filtering
        if topK > 0 && topK < logits.shape[0] {
            let sorted = MLX.sorted(sampledLogits)
            let threshold = sorted[sorted.shape[0] - topK]
            let mask = sampledLogits .< threshold
            sampledLogits = MLX.where(mask, MLXArray(-Float.infinity), sampledLogits)
        }
        
        // Sample
        if temperature <= 0 {
            // Greedy
            return MLX.argMax(sampledLogits).item(Int32.self)
        } else {
            // Categorical sampling
            let probs = softmax(sampledLogits)
            let sampled = MLXRandom.categorical(probs.expandedDimensions(axis: 0))
            return sampled.item(Int32.self)
        }
    }
    
    /// Compute training loss given target hidden states and target logits
    /// This is the key training objective from FastRL/EAGLE
    public func computeLoss(
        targetHiddenStates: MLXArray,  // [batch, seq, hidden] - from target during inference
        inputTokenIds: MLXArray,        // [batch, seq] - input tokens
        targetLogits: MLXArray          // [batch, seq, vocab] - target model's logits
    ) -> MLXArray {
        // Forward through drafter
        let predictedLogits = self.callAsFunction(hiddenState: targetHiddenStates, prevTokenId: inputTokenIds)
        
        // KL divergence loss between predicted and target distributions
        let targetProbs = softmax(targetLogits)
        let predictedLogProbs = MLX.log(softmax(predictedLogits) + 1e-10)
        
        // KL(target || predicted) = sum(target * log(target/predicted))
        let klLoss = MLX.sum(targetProbs * (MLX.log(targetProbs + 1e-10) - predictedLogProbs), axis: -1)
        
        return MLX.mean(klLoss)
    }
}

/// Actor managing the SingleLayerEAGLEDrafter with FastRL-style training
public actor FastRLDrafterActor {
    private var drafter: SingleLayerEAGLEDrafter
    private var optimizer: Adam
    private let config: SingleLayerDrafterConfig
    private let loadMonitor: GPULoadMonitor
    
    // Training state
    private var trainingStep: Int = 0
    private var lastLoss: Float = 0.0
    private var isTraining: Bool = false
    
    // Statistics
    private var totalDrafts: Int = 0
    private var acceptedDrafts: Int = 0
    
    // Data buffer for online training
    private var trainingBuffer: [(hiddenStates: MLXArray, tokenIds: MLXArray, targetLogits: MLXArray)] = []
    private let maxBufferSize: Int = 256
    
    public init(
        config: SingleLayerDrafterConfig,
        learningRate: Float = 1e-4,
        loadMonitor: GPULoadMonitor
    ) {
        self.config = config
        self.drafter = SingleLayerEAGLEDrafter(config: config)
        self.optimizer = Adam(learningRate: learningRate)
        self.loadMonitor = loadMonitor
    }
    
    /// Generate draft tokens
    public func generateDrafts(
        hiddenState: MLXArray,
        prevToken: MLXArray,
        count: Int,
        temperature: Float = 1.0,
        topK: Int = 8
    ) -> [Int32] {
        let (tokens, _) = drafter.generateDrafts(
            initialHidden: hiddenState,
            prevTokenId: prevToken,
            numDrafts: count,
            temperature: temperature,
            topK: topK
        )
        totalDrafts += count
        return tokens
    }
    
    /// Generate drafts with log probabilities for lossless verification
    public func generateDraftsWithLogProbs(
        hiddenState: MLXArray,
        prevToken: MLXArray,
        count: Int,
        temperature: Float = 1.0,
        topK: Int = 8
    ) -> (tokens: [Int32], logProbs: [Float]) {
        var draftTokens: [Int32] = []
        var logProbs: [Float] = []
        var currentHidden = hiddenState
        var currentToken = prevToken
        
        for _ in 0..<count {
            let logits = drafter(hiddenState: currentHidden, prevTokenId: currentToken).squeezed()
            let scaled = temperature > 0 ? logits / temperature : logits
            let probs = softmax(scaled)
            let nextToken = sampleToken(logits: logits, temperature: temperature, topK: topK)
            draftTokens.append(nextToken)
            let prob = probs[Int(nextToken)].item(Float.self)
            logProbs.append(log(max(prob, 1e-8)))
            
            let nextTokenArray = MLXArray([nextToken])
            // Update hidden by a forward pass on the predicted token embedding
            currentHidden = drafter(hiddenState: currentHidden, prevTokenId: nextTokenArray)
            currentToken = nextTokenArray
        }
        
        totalDrafts += count
        return (draftTokens, logProbs)
    }
    
    /// Get logits for a single position
    public func getLogits(hiddenState: MLXArray, prevToken: MLXArray) -> MLXArray {
        drafter(hiddenState: hiddenState, prevTokenId: prevToken)
    }
    
    private func sampleToken(logits: MLXArray, temperature: Float, topK: Int) -> Int32 {
        var processed = logits
        if topK > 0 && topK < logits.shape[0] {
            let sorted = MLX.sorted(processed)
            let threshold = sorted[sorted.shape[0] - topK]
            let mask = processed .< threshold
            processed = MLX.where(mask, MLXArray(-Float.infinity), processed)
        }
        return sampleToken(logits: processed, temperature: temperature)
    }
    
    private func sampleToken(logits: MLXArray, temperature: Float) -> Int32 {
        if temperature <= 0 {
            return Int32(MLX.argMax(logits).item(Int32.self))
        }
        let scaled = logits / temperature
        let probs = MLX.softmax(scaled)
        let sample = MLXRandom.categorical(probs.expandedDimensions(axis: 0))
        return sample.item(Int32.self)
    }
    
    /// Collect training data from target model inference
    /// This is called during inference to collect (hidden_state, token, target_logits) tuples
    public func collectTrainingData(
        hiddenStates: MLXArray,
        tokenIds: MLXArray,
        targetLogits: MLXArray
    ) {
        // Add to buffer
        trainingBuffer.append((hiddenStates, tokenIds, targetLogits))
        
        // Evict oldest if over capacity
        if trainingBuffer.count > maxBufferSize {
            trainingBuffer.removeFirst()
        }
    }
    
    /// Train on collected data (called during idle GPU time)
    public func trainStep(batchSize: Int = 8) async -> Float {
        // Check GPU load
        let isUnderLoad = await loadMonitor.checkUnderLoad()
        if isUnderLoad || trainingBuffer.isEmpty {
            return lastLoss
        }
        
        isTraining = true
        defer { isTraining = false }
        
        // Yield to inference
        await Task.yield()
        
        // Sample a batch
        let sampleCount = min(batchSize, trainingBuffer.count)
        var batchLoss: Float = 0.0
        
        for _ in 0..<sampleCount {
            let idx = Int.random(in: 0..<trainingBuffer.count)
            let sample = trainingBuffer[idx]
            
            let loss = drafter.computeLoss(
                targetHiddenStates: sample.hiddenStates,
                inputTokenIds: sample.tokenIds,
                targetLogits: sample.targetLogits
            )
            
            batchLoss += loss.item(Float.self)
        }
        
        lastLoss = batchLoss / Float(sampleCount)
        trainingStep += 1
        
        return lastLoss
    }
    
    /// Record acceptance statistics
    public func recordAcceptance(accepted: Int, total: Int) {
        acceptedDrafts += accepted
    }
    
    /// Get acceptance rate
    public func getAcceptanceRate() -> Float {
        guard totalDrafts > 0 else { return 0 }
        return Float(acceptedDrafts) / Float(totalDrafts)
    }
    
    /// Get training step
    public func getTrainingStep() -> Int {
        trainingStep
    }
    
    /// Get last loss
    public func getLastLoss() -> Float {
        lastLoss
    }
    
    /// Get buffer size
    public func getBufferSize() -> Int {
        trainingBuffer.count
    }
    
    /// Save drafter weights
    public func saveWeights(to url: URL) throws {
        let flatParams = drafter.parameters().flattened()
        var weights: [String: MLXArray] = [:]
        for (key, value) in flatParams {
            weights[key] = value
        }
        try save(arrays: weights, url: url)
    }
    
    /// Load drafter weights
    public func loadWeights(from url: URL) throws {
        let weights = try loadArrays(url: url)
        // Update drafter parameters directly
        let nestedWeights = ModuleParameters.unflattened(weights)
        drafter.update(parameters: nestedWeights)
        eval(drafter.parameters())
    }
    
    /// Clear training buffer
    public func clearBuffer() {
        trainingBuffer.removeAll()
    }
}

/// LoRA adapter for efficient fine-tuning
public class LoRAAdapter: Module, @unchecked Sendable {
    let loraA: Linear
    let loraB: Linear
    let scale: Float
    let rank: Int
    
    public init(inputDim: Int, outputDim: Int, rank: Int = 8, alpha: Float = 16.0) {
        self.rank = rank
        self.scale = alpha / Float(rank)
        self.loraA = Linear(inputDim, rank, bias: false)
        self.loraB = Linear(rank, outputDim, bias: false)
        super.init()
        
        // LoRA B is initialized to zero by Linear layer by default for stable training start
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // LoRA: W' = W + BA * scale
        let loraOutput = loraB(loraA(x))
        return loraOutput * scale
    }
}

/// Lightweight drafter model for speculative decoding
public class DrafterModel: Module, @unchecked Sendable {
    let config: DrafterConfig
    let embedTokens: Embedding
    let lmHead: Linear
    let eagleHead: EAGLEHead?
    let loraAdapters: [String: LoRAAdapter]
    
    public init(config: DrafterConfig, enableEAGLE: Bool = true, enableLoRA: Bool = false) {
        self.config = config
        self.embedTokens = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.lmHead = Linear(config.hiddenSize, config.vocabSize, bias: false)
        self.eagleHead = enableEAGLE ? EAGLEHead(hiddenSize: config.hiddenSize) : nil
        
        // Create LoRA adapters if enabled
        if enableLoRA {
            var adapters: [String: LoRAAdapter] = [:]
            adapters["lm_head"] = LoRAAdapter(
                inputDim: config.hiddenSize,
                outputDim: config.vocabSize,
                rank: 16
            )
            self.loraAdapters = adapters
        } else {
            self.loraAdapters = [:]
        }
        
        super.init()
    }
    
    /// Generate next token logits from hidden states
    public func callAsFunction(
        _ hiddenStates: MLXArray,
        previousEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        var x = hiddenStates
        
        // Apply EAGLE head if available and we have previous embeddings
        if let eagle = eagleHead, let prevEmbed = previousEmbeddings {
            x = eagle(x, embeddings: prevEmbed)
        }
        
        // Get logits
        var logits = lmHead(x)
        
        // Apply LoRA if available
        if let loraHead = loraAdapters["lm_head"] {
            logits = logits + loraHead(x)
        }
        
        return logits
    }
    
    /// Get embeddings for token ids
    public func getEmbeddings(_ tokenIds: MLXArray) -> MLXArray {
        embedTokens(tokenIds)
    }
}

/// Actor managing the Drafter model with thread-safe weight updates
public actor DrafterActor {
    private var model: DrafterModel
    private var optimizer: Adam
    private let config: DrafterConfig
    private let loadMonitor: GPULoadMonitor
    
    // Training state
    private var trainingStep: Int = 0
    private var lastLoss: Float = 0.0
    private var isTrainingEnabled: Bool
    
    // Statistics
    private var totalDrafts: Int = 0
    private var acceptedDrafts: Int = 0
    
    public init(
        config: DrafterConfig,
        enableEAGLE: Bool,
        enableLoRA: Bool,
        learningRate: Float = 1e-4,
        loadMonitor: GPULoadMonitor
    ) {
        self.config = config
        self.model = DrafterModel(config: config, enableEAGLE: enableEAGLE, enableLoRA: enableLoRA)
        self.optimizer = Adam(learningRate: learningRate)
        self.loadMonitor = loadMonitor
        self.isTrainingEnabled = enableEAGLE || enableLoRA
    }
    

    // Inference (High Priority)
    
    /// Generate draft tokens for speculation
    /// - Parameters:
    ///   - hiddenStates: Hidden states from target model
    ///   - previousTokens: Previously generated tokens for embeddings
    ///   - count: Number of draft tokens to generate (k)
    ///   - temperature: Sampling temperature
    /// - Returns: Array of draft token ids
    public func generateDraft(
        hiddenStates: MLXArray,
        previousTokens: MLXArray,
        count k: Int = 5,
        temperature: Float = 0.7
    ) -> [Int32] {
        var draftTokens: [Int32] = []
        var currentHidden = hiddenStates
        var currentTokens = previousTokens
        
        for _ in 0..<k {
            // Get embeddings for context
            let embeddings = model.getEmbeddings(currentTokens)
            let lastEmbedding = embeddings[embeddings.shape[0] - 1]
            
            // Get logits
            let lastHidden = currentHidden[currentHidden.shape[0] - 1]
            let logits = model(lastHidden.expandedDimensions(axis: 0), previousEmbeddings: lastEmbedding.expandedDimensions(axis: 0))
            
            // Sample next token
            let nextToken = sampleToken(logits: logits.squeezed(), temperature: temperature)
            draftTokens.append(nextToken)
            
            // Update for next iteration
            let nextTokenArray = MLXArray([nextToken])
            let nextEmbedding = model.getEmbeddings(nextTokenArray)
            
            // Simple hidden state update (full model would have transformer layers)
            currentHidden = MLX.concatenated([currentHidden, nextEmbedding], axis: 0)
            currentTokens = MLX.concatenated([currentTokens, nextTokenArray], axis: 0)
        }
        
        totalDrafts += k
        return draftTokens
    }
    
    /// Generate drafts with log probabilities for lossless verification
    public func generateDraftWithLogProbs(
        hiddenStates: MLXArray,
        previousTokens: MLXArray,
        count k: Int = 5,
        temperature: Float = 0.7
    ) -> (tokens: [Int32], logProbs: [Float]) {
        var draftTokens: [Int32] = []
        var logProbs: [Float] = []
        var currentHidden = hiddenStates
        var currentTokens = previousTokens
        
        for _ in 0..<k {
            let embeddings = model.getEmbeddings(currentTokens)
            let lastEmbedding = embeddings[embeddings.shape[0] - 1]
            let lastHidden = currentHidden[currentHidden.shape[0] - 1]
            let logits = model(lastHidden.expandedDimensions(axis: 0), previousEmbeddings: lastEmbedding.expandedDimensions(axis: 0)).squeezed()
            
            // Temperature scaling
            let scaled = temperature > 0 ? logits / temperature : logits
            let probs = softmax(scaled)
            let nextToken = sampleToken(logits: logits, temperature: temperature)
            draftTokens.append(nextToken)
            
            let tokenProb = probs[Int(nextToken)].item(Float.self)
            logProbs.append(log(max(tokenProb, 1e-8)))
            
            let nextTokenArray = MLXArray([nextToken])
            let nextEmbedding = model.getEmbeddings(nextTokenArray)
            currentHidden = MLX.concatenated([currentHidden, nextEmbedding], axis: 0)
            currentTokens = MLX.concatenated([currentTokens, nextTokenArray], axis: 0)
        }
        
        totalDrafts += k
        return (draftTokens, logProbs)
    }
    
    /// Sample a token from logits
    private func sampleToken(logits: MLXArray, temperature: Float) -> Int32 {
        if temperature <= 0 {
            // Greedy sampling
            return Int32(MLX.argMax(logits).item(Int32.self))
        }
        
        // Temperature-scaled sampling
        let scaledLogits = logits / temperature
        let probs = MLX.softmax(scaledLogits)
        
        // Sample from distribution
        let sample = MLXRandom.categorical(probs.expandedDimensions(axis: 0))
        return sample.item(Int32.self)
    }
    

    // Training (Background Priority)
    
    /// Apply gradients from training - called from background task
    /// Uses cooperative yielding for M1/M2 compatibility
    public func applyGradients(grads: ModuleParameters) async {
        // Check if we should skip due to GPU load
        let isUnderLoad = await loadMonitor.checkUnderLoad()
        if isUnderLoad {
            return  // Skip training step to maintain inference performance
        }
        
        guard isTrainingEnabled else { return }
        
        // Yield to allow inference tasks priority
        await Task.yield()
        
        // Update with optimizer
        optimizer.update(model: model, gradients: grads)
        
        // Evaluate updated parameters
        eval(model.parameters())
        
        trainingStep += 1
    }
    
    /// Train on a batch of samples using EAGLE-style hidden state fusion
    /// The training objective combines:
    /// 1. Token prediction loss (cross-entropy with target logits)
    /// 2. Hidden state alignment loss (optional, for EAGLE training)
    public func trainStep(samples: [TrainingSample]) async -> Float {
        guard isTrainingEnabled, !samples.isEmpty else { return 0 }
        
        let isUnderLoad = await loadMonitor.checkUnderLoad()
        if isUnderLoad {
            return lastLoss
        }
        
        // Yield for inference priority
        await Task.yield()
        
        var totalLoss: Float = 0.0
        
        for sample in samples {
            // Get token embeddings for the input sequence
            let embeddings = model.getEmbeddings(sample.inputIds)
            
            // Forward through drafter model with EAGLE head
            let predictedLogits = model(sample.hiddenStates, previousEmbeddings: embeddings)
            
            // Primary loss: Cross-entropy with target logits (KL divergence proxy)
            let targetProbs = softmax(sample.targetLogits)
            let predictedProbs = softmax(predictedLogits)
            let logPredictedProbs = log(predictedProbs + 1e-10)  // Add small epsilon for numerical stability
            let ceLoss = -sum(targetProbs * logPredictedProbs)
            
            // EAGLE-style hidden state alignment loss (if EAGLE head is enabled)
            // This helps the drafter predict hidden states that match the target
            var hiddenLoss: MLXArray = MLXArray(0.0)
            if model.eagleHead != nil {
                // Get the predicted hidden states from EAGLE head
                if let eagle = model.eagleHead {
                    let predictedHidden = eagle(sample.hiddenStates, embeddings: embeddings)
                    
                    // Use MSE loss for hidden state alignment
                    // The target hidden states are from the sample
                    let diff = predictedHidden - sample.hiddenStates
                    hiddenLoss = mean(diff * diff)
                }
            }
            
            // Combine losses with weighting
            // Token prediction is primary, hidden alignment is secondary
            let sampleLoss = ceLoss + 0.1 * hiddenLoss
            let normalizedLoss = sampleLoss / Float(predictedLogits.shape[0])
            totalLoss += normalizedLoss.item(Float.self)
        }
        
        lastLoss = totalLoss / Float(samples.count)
        trainingStep += 1
        
        return lastLoss
    }
    
    /// Train with explicit EAGLE loss using collected hidden states
    /// This is the preferred training method when we have paired (input_hidden, target_hidden) data
    public func trainStepEAGLE(
        inputHiddenStates: MLXArray,
        targetHiddenStates: MLXArray,
        inputTokens: MLXArray,
        targetLogits: MLXArray
    ) async -> Float {
        guard isTrainingEnabled else { return 0 }
        guard let eagle = model.eagleHead else {
            // Fall back to standard training if no EAGLE head
            let sample = TrainingSample(
                hiddenStates: inputHiddenStates,
                targetLogits: targetLogits,
                inputIds: inputTokens
            )
            return await trainStep(samples: [sample])
        }
        
        let isUnderLoad = await loadMonitor.checkUnderLoad()
        if isUnderLoad {
            return lastLoss
        }
        
        await Task.yield()
        
        // Get embeddings
        let embeddings = model.getEmbeddings(inputTokens)
        
        // Forward through EAGLE head to predict next hidden states
        let predictedHidden = eagle(inputHiddenStates, embeddings: embeddings)
        
        // Compute EAGLE loss
        let loss = EAGLEHead.computeEAGLELoss(
            predictedHidden: predictedHidden,
            targetHidden: targetHiddenStates,
            targetLogits: targetLogits,
            lmHead: model.lmHead
        )
        
        lastLoss = loss.item(Float.self)
        trainingStep += 1
        
        return lastLoss
    }
    

    // Statistics
    
    /// Record draft acceptance
    public func recordAcceptance(accepted: Int, total: Int) {
        acceptedDrafts += accepted
        // totalDrafts already updated in generateDraft
    }
    
    /// Get acceptance rate
    public func getAcceptanceRate() -> Float {
        guard totalDrafts > 0 else { return 0 }
        return Float(acceptedDrafts) / Float(totalDrafts)
    }
    
    /// Get current training step
    public func getTrainingStep() -> Int {
        trainingStep
    }
    
    /// Get last training loss
    public func getLastLoss() -> Float {
        lastLoss
    }
    

    // Model Management
    
    /// Save model weights to disk using MLX safetensors format.
    /// Call this periodically during training to checkpoint progress.
    public func saveWeights(to url: URL) throws {
        let flatParams = model.parameters().flattened()
        var weights: [String: MLXArray] = [:]
        for (key, value) in flatParams {
            weights[key] = value
        }
        try save(arrays: weights, url: url)
    }
    
    /// Load model weights from disk.
    /// Use to restore from a checkpoint or load pre-trained weights.
    public func loadWeights(from url: URL) throws {
        let loaded = try loadArrays(url: url)
        
        // Convert to ModuleParameters format
        var params = ModuleParameters()
        for (key, array) in loaded {
            params[key] = .value(array)
        }
        model.update(parameters: params)
        
        // Re-evaluate parameters
        eval(model.parameters())
    }
    
    /// Save LoRA adapters only (for efficient checkpointing)
    public func saveLoRAWeights(to url: URL) throws {
        guard !model.loraAdapters.isEmpty else { return }
        
        var loraWeights: [String: MLXArray] = [:]
        for (name, adapter) in model.loraAdapters {
            let adapterParams = adapter.parameters().flattened()
            for (key, value) in adapterParams {
                loraWeights["\(name).\(key)"] = value
            }
        }
        try save(arrays: loraWeights, url: url)
    }
    
    /// Load LoRA adapters only
    public func loadLoRAWeights(from url: URL) throws {
        guard !model.loraAdapters.isEmpty else { return }
        
        let loaded = try loadArrays(url: url)
        
        for (name, adapter) in model.loraAdapters {
            var adapterParams = ModuleParameters()
            for (key, array) in loaded {
                if key.hasPrefix("\(name).") {
                    let subKey = String(key.dropFirst(name.count + 1))
                    adapterParams[subKey] = .value(array)
                }
            }
            adapter.update(parameters: adapterParams)
        }
        
        eval(model.parameters())
    }
    
    /// Get model for direct access (inference only)
    public func getModel() -> DrafterModel {
        model
    }
    
    /// Check if training is enabled
    public func isTraining() -> Bool {
        isTrainingEnabled
    }
    
    /// Enable/disable training
    public func setTrainingEnabled(_ enabled: Bool) {
        isTrainingEnabled = enabled
    }
}

// Multi-Drafter Ensemble

/// Strategy for selecting drafters in an ensemble
public enum DrafterSelectionStrategy: Sendable {
    /// Round-robin selection
    case roundRobin
    /// Select based on historical acceptance rate
    case bestPerformer
    /// Domain-based routing (code, math, general, etc.)
    case domainBased
    /// Random selection with weighted probabilities
    case weighted([Float])
    /// Use all drafters and vote
    case voting
}

/// Specialization domain for domain-based routing
public enum DrafterDomain: String, Sendable, CaseIterable {
    case general = "general"
    case code = "code"
    case math = "math"
    case reasoning = "reasoning"
    case creative = "creative"
}

/// Performance statistics for a drafter
public struct DrafterPerformanceStats: Sendable {
    public let name: String
    public let totalDrafts: Int
    public let acceptedDrafts: Int
    public let averageAcceptanceRate: Float
    public let recentAcceptanceRate: Float  // Last 100 drafts
    public let domain: DrafterDomain?
    
    public init(
        name: String,
        totalDrafts: Int,
        acceptedDrafts: Int,
        recentAcceptanceRate: Float,
        domain: DrafterDomain? = nil
    ) {
        self.name = name
        self.totalDrafts = totalDrafts
        self.acceptedDrafts = acceptedDrafts
        self.averageAcceptanceRate = totalDrafts > 0 ? Float(acceptedDrafts) / Float(totalDrafts) : 0
        self.recentAcceptanceRate = recentAcceptanceRate
        self.domain = domain
    }
}

/// Manages an ensemble of drafter models
public actor DrafterEnsemble {
    /// Individual drafters with their metadata
    private var drafters: [(drafter: DrafterActor, name: String, domain: DrafterDomain?)] = []
    
    /// Selection strategy
    private var strategy: DrafterSelectionStrategy
    
    /// Performance tracking
    private var performanceHistory: [String: [Float]] = [:]
    private let historySize = 100
    
    /// Round-robin index
    private var roundRobinIndex = 0
    
    /// Load monitor for ensemble coordination
    private let loadMonitor: GPULoadMonitor
    
    public init(
        strategy: DrafterSelectionStrategy = .bestPerformer,
        loadMonitor: GPULoadMonitor
    ) {
        self.strategy = strategy
        self.loadMonitor = loadMonitor
    }
    
    /// Add a drafter to the ensemble
    public func addDrafter(
        _ drafter: DrafterActor,
        name: String,
        domain: DrafterDomain? = nil
    ) {
        drafters.append((drafter, name, domain))
        performanceHistory[name] = []
    }
    
    /// Remove a drafter from the ensemble
    public func removeDrafter(named name: String) {
        drafters.removeAll { $0.name == name }
        performanceHistory.removeValue(forKey: name)
    }
    
    /// Get all drafter names
    public func getDrafterNames() -> [String] {
        drafters.map { $0.name }
    }
    
    /// Set selection strategy
    public func setStrategy(_ strategy: DrafterSelectionStrategy) {
        self.strategy = strategy
    }
    
    /// Select a drafter based on the current strategy
    public func selectDrafter(context: String? = nil) async -> DrafterActor? {
        guard !drafters.isEmpty else { return nil }
        
        switch strategy {
        case .roundRobin:
            return selectRoundRobin()
            
        case .bestPerformer:
            return await selectBestPerformer()
            
        case .domainBased:
            return selectByDomain(context: context)
            
        case .weighted(let weights):
            return selectWeighted(weights: weights)
            
        case .voting:
            // For voting, return the first drafter (actual voting happens in generateWithVoting)
            return drafters.first?.drafter
        }
    }
    
    /// Generate draft tokens using the selected strategy
    public func generateDraft(
        hiddenStates: MLXArray,
        previousTokens: MLXArray,
        count: Int,
        temperature: Float,
        context: String? = nil
    ) async -> [Int32] {
        switch strategy {
        case .voting:
            return await generateWithVoting(
                hiddenStates: hiddenStates,
                previousTokens: previousTokens,
                count: count,
                temperature: temperature
            )
            
        default:
            guard let drafter = await selectDrafter(context: context) else {
                return []
            }
            return await drafter.generateDraft(
                hiddenStates: hiddenStates,
                previousTokens: previousTokens,
                count: count,
                temperature: temperature
            )
        }
    }
    
    /// Generate using voting from all drafters
    private func generateWithVoting(
        hiddenStates: MLXArray,
        previousTokens: MLXArray,
        count: Int,
        temperature: Float
    ) async -> [Int32] {
        // Collect drafts from all drafters
        var allDrafts: [[Int32]] = []
        
        await withTaskGroup(of: [Int32].self) { group in
            for (drafter, _, _) in drafters {
                group.addTask {
                    await drafter.generateDraft(
                        hiddenStates: hiddenStates,
                        previousTokens: previousTokens,
                        count: count,
                        temperature: temperature
                    )
                }
            }
            
            for await draft in group {
                allDrafts.append(draft)
            }
        }
        
        // Vote on each position
        var result: [Int32] = []
        for position in 0..<count {
            var votes: [Int32: Int] = [:]
            for draft in allDrafts {
                if position < draft.count {
                    votes[draft[position], default: 0] += 1
                }
            }
            
            // Select token with most votes
            if let winner = votes.max(by: { $0.value < $1.value })?.key {
                result.append(winner)
            } else {
                break
            }
        }
        
        return result
    }
    
    /// Round-robin selection
    private func selectRoundRobin() -> DrafterActor {
        let selected = drafters[roundRobinIndex].drafter
        roundRobinIndex = (roundRobinIndex + 1) % drafters.count
        return selected
    }
    
    /// Select the best performing drafter
    private func selectBestPerformer() async -> DrafterActor {
        var bestDrafter = drafters.first!.drafter
        var bestRate: Float = -1
        
        for (drafter, name, _) in drafters {
            let rate = getRecentAcceptanceRate(for: name)
            if rate > bestRate {
                bestRate = rate
                bestDrafter = drafter
            }
        }
        
        return bestDrafter
    }
    
    /// Select drafter by domain
    private func selectByDomain(context: String?) -> DrafterActor {
        guard let context = context else {
            return drafters.first!.drafter
        }
        
        // Simple domain detection
        let domain = detectDomain(from: context)
        
        // Find drafter specialized for this domain
        for (drafter, _, drafterDomain) in drafters {
            if drafterDomain == domain {
                return drafter
            }
        }
        
        // Fall back to general or first drafter
        for (drafter, _, drafterDomain) in drafters {
            if drafterDomain == .general {
                return drafter
            }
        }
        
        return drafters.first!.drafter
    }
    
    /// Select drafter with weighted random selection
    private func selectWeighted(weights: [Float]) -> DrafterActor {
        let normalizedWeights = weights.prefix(drafters.count)
        let totalWeight = normalizedWeights.reduce(0, +)
        let random = Float.random(in: 0..<totalWeight)
        
        var cumulative: Float = 0
        for (index, weight) in normalizedWeights.enumerated() {
            cumulative += weight
            if random < cumulative {
                return drafters[index].drafter
            }
        }
        
        return drafters.last!.drafter
    }
    
    /// Detect domain from context
    private func detectDomain(from context: String) -> DrafterDomain {
        let lowered = context.lowercased()
        
        // Code detection
        if lowered.contains("function") || lowered.contains("def ") ||
           lowered.contains("class ") || lowered.contains("import ") ||
           lowered.contains("```") {
            return .code
        }
        
        // Math detection
        if lowered.contains("calculate") || lowered.contains("equation") ||
           lowered.contains("solve") || lowered.contains("prove") ||
           context.contains("=") && context.contains("+") {
            return .math
        }
        
        // Reasoning detection
        if lowered.contains("explain") || lowered.contains("why") ||
           lowered.contains("reason") || lowered.contains("because") {
            return .reasoning
        }
        
        // Creative detection
        if lowered.contains("write a story") || lowered.contains("poem") ||
           lowered.contains("creative") || lowered.contains("imagine") {
            return .creative
        }
        
        return .general
    }
    
    /// Record acceptance rate for a drafter
    public func recordAcceptance(drafterName: String, accepted: Int, total: Int) {
        guard total > 0 else { return }
        
        let rate = Float(accepted) / Float(total)
        
        if var history = performanceHistory[drafterName] {
            history.append(rate)
            if history.count > historySize {
                history.removeFirst()
            }
            performanceHistory[drafterName] = history
        }
    }
    
    /// Get recent acceptance rate for a drafter
    public func getRecentAcceptanceRate(for name: String) -> Float {
        guard let history = performanceHistory[name], !history.isEmpty else {
            return 0.5  // Default rate for new drafters
        }
        return history.reduce(0, +) / Float(history.count)
    }
    
    /// Get performance statistics for all drafters
    public func getPerformanceStats() async -> [DrafterPerformanceStats] {
        var stats: [DrafterPerformanceStats] = []
        
        for (drafter, name, domain) in drafters {
            let acceptanceRate = await drafter.getAcceptanceRate()
            let recentRate = getRecentAcceptanceRate(for: name)
            
            // Estimate total drafts from rate (simplified)
            let totalDrafts = performanceHistory[name]?.count ?? 0
            let acceptedDrafts = Int(Float(totalDrafts) * acceptanceRate)
            
            stats.append(DrafterPerformanceStats(
                name: name,
                totalDrafts: totalDrafts * 5,  // Assuming ~5 tokens per draft
                acceptedDrafts: acceptedDrafts * 5,
                recentAcceptanceRate: recentRate,
                domain: domain
            ))
        }
        
        return stats
    }
    
    /// Train all drafters with shared training data
    public func trainAll(samples: [TrainingSample]) async {
        await withTaskGroup(of: Void.self) { group in
            for (drafter, _, _) in drafters {
                group.addTask {
                    let _ = await drafter.trainStep(samples: samples)
                }
            }
        }
    }
    
    /// Get the number of drafters in the ensemble
    public func count() -> Int {
        drafters.count
    }
}
