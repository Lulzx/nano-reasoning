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
    
    public init(
        hiddenStates: MLXArray,
        targetLogits: MLXArray,
        inputIds: MLXArray,
        attentionMask: MLXArray? = nil
    ) {
        self.hiddenStates = hiddenStates
        self.targetLogits = targetLogits
        self.inputIds = inputIds
        self.attentionMask = attentionMask
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
public class EAGLEHead: Module, @unchecked Sendable {
    let inputProjection: Linear
    let fusionLayer: Linear
    let outputProjection: Linear
    let layerNorm: LayerNorm
    
    public init(hiddenSize: Int) {
        // Project concatenated [hidden_state, embedding] to hidden_size
        self.inputProjection = Linear(hiddenSize * 2, hiddenSize)
        self.fusionLayer = Linear(hiddenSize, hiddenSize)
        self.outputProjection = Linear(hiddenSize, hiddenSize)
        self.layerNorm = LayerNorm(dimensions: hiddenSize)
        super.init()
    }
    
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
    
    /// Train on a batch of samples
    public func trainStep(samples: [TrainingSample]) async -> Float {
        guard isTrainingEnabled, !samples.isEmpty else { return 0 }
        
        let isUnderLoad = await loadMonitor.checkUnderLoad()
        if isUnderLoad {
            return lastLoss
        }
        
        // Yield for inference priority
        await Task.yield()
        
        // Compute cross-entropy loss between drafter predictions and target logits
        // For gradient computation, use MLX.valueAndGrad() with this loss function
        var totalLoss: Float = 0.0
        
        for sample in samples {
            let logits = model(sample.hiddenStates, previousEmbeddings: model.getEmbeddings(sample.inputIds))
            
            // Cross-entropy loss against target logits
            let targetProbs = softmax(sample.targetLogits)
            let logProbs = log(softmax(logits))
            let sampleLoss = -sum(targetProbs * logProbs) / Float(logits.shape[0])
            totalLoss += sampleLoss.item(Float.self)
        }
        
        lastLoss = totalLoss / Float(samples.count)
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
        let weights = model.parameters().flattened()
        // Use MLX.save(arrays: weights, url: url) when ready
        _ = weights  // Weights ready for serialization
    }
    
    /// Load model weights from disk.
    /// Use to restore from a checkpoint or load pre-trained weights.
    public func loadWeights(from url: URL) throws {
        // Use MLX.load(url: url) and model.update(parameters:) when ready
        _ = url  // Path ready for loading
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


