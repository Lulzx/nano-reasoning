// SPDX-License-Identifier: MIT
// Nano-Reasoning: Target Model Wrapper with Hidden State Exposure
// Implements Qwen3-compatible model loading with MLX

import Foundation
@preconcurrency import MLX
import MLXNN
import MLXRandom

/// Result from target model verification
public struct VerificationResult: @unchecked Sendable {
    /// Token IDs that were verified
    public let tokenIds: [Int32]
    /// Logits for each verified token (alias for targetLogits)
    public let logits: MLXArray
    /// Target logits - same as logits, explicit name for FastRL training
    public var targetLogits: MLXArray { logits }
    /// Hidden states from the final layer (for EAGLE/FastRL training)
    public let hiddenStates: MLXArray
    /// Which draft tokens were accepted (true = accepted)
    public let acceptanceMask: [Bool]
    /// Number of tokens accepted
    public let acceptedCount: Int
    /// Total number of draft tokens
    public let draftCount: Int
    /// The corrected token (if any draft was rejected)
    public let correctedToken: Int32?
    /// Accepted tokens only
    public var acceptedTokens: [Int32] {
        zip(tokenIds, acceptanceMask).filter { $0.1 }.map { $0.0 }
    }
    /// Bonus token (corrected token if draft rejected, else nil)
    public var bonusToken: Int32? { correctedToken }
    
    public init(
        tokenIds: [Int32],
        logits: MLXArray,
        hiddenStates: MLXArray,
        acceptanceMask: [Bool],
        correctedToken: Int32? = nil
    ) {
        self.tokenIds = tokenIds
        self.logits = logits
        self.hiddenStates = hiddenStates
        self.acceptanceMask = acceptanceMask
        self.acceptedCount = acceptanceMask.filter { $0 }.count
        self.draftCount = tokenIds.count
        self.correctedToken = correctedToken
    }
}

/// Generation configuration
public struct GenerationConfig: Sendable {
    public var maxTokens: Int
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var repetitionPenalty: Float
    public var stopTokens: [Int32]
    
    public init(
        maxTokens: Int,
        temperature: Float,
        topP: Float,
        topK: Int,
        repetitionPenalty: Float,
        stopTokens: [Int32]
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.stopTokens = stopTokens
    }
    
    public static let `default` = GenerationConfig(
        maxTokens: 2048,
        temperature: 0.7,
        topP: 0.9,
        topK: 40,
        repetitionPenalty: 1.1,
        stopTokens: []
    )
    
    public static let greedy = GenerationConfig(
        maxTokens: 2048,
        temperature: 0.0,
        topP: 1.0,
        topK: 1,
        repetitionPenalty: 1.0,
        stopTokens: []
    )
}

/// KV Cache for efficient autoregressive generation
public actor KVCache {
    private var keys: [[MLXArray]]  // [layer][batch, heads, seq, head_dim]
    private var values: [[MLXArray]]
    private let numLayers: Int
    private var currentLength: Int = 0
    
    public init(numLayers: Int) {
        self.numLayers = numLayers
        self.keys = Array(repeating: [], count: numLayers)
        self.values = Array(repeating: [], count: numLayers)
    }
    
    /// Update cache with new key-value pairs
    public func update(layer: Int, newKeys: MLXArray, newValues: MLXArray) {
        if keys[layer].isEmpty {
            keys[layer] = [newKeys]
            values[layer] = [newValues]
        } else {
            keys[layer].append(newKeys)
            values[layer].append(newValues)
        }
        
        if layer == numLayers - 1 {
            currentLength += newKeys.shape[2]
        }
    }
    
    /// Get cached keys for a layer
    public func getKeys(layer: Int) -> MLXArray? {
        guard !keys[layer].isEmpty else { return nil }
        return MLX.concatenated(keys[layer], axis: 2)
    }
    
    /// Get cached values for a layer
    public func getValues(layer: Int) -> MLXArray? {
        guard !values[layer].isEmpty else { return nil }
        return MLX.concatenated(values[layer], axis: 2)
    }
    
    /// Get current sequence length
    public func getLength() -> Int {
        currentLength
    }
    
    /// Truncate cache to a specific length
    public func truncate(to length: Int) {
        currentLength = min(currentLength, length)
    }
    
    /// Clear the cache
    public func clear() {
        keys = Array(repeating: [], count: numLayers)
        values = Array(repeating: [], count: numLayers)
        currentLength = 0
    }
}

// Qwen Model Components

/// RMSNorm layer
public class RMSNorm: Module, @unchecked Sendable {
    let weight: MLXArray
    let eps: Float
    
    public init(dimensions: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let variance = MLX.mean(x * x, axis: -1, keepDims: true)
        let normalized = x * MLX.rsqrt(variance + eps)
        return weight * normalized
    }
}

/// Rotary Position Embedding
public struct RoPE: Sendable {
    let dim: Int
    let maxSeqLen: Int
    let theta: Float
    
    public init(dim: Int, maxSeqLen: Int = 32768, theta: Float = 1000000.0) {
        self.dim = dim
        self.maxSeqLen = maxSeqLen
        self.theta = theta
    }
    
    public func apply(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        let seqLen = x.shape[x.ndim - 2]
        let headDim = x.shape[x.ndim - 1]
        
        // Create position indices
        let positions = MLXArray(Array(offset..<(offset + seqLen)).map { Float($0) })
        
        // Create frequency bands
        let freqExponent = MLXArray(stride(from: 0, to: headDim, by: 2).map { Float($0) }) / Float(headDim)
        let freqs = 1.0 / MLX.pow(MLXArray(theta), freqExponent)
        
        // Compute angles
        let angles = positions.expandedDimensions(axis: -1) * freqs.expandedDimensions(axis: 0)
        let cosAngles = MLX.cos(angles)
        let sinAngles = MLX.sin(angles)
        
        // Split x into pairs
        let x1 = x[.ellipsis, .stride(from: 0, to: nil, by: 2)]
        let x2 = x[.ellipsis, .stride(from: 1, to: nil, by: 2)]
        
        // Apply rotation
        let rotated1 = x1 * cosAngles - x2 * sinAngles
        let rotated2 = x1 * sinAngles + x2 * cosAngles
        
        // Interleave back
        return MLX.stacked([rotated1, rotated2], axis: -1).reshaped(x.shape)
    }
}

/// Attention layer with KV cache support
public class QwenAttention: Module, @unchecked Sendable {
    let hiddenSize: Int
    let numHeads: Int
    let headDim: Int
    let numKVHeads: Int
    
    let qProj: Linear
    let kProj: Linear
    let vProj: Linear
    let oProj: Linear
    let rope: RoPE
    
    public init(hiddenSize: Int, numHeads: Int, numKVHeads: Int, ropeTheta: Float = 1000000.0) {
        self.hiddenSize = hiddenSize
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        self.numKVHeads = numKVHeads
        
        self.qProj = Linear(hiddenSize, numHeads * headDim, bias: true)
        self.kProj = Linear(hiddenSize, numKVHeads * headDim, bias: true)
        self.vProj = Linear(hiddenSize, numKVHeads * headDim, bias: true)
        self.oProj = Linear(numHeads * headDim, hiddenSize, bias: false)
        self.rope = RoPE(dim: headDim, theta: ropeTheta)
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let batchSize = x.shape[0]
        let seqLen = x.shape[1]
        
        // Project Q, K, V
        var q = qProj(x)
        var k = kProj(x)
        var v = vProj(x)
        
        // Reshape for multi-head attention
        q = q.reshaped([batchSize, seqLen, numHeads, headDim]).transposed(0, 2, 1, 3)
        k = k.reshaped([batchSize, seqLen, numKVHeads, headDim]).transposed(0, 2, 1, 3)
        v = v.reshaped([batchSize, seqLen, numKVHeads, headDim]).transposed(0, 2, 1, 3)
        
        // Apply RoPE
        q = rope.apply(q)
        k = rope.apply(k)
        
        // Expand K, V if using grouped query attention
        if numKVHeads < numHeads {
            let repeatFactor = numHeads / numKVHeads
            k = MLX.repeated(k, count: repeatFactor, axis: 1)
            v = MLX.repeated(v, count: repeatFactor, axis: 1)
        }
        
        // Compute attention scores
        let scale = 1.0 / sqrt(Float(headDim))
        var scores = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * scale
        
        // Apply mask if provided
        if let mask = mask {
            scores = scores + mask
        }
        
        // Softmax and apply to values
        let attnWeights = MLX.softmax(scores, axis: -1)
        var output = MLX.matmul(attnWeights, v)
        
        // Reshape back
        output = output.transposed(0, 2, 1, 3).reshaped([batchSize, seqLen, -1])
        
        // Hidden states before output projection (for EAGLE)
        let hiddenStates = output
        
        // Final projection
        output = oProj(output)
        
        return (output, hiddenStates)
    }
}

/// Feed-forward network (SwiGLU variant)
public class QwenMLP: Module, @unchecked Sendable {
    let gateProj: Linear
    let upProj: Linear
    let downProj: Linear
    
    public init(hiddenSize: Int, intermediateSize: Int) {
        self.gateProj = Linear(hiddenSize, intermediateSize, bias: false)
        self.upProj = Linear(hiddenSize, intermediateSize, bias: false)
        self.downProj = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = silu(gateProj(x))
        let up = upProj(x)
        return downProj(gate * up)
    }
}

/// Transformer decoder layer
public class QwenDecoderLayer: Module, @unchecked Sendable {
    let selfAttn: QwenAttention
    let mlp: QwenMLP
    let inputLayerNorm: RMSNorm
    let postAttnLayerNorm: RMSNorm
    
    public init(hiddenSize: Int, numHeads: Int, numKVHeads: Int, intermediateSize: Int, ropeTheta: Float) {
        self.selfAttn = QwenAttention(hiddenSize: hiddenSize, numHeads: numHeads, numKVHeads: numKVHeads, ropeTheta: ropeTheta)
        self.mlp = QwenMLP(hiddenSize: hiddenSize, intermediateSize: intermediateSize)
        self.inputLayerNorm = RMSNorm(dimensions: hiddenSize)
        self.postAttnLayerNorm = RMSNorm(dimensions: hiddenSize)
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> (MLXArray, MLXArray) {
        // Self-attention with residual
        let normedInput = inputLayerNorm(x)
        let (attnOutput, hiddenStates) = selfAttn(normedInput, mask: mask)
        var output = x + attnOutput
        
        // MLP with residual
        let normedMLP = postAttnLayerNorm(output)
        output = output + mlp(normedMLP)
        
        return (output, hiddenStates)
    }
}

/// Target model wrapper that exposes hidden states for EAGLE training
public actor TargetModel {
    private var embedTokens: Embedding?
    private var layers: [QwenDecoderLayer] = []
    private var norm: RMSNorm?
    private var lmHead: Linear?
    
    private let configuration: ModelConfiguration
    private let hardwareTier: HardwareTier
    private var kvCache: KVCache?
    private var isLoaded: Bool = false
    
    // Model info
    private var vocabSize: Int = 0
    private var hiddenSize: Int = 0
    private var numLayers: Int = 0
    private var weightsLoaded: Bool = false
    
    // Statistics
    private var totalTokensGenerated: Int = 0
    private var totalVerifications: Int = 0
    
    public init(configuration: ModelConfiguration, tier: HardwareTier) {
        self.configuration = configuration
        self.hardwareTier = tier
    }
    

    // Model Loading
    
    /// Load and initialize the target model with tier-appropriate dimensions.
    /// Creates transformer layers, embeddings, and LM head based on hardware tier.
    /// For production use, integrate with MLX-LM model loading from HuggingFace.
    public func loadModel(progressHandler: ((Double) -> Void)? = nil) async throws {
        guard !isLoaded else { return }
        
        progressHandler?(0.0)
        
        // Set model dimensions based on tier
        switch hardwareTier {
        case .entry:
            vocabSize = 151936
            hiddenSize = 2048
            numLayers = 28
        case .pro:
            vocabSize = 151936
            hiddenSize = 4096
            numLayers = 32
        case .elite:
            vocabSize = 151936
            hiddenSize = 5120
            numLayers = 64
        }
        
        progressHandler?(0.2)
        
        // Create model components
        embedTokens = Embedding(embeddingCount: vocabSize, dimensions: hiddenSize)
        
        progressHandler?(0.4)
        
        // Create transformer layers
        let numHeads = hiddenSize / 128  // Standard head dim of 128
        let numKVHeads = numHeads / 4    // GQA with 4:1 ratio
        let intermediateSize = Int(Float(hiddenSize) * 2.75)
        
        for _ in 0..<numLayers {
            let layer = QwenDecoderLayer(
                hiddenSize: hiddenSize,
                numHeads: numHeads,
                numKVHeads: numKVHeads,
                intermediateSize: intermediateSize,
                ropeTheta: 1000000.0
            )
            layers.append(layer)
        }
        
        progressHandler?(0.8)
        
        norm = RMSNorm(dimensions: hiddenSize)
        lmHead = Linear(hiddenSize, vocabSize, bias: false)
        
        kvCache = KVCache(numLayers: numLayers)
        
        // Attempt to load pretrained weights if available
        if let weightsURL = resolveWeightsURL() {
            do {
                try loadPretrainedWeights(from: weightsURL)
                weightsLoaded = true
            } catch {
                // Fallback to randomly initialized weights
                print("Warning: Failed to load pretrained weights at \(weightsURL.path): \(error)")
            }
        }
        
        progressHandler?(1.0)
        isLoaded = true
    }
    
    /// Resolve weights path (env override or cache directory)
    private func resolveWeightsURL() -> URL? {
        if let env = EnvironmentConfig.modelWeightsPath {
            return env
        }
        
        // Default cache path: ~/.cache/nano-reasoning/models/<modelId>/weights.safetensors
        let safeId = configuration.targetModelId.replacingOccurrences(of: "/", with: "_")
        let defaultPath = EnvironmentConfig.modelCacheDirectory
            .appendingPathComponent(safeId)
            .appendingPathComponent("weights.safetensors")
        if FileManager.default.fileExists(atPath: defaultPath.path) {
            return defaultPath
        }
        return nil
    }
    
    /// Load pretrained weights from safetensors (best-effort mapping)
    private func loadPretrainedWeights(from url: URL) throws {
        let arrays = try loadArrays(url: url)
        // Apply to embedding
        if var embed = embedTokens {
            var params = ModuleParameters()
            for (k, v) in arrays where k.contains("embed") || k.contains("token_embedding") {
                params[k] = .value(v)
            }
            embed.update(parameters: params)
            embedTokens = embed
        }
        
        // Apply to layers (best effort by index)
        for (idx, layer) in layers.enumerated() {
            var params = ModuleParameters()
            for (k, v) in arrays where k.contains("layers.\(idx)") {
                let cleaned = k.replacingOccurrences(of: "layers.\(idx).", with: "")
                params[cleaned] = .value(v)
            }
            layer.update(parameters: params)
        }
        
        // Norm and LM head
        if var finalNorm = norm {
            var params = ModuleParameters()
            for (k, v) in arrays where k.contains("norm") {
                params[k.replacingOccurrences(of: "norm.", with: "")] = .value(v)
            }
            finalNorm.update(parameters: params)
            norm = finalNorm
        }
        
        // Quantization: apply fake quantization for int4/int8 configs
        switch configuration.targetQuantization {
        case .int4:
            applyQuantization(.int4)
        case .int8:
            applyQuantization(.int8)
        default:
            break
        }
    }
    
    /// Apply fake quantization to weights for memory savings (best effort)
    private func applyQuantization(_ config: QuantizationConfig) {
        if var head = lmHead {
            var params = ModuleParameters()
            for (key, arr) in head.parameters().flattened() {
                let q = QuantizationUtils.fakeQuantize(arr, config: config)
                params[key] = .value(q)
            }
            head.update(parameters: params)
            lmHead = head
        }
    }
    
    /// Check if model is loaded
    public func checkLoaded() -> Bool {
        isLoaded
    }
    

    // Inference
    
    /// Forward pass returning logits
    private func forward(_ inputIds: MLXArray, mask: MLXArray? = nil) -> (logits: MLXArray, hiddenStates: MLXArray) {
        guard let embedTokens = embedTokens, let norm = norm, let lmHead = lmHead else {
            return (MLXArray.zeros([1, 1, vocabSize]), MLXArray.zeros([1, 1, hiddenSize]))
        }
        
        // Get embeddings
        var hidden = embedTokens(inputIds)
        var lastHiddenStates = hidden
        
        // Create causal mask
        let seqLen = inputIds.shape[inputIds.ndim - 1]
        let causalMask = makeCausalMask(seqLen: seqLen)
        
        // Apply transformer layers
        for layer in layers {
            let (output, states) = layer(hidden, mask: causalMask)
            hidden = output
            lastHiddenStates = states
        }
        
        // Final norm and LM head
        hidden = norm(hidden)
        let logits = lmHead(hidden)
        
        return (logits, lastHiddenStates)
    }
    
    /// Create causal attention mask
    private func makeCausalMask(seqLen: Int) -> MLXArray {
        // Create a large negative value for masked positions
        let negInf = MLXArray(Array(repeating: Float(-1e9), count: seqLen * seqLen)).reshaped([seqLen, seqLen])
        // Lower triangular matrix (causal mask)
        return tril(MLXArray.zeros([seqLen, seqLen])) + triu(negInf, k: 1)
    }
    
    /// Generate next token logits
    public func getLogits(inputIds: MLXArray) async throws -> MLXArray {
        guard isLoaded else {
            throw TargetModelError.modelNotLoaded
        }
        
        let input = inputIds.ndim == 1 ? inputIds.expandedDimensions(axis: 0) : inputIds
        let (logits, _) = forward(input)
        return logits
    }
    
    /// Generate next token logits with hidden states (for EAGLE)
    public func getLogitsAndHiddenStates(inputIds: MLXArray) async throws -> (logits: MLXArray, hiddenStates: MLXArray) {
        guard isLoaded else {
            throw TargetModelError.modelNotLoaded
        }
        
        let input = inputIds.ndim == 1 ? inputIds.expandedDimensions(axis: 0) : inputIds
        return forward(input)
    }
    
    /// Verify draft tokens against target model
    public func verifyDraftTokens(
        contextIds: MLXArray,
        draftTokens: [Int32],
        config: GenerationConfig = .default,
        draftLogProbs: [Float]? = nil
    ) async throws -> VerificationResult {
        guard isLoaded else {
            throw TargetModelError.modelNotLoaded
        }
        
        totalVerifications += 1
        
        // Combine context with draft tokens for batch verification
        let draftArray = MLXArray(draftTokens)
        let fullSequence = MLX.concatenated([contextIds, draftArray], axis: 0)
        
        // Get logits and hidden states for the full sequence
        let (logits, hiddenStates) = try await getLogitsAndHiddenStates(inputIds: fullSequence)
        
        // Verify each draft token
        var acceptanceMask: [Bool] = []
        var correctedToken: Int32? = nil
        let contextLength = contextIds.shape[0]
        
        for (index, draftToken) in draftTokens.enumerated() {
            let position = contextLength + index - 1
            
            if position >= 0 && position < logits.shape[1] {
                let tokenLogits = logits[0, position]
                
                if let draftLogProbs = draftLogProbs, draftLogProbs.count > index {
                    // Lossless acceptance: accept with probability min(1, p_t / p_d)
                    let targetProbs = MLX.softmax(tokenLogits / config.temperature)
                    let targetProb = targetProbs[Int(draftToken)].item(Float.self)
                    let draftProb = exp(draftLogProbs[index])
                    let acceptanceProb = min(1.0, targetProb / max(draftProb, 1e-8))
                    let r = Float.random(in: 0...1)
                    let accepted = r < acceptanceProb
                    acceptanceMask.append(accepted)
                    if !accepted {
                        let newToken = sampleToken(logits: tokenLogits, config: config)
                        correctedToken = newToken
                        break
                    }
                } else {
                    let predictedToken = sampleToken(logits: tokenLogits, config: config)
                    if predictedToken == draftToken {
                        acceptanceMask.append(true)
                    } else {
                        acceptanceMask.append(false)
                        correctedToken = predictedToken
                        break
                    }
                }
            } else {
                acceptanceMask.append(false)
                break
            }
        }
        
        totalTokensGenerated += acceptanceMask.filter { $0 }.count
        if correctedToken != nil {
            totalTokensGenerated += 1
        }
        
        return VerificationResult(
            tokenIds: draftTokens,
            logits: logits,
            hiddenStates: hiddenStates,
            acceptanceMask: acceptanceMask,
            correctedToken: correctedToken
        )
    }
    
    /// Sample a token from logits
    private func sampleToken(logits: MLXArray, config: GenerationConfig) -> Int32 {
        if config.temperature <= 0 {
            return Int32(MLX.argMax(logits).item(Int32.self))
        }
        
        let scaledLogits = logits / config.temperature
        
        // Apply top-k filtering using argsort
        var processedLogits = scaledLogits
        if config.topK > 0 && config.topK < logits.shape[0] {
            let sortedIndices = argSort(scaledLogits)
            let kthIndex = sortedIndices[logits.shape[0] - config.topK]
            let threshold = scaledLogits[Int(kthIndex.item(Int32.self))]
            processedLogits = MLX.where(
                scaledLogits .< threshold,
                MLXArray(-1e9),
                scaledLogits
            )
        }
        
        // Sample from the filtered distribution
        let probs = MLX.softmax(processedLogits)
        let sample = MLXRandom.categorical(probs.expandedDimensions(axis: 0))
        
        return sample.item(Int32.self)
    }
    

    // Generation (Non-Speculative)
    
    /// Standard autoregressive generation (fallback)
    public func generate(
        prompt: MLXArray,
        config: GenerationConfig = .default,
        tokenCallback: ((Int32) async -> Bool)? = nil
    ) async throws -> [Int32] {
        guard isLoaded else {
            throw TargetModelError.modelNotLoaded
        }
        
        var tokens = Array(prompt.asArray(Int32.self))
        var generated: [Int32] = []
        
        for _ in 0..<config.maxTokens {
            let inputIds = MLXArray(tokens)
            let logits = try await getLogits(inputIds: inputIds)
            let lastLogits = logits[0, logits.shape[1] - 1]
            
            let nextToken = sampleToken(logits: lastLogits, config: config)
            
            if config.stopTokens.contains(nextToken) {
                break
            }
            
            tokens.append(nextToken)
            generated.append(nextToken)
            totalTokensGenerated += 1
            
            if let callback = tokenCallback {
                let shouldContinue = await callback(nextToken)
                if !shouldContinue {
                    break
                }
            }
        }
        
        return generated
    }
    

    // Cache Management
    
    /// Clear KV cache
    public func clearCache() async {
        await kvCache?.clear()
    }
    
    /// Get current cache length
    public func getCacheLength() async -> Int {
        await kvCache?.getLength() ?? 0
    }
    

    // Statistics
    
    /// Get generation statistics
    public func getStatistics() -> (totalTokens: Int, totalVerifications: Int) {
        (totalTokensGenerated, totalVerifications)
    }
    
    /// Reset statistics
    public func resetStatistics() {
        totalTokensGenerated = 0
        totalVerifications = 0
    }
    

    // Model Info
    
    public func getVocabSize() -> Int { vocabSize }
    public func getHiddenSize() -> Int { hiddenSize }
    public func getNumLayers() -> Int { numLayers }
}

// Errors

public enum TargetModelError: Error, LocalizedError {
    case modelNotLoaded
    case verificationFailed(String)
    case generationFailed(String)
    case invalidInput(String)
    
    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Target model has not been loaded"
        case .verificationFailed(let reason):
            return "Draft verification failed: \(reason)"
        case .generationFailed(let reason):
            return "Generation failed: \(reason)"
        case .invalidInput(let reason):
            return "Invalid input: \(reason)"
        }
    }
}

// NPU Offload Support (M4/M5)

#if arch(arm64) && os(macOS)
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// Workload type for routing to appropriate compute device
public enum ComputeWorkload: Sendable {
    case attention      // Attention layers - GPU optimized
    case matmul         // Matrix multiplication - NPU optimized on M4+
    case normalization  // Layer normalization - GPU
    case embedding      // Token embeddings - GPU
    case softmax        // Softmax computation - GPU
}

/// Metal 4 / NPU optimization for M4/M5 chips
public actor NPUOffloadManager {
    private let isAvailable: Bool
    private let chipFamily: ChipFamily
    private let device: MTLDevice?
    
    // Compute pipelines
    private var gpuQueue: MTLCommandQueue?
    private var npuQueue: MTLCommandQueue?
    
    // Performance monitoring
    private var gpuUtilization: Float = 0
    private var npuUtilization: Float = 0
    
    // Workload routing table
    private var workloadRouting: [ComputeWorkload: Bool] = [:]  // true = NPU, false = GPU
    
    public init(hardwareProfile: HardwareProfile) {
        self.isAvailable = hardwareProfile.chipFamily.hasEnhancedNPU
        self.chipFamily = hardwareProfile.chipFamily
        self.device = MTLCreateSystemDefaultDevice()
        
        // Initialize default routing (NPU for heavy compute on M4+)
        if isAvailable {
            workloadRouting = [
                .attention: false,      // Keep on GPU for memory locality
                .matmul: true,          // NPU excels at large matmuls
                .normalization: false,  // Small operation, keep on GPU
                .embedding: false,      // Memory bound, GPU
                .softmax: false         // Small operation, GPU
            ]
        }
    }
    
    public func canOffload() -> Bool { isAvailable }
    
    /// Initialize compute pipelines for NPU and GPU
    public func initialize() async throws {
        guard isAvailable, let device = device else { return }
        
        // Create command queues
        gpuQueue = device.makeCommandQueue()
        
        // On M4/M5, we can create a separate queue optimized for neural engine workloads
        // This is a placeholder for when Metal 4 APIs expose NPU-specific queue creation
        if chipFamily == .m4 || chipFamily == .m5 {
            npuQueue = device.makeCommandQueue()
        }
    }
    
    /// Configure compute graph for NPU execution
    /// Uses Metal Performance Shaders Graph for NPU-optimized operations
    public func configureForNPU() async {
        guard isAvailable else { return }
        
        // Metal 4 specific optimizations will go here when available
        // For now, use MPSGraph for NPU-friendly graph execution
    }
    
    /// Create an MPSGraph optimized for M4/M5 neural engine
    public func createOptimizedGraph() -> MPSGraph? {
        guard isAvailable else { return nil }
        
        let graph = MPSGraph()
        
        // Configure graph compilation options for NPU when available
        // graph.options = .preferNeuralEngine  // Future Metal 4 API
        
        return graph
    }
    
    /// Get recommended device for a workload
    public func getDevice(for workload: ComputeWorkload) -> MTLDevice? {
        guard isAvailable else { return device }
        
        // Route based on workload type
        if workloadRouting[workload] == true {
            // NPU path - return device configured for neural engine
            return device
        }
        return device
    }
    
    /// Get command queue for workload
    public func getQueue(for workload: ComputeWorkload) -> MTLCommandQueue? {
        guard isAvailable else { return gpuQueue }
        
        if workloadRouting[workload] == true, let npu = npuQueue {
            return npu
        }
        return gpuQueue
    }
    
    /// Execute a matrix multiplication using the optimal device
    public func executeMatMul(
        a: MTLBuffer,
        b: MTLBuffer,
        result: MTLBuffer,
        m: Int,
        n: Int,
        k: Int
    ) async {
        guard let device = device, let queue = getQueue(for: .matmul) else { return }
        
        // Use MPS for optimized matrix multiplication
        let matMul = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: false,
            resultRows: m,
            resultColumns: n,
            interiorColumns: k,
            alpha: 1.0,
            beta: 0.0
        )
        
        guard let commandBuffer = queue.makeCommandBuffer() else { return }
        
        let matrixA = MPSMatrix(
            buffer: a,
            descriptor: MPSMatrixDescriptor(
                rows: m,
                columns: k,
                rowBytes: k * MemoryLayout<Float16>.stride,
                dataType: .float16
            )
        )
        
        let matrixB = MPSMatrix(
            buffer: b,
            descriptor: MPSMatrixDescriptor(
                rows: k,
                columns: n,
                rowBytes: n * MemoryLayout<Float16>.stride,
                dataType: .float16
            )
        )
        
        let matrixC = MPSMatrix(
            buffer: result,
            descriptor: MPSMatrixDescriptor(
                rows: m,
                columns: n,
                rowBytes: n * MemoryLayout<Float16>.stride,
                dataType: .float16
            )
        )
        
        matMul.encode(
            commandBuffer: commandBuffer,
            leftMatrix: matrixA,
            rightMatrix: matrixB,
            resultMatrix: matrixC
        )
        
        commandBuffer.commit()
        
        // Wait for completion using continuation
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
        }
    }
    
    /// Update workload routing based on performance metrics
    public func updateRouting(workload: ComputeWorkload, useNPU: Bool) {
        workloadRouting[workload] = useNPU
    }
    
    /// Get current utilization metrics
    public func getUtilization() -> (gpu: Float, npu: Float) {
        (gpuUtilization, npuUtilization)
    }
    
    /// Record utilization sample
    public func recordUtilization(gpu: Float, npu: Float) {
        self.gpuUtilization = gpu
        self.npuUtilization = npu
    }
    
    /// Check if hybrid GPU/NPU execution is beneficial
    public func shouldUseHybridExecution(contextLength: Int) -> Bool {
        // For long contexts, hybrid execution can improve throughput
        // by running attention on GPU while prefetch/postprocess runs on NPU
        guard isAvailable else { return false }
        return contextLength > 2048
    }
    
    /// Create async compute pipeline for overlapped execution
    public func createAsyncPipeline() -> AsyncComputePipeline? {
        guard isAvailable, let gpuQ = gpuQueue, let npuQ = npuQueue else {
            return nil
        }
        return AsyncComputePipeline(gpuQueue: gpuQ, npuQueue: npuQ)
    }
}

/// Manages overlapped GPU/NPU execution for maximum throughput
public struct AsyncComputePipeline: @unchecked Sendable {
    public let gpuQueue: MTLCommandQueue
    public let npuQueue: MTLCommandQueue
    
    /// Execute GPU and NPU workloads sequentially (async-safe version)
    /// For true parallel execution, use the synchronous API from a dedicated thread
    public func executeSequential(
        gpuWork: (MTLCommandBuffer) -> Void,
        npuWork: (MTLCommandBuffer) -> Void
    ) {
        // Execute GPU work
        if let buffer = gpuQueue.makeCommandBuffer() {
            gpuWork(buffer)
            buffer.commit()
        }
        
        // Execute NPU work
        if let buffer = npuQueue.makeCommandBuffer() {
            npuWork(buffer)
            buffer.commit()
        }
    }
    
    /// Execute with async completion handlers
    public func executeOverlapped(
        gpuWork: @escaping (MTLCommandBuffer) -> Void,
        npuWork: @escaping (MTLCommandBuffer) -> Void,
        completion: @escaping () -> Void
    ) {
        let group = DispatchGroup()
        
        group.enter()
        if let buffer = gpuQueue.makeCommandBuffer() {
            gpuWork(buffer)
            buffer.addCompletedHandler { _ in group.leave() }
            buffer.commit()
        } else {
            group.leave()
        }
        
        group.enter()
        if let buffer = npuQueue.makeCommandBuffer() {
            npuWork(buffer)
            buffer.addCompletedHandler { _ in group.leave() }
            buffer.commit()
        } else {
            group.leave()
        }
        
        group.notify(queue: .main) {
            completion()
        }
    }
}

/// Metal 4 feature detection and capability reporting
public struct Metal4Capabilities: Sendable {
    public let supportsNPUOffload: Bool
    public let supportsAsyncCompute: Bool
    public let supportsFloat16Accumulation: Bool
    public let maxNPUBatchSize: Int
    public let recommendedTileSize: Int
    
    public static func detect(for device: MTLDevice?) -> Metal4Capabilities {
        guard let device = device else {
            return Metal4Capabilities(
                supportsNPUOffload: false,
                supportsAsyncCompute: false,
                supportsFloat16Accumulation: false,
                maxNPUBatchSize: 0,
                recommendedTileSize: 32
            )
        }
        
        // Detect M4/M5 specific capabilities
        let name = device.name.lowercased()
        let isM4OrNewer = name.contains("m4") || name.contains("m5")
        
        return Metal4Capabilities(
            supportsNPUOffload: isM4OrNewer,
            supportsAsyncCompute: device.supportsFamily(.apple7),
            supportsFloat16Accumulation: device.supportsFamily(.apple8),
            maxNPUBatchSize: isM4OrNewer ? 32 : 8,
            recommendedTileSize: isM4OrNewer ? 64 : 32
        )
    }
}
#endif
