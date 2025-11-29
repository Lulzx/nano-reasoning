// SPDX-License-Identifier: MIT
// Nano-Reasoning: Quantization Support for INT8 Training
// Implements quantization-aware training (QAT) for memory efficiency

import Foundation
@preconcurrency import MLX
import MLXNN
import MLXRandom

/// Quantization configuration
public struct QuantizationConfig: Sendable {
    /// Number of bits for quantization
    public let bits: Int
    /// Group size for group-wise quantization
    public let groupSize: Int
    /// Whether to use symmetric quantization
    public let symmetric: Bool
    /// Calibration method
    public let calibrationMethod: CalibrationMethod
    
    public enum CalibrationMethod: Sendable {
        case minMax       // Use min/max of activations
        case percentile   // Use percentile (e.g., 99.9th)
        case mse          // Minimize mean squared error
    }
    
    public static let int8 = QuantizationConfig(
        bits: 8,
        groupSize: 128,
        symmetric: true,
        calibrationMethod: .minMax
    )
    
    public static let int4 = QuantizationConfig(
        bits: 4,
        groupSize: 32,
        symmetric: false,
        calibrationMethod: .percentile
    )
}

/// Quantization parameters for a tensor
public struct QuantizationParams: @unchecked Sendable {
    /// Scale factor for dequantization
    public let scale: MLXArray
    /// Zero point for asymmetric quantization
    public let zeroPoint: MLXArray?
    /// Number of bits
    public let bits: Int
    
    public init(scale: MLXArray, zeroPoint: MLXArray? = nil, bits: Int = 8) {
        self.scale = scale
        self.zeroPoint = zeroPoint
        self.bits = bits
    }
}

/// Quantization utilities for INT8 training
public struct QuantizationUtils {
    
    /// Compute quantization parameters from tensor statistics
    public static func computeParams(
        for tensor: MLXArray,
        config: QuantizationConfig
    ) -> QuantizationParams {
        let minVal = MLX.min(tensor)
        let maxVal = MLX.max(tensor)
        
        let qmin = Float(-(1 << (config.bits - 1)))
        let qmax = Float((1 << (config.bits - 1)) - 1)
        
        if config.symmetric {
            // Symmetric quantization: zeroPoint = 0
            let absMax = MLX.maximum(MLX.abs(minVal), MLX.abs(maxVal))
            let scale = absMax / MLXArray(qmax)
            
            return QuantizationParams(
                scale: scale,
                zeroPoint: nil,
                bits: config.bits
            )
        } else {
            // Asymmetric quantization
            let scale = (maxVal - minVal) / MLXArray(qmax - qmin)
            let zeroPoint = MLX.round(MLXArray(qmin) - minVal / scale)
            
            return QuantizationParams(
                scale: scale,
                zeroPoint: zeroPoint,
                bits: config.bits
            )
        }
    }
    
    /// Quantize a tensor to INT8
    public static func quantize(
        _ tensor: MLXArray,
        params: QuantizationParams
    ) -> MLXArray {
        let qmin = Float(-(1 << (params.bits - 1)))
        let qmax = Float((1 << (params.bits - 1)) - 1)
        
        var quantized: MLXArray
        if let zeroPoint = params.zeroPoint {
            quantized = MLX.round(tensor / params.scale) + zeroPoint
        } else {
            quantized = MLX.round(tensor / params.scale)
        }
        
        // Clamp to valid range
        quantized = MLX.clip(quantized, min: MLXArray(qmin), max: MLXArray(qmax))
        
        return quantized
    }
    
    /// Dequantize from INT8 back to floating point
    public static func dequantize(
        _ quantized: MLXArray,
        params: QuantizationParams
    ) -> MLXArray {
        if let zeroPoint = params.zeroPoint {
            return (quantized - zeroPoint) * params.scale
        } else {
            return quantized * params.scale
        }
    }
    
    /// Fake quantization for QAT (quantize then immediately dequantize)
    /// This simulates quantization effects during training while keeping gradients
    public static func fakeQuantize(
        _ tensor: MLXArray,
        config: QuantizationConfig
    ) -> MLXArray {
        let params = computeParams(for: tensor, config: config)
        let quantized = quantize(tensor, params: params)
        return dequantize(quantized, params: params)
    }
    
    /// Group-wise quantization for better accuracy
    public static func quantizeGroupwise(
        _ tensor: MLXArray,
        config: QuantizationConfig
    ) -> (quantized: MLXArray, scales: MLXArray, zeroPoints: MLXArray?) {
        let shape = tensor.shape
        let lastDim = shape[shape.count - 1]
        let numGroups = (lastDim + config.groupSize - 1) / config.groupSize
        
        var quantizedGroups: [MLXArray] = []
        var scales: [MLXArray] = []
        var zeroPoints: [MLXArray] = []
        
        for g in 0..<numGroups {
            let start = g * config.groupSize
            let end = min(start + config.groupSize, lastDim)
            
            // Extract group
            let group = tensor[.ellipsis, start..<end]
            
            // Compute params for this group
            let params = computeParams(for: group, config: config)
            
            // Quantize group
            let qGroup = quantize(group, params: params)
            quantizedGroups.append(qGroup)
            scales.append(params.scale)
            
            if let zp = params.zeroPoint {
                zeroPoints.append(zp)
            }
        }
        
        let quantized = MLX.concatenated(quantizedGroups, axis: -1)
        let scaleArray = MLX.stacked(scales, axis: 0)
        let zpArray = zeroPoints.isEmpty ? nil : MLX.stacked(zeroPoints, axis: 0)
        
        return (quantized, scaleArray, zpArray)
    }
}

/// INT8 Linear layer with quantization-aware training
public class QuantizedLinear: Module, @unchecked Sendable {
    let weight: MLXArray
    let bias: MLXArray?
    let config: QuantizationConfig
    
    // Cached quantization parameters (updated during calibration)
    private var weightParams: QuantizationParams?
    private var inputParams: QuantizationParams?
    
    // Training mode flag
    private var isCalibrating: Bool = false
    
    // Statistics for calibration
    private var runningMin: MLXArray?
    private var runningMax: MLXArray?
    private var calibrationSamples: Int = 0
    
    public init(
        inputDimensions: Int,
        outputDimensions: Int,
        bias: Bool = true,
        config: QuantizationConfig = .int8
    ) {
        // Initialize with He initialization
        let scale = sqrt(2.0 / Float(inputDimensions))
        self.weight = MLXRandom.normal([outputDimensions, inputDimensions]) * scale
        self.bias = bias ? MLXArray.zeros([outputDimensions]) : nil
        self.config = config
        super.init()
    }
    
    /// Initialize from existing Linear layer
    public init(from linear: Linear, config: QuantizationConfig = .int8) {
        self.weight = linear.weight
        self.bias = linear.bias
        self.config = config
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var processedWeight = weight
        
        // Apply fake quantization during training
        if isCalibrating {
            // Update running statistics
            updateCalibrationStats(x)
        }
        
        // Apply fake quantization to weights
        processedWeight = QuantizationUtils.fakeQuantize(weight, config: config)
        
        // Standard linear computation
        var output = MLX.matmul(x, processedWeight.T)
        
        if let bias = bias {
            output = output + bias
        }
        
        return output
    }
    
    /// Start calibration mode
    public func startCalibration() {
        isCalibrating = true
        runningMin = nil
        runningMax = nil
        calibrationSamples = 0
    }
    
    /// Stop calibration and compute final quantization parameters
    public func stopCalibration() {
        isCalibrating = false
        
        // Compute weight quantization parameters
        weightParams = QuantizationUtils.computeParams(for: weight, config: config)
        
        // Compute input quantization parameters if we have statistics
        if let minVal = runningMin, let maxVal = runningMax {
            let range = maxVal - minVal
            let scale = range / MLXArray(Float((1 << config.bits) - 1))
            inputParams = QuantizationParams(scale: scale, zeroPoint: nil, bits: config.bits)
        }
    }
    
    /// Update calibration statistics
    private func updateCalibrationStats(_ input: MLXArray) {
        let minVal = MLX.min(input)
        let maxVal = MLX.max(input)
        
        if let rMin = runningMin, let rMax = runningMax {
            runningMin = MLX.minimum(rMin, minVal)
            runningMax = MLX.maximum(rMax, maxVal)
        } else {
            runningMin = minVal
            runningMax = maxVal
        }
        
        calibrationSamples += 1
    }
    
    /// Get quantized weights for inference
    public func getQuantizedWeights() -> (MLXArray, QuantizationParams)? {
        guard let params = weightParams else { return nil }
        let quantized = QuantizationUtils.quantize(weight, params: params)
        return (quantized, params)
    }
}

/// Quantized drafter model for memory-efficient training
public class QuantizedDrafterModel: Module, @unchecked Sendable {
    let config: DrafterConfig
    let quantConfig: QuantizationConfig
    let embedTokens: Embedding
    let lmHead: QuantizedLinear
    let eagleHead: EAGLEHead?
    
    // Quantization state
    private var isCalibrated: Bool = false
    
    public init(
        config: DrafterConfig,
        quantConfig: QuantizationConfig = .int8,
        enableEAGLE: Bool = true
    ) {
        self.config = config
        self.quantConfig = quantConfig
        
        // Embeddings are not quantized (small overhead, need precision)
        self.embedTokens = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        
        // LM head is quantized
        self.lmHead = QuantizedLinear(
            inputDimensions: config.hiddenSize,
            outputDimensions: config.vocabSize,
            bias: false,
            config: quantConfig
        )
        
        self.eagleHead = enableEAGLE ? EAGLEHead(hiddenSize: config.hiddenSize) : nil
        
        super.init()
    }
    
    /// Convert from a regular DrafterModel
    public init(from drafterModel: DrafterModel, quantConfig: QuantizationConfig = .int8) {
        self.config = drafterModel.config
        self.quantConfig = quantConfig
        self.embedTokens = drafterModel.embedTokens
        self.lmHead = QuantizedLinear(from: drafterModel.lmHead, config: quantConfig)
        self.eagleHead = drafterModel.eagleHead
        super.init()
    }
    
    public func callAsFunction(
        _ hiddenStates: MLXArray,
        previousEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        var x = hiddenStates
        
        // Apply EAGLE head if available
        if let eagle = eagleHead, let prevEmbed = previousEmbeddings {
            x = eagle(x, embeddings: prevEmbed)
        }
        
        // Get logits through quantized layer
        return lmHead(x)
    }
    
    /// Get embeddings for token ids
    public func getEmbeddings(_ tokenIds: MLXArray) -> MLXArray {
        embedTokens(tokenIds)
    }
    
    /// Start calibration pass
    public func startCalibration() {
        lmHead.startCalibration()
    }
    
    /// Stop calibration and finalize quantization
    public func stopCalibration() {
        lmHead.stopCalibration()
        isCalibrated = true
    }
    
    /// Check if model is calibrated
    public func calibrated() -> Bool {
        isCalibrated
    }
}

/// Mixed-precision training utilities
public struct MixedPrecisionTraining {
    
    /// Loss scaling for FP16/INT8 training stability
    public struct LossScaler {
        private var scale: Float
        private var growthFactor: Float
        private var backoffFactor: Float
        private var growthInterval: Int
        private var stepsSinceGrowth: Int = 0
        private var consecutiveOverflows: Int = 0
        
        public init(
            initialScale: Float = 65536.0,
            growthFactor: Float = 2.0,
            backoffFactor: Float = 0.5,
            growthInterval: Int = 2000
        ) {
            self.scale = initialScale
            self.growthFactor = growthFactor
            self.backoffFactor = backoffFactor
            self.growthInterval = growthInterval
        }
        
        /// Scale gradients before backward pass
        public func scaleGradients(_ gradients: [String: MLXArray]) -> [String: MLXArray] {
            gradients.mapValues { $0 * scale }
        }
        
        /// Unscale gradients after backward pass
        public func unscaleGradients(_ gradients: [String: MLXArray]) -> [String: MLXArray] {
            gradients.mapValues { $0 / scale }
        }
        
        /// Check for overflow and update scale
        public mutating func update(hadOverflow: Bool) {
            if hadOverflow {
                scale *= backoffFactor
                consecutiveOverflows += 1
                stepsSinceGrowth = 0
            } else {
                consecutiveOverflows = 0
                stepsSinceGrowth += 1
                
                if stepsSinceGrowth >= growthInterval {
                    scale *= growthFactor
                    stepsSinceGrowth = 0
                }
            }
        }
        
        /// Check if gradients contain overflow (NaN or Inf)
        public func checkOverflow(_ gradients: [String: MLXArray]) -> Bool {
            for (_, grad) in gradients {
                let hasNaN = MLX.any(MLX.isNaN(grad)).item(Bool.self)
                let hasInf = MLX.any(MLX.isInf(grad)).item(Bool.self)
                if hasNaN || hasInf {
                    return true
                }
            }
            return false
        }
        
        public func getScale() -> Float { scale }
    }
}

/// Extension to DrafterActor for quantized training support
extension DrafterActor {
    
    /// Create a quantized version of the current model
    public func createQuantizedModel(config: QuantizationConfig = .int8) -> QuantizedDrafterModel {
        let model = getModel()
        return QuantizedDrafterModel(from: model, quantConfig: config)
    }
    
    /// Run calibration pass with sample data
    public func calibrate(samples: [TrainingSample]) async {
        let quantModel = createQuantizedModel()
        quantModel.startCalibration()
        
        // Run forward passes to collect statistics
        for sample in samples {
            let embeddings = quantModel.getEmbeddings(sample.inputIds)
            let _ = quantModel(sample.hiddenStates, previousEmbeddings: embeddings)
        }
        
        quantModel.stopCalibration()
    }
}
