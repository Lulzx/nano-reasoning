// SPDX-License-Identifier: MIT
// Nano-Reasoning: Hardware Detection and Tier Selection
// Supports M1/M2/M3/M4/M5 Apple Silicon chips

import Foundation
import Metal

/// Hardware tier classification based on chip capabilities and memory
public enum HardwareTier: Sendable, CustomStringConvertible {
    /// Entry tier: M1/M2 with 16GB - Frozen speculator, no training
    case entry
    /// Pro tier: M1/M2/M3 Pro/Max with 24GB+ - Adaptive LoRA training
    case pro
    /// Elite tier: M4/M5 with 36GB+ - Full EAGLE with NPU offload
    case elite
    
    public var description: String {
        switch self {
        case .entry: return "Entry (Frozen Speculator)"
        case .pro: return "Pro (Adaptive LoRA)"
        case .elite: return "Elite (Full EAGLE)"
        }
    }
    
    /// Whether training is enabled for this tier
    public var trainingEnabled: Bool {
        switch self {
        case .entry: return false
        case .pro, .elite: return true
        }
    }
    
    /// Whether NPU offload is available
    public var npuOffloadAvailable: Bool {
        self == .elite
    }
    
    /// Recommended draft count (k)
    public var draftCount: Int {
        switch self {
        case .entry: return 3
        case .pro: return 5
        case .elite: return 7
        }
    }
    
    /// Training buffer capacity based on tier
    public var bufferCapacity: Int {
        switch self {
        case .entry: return 0
        case .pro: return 50
        case .elite: return 200
        }
    }
}

/// Chip family detection for optimization paths
public enum ChipFamily: Sendable, CustomStringConvertible {
    case m1
    case m2
    case m3
    case m4
    case m5
    case unknown
    
    public var description: String {
        switch self {
        case .m1: return "M1"
        case .m2: return "M2"
        case .m3: return "M3"
        case .m4: return "M4"
        case .m5: return "M5"
        case .unknown: return "Unknown"
        }
    }
    
    /// Whether Metal 4 features are available
    public var supportsMetal4: Bool {
        switch self {
        case .m4, .m5: return true
        default: return false
        }
    }
    
    /// Whether dedicated Neural Accelerators are available for LLM offload
    public var hasEnhancedNPU: Bool {
        switch self {
        case .m4, .m5: return true
        default: return false
        }
    }
}

/// Chip variant (base, Pro, Max, Ultra)
public enum ChipVariant: Sendable, CustomStringConvertible {
    case base
    case pro
    case max
    case ultra
    
    public var description: String {
        switch self {
        case .base: return "Base"
        case .pro: return "Pro"
        case .max: return "Max"
        case .ultra: return "Ultra"
        }
    }
    
    /// GPU core multiplier for performance estimation
    public var gpuMultiplier: Double {
        switch self {
        case .base: return 1.0
        case .pro: return 1.5
        case .max: return 2.5
        case .ultra: return 5.0
        }
    }
}

/// Complete hardware profile for the current system
public struct HardwareProfile: Sendable, CustomStringConvertible {
    public let tier: HardwareTier
    public let chipFamily: ChipFamily
    public let chipVariant: ChipVariant
    public let totalMemoryGB: Int
    public let gpuName: String
    public let gpuCoreCount: Int?
    public let isUnifiedMemory: Bool
    
    public var description: String {
        """
        Hardware Profile:
          Tier: \(tier)
          Chip: \(chipFamily) \(chipVariant)
          GPU: \(gpuName)
          Memory: \(totalMemoryGB) GB (Unified: \(isUnifiedMemory))
          GPU Cores: \(gpuCoreCount.map(String.init) ?? "Unknown")
        """
    }
    
    /// Check if system meets minimum requirements
    public var meetsMinimumRequirements: Bool {
        totalMemoryGB >= 16 && isUnifiedMemory
    }
}

/// Hardware detection and profiling
public struct HardwareDetector: Sendable {
    
    public init() {}
    
    /// Detect current system hardware profile
    public func detectHardware() -> HardwareProfile {
        let memoryBytes = ProcessInfo.processInfo.physicalMemory
        let memoryGB = Int(memoryBytes / (1024 * 1024 * 1024))
        
        // Get Metal device info
        let device = MTLCreateSystemDefaultDevice()
        let gpuName = device?.name ?? "Unknown GPU"
        
        // Parse chip family and variant from GPU name
        let (family, variant) = parseChipInfo(from: gpuName)
        
        // Determine tier based on memory and chip
        let tier = determineTier(memoryGB: memoryGB, family: family, variant: variant)
        
        // Estimate GPU core count (Apple doesn't expose this directly)
        let coreCount = estimateGPUCores(family: family, variant: variant)
        
        return HardwareProfile(
            tier: tier,
            chipFamily: family,
            chipVariant: variant,
            totalMemoryGB: memoryGB,
            gpuName: gpuName,
            gpuCoreCount: coreCount,
            isUnifiedMemory: device?.hasUnifiedMemory ?? false
        )
    }
    
    /// Parse chip family and variant from Metal device name
    private func parseChipInfo(from gpuName: String) -> (ChipFamily, ChipVariant) {
        let name = gpuName.lowercased()
        
        // Detect family
        let family: ChipFamily
        if name.contains("m5") {
            family = .m5
        } else if name.contains("m4") {
            family = .m4
        } else if name.contains("m3") {
            family = .m3
        } else if name.contains("m2") {
            family = .m2
        } else if name.contains("m1") {
            family = .m1
        } else {
            family = .unknown
        }
        
        // Detect variant
        let variant: ChipVariant
        if name.contains("ultra") {
            variant = .ultra
        } else if name.contains("max") {
            variant = .max
        } else if name.contains("pro") {
            variant = .pro
        } else {
            variant = .base
        }
        
        return (family, variant)
    }
    
    /// Determine hardware tier based on memory and chip
    private func determineTier(memoryGB: Int, family: ChipFamily, variant: ChipVariant) -> HardwareTier {
        // Elite: M4/M5 with 36GB+
        if (family == .m4 || family == .m5) && memoryGB >= 36 {
            return .elite
        }
        
        // Pro: 24GB+ with Pro/Max/Ultra variants OR M3+ base with 24GB+
        if memoryGB >= 24 {
            if variant != .base || family == .m3 || family == .m4 || family == .m5 {
                return .pro
            }
        }
        
        // Entry: Everything else (16GB minimum)
        return .entry
    }
    
    /// Estimate GPU core count based on chip family and variant
    private func estimateGPUCores(family: ChipFamily, variant: ChipVariant) -> Int? {
        // These are approximate values based on known chip configurations
        switch (family, variant) {
        case (.m1, .base): return 8
        case (.m1, .pro): return 16
        case (.m1, .max): return 32
        case (.m1, .ultra): return 64
        case (.m2, .base): return 10
        case (.m2, .pro): return 19
        case (.m2, .max): return 38
        case (.m2, .ultra): return 76
        case (.m3, .base): return 10
        case (.m3, .pro): return 18
        case (.m3, .max): return 40
        case (.m4, .base): return 10
        case (.m4, .pro): return 20
        case (.m4, .max): return 40
        case (.m5, .base): return 12  // Estimated
        case (.m5, .pro): return 24   // Estimated
        case (.m5, .max): return 48   // Estimated
        case (.m5, .ultra): return 96 // Estimated
        default: return nil
        }
    }
}

/// Model configuration based on hardware tier
public struct ModelConfiguration: Sendable {
    public let targetModelId: String
    public let drafterModelId: String
    public let targetQuantization: Quantization
    public let drafterQuantization: Quantization
    public let maxContextLength: Int
    public let batchSize: Int
    
    public enum Quantization: String, Sendable {
        case fp16 = "fp16"
        case int8 = "int8"
        case int4 = "int4"
    }
    
    /// Create configuration for a given hardware tier
    public static func forTier(_ tier: HardwareTier) -> ModelConfiguration {
        switch tier {
        case .entry:
            return ModelConfiguration(
                targetModelId: "mlx-community/Qwen2.5-3B-Instruct-4bit",
                drafterModelId: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                targetQuantization: .int4,
                drafterQuantization: .int4,
                maxContextLength: 4096,
                batchSize: 1
            )
        case .pro:
            return ModelConfiguration(
                targetModelId: "mlx-community/Qwen2.5-7B-Instruct-4bit",
                drafterModelId: "mlx-community/Qwen2.5-0.5B-Instruct",
                targetQuantization: .int4,
                drafterQuantization: .fp16,
                maxContextLength: 8192,
                batchSize: 2
            )
        case .elite:
            return ModelConfiguration(
                targetModelId: "mlx-community/Qwen2.5-32B-Instruct-4bit",
                drafterModelId: "mlx-community/Qwen2.5-0.5B-Instruct",
                targetQuantization: .int4,
                drafterQuantization: .fp16,
                maxContextLength: 16384,
                batchSize: 4
            )
        }
    }
}

/// GPU load monitoring for adaptive behavior
public actor GPULoadMonitor {
    private var recentLoadSamples: [Double] = []
    private let maxSamples = 10
    private var isUnderLoad: Bool = false
    
    /// Record a GPU load sample (0.0 to 1.0)
    public func recordLoad(_ load: Double) {
        recentLoadSamples.append(load)
        if recentLoadSamples.count > maxSamples {
            recentLoadSamples.removeFirst()
        }
        
        // Update load status
        let averageLoad = recentLoadSamples.reduce(0, +) / Double(recentLoadSamples.count)
        isUnderLoad = averageLoad > 0.85
    }
    
    /// Check if GPU is under heavy load
    public func checkUnderLoad() -> Bool {
        isUnderLoad
    }
    
    /// Get current average load
    public func getAverageLoad() -> Double {
        guard !recentLoadSamples.isEmpty else { return 0.0 }
        return recentLoadSamples.reduce(0, +) / Double(recentLoadSamples.count)
    }
    
    /// Reset load monitoring
    public func reset() {
        recentLoadSamples.removeAll()
        isUnderLoad = false
    }
}
