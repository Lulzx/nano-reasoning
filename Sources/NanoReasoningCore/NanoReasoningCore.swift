// SPDX-License-Identifier: MIT
// Nano-Reasoning: FastRL Adaptive Drafter for Apple Silicon
// Main module export file

@_exported import Foundation
@_exported import MLX

/// Nano-Reasoning library version
public let version = "1.0.0"

/// Quick start helper for common use cases
public struct NanoReasoning {
    
    /// Create a ready-to-use orchestrator with automatic hardware detection
    public static func createOrchestrator(
        generationConfig: GenerationConfig = .default
    ) async throws -> Orchestrator {
        let orchestrator = Orchestrator(generationConfig: generationConfig)
        try await orchestrator.initialize { stage, progress in
            print("[\(Int(progress * 100))%] \(stage)")
        }
        return orchestrator
    }
    
    /// Get system information
    public static func getSystemInfo() -> HardwareProfile {
        let detector = HardwareDetector()
        return detector.detectHardware()
    }
    
    /// Check if system meets minimum requirements
    public static func checkRequirements() -> (meets: Bool, message: String) {
        let profile = getSystemInfo()
        
        if !profile.meetsMinimumRequirements {
            return (false, "System does not meet minimum requirements. Need 16GB unified memory on Apple Silicon.")
        }
        
        return (true, "System meets requirements: \(profile.tier)")
    }
}
