// SPDX-License-Identifier: MIT
// Nano-Reasoning: Unit Tests

import XCTest
@testable import NanoReasoningCore
import MLX

final class NanoReasoningTests: XCTestCase {
    
    // MARK: - Hardware Detection Tests
    
    func testHardwareDetection() {
        let detector = HardwareDetector()
        let profile = detector.detectHardware()
        
        // Should detect a valid profile
        XCTAssertGreaterThan(profile.totalMemoryGB, 0)
        XCTAssertFalse(profile.gpuName.isEmpty)
        
        // Apple Silicon should have unified memory
        #if arch(arm64)
        XCTAssertTrue(profile.isUnifiedMemory)
        #endif
    }
    
    func testHardwareTierClassification() {
        // Entry tier: 16GB
        let entryProfile = HardwareProfile(
            tier: .entry,
            chipFamily: .m1,
            chipVariant: .base,
            totalMemoryGB: 16,
            gpuName: "Apple M1",
            gpuCoreCount: 8,
            isUnifiedMemory: true
        )
        XCTAssertEqual(entryProfile.tier, .entry)
        XCTAssertFalse(entryProfile.tier.trainingEnabled)
        
        // Pro tier: 24GB+
        let proProfile = HardwareProfile(
            tier: .pro,
            chipFamily: .m2,
            chipVariant: .pro,
            totalMemoryGB: 32,
            gpuName: "Apple M2 Pro",
            gpuCoreCount: 19,
            isUnifiedMemory: true
        )
        XCTAssertEqual(proProfile.tier, .pro)
        XCTAssertTrue(proProfile.tier.trainingEnabled)
        
        // Elite tier: M4/M5 with 36GB+
        let eliteProfile = HardwareProfile(
            tier: .elite,
            chipFamily: .m4,
            chipVariant: .max,
            totalMemoryGB: 64,
            gpuName: "Apple M4 Max",
            gpuCoreCount: 40,
            isUnifiedMemory: true
        )
        XCTAssertEqual(eliteProfile.tier, .elite)
        XCTAssertTrue(eliteProfile.tier.trainingEnabled)
        XCTAssertTrue(eliteProfile.npuOffloadAvailable)
    }
    
    func testModelConfigurationForTiers() {
        let entryConfig = ModelConfiguration.forTier(.entry)
        XCTAssertTrue(entryConfig.targetModelId.contains("3B") || entryConfig.targetModelId.contains("4B"))
        XCTAssertEqual(entryConfig.targetQuantization, .int4)
        XCTAssertEqual(entryConfig.drafterQuantization, .int4)
        
        let proConfig = ModelConfiguration.forTier(.pro)
        XCTAssertTrue(proConfig.targetModelId.contains("7B") || proConfig.targetModelId.contains("8B"))
        XCTAssertEqual(proConfig.drafterQuantization, .fp16)
        
        let eliteConfig = ModelConfiguration.forTier(.elite)
        XCTAssertTrue(eliteConfig.targetModelId.contains("32B"))
    }
    
    // MARK: - Training Buffer Tests
    
    func testTrainingBufferCapacity() async {
        let buffer = TrainingBuffer(capacity: 5)
        
        for i in 0..<10 {
            let sample = TrainingSample(
                hiddenStates: MLXArray([Float](repeating: Float(i), count: 10)),
                targetLogits: MLXArray([Float](repeating: Float(i), count: 10)),
                inputIds: MLXArray([Int32(i)])
            )
            await buffer.push(sample: sample)
        }
        
        // Buffer should not exceed capacity
        let count = await buffer.count()
        XCTAssertLessThanOrEqual(count, 5)
    }
    
    func testTrainingBufferPriority() async {
        let buffer = TrainingBuffer(capacity: 3)
        
        // Add low priority samples
        for i in 0..<3 {
            let sample = TrainingSample(
                hiddenStates: MLXArray([Float(i)]),
                targetLogits: MLXArray([Float(i)]),
                inputIds: MLXArray([Int32(i)])
            )
            await buffer.push(sample: sample, priority: .low)
        }
        
        // Add high priority sample - should replace a low priority one
        let highPrioritySample = TrainingSample(
            hiddenStates: MLXArray([Float(100)]),
            targetLogits: MLXArray([Float(100)]),
            inputIds: MLXArray([Int32(100)])
        )
        let accepted = await buffer.push(sample: highPrioritySample, priority: .high)
        XCTAssertTrue(accepted)
        
        // Pop should return high priority first
        let popped = await buffer.pop()
        XCTAssertNotNil(popped)
    }
    
    func testTrainingBufferStatistics() async {
        let buffer = TrainingBuffer(capacity: 5)
        
        // Push some samples
        for i in 0..<3 {
            let sample = TrainingSample(
                hiddenStates: MLXArray([Float(i)]),
                targetLogits: MLXArray([Float(i)]),
                inputIds: MLXArray([Int32(i)])
            )
            await buffer.push(sample: sample)
        }
        
        let stats = await buffer.getStatistics()
        XCTAssertEqual(stats.currentSize, 3)
        XCTAssertEqual(stats.capacity, 5)
        XCTAssertEqual(stats.totalPushed, 3)
        XCTAssertEqual(stats.totalDropped, 0)
    }
    
    // MARK: - GPU Load Monitor Tests
    
    func testGPULoadMonitor() async {
        let monitor = GPULoadMonitor()
        
        // Record low load samples
        for _ in 0..<5 {
            await monitor.recordLoad(0.3)
        }
        
        let underLoad = await monitor.checkUnderLoad()
        XCTAssertFalse(underLoad)
        
        // Record high load samples
        for _ in 0..<10 {
            await monitor.recordLoad(0.95)
        }
        
        let underHighLoad = await monitor.checkUnderLoad()
        XCTAssertTrue(underHighLoad)
    }
    
    // MARK: - Configuration Tests
    
    func testConfigurationValidation() {
        var config = NanoReasoningConfig.default
        
        // Valid config should not throw
        XCTAssertNoThrow(try config.validate())
        
        // Invalid maxTokens
        config.generation.maxTokens = -1
        XCTAssertThrowsError(try config.validate())
        config.generation.maxTokens = 100
        
        // Invalid temperature
        config.generation.temperature = -0.5
        XCTAssertThrowsError(try config.validate())
        config.generation.temperature = 0.7
        
        // Invalid topP
        config.generation.topP = 1.5
        XCTAssertThrowsError(try config.validate())
    }
    
    func testConfigurationBuilder() {
        let config = ConfigurationBuilder()
            .withTier("pro")
            .withTemperature(0.5)
            .withMaxTokens(1024)
            .withTrainingEnabled(true)
            .withLearningRate(5e-5)
            .build()
        
        XCTAssertEqual(config.hardware.forceTier, "pro")
        XCTAssertEqual(config.generation.temperature, 0.5)
        XCTAssertEqual(config.generation.maxTokens, 1024)
        XCTAssertTrue(config.training.enabled)
        XCTAssertEqual(config.training.learningRate, 5e-5)
    }
    
    // MARK: - Speculative Config Tests
    
    func testSpeculativeConfigForTiers() {
        let entryConfig = SpeculativeConfig.forTier(.entry)
        XCTAssertEqual(entryConfig.draftCount, 3)
        XCTAssertFalse(entryConfig.useTreeSpeculation)
        
        let proConfig = SpeculativeConfig.forTier(.pro)
        XCTAssertEqual(proConfig.draftCount, 5)
        
        let eliteConfig = SpeculativeConfig.forTier(.elite)
        XCTAssertEqual(eliteConfig.draftCount, 7)
        XCTAssertTrue(eliteConfig.useTreeSpeculation)
    }
    
    // MARK: - Token Sampler Tests
    
    func testGreedySampling() {
        let logits = MLXArray([0.1, 0.5, 0.2, 0.8, 0.3])
        let sampled = TokenSampler.greedy(logits)
        XCTAssertEqual(sampled, 3)  // Index of max value (0.8)
    }
    
    func testTemperatureSampling() {
        let logits = MLXArray([10.0, 0.0, 0.0, 0.0, 0.0])
        
        // With very low temperature, should almost always sample index 0
        var samples: [Int32] = []
        for _ in 0..<10 {
            samples.append(TokenSampler.sample(logits, temperature: 0.01))
        }
        
        let zeroCount = samples.filter { $0 == 0 }.count
        XCTAssertGreaterThan(zeroCount, 8)  // Should be mostly 0s
    }
    
    // MARK: - Model Registry Tests
    
    func testModelRegistry() {
        XCTAssertFalse(ModelRegistry.targetModels.isEmpty)
        XCTAssertFalse(ModelRegistry.drafterModels.isEmpty)
        
        // Check recommended models
        let target16GB = ModelRegistry.recommendedTarget(forMemoryGB: 16)
        XCTAssertNotNil(target16GB)
        
        let target64GB = ModelRegistry.recommendedTarget(forMemoryGB: 64)
        XCTAssertNotNil(target64GB)
        // Larger memory should allow larger model
        XCTAssertGreaterThan(target64GB!.memoryRequired, target16GB!.memoryRequired)
        
        let entryDrafter = ModelRegistry.recommendedDrafter(forTier: .entry)
        XCTAssertNotNil(entryDrafter)
        XCTAssertEqual(entryDrafter?.quantization, "4-bit")
    }
    
    // MARK: - Drafter Config Tests
    
    func testDrafterConfig() {
        let config = DrafterConfig.qwen3_0_6B
        
        XCTAssertEqual(config.hiddenSize, 1024)
        XCTAssertEqual(config.numLayers, 28)
        XCTAssertEqual(config.vocabSize, 151936)
    }
    
    // MARK: - Generation Config Tests
    
    func testGenerationConfigDefaults() {
        let defaultConfig = GenerationConfig.default
        XCTAssertEqual(defaultConfig.maxTokens, 2048)
        XCTAssertEqual(defaultConfig.temperature, 0.7)
        
        let greedyConfig = GenerationConfig.greedy
        XCTAssertEqual(greedyConfig.temperature, 0.0)
        XCTAssertEqual(greedyConfig.topK, 1)
    }
    
    // MARK: - Training Config Tests
    
    func testTrainingConfigForTiers() {
        let entryConfig = TrainingConfig.forTier(.entry)
        XCTAssertEqual(entryConfig.learningRate, 0)  // No training
        
        let proConfig = TrainingConfig.forTier(.pro)
        XCTAssertGreaterThan(proConfig.learningRate, 0)
        XCTAssertEqual(proConfig.batchSize, 2)
        
        let eliteConfig = TrainingConfig.forTier(.elite)
        XCTAssertGreaterThan(eliteConfig.batchSize, proConfig.batchSize)
    }
}

// MARK: - Performance Tests

final class NanoReasoningPerformanceTests: XCTestCase {
    
    func testBufferPerformance() async {
        let buffer = TrainingBuffer(capacity: 1000)
        
        measure {
            let expectation = XCTestExpectation(description: "Buffer operations")
            
            Task {
                // Push 1000 samples
                for i in 0..<1000 {
                    let sample = TrainingSample(
                        hiddenStates: MLXArray([Float](repeating: Float(i), count: 256)),
                        targetLogits: MLXArray([Float](repeating: Float(i), count: 32000)),
                        inputIds: MLXArray([Int32(i)])
                    )
                    await buffer.push(sample: sample)
                }
                
                // Pop 500 samples
                for _ in 0..<500 {
                    _ = await buffer.pop()
                }
                
                expectation.fulfill()
            }
            
            wait(for: [expectation], timeout: 10.0)
        }
    }
}

// MARK: - Integration Tests

final class NanoReasoningIntegrationTests: XCTestCase {
    
    func testSystemRequirementsCheck() {
        let (meets, message) = NanoReasoning.checkRequirements()
        
        // On Apple Silicon with sufficient memory, should pass
        #if arch(arm64)
        if ProcessInfo.processInfo.physicalMemory >= 16 * 1024 * 1024 * 1024 {
            XCTAssertTrue(meets, message)
        }
        #endif
    }
    
    func testSystemInfoRetrieval() {
        let profile = NanoReasoning.getSystemInfo()
        
        XCTAssertGreaterThan(profile.totalMemoryGB, 0)
        XCTAssertFalse(profile.gpuName.isEmpty)
    }
}
