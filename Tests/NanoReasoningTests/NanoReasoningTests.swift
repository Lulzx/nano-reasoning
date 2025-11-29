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

// MARK: - Model Loading Tests

final class ModelLoadingTests: XCTestCase {
    
    // MARK: - ModelDownloader Tests
    
    func testModelDownloadConfigCreation() {
        let config = ModelDownloadConfig(modelId: "Qwen/Qwen3-0.6B")
        XCTAssertEqual(config.modelId, "Qwen/Qwen3-0.6B")
        XCTAssertEqual(config.revision, "main")
        XCTAssertFalse(config.forceRedownload)
    }
    
    func testModelDownloadConfigWithCustomCache() {
        let customCache = URL(fileURLWithPath: "/tmp/test-cache")
        let config = ModelDownloadConfig(
            modelId: "Qwen/Qwen3-0.6B",
            revision: "main",
            cacheDirectory: customCache,
            forceRedownload: true
        )
        XCTAssertEqual(config.cacheDirectory, customCache)
        XCTAssertTrue(config.forceRedownload)
    }
    
    func testDownloadProgressDescription() {
        let progress = DownloadProgress(
            bytesDownloaded: 500_000_000,
            totalBytes: 1_000_000_000,
            currentFile: "model.safetensors",
            filesCompleted: 2,
            totalFiles: 4
        )
        
        XCTAssertEqual(progress.fractionCompleted, 0.5, accuracy: 0.01)
        XCTAssertTrue(progress.description.contains("500.0"))
        XCTAssertTrue(progress.description.contains("2/4"))
    }
    
    func testDownloadProgressZeroTotal() {
        let progress = DownloadProgress(
            bytesDownloaded: 0,
            totalBytes: 0,
            currentFile: "starting",
            filesCompleted: 0,
            totalFiles: 0
        )
        
        XCTAssertEqual(progress.fractionCompleted, 0.0)
    }
    
    func testModelDownloadErrorDescriptions() {
        let networkError = ModelDownloadError.networkError("Connection failed")
        XCTAssertTrue(networkError.localizedDescription.contains("Network"))
        
        let fileNotFound = ModelDownloadError.fileNotFound("model.safetensors")
        XCTAssertTrue(fileNotFound.localizedDescription.contains("not found"))
        
        let diskSpace = ModelDownloadError.insufficientDiskSpace(required: 10_000_000_000, available: 1_000_000_000)
        XCTAssertTrue(diskSpace.localizedDescription.contains("GB"))
    }
    
    func testListCachedModels() async {
        let downloader = ModelDownloader.shared
        let cached = await downloader.listCachedModels()
        
        // Should return an array (may be empty if no models cached)
        XCTAssertNotNil(cached)
    }
    
    func testGetCacheSize() async {
        let downloader = ModelDownloader.shared
        let size = await downloader.getCacheSize()
        
        // Cache size should be non-negative
        XCTAssertGreaterThanOrEqual(size, 0)
    }
    
    // MARK: - ModelWeightConfig Tests
    
    func testModelWeightConfigForQwen3Tiers() {
        let entryConfig = ModelWeightConfig.forQwen3(tier: .entry)
        XCTAssertEqual(entryConfig.vocabSize, 151936)
        XCTAssertEqual(entryConfig.hiddenSize, 2560)
        XCTAssertEqual(entryConfig.numLayers, 36)
        
        let proConfig = ModelWeightConfig.forQwen3(tier: .pro)
        XCTAssertEqual(proConfig.vocabSize, 152064)
        XCTAssertEqual(proConfig.hiddenSize, 3584)
        XCTAssertEqual(proConfig.numLayers, 28)
        
        let eliteConfig = ModelWeightConfig.forQwen3(tier: .elite)
        XCTAssertEqual(eliteConfig.vocabSize, 152064)
        XCTAssertEqual(eliteConfig.hiddenSize, 5120)
        XCTAssertEqual(eliteConfig.numLayers, 64)
    }
    
    func testQwen3_0_6BConfig() {
        let config = ModelWeightConfig.qwen3_0_6B
        XCTAssertEqual(config.vocabSize, 151936)
        XCTAssertEqual(config.hiddenSize, 1024)
        XCTAssertEqual(config.numLayers, 28)
        XCTAssertEqual(config.numHeads, 16)
        XCTAssertEqual(config.numKVHeads, 2)
    }
    
    // MARK: - WeightLoader Tests
    
    func testWeightKeyMapping() {
        let originalWeights: [String: MLXArray] = [
            "model.embed_tokens.weight": MLXArray([1.0, 2.0]),
            "model.layers.0.self_attn.q_proj.weight": MLXArray([3.0, 4.0]),
            "model.layers.0.input_layernorm.weight": MLXArray([5.0, 6.0])
        ]
        
        let mapped = WeightLoader.mapWeightKeys(originalWeights)
        
        // "model." prefix should be removed
        XCTAssertNotNil(mapped["embed_tokens.weight"])
        XCTAssertNil(mapped["model.embed_tokens.weight"])
        
        // Keys should be mapped to camelCase
        XCTAssertNotNil(mapped["layers.0.selfAttn.qProj.weight"])
        XCTAssertNotNil(mapped["layers.0.inputLayerNorm.weight"])
    }
    
    // MARK: - DrafterError Tests
    
    func testDrafterErrorDescriptions() {
        let shapeMismatch = DrafterError.shapeMismatch("Expected [100, 200], got [50, 100]")
        XCTAssertTrue(shapeMismatch.localizedDescription.contains("Shape mismatch"))
        
        let weightLoadingFailed = DrafterError.weightLoadingFailed("File not found")
        XCTAssertTrue(weightLoadingFailed.localizedDescription.contains("Weight loading failed"))
        
        let modelNotLoaded = DrafterError.modelNotLoaded
        XCTAssertTrue(modelNotLoaded.localizedDescription.contains("not been loaded"))
        
        let configMismatch = DrafterError.configurationMismatch("Vocab size differs")
        XCTAssertTrue(configMismatch.localizedDescription.contains("Configuration mismatch"))
    }
    
    // MARK: - TargetModel Tests
    
    func testTargetModelInitialization() async throws {
        let config = ModelConfiguration.forTier(.entry)
        let model = TargetModel(configuration: config, tier: .entry)
        
        // Before loading, model should not be loaded
        let isLoaded = await model.checkLoaded()
        XCTAssertFalse(isLoaded)
    }
    
    func testTargetModelDimensions() async throws {
        let config = ModelConfiguration.forTier(.entry)
        let model = TargetModel(configuration: config, tier: .entry)
        
        // Attempt to load (will use tier-based initialization without network)
        do {
            try await model.loadModel()
        } catch {
            // Expected to fail without network/weights, but model should be partially initialized
        }
        
        // Check dimensions are set (even if loading failed, tier-based defaults should be used)
        let vocabSize = await model.getVocabSize()
        let hiddenSize = await model.getHiddenSize()
        let numLayers = await model.getNumLayers()
        
        // Should have valid dimensions from tier-based defaults
        XCTAssertGreaterThan(vocabSize, 0)
        XCTAssertGreaterThan(hiddenSize, 0)
        XCTAssertGreaterThan(numLayers, 0)
    }
    
    func testTargetModelStatistics() async throws {
        let config = ModelConfiguration.forTier(.entry)
        let model = TargetModel(configuration: config, tier: .entry)
        
        // Initial statistics should be zero
        let (totalTokens, totalVerifications) = await model.getStatistics()
        XCTAssertEqual(totalTokens, 0)
        XCTAssertEqual(totalVerifications, 0)
        
        // Reset should not throw
        await model.resetStatistics()
        
        let (afterResetTokens, afterResetVerifications) = await model.getStatistics()
        XCTAssertEqual(afterResetTokens, 0)
        XCTAssertEqual(afterResetVerifications, 0)
    }
    
    // MARK: - DrafterActor Tests
    
    func testFastRLDrafterActorInitialization() async throws {
        let config = SingleLayerDrafterConfig.forQwen3(size: "0.6B")
        let loadMonitor = GPULoadMonitor()
        
        let drafter = FastRLDrafterActor(
            config: config,
            learningRate: 1e-4,
            loadMonitor: loadMonitor
        )
        
        // Initial state
        let isLoaded = await drafter.isWeightsLoaded()
        XCTAssertFalse(isLoaded)
        
        let modelId = await drafter.getModelId()
        XCTAssertNil(modelId)
        
        let acceptanceRate = await drafter.getAcceptanceRate()
        XCTAssertEqual(acceptanceRate, 0)
        
        let trainingStep = await drafter.getTrainingStep()
        XCTAssertEqual(trainingStep, 0)
    }
    
    func testFastRLDrafterActorBufferManagement() async throws {
        let config = SingleLayerDrafterConfig.forQwen3(size: "0.6B")
        let loadMonitor = GPULoadMonitor()
        
        let drafter = FastRLDrafterActor(
            config: config,
            learningRate: 1e-4,
            loadMonitor: loadMonitor
        )
        
        // Collect training data
        await drafter.collectTrainingData(
            hiddenStates: MLXArray([Float](repeating: 0.1, count: 1024)),
            tokenIds: MLXArray([Int32(1), Int32(2), Int32(3)]),
            targetLogits: MLXArray([Float](repeating: 0.01, count: 151936))
        )
        
        let bufferSize = await drafter.getBufferSize()
        XCTAssertEqual(bufferSize, 1)
        
        // Clear buffer
        await drafter.clearBuffer()
        let clearedSize = await drafter.getBufferSize()
        XCTAssertEqual(clearedSize, 0)
    }
    
    func testDrafterActorInitialization() async throws {
        let config = DrafterConfig.qwen3_0_6B
        let loadMonitor = GPULoadMonitor()
        
        let drafter = DrafterActor(
            config: config,
            enableEAGLE: true,
            enableLoRA: false,
            learningRate: 1e-4,
            loadMonitor: loadMonitor
        )
        
        // Initial state
        let isLoaded = await drafter.isWeightsLoaded()
        XCTAssertFalse(isLoaded)
        
        let modelId = await drafter.getModelId()
        XCTAssertNil(modelId)
        
        let acceptanceRate = await drafter.getAcceptanceRate()
        XCTAssertEqual(acceptanceRate, 0)
        
        let isTraining = await drafter.isTraining()
        XCTAssertTrue(isTraining) // EAGLE is enabled
    }
    
    func testDrafterActorTrainingToggle() async throws {
        let config = DrafterConfig.qwen3_0_6B
        let loadMonitor = GPULoadMonitor()
        
        let drafter = DrafterActor(
            config: config,
            enableEAGLE: true,
            enableLoRA: false,
            learningRate: 1e-4,
            loadMonitor: loadMonitor
        )
        
        // Initially enabled (EAGLE is on)
        var isTraining = await drafter.isTraining()
        XCTAssertTrue(isTraining)
        
        // Disable training
        await drafter.setTrainingEnabled(false)
        isTraining = await drafter.isTraining()
        XCTAssertFalse(isTraining)
        
        // Re-enable training
        await drafter.setTrainingEnabled(true)
        isTraining = await drafter.isTraining()
        XCTAssertTrue(isTraining)
    }
    
    // MARK: - SingleLayerDrafterConfig Tests
    
    func testSingleLayerDrafterConfigForQwen3Sizes() {
        let config06B = SingleLayerDrafterConfig.forQwen3(size: "0.6B")
        XCTAssertEqual(config06B.hiddenSize, 1024)
        XCTAssertEqual(config06B.vocabSize, 151936)
        
        let config4B = SingleLayerDrafterConfig.forQwen3(size: "4B")
        XCTAssertEqual(config4B.hiddenSize, 3072)
        XCTAssertEqual(config4B.vocabSize, 152064)
        
        let config7B = SingleLayerDrafterConfig.forQwen3(size: "7B")
        XCTAssertEqual(config7B.hiddenSize, 4096)
        XCTAssertEqual(config7B.vocabSize, 152064)
        
        let config32B = SingleLayerDrafterConfig.forQwen3(size: "32B")
        XCTAssertEqual(config32B.hiddenSize, 5120)
        XCTAssertEqual(config32B.vocabSize, 152064)
        
        // Unknown size should default to 7B
        let configUnknown = SingleLayerDrafterConfig.forQwen3(size: "unknown")
        XCTAssertEqual(configUnknown.hiddenSize, 4096)
    }
}
