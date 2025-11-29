// SPDX-License-Identifier: MIT
// Nano-Reasoning: CLI Entry Point
// FastRL Adaptive Drafter for Apple Silicon

import ArgumentParser
import Foundation
import NanoReasoningCore

@main
struct NanoReasoningCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "nano-reasoning",
        abstract: "FastRL Adaptive Drafter for Apple Silicon",
        version: NanoReasoningCore.version,
        subcommands: [
            Generate.self,
            Info.self,
            Benchmark.self,
            Chat.self,
        ],
        defaultSubcommand: Chat.self
    )
}

// Generate Command

struct Generate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "generate",
        abstract: "Generate text from a prompt"
    )
    
    @Argument(help: "The prompt to generate from")
    var prompt: String
    
    @Option(name: .shortAndLong, help: "Maximum tokens to generate")
    var maxTokens: Int = 512
    
    @Option(name: .shortAndLong, help: "Temperature for sampling (0.0 = greedy)")
    var temperature: Float = 0.7
    
    @Option(name: .long, help: "Top-p (nucleus) sampling parameter")
    var topP: Float = 0.9
    
    @Option(name: .long, help: "Top-k sampling parameter")
    var topK: Int = 40
    
    @Flag(name: .long, help: "Disable speculative decoding")
    var noSpeculation: Bool = false
    
    @Flag(name: .long, help: "Show generation statistics")
    var stats: Bool = false
    
    func run() async throws {
        // Check requirements
        let (meets, message) = NanoReasoning.checkRequirements()
        if !meets {
            print("Error: \(message)")
            throw ExitCode.failure
        }
        
        // Create configuration
        let config = GenerationConfig(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            topK: topK,
            repetitionPenalty: 1.1,
            stopTokens: []
        )
        
        // Initialize orchestrator
        print("Initializing Nano-Reasoning...")
        let orchestrator = try await NanoReasoning.createOrchestrator(generationConfig: config)
        
        print("\nGenerating...\n")
        print("─" * 60)
        
        // Generate with streaming output
        let startTime = Date()
        
        if noSpeculation {
            _ = try await orchestrator.generateStandard(prompt: prompt, maxTokens: maxTokens) { token in
                print(token, terminator: "")
                fflush(stdout)
                return true
            }
        } else {
            _ = try await orchestrator.generate(prompt: prompt, maxTokens: maxTokens) { token in
                print(token, terminator: "")
                fflush(stdout)
                return true
            }
        }
        
        let elapsed = Date().timeIntervalSince(startTime)
        
        print("\n")
        print("─" * 60)
        
        // Show statistics
        if stats {
            if let sessionStats = await orchestrator.getSessionStatistics() {
                print("\nStatistics:")
                print("  Total tokens: \(sessionStats.totalTokens)")
                print("  Drafted tokens: \(sessionStats.draftedTokens)")
                print("  Accepted tokens: \(sessionStats.acceptedTokens)")
                print("  Acceptance rate: \(String(format: "%.1f%%", sessionStats.averageAcceptanceRate * 100))")
                print("  Tokens/second: \(String(format: "%.1f", sessionStats.tokensPerSecond))")
                print("  Time: \(String(format: "%.2fs", elapsed))")
            }
        }
    }
}

// Info Command

struct Info: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "info",
        abstract: "Display system information and configuration"
    )
    
    func run() async throws {
        let profile = NanoReasoning.getSystemInfo()
        
        print("╔══════════════════════════════════════════════════════════╗")
        print("║            Nano-Reasoning System Information              ║")
        print("╠══════════════════════════════════════════════════════════╣")
        print("║ Version: \(NanoReasoningCore.version.padding(toLength: 48, withPad: " ", startingAt: 0))║")
        print("╠══════════════════════════════════════════════════════════╣")
        print("║ Hardware Profile:                                        ║")
        print("║   Chip: \(profile.chipFamily) \(profile.chipVariant)".padding(toLength: 60, withPad: " ", startingAt: 0) + "║")
        print("║   GPU: \(profile.gpuName)".padding(toLength: 60, withPad: " ", startingAt: 0) + "║")
        print("║   Memory: \(profile.totalMemoryGB) GB".padding(toLength: 58, withPad: " ", startingAt: 0) + "║")
        print("║   Unified Memory: \(profile.isUnifiedMemory ? "Yes" : "No")".padding(toLength: 50, withPad: " ", startingAt: 0) + "║")
        print("╠══════════════════════════════════════════════════════════╣")
        print("║ Configuration:                                           ║")
        print("║   Tier: \(profile.tier)".padding(toLength: 60, withPad: " ", startingAt: 0) + "║")
        print("║   Training Enabled: \(profile.tier.trainingEnabled ? "Yes" : "No")".padding(toLength: 48, withPad: " ", startingAt: 0) + "║")
        print("║   NPU Offload: \(profile.tier.npuOffloadAvailable ? "Available" : "Not Available")".padding(toLength: 53, withPad: " ", startingAt: 0) + "║")
        print("║   Draft Count (k): \(profile.tier.draftCount)".padding(toLength: 49, withPad: " ", startingAt: 0) + "║")
        print("╠══════════════════════════════════════════════════════════╣")
        
        let modelConfig = ModelConfiguration.forTier(profile.tier)
        print("║ Models:                                                  ║")
        print("║   Target: \(modelConfig.targetModelId)".padding(toLength: 58, withPad: " ", startingAt: 0) + "║")
        print("║   Drafter: \(modelConfig.drafterModelId)".padding(toLength: 57, withPad: " ", startingAt: 0) + "║")
        print("╚══════════════════════════════════════════════════════════╝")
        
        // Requirements check
        let (meets, message) = NanoReasoning.checkRequirements()
        if meets {
            print("\n✓ System meets all requirements")
        } else {
            print("\n✗ \(message)")
        }
    }
}

// Benchmark Command

struct Benchmark: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "benchmark",
        abstract: "Run performance benchmarks"
    )
    
    @Option(name: .shortAndLong, help: "Number of iterations")
    var iterations: Int = 5
    
    @Option(name: .shortAndLong, help: "Tokens to generate per iteration")
    var tokens: Int = 100
    
    @Flag(name: .long, help: "Compare with non-speculative generation")
    var compare: Bool = false
    
    func run() async throws {
        let (meets, message) = NanoReasoning.checkRequirements()
        if !meets {
            print("Error: \(message)")
            throw ExitCode.failure
        }
        
        print("Nano-Reasoning Benchmark")
        print("========================")
        print("Iterations: \(iterations)")
        print("Tokens per iteration: \(tokens)")
        print()
        
        let config = GenerationConfig(
            maxTokens: tokens,
            temperature: 0.7,
            topP: 0.9,
            topK: 40,
            repetitionPenalty: 1.1,
            stopTokens: []
        )
        
        print("Initializing...")
        let orchestrator = try await NanoReasoning.createOrchestrator(generationConfig: config)
        
        let testPrompt = "Explain the concept of machine learning in simple terms:"
        
        // Speculative benchmark
        print("\nRunning speculative decoding benchmark...")
        var specTimes: [Double] = []
        var specTokensPerSec: [Float] = []
        
        for i in 1...iterations {
            print("  Iteration \(i)/\(iterations)...", terminator: "")
            fflush(stdout)
            
            let start = Date()
            _ = try await orchestrator.generate(prompt: testPrompt, maxTokens: tokens)
            let elapsed = Date().timeIntervalSince(start)
            
            specTimes.append(elapsed)
            if let stats = await orchestrator.getSessionStatistics() {
                specTokensPerSec.append(stats.tokensPerSecond)
            }
            
            print(" \(String(format: "%.2fs", elapsed))")
            await orchestrator.resetStatistics()
        }
        
        // Print speculative results
        let avgSpecTime = specTimes.reduce(0, +) / Double(specTimes.count)
        let avgSpecTPS = specTokensPerSec.reduce(0, +) / Float(specTokensPerSec.count)
        
        print("\nSpeculative Decoding Results:")
        print("  Average time: \(String(format: "%.2fs", avgSpecTime))")
        print("  Average tokens/sec: \(String(format: "%.1f", avgSpecTPS))")
        
        // Compare with standard generation
        if compare {
            print("\nRunning standard generation benchmark...")
            var stdTimes: [Double] = []
            
            for i in 1...iterations {
                print("  Iteration \(i)/\(iterations)...", terminator: "")
                fflush(stdout)
                
                let start = Date()
                _ = try await orchestrator.generateStandard(prompt: testPrompt, maxTokens: tokens)
                let elapsed = Date().timeIntervalSince(start)
                
                stdTimes.append(elapsed)
                print(" \(String(format: "%.2fs", elapsed))")
            }
            
            let avgStdTime = stdTimes.reduce(0, +) / Double(stdTimes.count)
            let avgStdTPS = Float(tokens) / Float(avgStdTime)
            
            print("\nStandard Generation Results:")
            print("  Average time: \(String(format: "%.2fs", avgStdTime))")
            print("  Average tokens/sec: \(String(format: "%.1f", avgStdTPS))")
            
            let speedup = avgStdTime / avgSpecTime
            print("\nSpeedup: \(String(format: "%.2fx", speedup))")
        }
        
        // Final statistics
        let (_, drafted, accepted, _, rate) = await orchestrator.getCumulativeStatistics()
        print("\nOverall Statistics:")
        print("  Total drafted: \(drafted)")
        print("  Total accepted: \(accepted)")
        print("  Acceptance rate: \(String(format: "%.1f%%", rate * 100))")
    }
}

// Chat Command

struct Chat: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "chat",
        abstract: "Interactive chat mode"
    )
    
    @Option(name: .shortAndLong, help: "Maximum tokens per response")
    var maxTokens: Int = 1024
    
    @Option(name: .shortAndLong, help: "Temperature for sampling")
    var temperature: Float = 0.7
    
    @Flag(name: .long, help: "Show performance statistics")
    var stats: Bool = false
    
    func run() async throws {
        let (meets, message) = NanoReasoning.checkRequirements()
        if !meets {
            print("Error: \(message)")
            throw ExitCode.failure
        }
        
        let profile = NanoReasoning.getSystemInfo()
        
        print("""
        ╔══════════════════════════════════════════════════════════╗
        ║              Nano-Reasoning Interactive Chat              ║
        ║                   FastRL on Apple Silicon                 ║
        ╠══════════════════════════════════════════════════════════╣
        ║  Hardware: \(profile.chipFamily) \(profile.chipVariant) (\(profile.totalMemoryGB)GB)
        ║  Tier: \(profile.tier)
        ║  Training: \(profile.tier.trainingEnabled ? "Enabled" : "Disabled")
        ╠══════════════════════════════════════════════════════════╣
        ║  Commands:                                               ║
        ║    /quit, /exit  - Exit the chat                         ║
        ║    /stats        - Show generation statistics            ║
        ║    /clear        - Clear conversation history            ║
        ║    /help         - Show this help message                ║
        ╚══════════════════════════════════════════════════════════╝
        """)
        
        let config = GenerationConfig(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: 0.9,
            topK: 40,
            repetitionPenalty: 1.1,
            stopTokens: []
        )
        
        print("\nInitializing models...")
        let orchestrator = try await NanoReasoning.createOrchestrator(generationConfig: config)
        
        // Start background training if available
        if profile.tier.trainingEnabled {
            await orchestrator.startTraining()
            print("Background training started.")
        }
        
        print("\nReady! Type your message and press Enter.\n")
        
        var conversationHistory: [String] = []
        
        while true {
            print("You: ", terminator: "")
            fflush(stdout)
            
            guard let input = readLine()?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !input.isEmpty else {
                continue
            }
            
            // Handle commands
            if input.hasPrefix("/") {
                let command = input.lowercased()
                switch command {
                case "/quit", "/exit":
                    print("\nGoodbye!")
                    await orchestrator.stopTraining()
                    return
                    
                case "/stats":
                    await printStats(orchestrator)
                    continue
                    
                case "/clear":
                    conversationHistory.removeAll()
                    await orchestrator.resetStatistics()
                    print("Conversation cleared.\n")
                    continue
                    
                case "/help":
                    printHelp()
                    continue
                    
                default:
                    print("Unknown command. Type /help for available commands.\n")
                    continue
                }
            }
            
            // Add to conversation
            conversationHistory.append("User: \(input)")
            
            // Build prompt with context
            let prompt = buildPrompt(history: conversationHistory)
            
            print("\nAssistant: ", terminator: "")
            fflush(stdout)
            
            let startTime = Date()
            var response = ""
            
            do {
                response = try await orchestrator.generate(prompt: prompt, maxTokens: maxTokens) { token in
                    print(token, terminator: "")
                    fflush(stdout)
                    return true
                }
                
                let elapsed = Date().timeIntervalSince(startTime)
                
                print("\n")
                
                if stats {
                    if let sessionStats = await orchestrator.getSessionStatistics() {
                        print("[\(String(format: "%.1f", sessionStats.tokensPerSecond)) tok/s, " +
                              "\(String(format: "%.0f%%", sessionStats.averageAcceptanceRate * 100)) accepted, " +
                              "\(String(format: "%.2fs", elapsed))]")
                    }
                    print()
                }
                
                // Add response to history
                conversationHistory.append("Assistant: \(response)")
                
            } catch {
                print("\nError: \(error.localizedDescription)\n")
            }
        }
    }
    
    private func buildPrompt(history: [String]) -> String {
        // Simple prompt building - could be enhanced with chat template
        let context = history.suffix(10).joined(separator: "\n")
        return "\(context)\nAssistant:"
    }
    
    private func printStats(_ orchestrator: Orchestrator) async {
        print("\n--- Generation Statistics ---")
        
        let (total, drafted, accepted, verifications, rate) = await orchestrator.getCumulativeStatistics()
        print("Total tokens: \(total)")
        print("Drafted: \(drafted)")
        print("Accepted: \(accepted)")
        print("Verifications: \(verifications)")
        print("Acceptance rate: \(String(format: "%.1f%%", rate * 100))")
        
        if let training = await orchestrator.getTrainingStatistics() {
            print("\n--- Training Statistics ---")
            print("Training step: \(training.step)")
            print("Loss: \(String(format: "%.4f", training.loss))")
            print("Drafter acceptance: \(String(format: "%.1f%%", training.acceptanceRate * 100))")
        }
        
        if let buffer = await orchestrator.getBufferStatistics() {
            print("\n--- Buffer Statistics ---")
            print("Buffer size: \(buffer.currentSize)/\(buffer.capacity)")
            print("Utilization: \(String(format: "%.1f%%", buffer.utilizationRate * 100))")
            print("Drop rate: \(String(format: "%.1f%%", buffer.dropRate * 100))")
        }
        
        print()
    }
    
    private func printHelp() {
        print("""
        
        Available Commands:
          /quit, /exit  - Exit the chat
          /stats        - Show detailed generation and training statistics
          /clear        - Clear conversation history and reset statistics
          /help         - Show this help message
        
        Tips:
          - The drafter model learns from rejections during generation
          - Acceptance rate improves over time with training enabled
          - Use /stats to monitor the learning progress
        
        """)
    }
}

// String Extension

extension String {
    static func * (left: String, right: Int) -> String {
        String(repeating: left, count: right)
    }
}
