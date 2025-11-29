// SPDX-License-Identifier: MIT
// Nano-Reasoning: HuggingFace Tokenizer Integration
// Provides proper BPE/WordPiece tokenization for Qwen models

import Foundation
import Tokenizers
import Hub

/// Tokenizer configuration for model compatibility
public struct TokenizerConfig: Sendable {
    public let modelId: String
    public let padTokenId: Int32
    public let eosTokenId: Int32
    public let bosTokenId: Int32?
    public let chatTemplate: String?
    
    public static let qwen3 = TokenizerConfig(
        modelId: "Qwen/Qwen3-0.6B-Instruct",
        padTokenId: 151643,
        eosTokenId: 151645,
        bosTokenId: nil,
        chatTemplate: """
        {%- for message in messages %}
        {%- if message['role'] == 'system' %}
        <|im_start|>system
        {{ message['content'] }}<|im_end|>
        {%- elif message['role'] == 'user' %}
        <|im_start|>user
        {{ message['content'] }}<|im_end|>
        {%- elif message['role'] == 'assistant' %}
        <|im_start|>assistant
        {{ message['content'] }}<|im_end|>
        {%- endif %}
        {%- endfor %}
        {%- if add_generation_prompt %}
        <|im_start|>assistant
        {%- endif %}
        """
    )
}

/// Chat message for template formatting
public struct ChatMessage: Sendable {
    public let role: String
    public let content: String
    
    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
    
    public static func system(_ content: String) -> ChatMessage {
        ChatMessage(role: "system", content: content)
    }
    
    public static func user(_ content: String) -> ChatMessage {
        ChatMessage(role: "user", content: content)
    }
    
    public static func assistant(_ content: String) -> ChatMessage {
        ChatMessage(role: "assistant", content: content)
    }
}

/// Tokenizer wrapper for HuggingFace tokenizers
public actor TokenizerManager {
    private var tokenizer: Tokenizer?
    private let config: TokenizerConfig
    private var isLoaded: Bool = false
    
    // Cache for common tokens
    private var specialTokenCache: [String: Int32] = [:]
    
    public init(config: TokenizerConfig = .qwen3) {
        self.config = config
    }
    
    /// Load the tokenizer from HuggingFace Hub
    public func load() async throws {
        guard !isLoaded else { return }
        
        do {
            tokenizer = try await AutoTokenizer.from(pretrained: config.modelId)
            isLoaded = true
            
            // Cache special tokens
            specialTokenCache["<|im_start|>"] = try? encode("<|im_start|>").first
            specialTokenCache["<|im_end|>"] = try? encode("<|im_end|>").first
            specialTokenCache["<|endoftext|>"] = config.eosTokenId
        } catch {
            throw TokenizerError.loadFailed(config.modelId, error.localizedDescription)
        }
    }
    
    /// Encode text to token IDs
    public func encode(_ text: String, addSpecialTokens: Bool = true) throws -> [Int32] {
        guard let tokenizer = tokenizer else {
            throw TokenizerError.notLoaded
        }
        
        let encoded = tokenizer.encode(text: text)
        return encoded.map { Int32($0) }
    }
    
    /// Encode with truncation and padding
    public func encode(
        _ text: String,
        maxLength: Int,
        padding: Bool = false,
        truncation: Bool = true
    ) throws -> [Int32] {
        var tokens = try encode(text)
        
        // Truncate if needed
        if truncation && tokens.count > maxLength {
            tokens = Array(tokens.prefix(maxLength))
        }
        
        // Pad if needed
        if padding && tokens.count < maxLength {
            let padCount = maxLength - tokens.count
            tokens.append(contentsOf: Array(repeating: config.padTokenId, count: padCount))
        }
        
        return tokens
    }
    
    /// Decode token IDs to text
    public func decode(_ tokens: [Int32], skipSpecialTokens: Bool = true) throws -> String {
        guard let tokenizer = tokenizer else {
            throw TokenizerError.notLoaded
        }
        
        let intTokens = tokens.map { Int($0) }
        return tokenizer.decode(tokens: intTokens)
    }
    
    /// Decode a single token
    public func decodeToken(_ token: Int32) throws -> String {
        try decode([token], skipSpecialTokens: false)
    }
    
    /// Apply chat template to messages
    public func applyChatTemplate(
        messages: [ChatMessage],
        addGenerationPrompt: Bool = true
    ) throws -> String {
        // Build formatted string using Qwen chat format
        var result = ""
        
        for message in messages {
            result += "<|im_start|>\(message.role)\n\(message.content)<|im_end|>\n"
        }
        
        if addGenerationPrompt {
            result += "<|im_start|>assistant\n"
        }
        
        return result
    }
    
    /// Encode chat messages
    public func encodeChat(
        messages: [ChatMessage],
        addGenerationPrompt: Bool = true
    ) throws -> [Int32] {
        let formatted = try applyChatTemplate(messages: messages, addGenerationPrompt: addGenerationPrompt)
        return try encode(formatted)
    }
    
    /// Get vocabulary size (uses config default since tokenizer doesn't expose this)
    public func vocabSize() -> Int {
        // Qwen3 vocabulary size
        151936
    }
    
    /// Check if token is EOS
    public func isEOS(_ token: Int32) -> Bool {
        token == config.eosTokenId
    }
    
    /// Check if token is special
    public func isSpecialToken(_ token: Int32) -> Bool {
        token == config.eosTokenId ||
        token == config.padTokenId ||
        (config.bosTokenId != nil && token == config.bosTokenId)
    }
    
    /// Get EOS token ID
    public func getEOSTokenId() -> Int32 {
        config.eosTokenId
    }
    
    /// Get PAD token ID
    public func getPadTokenId() -> Int32 {
        config.padTokenId
    }
}

/// Tokenizer errors
public enum TokenizerError: Error, LocalizedError {
    case notLoaded
    case loadFailed(String, String)
    case encodingFailed(String)
    case decodingFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .notLoaded:
            return "Tokenizer has not been loaded"
        case .loadFailed(let modelId, let reason):
            return "Failed to load tokenizer '\(modelId)': \(reason)"
        case .encodingFailed(let reason):
            return "Encoding failed: \(reason)"
        case .decodingFailed(let reason):
            return "Decoding failed: \(reason)"
        }
    }
}

/// Batch tokenization utilities
public struct BatchTokenizer {
    
    /// Encode multiple texts in parallel
    public static func encodeBatch(
        texts: [String],
        tokenizer: TokenizerManager,
        maxLength: Int? = nil
    ) async throws -> [[Int32]] {
        try await withThrowingTaskGroup(of: (Int, [Int32]).self) { group in
            for (index, text) in texts.enumerated() {
                group.addTask {
                    let tokens: [Int32]
                    if let maxLen = maxLength {
                        tokens = try await tokenizer.encode(text, maxLength: maxLen)
                    } else {
                        tokens = try await tokenizer.encode(text)
                    }
                    return (index, tokens)
                }
            }
            
            var results: [(Int, [Int32])] = []
            for try await result in group {
                results.append(result)
            }
            
            return results.sorted { $0.0 < $1.0 }.map { $0.1 }
        }
    }
    
    /// Pad a batch of token sequences to the same length
    public static func padBatch(
        sequences: [[Int32]],
        padTokenId: Int32,
        maxLength: Int? = nil
    ) -> (paddedSequences: [[Int32]], attentionMask: [[Int32]]) {
        let targetLength = maxLength ?? sequences.map { $0.count }.max() ?? 0
        
        var paddedSequences: [[Int32]] = []
        var attentionMask: [[Int32]] = []
        
        for sequence in sequences {
            let padCount = max(0, targetLength - sequence.count)
            let padded = sequence + Array(repeating: padTokenId, count: padCount)
            let mask = Array(repeating: Int32(1), count: sequence.count) +
                      Array(repeating: Int32(0), count: padCount)
            
            paddedSequences.append(Array(padded.prefix(targetLength)))
            attentionMask.append(Array(mask.prefix(targetLength)))
        }
        
        return (paddedSequences, attentionMask)
    }
}
