// SPDX-License-Identifier: MIT
// Nano-Reasoning: Ring Buffer for Training Samples
// Implements backpressure handling for M1/M2 GPU saturation

import Foundation
import MLX

/// Statistics for the training buffer
public struct BufferStatistics: Sendable {
    public let currentSize: Int
    public let capacity: Int
    public let totalPushed: Int
    public let totalDropped: Int
    public let totalConsumed: Int
    
    public var utilizationRate: Float {
        guard capacity > 0 else { return 0 }
        return Float(currentSize) / Float(capacity)
    }
    
    public var dropRate: Float {
        guard totalPushed > 0 else { return 0 }
        return Float(totalDropped) / Float(totalPushed)
    }
}

/// Priority levels for training samples
public enum SamplePriority: Int, Comparable, Sendable {
    case low = 0
    case normal = 1
    case high = 2
    
    public static func < (lhs: SamplePriority, rhs: SamplePriority) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

/// Training sample with metadata
public struct PrioritizedSample: Sendable {
    public let sample: TrainingSample
    public let priority: SamplePriority
    public let timestamp: Date
    public let rejectionCount: Int  // How many tokens were rejected (higher = more valuable for learning)
    
    public init(
        sample: TrainingSample,
        priority: SamplePriority = .normal,
        rejectionCount: Int = 0
    ) {
        self.sample = sample
        self.priority = priority
        self.timestamp = Date()
        self.rejectionCount = rejectionCount
    }
}

/// Ring buffer actor for decoupling training sample production from consumption
/// Implements backpressure by dropping oldest samples when full
public actor TrainingBuffer {
    private var buffer: [PrioritizedSample] = []
    private let capacity: Int
    private let minBatchSize: Int
    
    // Statistics
    private var totalPushed: Int = 0
    private var totalDropped: Int = 0
    private var totalConsumed: Int = 0
    
    // Backpressure state
    private var isBackpressured: Bool = false
    private let backpressureThreshold: Float = 0.9
    
    /// Initialize buffer with capacity based on hardware tier
    public init(capacity: Int, minBatchSize: Int = 4) {
        self.capacity = max(capacity, 1)
        self.minBatchSize = minBatchSize
    }
    
    /// Convenience initializer from hardware tier
    public init(tier: HardwareTier, minBatchSize: Int = 4) {
        self.init(capacity: tier.bufferCapacity, minBatchSize: minBatchSize)
    }
    

    // Push Operations
    
    /// Push a single sample into the buffer
    /// Returns true if sample was accepted, false if dropped due to capacity
    @discardableResult
    public func push(sample: TrainingSample, priority: SamplePriority = .normal, rejectionCount: Int = 0) -> Bool {
        totalPushed += 1
        
        let prioritizedSample = PrioritizedSample(
            sample: sample,
            priority: priority,
            rejectionCount: rejectionCount
        )
        
        if buffer.count >= capacity {
            // Buffer full - implement smart dropping
            if let dropIndex = findDropCandidate(newPriority: priority) {
                buffer.remove(at: dropIndex)
                totalDropped += 1
            } else {
                // New sample is lowest priority, drop it
                totalDropped += 1
                return false
            }
        }
        
        buffer.append(prioritizedSample)
        updateBackpressureState()
        return true
    }
    
    /// Push multiple samples (batch operation)
    public func pushBatch(samples: [(TrainingSample, SamplePriority, Int)]) -> Int {
        var accepted = 0
        for (sample, priority, rejectionCount) in samples {
            if push(sample: sample, priority: priority, rejectionCount: rejectionCount) {
                accepted += 1
            }
        }
        return accepted
    }
    
    /// Push rejection data from speculative decoding verification
    public func pushRejectionData(
        hiddenStates: MLXArray,
        targetLogits: MLXArray,
        inputIds: MLXArray,
        rejectedCount: Int
    ) {
        let sample = TrainingSample(
            hiddenStates: hiddenStates,
            targetLogits: targetLogits,
            inputIds: inputIds
        )
        
        // Higher rejection count = higher priority (more valuable for learning)
        let priority: SamplePriority
        switch rejectedCount {
        case 0...1: priority = .low
        case 2...3: priority = .normal
        default: priority = .high
        }
        
        push(sample: sample, priority: priority, rejectionCount: rejectedCount)
    }
    

    // Pop Operations
    
    /// Pop a single sample (FIFO with priority consideration)
    public func pop() -> TrainingSample? {
        guard !buffer.isEmpty else { return nil }
        
        // Find highest priority sample
        let index = findHighestPrioritySample()
        let sample = buffer.remove(at: index)
        totalConsumed += 1
        updateBackpressureState()
        
        return sample.sample
    }
    
    /// Pop a batch of samples for training
    public func popBatch(maxSize: Int) -> [TrainingSample] {
        guard buffer.count >= minBatchSize else { return [] }
        
        let batchSize = min(maxSize, buffer.count)
        var samples: [TrainingSample] = []
        
        // Sort by priority and get top samples
        let sorted = buffer.enumerated()
            .sorted { $0.element.priority > $1.element.priority }
            .prefix(batchSize)
            .map { $0.offset }
            .sorted(by: >)  // Sort indices in descending order for safe removal
        
        for index in sorted {
            samples.append(buffer[index].sample)
            buffer.remove(at: index)
        }
        
        totalConsumed += samples.count
        updateBackpressureState()
        
        return samples
    }
    
    /// Peek at the next sample without removing
    public func peek() -> TrainingSample? {
        guard !buffer.isEmpty else { return nil }
        let index = findHighestPrioritySample()
        return buffer[index].sample
    }
    

    // Buffer Management
    
    /// Clear all samples
    public func clear() {
        let dropped = buffer.count
        buffer.removeAll()
        totalDropped += dropped
        updateBackpressureState()
    }
    
    /// Check if buffer has enough samples for a training batch
    public func hasTrainingBatch() -> Bool {
        buffer.count >= minBatchSize
    }
    
    /// Check if buffer is experiencing backpressure
    public func isUnderBackpressure() -> Bool {
        isBackpressured
    }
    
    /// Get current buffer size
    public func count() -> Int {
        buffer.count
    }
    
    /// Check if buffer is empty
    public func isEmpty() -> Bool {
        buffer.isEmpty
    }
    
    /// Check if buffer is full
    public func isFull() -> Bool {
        buffer.count >= capacity
    }
    

    // Statistics
    
    /// Get buffer statistics
    public func getStatistics() -> BufferStatistics {
        BufferStatistics(
            currentSize: buffer.count,
            capacity: capacity,
            totalPushed: totalPushed,
            totalDropped: totalDropped,
            totalConsumed: totalConsumed
        )
    }
    
    /// Reset statistics
    public func resetStatistics() {
        totalPushed = 0
        totalDropped = 0
        totalConsumed = 0
    }
    

    // Private Helpers
    
    /// Find the best candidate to drop when buffer is full
    private func findDropCandidate(newPriority: SamplePriority) -> Int? {
        // Find oldest sample with priority <= new sample's priority
        var candidateIndex: Int?
        var oldestTime: Date?
        
        for (index, prioritizedSample) in buffer.enumerated() {
            if prioritizedSample.priority <= newPriority {
                if oldestTime == nil || prioritizedSample.timestamp < oldestTime! {
                    oldestTime = prioritizedSample.timestamp
                    candidateIndex = index
                }
            }
        }
        
        return candidateIndex
    }
    
    /// Find the highest priority sample to pop
    private func findHighestPrioritySample() -> Int {
        guard !buffer.isEmpty else { return 0 }
        
        var bestIndex = 0
        var bestPriority = buffer[0].priority
        var bestRejectionCount = buffer[0].rejectionCount
        
        for (index, sample) in buffer.enumerated() {
            // Prioritize by: priority level, then rejection count, then age (FIFO for ties)
            if sample.priority > bestPriority ||
               (sample.priority == bestPriority && sample.rejectionCount > bestRejectionCount) {
                bestIndex = index
                bestPriority = sample.priority
                bestRejectionCount = sample.rejectionCount
            }
        }
        
        return bestIndex
    }
    
    /// Update backpressure state
    private func updateBackpressureState() {
        let utilization = Float(buffer.count) / Float(capacity)
        isBackpressured = utilization >= backpressureThreshold
    }
}

// Async Stream Support

extension TrainingBuffer {
    /// Create an async stream of training batches
    /// Uses nonisolated(unsafe) to avoid unnecessary actor isolation in stream
    public nonisolated func batchStream(batchSize: Int, pollInterval: Duration = .milliseconds(100)) -> AsyncStream<[TrainingSample]> {
        let buffer = self
        return AsyncStream { continuation in
            let task = Task {
                while !Task.isCancelled {
                    let batch = await buffer.popBatch(maxSize: batchSize)
                    if !batch.isEmpty {
                        continuation.yield(batch)
                    }
                    try? await Task.sleep(for: pollInterval)
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
}

// Multi-Buffer Strategy

/// Manages multiple buffers for different sample types
public actor MultiTrainingBuffer {
    private var buffers: [String: TrainingBuffer] = [:]
    private let defaultCapacity: Int
    
    public init(defaultCapacity: Int = 50) {
        self.defaultCapacity = defaultCapacity
    }
    
    /// Get or create a buffer for a specific category
    public func buffer(for category: String) -> TrainingBuffer {
        if let existing = buffers[category] {
            return existing
        }
        let newBuffer = TrainingBuffer(capacity: defaultCapacity)
        buffers[category] = newBuffer
        return newBuffer
    }
    
    /// Push to a specific category
    public func push(sample: TrainingSample, category: String, priority: SamplePriority = .normal) async {
        let buf = buffer(for: category)
        await buf.push(sample: sample, priority: priority)
    }
    
    /// Pop from a specific category
    public func pop(from category: String) async -> TrainingSample? {
        guard let buf = buffers[category] else { return nil }
        return await buf.pop()
    }
    
    /// Get aggregated statistics
    public func getAggregatedStatistics() async -> [String: BufferStatistics] {
        var stats: [String: BufferStatistics] = [:]
        for (category, buffer) in buffers {
            stats[category] = await buffer.getStatistics()
        }
        return stats
    }
}
