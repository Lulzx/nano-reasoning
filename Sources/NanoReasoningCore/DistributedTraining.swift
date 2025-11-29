// SPDX-License-Identifier: MIT
// Nano-Reasoning: Distributed Training Across Multiple Macs
// Implements gradient synchronization and data parallelism

import Foundation
import Network
@preconcurrency import MLX
import GRPC
import NIOCore
import NIOPosix

// MARK: - Node Configuration

/// Role of a node in the distributed training cluster
public enum NodeRole: String, Sendable, Codable {
    case coordinator  // Primary node that orchestrates training
    case worker       // Worker nodes that compute gradients
}

/// Configuration for a distributed training node
public struct DistributedConfig: Sendable, Codable {
    /// Unique identifier for this node
    public let nodeId: String
    /// Role of this node
    public let role: NodeRole
    /// Port for gRPC server
    public let port: Int
    /// Coordinator address (for workers)
    public let coordinatorAddress: String?
    /// Number of workers expected (for coordinator)
    public let expectedWorkers: Int
    /// Gradient sync interval (in steps)
    public let syncInterval: Int
    /// Use gradient compression
    public let useCompression: Bool
    /// Ring AllReduce topology
    public let useRingAllReduce: Bool
    
    public init(
        nodeId: String = UUID().uuidString,
        role: NodeRole = .worker,
        port: Int = 50051,
        coordinatorAddress: String? = nil,
        expectedWorkers: Int = 2,
        syncInterval: Int = 1,
        useCompression: Bool = true,
        useRingAllReduce: Bool = true
    ) {
        self.nodeId = nodeId
        self.role = role
        self.port = port
        self.coordinatorAddress = coordinatorAddress
        self.expectedWorkers = expectedWorkers
        self.syncInterval = syncInterval
        self.useCompression = useCompression
        self.useRingAllReduce = useRingAllReduce
    }
    
    public static let coordinator = DistributedConfig(
        role: .coordinator,
        port: 50051,
        expectedWorkers: 2
    )
    
    public static let worker = DistributedConfig(
        role: .worker,
        coordinatorAddress: "localhost:50051"
    )
}

// MARK: - Peer Discovery with Bonjour

/// Bonjour-based peer discovery for local network Macs
public actor PeerDiscovery {
    private let serviceType = "_nanoreasoning._tcp"
    private let serviceDomain = "local."
    private var browser: NWBrowser?
    private var listener: NWListener?
    
    private var discoveredPeers: [String: NWEndpoint] = [:]
    private var peerUpdateHandler: ((String, NWEndpoint, Bool) -> Void)?
    
    private let nodeId: String
    private let port: UInt16
    
    public init(nodeId: String, port: UInt16) {
        self.nodeId = nodeId
        self.port = port
    }
    
    /// Start advertising this node and browsing for peers
    public func start() async throws {
        // Start advertising our service
        try startAdvertising()
        
        // Start browsing for peers
        startBrowsing()
    }
    
    /// Stop discovery
    public func stop() {
        browser?.cancel()
        listener?.cancel()
        browser = nil
        listener = nil
    }
    
    /// Get all discovered peers
    public func getPeers() -> [String: NWEndpoint] {
        discoveredPeers
    }
    
    /// Set handler for peer updates
    public func onPeerUpdate(_ handler: @escaping (String, NWEndpoint, Bool) -> Void) {
        peerUpdateHandler = handler
    }
    
    private func startAdvertising() throws {
        // Create TXT record with node info
        let txtRecord = NWTXTRecord(["nodeId": nodeId, "port": "\(port)"])
        
        // Create listener parameters
        let parameters = NWParameters.tcp
        parameters.includePeerToPeer = true
        
        // Create listener on the specified port
        listener = try NWListener(using: parameters, on: NWEndpoint.Port(rawValue: port)!)
        
        // Set service info
        listener?.service = NWListener.Service(name: nodeId, type: serviceType, domain: serviceDomain, txtRecord: txtRecord)
        
        listener?.stateUpdateHandler = { state in
            switch state {
            case .ready:
                print("Advertising on port \(self.port)")
            case .failed(let error):
                print("Listener failed: \(error)")
            default:
                break
            }
        }
        
        listener?.start(queue: .global())
    }
    
    private func startBrowsing() {
        let descriptor = NWBrowser.Descriptor.bonjour(type: serviceType, domain: serviceDomain)
        let parameters = NWParameters()
        parameters.includePeerToPeer = true
        
        browser = NWBrowser(for: descriptor, using: parameters)
        
        browser?.browseResultsChangedHandler = { [weak self] results, changes in
            Task {
                await self?.handleBrowseResults(results, changes: changes)
            }
        }
        
        browser?.stateUpdateHandler = { state in
            switch state {
            case .ready:
                print("Browsing for peers...")
            case .failed(let error):
                print("Browser failed: \(error)")
            default:
                break
            }
        }
        
        browser?.start(queue: .global())
    }
    
    private func handleBrowseResults(_ results: Set<NWBrowser.Result>, changes: Set<NWBrowser.Result.Change>) {
        for change in changes {
            switch change {
            case .added(let result):
                if case .service(let name, _, _, _) = result.endpoint {
                    if name != nodeId {  // Don't add ourselves
                        discoveredPeers[name] = result.endpoint
                        peerUpdateHandler?(name, result.endpoint, true)
                    }
                }
            case .removed(let result):
                if case .service(let name, _, _, _) = result.endpoint {
                    discoveredPeers.removeValue(forKey: name)
                    peerUpdateHandler?(name, result.endpoint, false)
                }
            default:
                break
            }
        }
    }
}

// MARK: - Gradient Synchronization

/// Sendable wrapper for gradient dictionaries
public struct SendableGradients: @unchecked Sendable {
    public var gradients: [String: MLXArray]
    
    public init(_ gradients: [String: MLXArray]) {
        self.gradients = gradients
    }
}

/// Gradient tensor for network transmission
public struct GradientTensor: Sendable {
    public let name: String
    public let shape: [Int]
    public let data: Data
    public let isCompressed: Bool
    
    public init(name: String, array: MLXArray, compress: Bool = true) {
        self.name = name
        self.shape = array.shape
        
        // Convert to Data
        let floatArray = array.asArray(Float.self)
        var data = Data(capacity: floatArray.count * MemoryLayout<Float>.size)
        floatArray.withUnsafeBufferPointer { buffer in
            data.append(buffer)
        }
        
        if compress {
            // Simple quantization compression (top-k sparsification + quantization)
            self.data = GradientCompression.compress(data)
            self.isCompressed = true
        } else {
            self.data = data
            self.isCompressed = false
        }
    }
    
    public func toMLXArray() -> MLXArray {
        let decompressed = isCompressed ? GradientCompression.decompress(data) : data
        
        let floatCount = decompressed.count / MemoryLayout<Float>.size
        var floats = [Float](repeating: 0, count: floatCount)
        _ = floats.withUnsafeMutableBytes { buffer in
            decompressed.copyBytes(to: buffer)
        }
        
        return MLXArray(floats).reshaped(shape)
    }
}

/// Gradient compression utilities
public struct GradientCompression {
    
    /// Compress gradients using top-k sparsification
    public static func compress(_ data: Data, topKRatio: Float = 0.1) -> Data {
        // For now, use simple zlib-style compression
        // In production, implement top-k sparsification + quantization
        return (try? (data as NSData).compressed(using: .lzfse) as Data) ?? data
    }
    
    /// Decompress gradients
    public static func decompress(_ data: Data) -> Data {
        return (try? (data as NSData).decompressed(using: .lzfse) as Data) ?? data
    }
}

/// AllReduce implementations for gradient synchronization
public actor AllReduceManager {
    private var peers: [String: GradientExchangeClient] = [:]
    private let nodeId: String
    private let useRing: Bool
    
    public init(nodeId: String, useRingTopology: Bool = true) {
        self.nodeId = nodeId
        self.useRing = useRingTopology
    }
    
    /// Register a peer for gradient exchange
    public func registerPeer(id: String, client: GradientExchangeClient) {
        peers[id] = client
    }
    
    /// Remove a peer
    public func removePeer(id: String) {
        peers.removeValue(forKey: id)
    }
    
    /// Perform AllReduce on gradients
    public func allReduce(gradients: SendableGradients) async throws -> SendableGradients {
        var localGradients = gradients.gradients
        
        if useRing {
            localGradients = try await ringAllReduce(gradients: localGradients)
        } else {
            localGradients = try await treeAllReduce(gradients: localGradients)
        }
        
        return SendableGradients(localGradients)
    }
    
    /// Ring AllReduce: O(n) communication complexity
    private func ringAllReduce(gradients: [String: MLXArray]) async throws -> [String: MLXArray] {
        let peerList = Array(peers.keys).sorted()
        let worldSize = peerList.count + 1
        
        guard worldSize > 1 else { return gradients }
        
        // Find our position in the ring
        var allNodes = peerList + [nodeId]
        allNodes.sort()
        let myIndex = allNodes.firstIndex(of: nodeId) ?? 0
        
        // Scatter-reduce phase
        var accumulated = gradients
        
        for _ in 0..<(worldSize - 1) {
            let sendTo = allNodes[(myIndex + 1) % worldSize]
            let recvFrom = allNodes[(myIndex - 1 + worldSize) % worldSize]
            
            // Skip if sending/receiving from ourselves
            if sendTo == nodeId || recvFrom == nodeId {
                continue
            }
            
            // Send gradients to next peer and receive from previous
            if let sendClient = peers[sendTo], let recvClient = peers[recvFrom] {
                // Convert gradients to tensors
                let tensors = accumulated.map { GradientTensor(name: $0.key, array: $0.value) }
                
                // Send async
                Task {
                    try await sendClient.sendGradients(tensors)
                }
                
                // Receive and accumulate
                let received = try await recvClient.receiveGradients()
                for tensor in received {
                    if let existing = accumulated[tensor.name] {
                        accumulated[tensor.name] = existing + tensor.toMLXArray()
                    }
                }
            }
        }
        
        // All-gather phase (to be implemented)
        // For now, divide by world size to get average
        for (key, value) in accumulated {
            accumulated[key] = value / Float(worldSize)
        }
        
        return accumulated
    }
    
    /// Tree-based AllReduce: Better for high-latency networks
    private func treeAllReduce(gradients: [String: MLXArray]) async throws -> [String: MLXArray] {
        // Simplified tree reduce: all workers send to coordinator, coordinator broadcasts back
        // This is a placeholder - full tree implementation would be more efficient
        
        var accumulated = gradients
        
        // Gather all gradients
        for (_, client) in peers {
            let received = try await client.receiveGradients()
            for tensor in received {
                if let existing = accumulated[tensor.name] {
                    accumulated[tensor.name] = existing + tensor.toMLXArray()
                }
            }
        }
        
        // Average
        let worldSize = peers.count + 1
        for (key, value) in accumulated {
            accumulated[key] = value / Float(worldSize)
        }
        
        // Broadcast back
        let tensors = accumulated.map { GradientTensor(name: $0.key, array: $0.value) }
        for (_, client) in peers {
            try await client.sendGradients(tensors)
        }
        
        return accumulated
    }
}

// MARK: - Gradient Exchange Client

/// Client for exchanging gradients with a peer
public actor GradientExchangeClient {
    private let endpoint: String
    private var channel: GRPCChannel?
    private let group: EventLoopGroup
    
    public init(endpoint: String) {
        self.endpoint = endpoint
        self.group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
    }
    
    public func connect() async throws {
        let target = ConnectionTarget.hostAndPort(
            endpoint.split(separator: ":").first.map(String.init) ?? "localhost",
            Int(endpoint.split(separator: ":").last ?? "50051") ?? 50051
        )
        
        channel = try GRPCChannelPool.with(
            target: target,
            transportSecurity: .plaintext,
            eventLoopGroup: group
        )
    }
    
    public func disconnect() async {
        try? await channel?.close().get()
        // Shutdown on a background thread to avoid blocking
        await withCheckedContinuation { continuation in
            DispatchQueue.global().async {
                try? self.group.syncShutdownGracefully()
                continuation.resume()
            }
        }
    }
    
    public func sendGradients(_ tensors: [GradientTensor]) async throws {
        // Placeholder for actual gRPC call
        // In production, this would use generated protobuf types
        _ = tensors
    }
    
    public func receiveGradients() async throws -> [GradientTensor] {
        // Placeholder for actual gRPC call
        return []
    }
}

// MARK: - Distributed Trainer

/// State of the distributed trainer
public enum DistributedTrainerState: Sendable {
    case idle
    case connecting
    case synchronizing
    case training
    case error(String)
}

/// Distributed training coordinator/worker
public actor DistributedTrainer {
    private let config: DistributedConfig
    private let drafter: DrafterActor
    private let buffer: TrainingBuffer
    
    private var state: DistributedTrainerState = .idle
    private var peerDiscovery: PeerDiscovery?
    private var allReduce: AllReduceManager
    private var connectedPeers: Set<String> = []
    
    // Training state
    private var globalStep: Int = 0
    private var localGradients: [String: MLXArray] = [:]
    
    public init(
        config: DistributedConfig,
        drafter: DrafterActor,
        buffer: TrainingBuffer
    ) {
        self.config = config
        self.drafter = drafter
        self.buffer = buffer
        self.allReduce = AllReduceManager(
            nodeId: config.nodeId,
            useRingTopology: config.useRingAllReduce
        )
    }
    
    /// Start distributed training
    public func start() async throws {
        state = .connecting
        
        // Start peer discovery
        peerDiscovery = PeerDiscovery(nodeId: config.nodeId, port: UInt16(config.port))
        
        // Store a reference to call from discovery handler
        let trainer = self
        await peerDiscovery?.onPeerUpdate { peerId, endpoint, added in
            Task { @MainActor in
                if added {
                    await trainer.handlePeerJoined(peerId, endpoint: endpoint)
                } else {
                    await trainer.handlePeerLeft(peerId)
                }
            }
        }
        
        try await peerDiscovery?.start()
        
        // Wait for expected workers if coordinator
        if config.role == .coordinator {
            try await waitForWorkers()
        } else if let coordinatorAddr = config.coordinatorAddress {
            try await connectToCoordinator(coordinatorAddr)
        }
        
        state = .training
        
        // Start training loop
        await trainingLoop()
    }
    
    /// Stop distributed training
    public func stop() async {
        await peerDiscovery?.stop()
        state = .idle
    }
    
    /// Get current state
    public func getState() -> DistributedTrainerState {
        state
    }
    
    /// Get global training step
    public func getGlobalStep() -> Int {
        globalStep
    }
    
    private func waitForWorkers() async throws {
        while connectedPeers.count < config.expectedWorkers {
            try await Task.sleep(for: .milliseconds(500))
        }
    }
    
    private func connectToCoordinator(_ address: String) async throws {
        let client = GradientExchangeClient(endpoint: address)
        try await client.connect()
        await allReduce.registerPeer(id: "coordinator", client: client)
    }
    
    private func handlePeerJoined(_ peerId: String, endpoint: NWEndpoint) async {
        connectedPeers.insert(peerId)
        
        // Create client for this peer
        // Note: Need to resolve endpoint to address:port
        let client = GradientExchangeClient(endpoint: "\(peerId):50051")
        try? await client.connect()
        await allReduce.registerPeer(id: peerId, client: client)
    }
    
    private func handlePeerLeft(_ peerId: String) async {
        connectedPeers.remove(peerId)
        await allReduce.removePeer(id: peerId)
    }
    
    private func trainingLoop() async {
        while case .training = state {
            // Get batch from buffer
            let batch = await buffer.popBatch(maxSize: 4)
            guard !batch.isEmpty else {
                try? await Task.sleep(for: .milliseconds(100))
                continue
            }
            
            // Compute local gradients
            _ = await drafter.trainStep(samples: batch)
            
            globalStep += 1
            
            // Synchronize gradients at specified interval
            if globalStep % config.syncInterval == 0 && !connectedPeers.isEmpty {
                await synchronizeGradients()
            }
            
            // Yield to prevent blocking
            await Task.yield()
        }
    }
    
    private func synchronizeGradients() async {
        state = .synchronizing
        
        do {
            // Get current model parameters (as proxy for gradients in this simplified version)
            let model = await drafter.getModel()
            let params = model.parameters().flattened()
            var gradients: [String: MLXArray] = [:]
            for (key, value) in params {
                gradients[key] = value
            }
            
            // AllReduce
            let sendable = SendableGradients(gradients)
            let averaged = try await allReduce.allReduce(gradients: sendable)
            
            // Apply averaged gradients (simplified - in production use proper gradient application)
            _ = averaged.gradients
            
            state = .training
        } catch {
            state = .error("Synchronization failed: \(error)")
        }
    }
}

// MARK: - Fault Tolerance

/// Handles node failures and recovery in distributed training
public actor FaultToleranceManager {
    private var healthChecks: [String: Date] = [:]
    private let timeout: TimeInterval
    private var failedNodes: Set<String> = []
    
    public init(timeout: TimeInterval = 30.0) {
        self.timeout = timeout
    }
    
    /// Record a heartbeat from a node
    public func recordHeartbeat(from nodeId: String) {
        healthChecks[nodeId] = Date()
        failedNodes.remove(nodeId)
    }
    
    /// Check for failed nodes
    public func checkHealth() -> [String] {
        let now = Date()
        var newlyFailed: [String] = []
        
        for (nodeId, lastHeartbeat) in healthChecks {
            if now.timeIntervalSince(lastHeartbeat) > timeout {
                if !failedNodes.contains(nodeId) {
                    failedNodes.insert(nodeId)
                    newlyFailed.append(nodeId)
                }
            }
        }
        
        return newlyFailed
    }
    
    /// Get list of failed nodes
    public func getFailedNodes() -> Set<String> {
        failedNodes
    }
    
    /// Handle node recovery
    public func handleRecovery(nodeId: String) {
        failedNodes.remove(nodeId)
        healthChecks[nodeId] = Date()
    }
}

// MARK: - Data Parallelism

/// Manages data partitioning across nodes
public actor DataPartitioner {
    private let worldSize: Int
    private let rank: Int
    
    public init(worldSize: Int, rank: Int) {
        self.worldSize = worldSize
        self.rank = rank
    }
    
    /// Get the shard of samples for this node
    public func getShard<T>(from data: [T]) -> [T] {
        let shardSize = (data.count + worldSize - 1) / worldSize
        let startIndex = rank * shardSize
        let endIndex = min(startIndex + shardSize, data.count)
        
        guard startIndex < data.count else { return [] }
        return Array(data[startIndex..<endIndex])
    }
    
    /// Get batch indices for this node
    public func getBatchIndices(totalBatches: Int, batchIndex: Int) -> Int? {
        let globalIndex = batchIndex * worldSize + rank
        return globalIndex < totalBatches ? globalIndex : nil
    }
}
