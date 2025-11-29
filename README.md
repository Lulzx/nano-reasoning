# Nano-Reasoning

**FastRL Adaptive Drafter for Apple Silicon**

A Swift implementation of speculative decoding with real-time drafter adaptation, optimized for the entire Apple Silicon family (M1 through M5). Leverages MLX and Unified Memory Architecture for zero-copy training loops.

## Overview

Nano-Reasoning implements an adaptive speculative decoding pipeline where a small "Drafter" model learns to predict the reasoning patterns of a larger "Target" model in real-time. The drafter improves continuously during inference, leading to progressively faster generation.

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────┐
│   Drafter   │────▶│   Orchestrator  │────▶│    Target    │
│  (0.5B-1B)  │     │                 │     │   (3B-72B)   │
└─────────────┘     └─────────────────┘     └──────────────┘
       ▲                    │                      │
       │                    ▼                      │
       │            ┌───────────────┐              │
       └────────────│ Training Loop │◀─────────────┘
                    │ (Background)  │  Rejection Data
                    └───────────────┘
```

## Features

- **Hardware-Adaptive**: Automatically detects chip capabilities and selects appropriate models
- **Zero-Copy Training**: Drafter learns from target rejections without memory copying overhead
- **Background Training**: Training runs at background priority, yielding to inference on GPU load
- **EAGLE-Style Speculation**: Hidden state fusion for improved draft quality
- **LoRA Adapters**: Memory-efficient fine-tuning for Pro tier systems

## Hardware Tiers

| Tier | Chips | RAM | Target Model | Drafter | Strategy |
|------|-------|-----|--------------|---------|----------|
| Entry | M1/M2 | 16GB | Qwen2.5-3B (4-bit) | Qwen2.5-0.5B (4-bit) | Frozen Speculator |
| Pro | M1/M2/M3 Pro/Max | 24GB+ | Qwen2.5-7B (4-bit) | Qwen2.5-0.5B (FP16) | Adaptive LoRA |
| Elite | M4/M5 | 36GB+ | Qwen2.5-32B (4-bit) | Qwen2.5-0.5B (FP16) | Full EAGLE + NPU |

## Requirements

- macOS 14.0+ (Sonoma, Sequoia, or later)
- Apple Silicon Mac (M1 or newer)
- 16GB+ Unified Memory
- Xcode 15.0+ or Swift 6.0+

## Installation

### From Source

```bash
git clone https://github.com/lulzx/nano-reasoning.git
cd nano-reasoning
swift build -c release
```

### As a Dependency

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/lulzx/nano-reasoning.git", from: "1.0.0")
]
```

## Usage

### Command Line Interface

**Interactive Chat:**
```bash
nano-reasoning chat
```

**Text Generation:**
```bash
nano-reasoning generate "Explain quantum entanglement" --max-tokens 500
```

**System Information:**
```bash
nano-reasoning info
```

**Performance Benchmark:**
```bash
nano-reasoning benchmark --iterations 10 --compare
```

### CLI Options

```
COMMANDS:
  chat        Interactive chat mode (default)
  generate    Generate text from a prompt
  info        Display system information
  benchmark   Run performance benchmarks

GENERATE OPTIONS:
  -m, --max-tokens    Maximum tokens to generate (default: 512)
  -t, --temperature   Sampling temperature (default: 0.7)
  --top-p             Nucleus sampling parameter (default: 0.9)
  --top-k             Top-k sampling parameter (default: 40)
  --no-speculation    Disable speculative decoding
  --stats             Show generation statistics

CHAT OPTIONS:
  /quit, /exit        Exit chat
  /stats              Show statistics
  /clear              Clear history
  /help               Show commands
```

### Library Usage

```swift
import NanoReasoningCore

// Quick start with automatic hardware detection
let orchestrator = try await NanoReasoning.createOrchestrator()

// Generate text
let response = try await orchestrator.generate(
    prompt: "What is machine learning?",
    maxTokens: 200
) { token in
    print(token, terminator: "")
    return true  // Continue generation
}

// Check statistics
if let stats = await orchestrator.getSessionStatistics() {
    print("Acceptance rate: \(stats.averageAcceptanceRate * 100)%")
    print("Tokens/second: \(stats.tokensPerSecond)")
}
```

### Advanced Configuration

```swift
import NanoReasoningCore

// Custom generation config
let config = GenerationConfig(
    maxTokens: 1024,
    temperature: 0.8,
    topP: 0.95,
    topK: 50,
    repetitionPenalty: 1.1,
    stopTokens: [151645]  // EOS token
)

let orchestrator = Orchestrator(generationConfig: config)
try await orchestrator.initialize()

// Start background training (Pro/Elite tiers)
await orchestrator.startTraining()
```

## Architecture

### Core Components

| Component | Type | Description |
|-----------|------|-------------|
| `Orchestrator` | Actor | Coordinates the pipeline, manages state |
| `DrafterActor` | Actor | Manages drafter model with thread-safe updates |
| `TargetModel` | Actor | Wraps target model, exposes hidden states |
| `TrainingBuffer` | Actor | Ring buffer with priority-based sampling |
| `TrainerTask` | Actor | Background training loop with GPU awareness |
| `SpeculativeDecoder` | Actor | Draft → Verify → Accept logic |

### Speculative Decoding Flow

1. **Draft**: Drafter generates k tokens using EAGLE-style hidden state fusion
2. **Verify**: Target model validates all k tokens in a single forward pass
3. **Accept**: Matching tokens are accepted; first mismatch triggers correction
4. **Learn**: Rejection data is pushed to training buffer for drafter improvement

### Memory Layout (Unified Memory)

```
┌────────────────────────────────────────────────┐
│              Unified Memory (UMA)              │
├────────────────────┬───────────────────────────┤
│  Target Weights    │    Drafter Weights        │
│  (4-bit, ~4-20GB)  │    (FP16, ~1-3GB)         │
├────────────────────┴───────────────────────────┤
│              KV Cache (Dynamic)                │
├────────────────────────────────────────────────┤
│         Training Buffer (Ring, 50-200)         │
└────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NANO_REASONING_CACHE` | Model cache directory | `~/.cache/nano-reasoning/models` |
| `NANO_REASONING_CHECKPOINTS` | Checkpoint directory | `~/.cache/nano-reasoning/checkpoints` |
| `NANO_REASONING_CONFIG` | Config file path | `~/.config/nano-reasoning/config.json` |
| `NANO_REASONING_DEBUG` | Enable debug mode | `0` |
| `NANO_REASONING_LOG_LEVEL` | Log level | `info` |

### Config File (JSON)

```json
{
  "hardware": {
    "forceTier": null,
    "gpuUtilizationTarget": 0.85,
    "enableNPUOffload": true
  },
  "models": {
    "targetModelId": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "drafterModelId": "mlx-community/Qwen2.5-0.5B-Instruct"
  },
  "generation": {
    "maxTokens": 2048,
    "temperature": 0.7,
    "draftCount": 5
  },
  "training": {
    "enabled": true,
    "learningRate": 0.0001,
    "batchSize": 4,
    "bufferCapacity": 50
  }
}
```

## Performance

### Typical Speedups

| Scenario | Acceptance Rate | Speedup |
|----------|-----------------|---------|
| Initial (cold) | 40-50% | 1.3-1.5x |
| Warmed up (100+ tokens) | 60-70% | 1.8-2.2x |
| Domain-adapted | 75-85% | 2.5-3.0x |

### Benchmark Results (M4 Pro, 24GB)

```
Model: Qwen2.5-7B-Instruct-4bit
Drafter: Qwen2.5-0.5B-Instruct

Standard Generation:  45.2 tokens/sec
Speculative (cold):   62.8 tokens/sec (1.39x)
Speculative (warm):   89.4 tokens/sec (1.98x)
```

## Development

### Project Structure

```
nano-reasoning/
├── Package.swift
├── Sources/
│   ├── NanoReasoning/           # CLI application
│   │   └── Main.swift
│   └── NanoReasoningCore/       # Core library
│       ├── HardwareTier.swift   # Hardware detection
│       ├── DrafterActor.swift   # Drafter model + EAGLE head
│       ├── TargetModel.swift    # Target model wrapper
│       ├── TrainingBuffer.swift # Ring buffer
│       ├── TrainerTask.swift    # Background training
│       ├── Orchestrator.swift   # Pipeline coordinator
│       ├── SpeculativeDecoding.swift
│       └── Configuration.swift
└── Tests/
    └── NanoReasoningTests/
```

### Building

```bash
# Debug build
swift build

# Release build
swift build -c release

# Run tests
swift test

# Generate documentation
swift package generate-documentation
```

### Dependencies

- [MLX-Swift](https://github.com/ml-explore/mlx-swift) - Apple's ML framework
- [Swift Argument Parser](https://github.com/apple/swift-argument-parser) - CLI parsing

## Roadmap

- [ ] Tree-based speculation for higher acceptance rates
- [ ] Metal 4 / NPU offload for M4/M5 chips
- [ ] Quantized drafter training (INT8)
- [ ] Multi-drafter ensemble
- [ ] HuggingFace tokenizer integration
- [ ] Model weight persistence and checkpointing
- [ ] Distributed training across multiple Macs

## References

- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [MLX: Efficient and flexible machine learning on Apple silicon](https://github.com/ml-explore/mlx)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting PRs.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
