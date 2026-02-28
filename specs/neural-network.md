# Neural Network Architecture

## 1. Overview

The neural network is a single dual-headed network `(p, v) = f_θ(s)` that takes a board position as input and outputs both a policy (move probabilities) and a value (position evaluation).

Architecture: **Residual network with Squeeze-and-Excitation (SE) blocks**, following the AlphaGo Zero/AlphaZero residual tower with the SE enhancement demonstrated by Leela Chess Zero.

The network is defined in Python/PyTorch. The C++ engine calls it via `torch::extension` or `libtorch` for inference. The architecture is designed to be **swappable** — MCTS and training only interact with the network through a standard interface, allowing future experiments with transformers or other architectures.

## 2. Network Interface

### Python Interface

```python
class AlphaZeroNetwork(nn.Module):
    """Abstract interface for all network architectures."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Board state tensor, shape (batch, C_in, H, W)
               C_in = game_config.input_channels
               H, W = game_config.board_shape

        Returns:
            policy_logits: Shape (batch, action_space_size). Raw logits (pre-softmax).
                           Illegal move masking is applied externally.
            value: Shape (batch, 1) for scalar head (tanh output, range [-1, 1]).
                   Shape (batch, 3) for WDL head (softmax output: [win, draw, loss]).
        """
        raise NotImplementedError
```

### C++ Inference Interface

```cpp
class NeuralNetInference {
public:
    // Batch inference. Input and output tensors are in unified memory.
    //
    // input: shape (batch_size, input_channels, board_rows, board_cols)
    // policy_out: shape (batch_size, action_space_size) — raw logits
    // value_out: shape (batch_size, 1) or (batch_size, 3)
    virtual void infer(
        const float* input, int batch_size,
        float* policy_out, float* value_out
    ) = 0;

    // Load new weights (called when training updates the model).
    virtual void load_weights(const std::string& path) = 0;
};
```

Since the DGX Spark uses unified memory, the input/output buffers are directly accessible by both CPU (which prepares the input) and GPU (which runs inference). No explicit memory transfer is required.

## 3. ResNet + SE Architecture

### Residual Tower

The network consists of:
1. **Initial convolutional block**: Transforms input planes to the internal filter count.
2. **N residual blocks with SE**: The main body of the network.
3. **Policy head**: Outputs move probabilities.
4. **Value head**: Outputs position evaluation.

### Configuration

| Parameter | Small (dev) | Medium (default) | Large |
|---|---|---|---|
| Residual blocks | 10 | 20 | 40 |
| Filters per layer | 128 | 256 | 256 |
| SE reduction ratio | 4 | 4 | 4 |
| Parameters (approx) | ~5M | ~25M | ~50M |

Start with the **small** configuration for development and correctness testing. Use **medium** for actual training runs. **Large** is available for extended training experiments.

All three configurations use the same code — only the `num_blocks` and `num_filters` hyperparameters change.

### Initial Convolutional Block

```
Input: (batch, C_in, H, W)
  │
  ├─ Conv2d(C_in, num_filters, kernel=3, stride=1, padding=1)
  ├─ BatchNorm2d(num_filters)
  └─ ReLU
  │
Output: (batch, num_filters, H, W)
```

### SE-Residual Block (repeated N times)

```
Input: x, shape (batch, F, H, W)    [F = num_filters]
  │
  ├─ Conv2d(F, F, kernel=3, stride=1, padding=1)
  ├─ BatchNorm2d(F)
  ├─ ReLU
  ├─ Conv2d(F, F, kernel=3, stride=1, padding=1)
  ├─ BatchNorm2d(F)
  │
  │  Squeeze-and-Excitation:
  ├─ GlobalAvgPool2d → (batch, F, 1, 1)
  ├─ FC(F, F // r)        [r = SE reduction ratio]
  ├─ ReLU
  ├─ FC(F // r, 2 * F)
  ├─ Split into (scale, bias), each (batch, F)
  ├─ scale = Sigmoid(scale)
  ├─ output = scale * conv_output + bias
  │
  ├─ Skip connection: output = output + x
  └─ ReLU
  │
Output: (batch, F, H, W)
```

Note: The SE block here uses the Leela Chess Zero variant with **bias** (scale + bias instead of just scale). The SE squeeze path produces 2F outputs which are split into a multiplicative scale (sigmoid-activated) and an additive bias. This is slightly more expressive than the standard SE block.

### Policy Head

The policy head converts the shared representation into move logits.

```
Input: (batch, F, H, W)    [from residual tower]
  │
  ├─ Conv2d(F, 32, kernel=1, stride=1)     [1x1 convolution to reduce channels]
  ├─ BatchNorm2d(32)
  ├─ ReLU
  ├─ Flatten → (batch, 32 * H * W)
  └─ Linear(32 * H * W, action_space_size)
  │
Output: (batch, action_space_size)   [raw logits, pre-softmax]
```

For chess: `32 * 8 * 8 = 2048 → 4672`
For Go: `32 * 19 * 19 = 11552 → 362`

The output is raw logits. Softmax is applied after illegal move masking (externally).

### Value Head — Scalar (Go)

```
Input: (batch, F, H, W)
  │
  ├─ Conv2d(F, 1, kernel=1, stride=1)      [1x1 convolution to single channel]
  ├─ BatchNorm2d(1)
  ├─ ReLU
  ├─ Flatten → (batch, H * W)
  ├─ Linear(H * W, 256)
  ├─ ReLU
  ├─ Linear(256, 1)
  └─ Tanh
  │
Output: (batch, 1)   [range: -1 (loss) to +1 (win)]
```

### Value Head — WDL (Chess)

```
Input: (batch, F, H, W)
  │
  ├─ Conv2d(F, 1, kernel=1, stride=1)
  ├─ BatchNorm2d(1)
  ├─ ReLU
  ├─ Flatten → (batch, H * W)
  ├─ Linear(H * W, 256)
  ├─ ReLU
  ├─ Linear(256, 3)
  └─ Softmax(dim=-1)
  │
Output: (batch, 3)   [probabilities: (win, draw, loss)]
```

For MCTS, the WDL output is converted to a scalar Q-value: `Q = win - loss`.

## 4. Loss Function

### Scalar Value (Go)

```
L = L_value + L_policy + c * ||θ||²

L_value  = (z - v)²                    [MSE between predicted value and game outcome]
L_policy = -Σ_a π_a * log(p_a)         [cross-entropy between MCTS policy and network policy]
c        = 1e-4                          [L2 regularization weight]
```

Where:
- `z ∈ {-1, 0, +1}`: game outcome from current player's perspective
- `v`: network value output (scalar)
- `π`: MCTS visit count distribution (search policy target)
- `p`: network policy output (after softmax)

### WDL Value (Chess)

```
L = L_value + L_policy + c * ||θ||²

L_value  = -Σ_k z_k * log(v_k)         [cross-entropy, k ∈ {win, draw, loss}]
L_policy = -Σ_a π_a * log(p_a)
c        = 1e-4
```

Where `z` is one-hot: `[1,0,0]` for win, `[0,1,0]` for draw, `[0,0,1]` for loss.

### Notes

- Policy and value losses are **weighted equally** (both are unit-scaled since rewards are in {-1, 0, +1}).
- L2 regularization applies to all network parameters (weights, not biases).
- The policy cross-entropy is computed only over legal actions (illegal actions are masked before log).

## 5. Training Configuration

| Parameter | Value | Notes |
|---|---|---|
| Optimizer | SGD with momentum | Per AlphaZero paper |
| Momentum | 0.9 | |
| Weight decay (L2) | 1e-4 | Applied as explicit L2 term in loss |
| Mini-batch size | 8192 (chess), 4096 (Go) | Tuned for GPU arithmetic intensity |
| Learning rate schedule | See tables below | Step decay, game-specific |
| Mixed precision | BF16 (PyTorch AMP) | Blackwell native BF16 tensor core support |
| Gradient scaling | AMP GradScaler | Prevents underflow in BF16 |

### Learning Rate Schedules

Schedules are adapted per-game based on total training steps and data generation rate.

**Chess** (125K total steps):

| Training steps | Learning rate |
|---|---|
| 0 – 7,000 | 0.2 |
| 7,000 – 9,000 | 0.02 |
| 9,000+ | 0.002 |

**Go** (350K total steps):

| Training steps (thousands) | Learning rate |
|---|---|
| 0 – 100 | 0.2 |
| 100 – 200 | 0.02 |
| 200 – 300 | 0.002 |
| 300+ | 0.0002 |

These are tuned for the DGX Spark's self-play data generation rate. The original AlphaZero paper used 200K/400K/600K step boundaries over 700K total steps.

## 6. Batch Normalization Folding

During **training**, batch normalization operates normally (computing running mean/variance statistics).

During **self-play inference**, batch normalization is **folded into the preceding convolution weights** to eliminate runtime overhead. This is a standard optimization used by Leela Chess Zero.

### Folding Procedure

For a convolution with weight `W`, bias `b`, followed by BatchNorm with parameters `γ` (scale), `β` (shift), `μ` (running mean), `σ²` (running variance), `ε` (epsilon):

```
W_folded = W * γ / sqrt(σ² + ε)
b_folded = (b - μ) * γ / sqrt(σ² + ε) + β
```

After folding, the convolution with `(W_folded, b_folded)` produces the same output as the original convolution + batch norm, with zero additional overhead.

### Implementation

- After each training checkpoint, export a "folded" version of the weights for inference.
- The inference model uses the folded weights directly — no BatchNorm layers.
- This is a one-time transformation per checkpoint, not a per-inference cost.

## 7. Precision Strategy

### Training

- Use **PyTorch Automatic Mixed Precision (AMP)** with BF16.
- Forward pass: convolutions and linear layers execute in BF16 on tensor cores.
- Loss computation: FP32 (for numerical stability).
- Backward pass: BF16 gradients, accumulated in FP32.
- Optimizer step: FP32 master weights updated from FP32 accumulated gradients.

```python
scaler = torch.amp.GradScaler()
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    policy_logits, value = model(states)
    loss = compute_loss(policy_logits, value, target_policy, target_value)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Inference

- **Phase 1**: BF16 inference using the folded model weights. Same precision as training but no backward pass.
- **Phase 2 (future optimization)**: FP8 inference via TensorRT or PyTorch's FP8 quantization. Requires calibration dataset to determine quantization scales. Potentially doubles inference throughput on Blackwell tensor cores.

## 8. Weight Initialization

- **Convolutional layers**: Kaiming (He) initialization with fan_out mode and ReLU nonlinearity.
- **Linear layers**: Kaiming initialization.
- **Batch normalization**: γ = 1, β = 0 (standard).
- **SE FC layers**: Xavier initialization.
- **Final policy linear layer**: Zero initialization (small initial policy entropy).
- **Final value linear layer**: Zero initialization.

## 9. Swappability for Future Architectures

The network architecture is decoupled from the rest of the system through the `AlphaZeroNetwork` interface. To add a new architecture (e.g., transformer):

1. Implement a new class inheriting from `AlphaZeroNetwork`.
2. Accept the same `GameConfig` and produce the same output shapes.
3. Register it in the network factory.
4. No changes needed to MCTS, self-play, training loop, or replay buffer.

### Transformer Adaptation Notes (for future reference)

- **Input**: Same `(batch, C_in, H, W)` tensor, reshaped to `(batch, H*W, C_in)` tokens.
- **Architecture**: Encoder-only transformer (per Leela BT4). Attention is O(n²) where n = H*W.
  - Chess: n = 64 (efficient)
  - Go: n = 361 (significant cost; may need efficient attention variants)
- **Policy head**: Per-token prediction (each token predicts moves from that square) or FC from pooled representation.
- **Value head**: CLS token or mean pooling → FC layers → scalar/WDL.
- **Positional encoding**: Learnable or Leela's "Smolgen" (position-dependent attention biases).
