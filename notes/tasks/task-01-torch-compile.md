# Task 01: Add `torch.compile` to Model

**Priority**: Highest — largest single performance improvement
**Expected impact**: 20-40% faster forward/backward passes
**Dependencies**: None
**Difficulty**: Low

## Background

Read `specs/*` for the full AlphaZero specification, and `notes/gpu_optimization.md` for the bottleneck analysis motivating this change.

The ResNetSE model (20 residual blocks with Squeeze-and-Excitation, 256 filters, ~35M parameters) launches hundreds of small CUDA kernels per forward pass. `torch.compile` with the inductor backend fuses these into fewer, larger kernels — reducing launch overhead and improving GPU memory access patterns.

The model is an ideal compile candidate:
- Purely static control flow (no `if` statements based on tensor values in `forward()`)
- No custom autograd functions
- Standard nn.Module ops only: Conv2d, BatchNorm2d, Linear, ReLU, sigmoid, chunk, unsqueeze
- No `torch.jit` usage anywhere in the codebase
- PyTorch 2.10.0+cu130 (torch.compile requires 2.0+)

The model is shared between inference (batch=384, no_grad, bf16 autocast) and training (batch=4096, grad enabled, bf16 autocast). `torch.compile` handles these different execution contexts via internal guards, compiling and caching separate graph variants for each.

## File to Modify

### `scripts/train.py`

**Function**: `build_training_runtime()` (line 397)

This function creates the model (line 413), optionally loads a checkpoint (line 430), then constructs the eval queue and self-play manager. The compiled model must be created before it's passed to other components.

**Current code** (lines 412-416):
```python
game_config = _resolve_game_config(config)
model = _build_model(active_dependencies, config, game_config)
training_config = _resolve_training_config(active_dependencies, config)
pipeline_config = active_dependencies.load_pipeline_config_from_config(config)
lr_schedule = active_dependencies.load_lr_schedule_from_config(config)
```

## Required Change

After line 413 (`model = _build_model(...)`), add torch.compile. Gate it behind a config option for flexibility:

```python
game_config = _resolve_game_config(config)
model = _build_model(active_dependencies, config, game_config)

# torch.compile for GPU kernel fusion
system = _section(config, "system")
compile_model = system.get("compile", True)  # Default: enabled
if compile_model:
    import torch
    model = torch.compile(model, mode="reduce-overhead")

training_config = _resolve_training_config(active_dependencies, config)
```

**Important ordering**: `torch.compile` must be applied:
- **After** `_build_model()` which creates the ResNetSE instance
- **Before** `load_training_checkpoint()` (line 430) which calls `model.load_state_dict()` — compiled models accept this normally
- **Before** `make_eval_queue_batch_evaluator()` (line 444) which captures the model reference in its closure

## Config Addition

In `configs/chess_1hr.yaml`, optionally add under the `system:` section:
```yaml
system:
  precision: "bf16"
  compile: true    # Enable torch.compile (default: true)
```

## What NOT to Change

- Do not compile the evaluator closure or loss functions — only the model itself benefits
- Do not add `fullgraph=True` — it's more fragile and the default `fullgraph=False` handles edge cases gracefully
- Do not use `dynamic=True` — the two fixed batch sizes (384 inference, 4096 training) will each trigger one cached compilation, which is optimal

## Verification

1. Run training for 100+ steps:
   ```bash
   python scripts/train.py --config configs/chess_1hr.yaml
   ```

2. Confirm no compilation errors in stderr (watch for `torch._dynamo` or `torch._inductor` warnings)

3. First 2 training steps will be slow (~5-30 seconds each) as compilation occurs for each batch size variant. Subsequent steps should be significantly faster.

4. Compare throughput:
   - Baseline: ~0.52 train steps/sec
   - Expected: ~0.65-0.75 train steps/sec (or higher)

5. Loss values should be numerically comparable to non-compiled runs (not identical due to kernel fusion changing floating-point ordering slightly, but within noise)

6. To disable for debugging, set `system.compile: false` in the YAML config

## Model Architecture Reference

The model being compiled (`python/alphazero/network/resnet_se.py`):

```python
class ResNetSE(nn.Module):
    def forward(self, x):
        features = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.residual_blocks:  # Static loop over ModuleList
            features = block(features)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value
```

Each `SEResidualBlock` contains: Conv2d → BatchNorm2d → ReLU → Conv2d → BatchNorm2d → SE(GlobalAvgPool → FC → ReLU → FC → split → sigmoid) → scale + bias → skip connection → ReLU.
