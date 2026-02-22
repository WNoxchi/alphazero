# Feature: Load Folded Checkpoints in Web UI

## Context

The training pipeline exports two checkpoint variants:
- **Regular** (`checkpoint_00010000.pt`): Full model with BatchNorm layers. Used to resume training.
- **Folded** (`checkpoint_00010000_folded.pt`): BatchNorm folded into Conv layers via `alphazero.network.bn_fold.export_folded_model()`. Produces identical output but is slightly faster for inference (fewer ops, no BN statistics).

The web UI currently only loads **regular** checkpoints because `scripts/play.py:build_play_runtime()` constructs a standard `ResNetSE` (which has BN layers) and calls `load_state_dict()` — folded weights don't match because BN keys are missing and Conv layers have unexpected bias terms.

## Goal

Allow the web UI to load folded checkpoints for slightly faster inference during watch mode. This is a nice-to-have optimization, not critical — the speed difference is small for interactive play.

## How Folded Checkpoints Differ

Regular state dict keys for a residual block:
```
residual_blocks.0.conv_1.weight      # no bias (bias=False in Conv2d)
residual_blocks.0.bn_1.weight
residual_blocks.0.bn_1.bias
residual_blocks.0.bn_1.running_mean
residual_blocks.0.bn_1.running_var
residual_blocks.0.bn_1.num_batches_tracked
```

Folded state dict keys for the same block:
```
residual_blocks.0.conv_1.weight      # modified weights (BN folded in)
residual_blocks.0.conv_1.bias        # new bias (BN shift folded in)
residual_blocks.0.bn_1.weight        # nn.Identity() — no parameters, but key may be absent
```

The folding is done by `bn_fold.py:fold_conv_bn_pair()` which replaces each `(Conv2d, BatchNorm2d)` pair with a single `Conv2d(bias=True)` and swaps the BN module with `nn.Identity()`.

## Implementation Approach

The cleanest approach is to add a `build_folded_model` path in the web layer (no changes to core alphazero code):

### 1. Detect folded vs regular checkpoint

In `web/model_manager.py`, check if the state dict contains BN keys:

```python
def _is_folded_checkpoint(state_dict: dict) -> bool:
    return "input_bn.weight" not in state_dict
```

### 2. Build appropriate model

When loading a folded checkpoint, build a standard `ResNetSE` and then fold it before loading weights:

```python
from alphazero.network.bn_fold import fold_batch_norms

model = ResNetSE(game_config, num_blocks=..., num_filters=..., se_reduction=...)
if is_folded:
    model = fold_batch_norms(model, inplace=True)
model.load_state_dict(state_dict)
```

This works because `fold_batch_norms` transforms the model architecture to match the folded state dict — Conv layers get `bias=True` and BN layers become `nn.Identity()`.

### 3. Update `list_models()` to prefer folded

In `web/model_manager.py`, restore the folded-preference logic:

```python
folded = path.parent / f"{kind}_{step_digits}_folded.pt"
actual_path = folded if folded.exists() else path
```

### 4. Bypass `build_play_runtime`

The current code uses `scripts/play.py:build_play_runtime()` which always builds a regular model. For folded support, `ModelManager.get_runtime()` would need to build the model directly instead of going through `build_play_runtime`. The relevant steps from `build_play_runtime` are:

1. `ResNetSE(game_config, ...)` — construct model
2. `load_checkpoint(path, model)` — load weights
3. `model.to(device).eval()` — move to GPU
4. `_build_evaluator(model=model, ...)` — wrap in eval function
5. Build `SearchConfig` and return `PlayRuntime`

All of these can be replicated in `model_manager.py`, inserting the `fold_batch_norms` call between steps 1 and 2.

## Files to Change

| File | Change |
|------|--------|
| `web/model_manager.py` | Add folded detection, build model directly instead of via `build_play_runtime`, restore folded preference in `list_models()` |

## Testing

- Load a folded checkpoint and a regular checkpoint of the same training step
- Run both against each other in watch mode — they should play identically (same eval scores, same moves) given the same random seed
