"""Loss functions for AlphaZero training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as functional
from torch import nn


DEFAULT_L2_WEIGHT = 1e-4
_EPSILON = 1e-8
_DISTRIBUTION_ATOL = 1e-4
ValueHeadType = Literal["scalar", "wdl"]


@dataclass(frozen=True, slots=True)
class LossComponents:
    """Decomposed losses used for optimization and metrics logging."""

    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    l2_loss: torch.Tensor


def _validate_probability_distribution(name: str, distribution: torch.Tensor) -> None:
    if distribution.ndim != 2:
        raise ValueError(f"{name} must have rank 2, got shape {tuple(distribution.shape)}")
    if torch.any(distribution < 0):
        raise ValueError(f"{name} must be non-negative")

    sums = distribution.sum(dim=-1)
    if not torch.allclose(
        sums,
        torch.ones_like(sums),
        atol=_DISTRIBUTION_ATOL,
        rtol=0.0,
    ):
        raise ValueError(f"{name} rows must sum to 1")


def _to_scalar_vector(name: str, values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[1] == 1:
        return values.squeeze(dim=1)
    raise ValueError(f"{name} must have shape (batch,) or (batch, 1), got {tuple(values.shape)}")


def policy_cross_entropy_loss(
    policy_logits: torch.Tensor,
    target_policy: torch.Tensor,
    *,
    legal_action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy between MCTS policy target and network policy logits."""

    if policy_logits.ndim != 2:
        raise ValueError(
            f"policy_logits must have shape (batch, action_space_size), got {tuple(policy_logits.shape)}"
        )
    if target_policy.shape != policy_logits.shape:
        raise ValueError(
            "target_policy must match policy_logits shape; "
            f"got {tuple(target_policy.shape)} vs {tuple(policy_logits.shape)}"
        )
    _validate_probability_distribution("target_policy", target_policy)

    if legal_action_mask is not None:
        if legal_action_mask.dtype is not torch.bool:
            raise TypeError(
                f"legal_action_mask must have dtype bool, got {legal_action_mask.dtype}"
            )
        if legal_action_mask.shape != policy_logits.shape:
            raise ValueError(
                "legal_action_mask must match policy_logits shape; "
                f"got {tuple(legal_action_mask.shape)} vs {tuple(policy_logits.shape)}"
            )
        if torch.any(~legal_action_mask.any(dim=-1)):
            raise ValueError("each sample must contain at least one legal action")
        if torch.any((target_policy > 0) & (~legal_action_mask)):
            raise ValueError(
                "target_policy assigns probability to at least one action marked illegal"
            )

    logits = policy_logits.to(dtype=torch.float32)
    targets = target_policy.to(dtype=torch.float32, device=logits.device)

    if legal_action_mask is not None:
        legal_action_mask = legal_action_mask.to(device=logits.device)
        logits = logits.masked_fill(~legal_action_mask, torch.finfo(logits.dtype).min)

    log_probabilities = functional.log_softmax(logits, dim=-1)
    return -(targets * log_probabilities).sum(dim=-1).mean()


def scalar_value_loss(value: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
    """MSE between scalar value head output and scalar game outcome targets."""

    predicted_values = _to_scalar_vector("value", value).to(dtype=torch.float32)
    target_values = _to_scalar_vector("target_value", target_value).to(
        dtype=torch.float32,
        device=predicted_values.device,
    )
    if predicted_values.shape != target_values.shape:
        raise ValueError(
            f"scalar value and target shapes must match, got {tuple(predicted_values.shape)} "
            f"and {tuple(target_values.shape)}"
        )
    return functional.mse_loss(predicted_values, target_values)


def wdl_value_loss(value: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
    """Cross-entropy between predicted and target WDL distributions."""

    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError(f"value must have shape (batch, 3), got {tuple(value.shape)}")
    if target_value.shape != value.shape:
        raise ValueError(
            "target_value must match WDL value shape; "
            f"got {tuple(target_value.shape)} vs {tuple(value.shape)}"
        )

    _validate_probability_distribution("value", value)
    _validate_probability_distribution("target_value", target_value)

    predicted_probabilities = value.to(dtype=torch.float32).clamp_min(_EPSILON)
    target_probabilities = target_value.to(
        dtype=torch.float32,
        device=predicted_probabilities.device,
    )
    return -(target_probabilities * torch.log(predicted_probabilities)).sum(dim=-1).mean()


def ownership_loss(
    predicted_logits: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Ownership BCE loss using {-1, 0, +1} targets mapped to {0, 0.5, 1}."""

    if predicted_logits.ndim != 2:
        raise ValueError(
            "predicted_logits must have shape (batch, board_area), "
            f"got {tuple(predicted_logits.shape)}"
        )
    if target.shape != predicted_logits.shape:
        raise ValueError(
            "target must match predicted_logits shape; "
            f"got {tuple(target.shape)} vs {tuple(predicted_logits.shape)}"
        )

    target_values = target.to(dtype=torch.float32, device=predicted_logits.device)
    if not bool(torch.isfinite(target_values).all()):
        raise ValueError("target must contain only finite values")
    if bool((target_values < -1.0).any()) or bool((target_values > 1.0).any()):
        raise ValueError("target values must lie in [-1, 1]")

    sample_weights = _to_scalar_vector("weights", weights).to(
        dtype=torch.float32,
        device=predicted_logits.device,
    )
    if tuple(sample_weights.shape) != (predicted_logits.shape[0],):
        raise ValueError(
            "weights must have one value per sample; "
            f"expected {(predicted_logits.shape[0],)}, got {tuple(sample_weights.shape)}"
        )
    if not bool(torch.isfinite(sample_weights).all()):
        raise ValueError("weights must be finite")
    if bool((sample_weights < 0.0).any()):
        raise ValueError("weights must be non-negative")

    target_probs = (1.0 + target_values) * 0.5
    scaled_logits = predicted_logits.to(dtype=torch.float32) * 2.0
    per_point = functional.binary_cross_entropy_with_logits(
        scaled_logits,
        target_probs,
        reduction="none",
    )
    per_sample = per_point.mean(dim=-1)
    return (per_sample * sample_weights).mean()


def l2_regularization_loss(model: nn.Module) -> torch.Tensor:
    """Compute explicit L2 regularization over all model parameters."""

    if not isinstance(model, nn.Module):
        raise TypeError(
            f"model must be a torch.nn.Module, got {type(model).__name__}"
        )

    first_parameter = next(model.parameters(), None)
    if first_parameter is None:
        return torch.tensor(0.0, dtype=torch.float32)

    l2_loss = torch.zeros((), dtype=torch.float32, device=first_parameter.device)
    for parameter in model.parameters():
        l2_loss = l2_loss + parameter.to(dtype=torch.float32).pow(2).sum()
    return l2_loss


def compute_loss_components(
    policy_logits: torch.Tensor,
    value: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    *,
    value_type: ValueHeadType,
    l2_weight: float = DEFAULT_L2_WEIGHT,
    model: nn.Module | None = None,
    legal_action_mask: torch.Tensor | None = None,
) -> LossComponents:
    """Compute all AlphaZero loss terms and return a decomposed breakdown."""

    if l2_weight < 0:
        raise ValueError(f"l2_weight must be non-negative, got {l2_weight}")

    policy_loss = policy_cross_entropy_loss(
        policy_logits,
        target_policy,
        legal_action_mask=legal_action_mask,
    )

    if value_type == "scalar":
        value_loss = scalar_value_loss(value, target_value)
    elif value_type == "wdl":
        value_loss = wdl_value_loss(value, target_value)
    else:
        raise ValueError(
            f"value_type must be 'scalar' or 'wdl', got {value_type!r}"
        )

    if model is None:
        l2_loss = torch.zeros((), dtype=torch.float32, device=policy_loss.device)
    else:
        l2_loss = l2_regularization_loss(model)
        if l2_loss.device != policy_loss.device:
            l2_loss = l2_loss.to(device=policy_loss.device)

    total_loss = policy_loss + value_loss + (float(l2_weight) * l2_loss)
    return LossComponents(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        l2_loss=l2_loss,
    )


def compute_loss(
    policy_logits: torch.Tensor,
    value: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    *,
    value_type: ValueHeadType,
    l2_weight: float = DEFAULT_L2_WEIGHT,
    model: nn.Module | None = None,
    legal_action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute total AlphaZero training loss: policy + value + l2_weight * L2."""

    return compute_loss_components(
        policy_logits,
        value,
        target_policy,
        target_value,
        value_type=value_type,
        l2_weight=l2_weight,
        model=model,
        legal_action_mask=legal_action_mask,
    ).total_loss


__all__ = [
    "DEFAULT_L2_WEIGHT",
    "LossComponents",
    "ValueHeadType",
    "policy_cross_entropy_loss",
    "scalar_value_loss",
    "wdl_value_loss",
    "ownership_loss",
    "l2_regularization_loss",
    "compute_loss_components",
    "compute_loss",
]
