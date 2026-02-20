"""Batch-normalization folding utilities for inference export."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


def fold_conv_bn_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Fold a BatchNorm2d layer into the preceding Conv2d layer.

    The returned convolution always includes an explicit bias term carrying the
    folded BatchNorm shift.
    """

    if not isinstance(conv, nn.Conv2d):
        raise TypeError(f"conv must be an nn.Conv2d, got {type(conv).__name__}")
    if not isinstance(bn, nn.BatchNorm2d):
        raise TypeError(f"bn must be an nn.BatchNorm2d, got {type(bn).__name__}")
    if conv.out_channels != bn.num_features:
        raise ValueError(
            "conv.out_channels and bn.num_features must match, "
            f"got {conv.out_channels} and {bn.num_features}"
        )
    if bn.running_mean is None or bn.running_var is None:
        raise ValueError(
            "batch norm must track running statistics (running_mean/running_var)"
        )

    device = conv.weight.device
    compute_dtype = torch.float64

    weight = conv.weight.detach().to(device=device, dtype=compute_dtype)
    if conv.bias is None:
        bias = torch.zeros(conv.out_channels, device=device, dtype=compute_dtype)
    else:
        bias = conv.bias.detach().to(device=device, dtype=compute_dtype)

    running_mean = bn.running_mean.detach().to(device=device, dtype=compute_dtype)
    running_var = bn.running_var.detach().to(device=device, dtype=compute_dtype)

    if bn.affine:
        gamma = bn.weight.detach().to(device=device, dtype=compute_dtype)
        beta = bn.bias.detach().to(device=device, dtype=compute_dtype)
    else:
        gamma = torch.ones(conv.out_channels, device=device, dtype=compute_dtype)
        beta = torch.zeros(conv.out_channels, device=device, dtype=compute_dtype)

    inv_std = gamma / torch.sqrt(running_var + float(bn.eps))
    folded_weight = weight * inv_std.view(-1, 1, 1, 1)
    folded_bias = (bias - running_mean) * inv_std + beta

    folded_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    )
    folded_conv.to(device=device, dtype=conv.weight.dtype)
    folded_conv.train(conv.training)

    with torch.no_grad():
        folded_conv.weight.copy_(folded_weight.to(dtype=conv.weight.dtype))
        folded_conv.bias.copy_(folded_bias.to(dtype=conv.weight.dtype))

    folded_conv.weight.requires_grad_(conv.weight.requires_grad)
    if conv.bias is None:
        folded_conv.bias.requires_grad_(conv.weight.requires_grad)
    else:
        folded_conv.bias.requires_grad_(conv.bias.requires_grad)

    return folded_conv


def has_batch_norm_layers(model: nn.Module) -> bool:
    """Return True when the module tree still contains BatchNorm2d layers."""

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    return any(isinstance(module, nn.BatchNorm2d) for module in model.modules())


def _candidate_bn_names(conv_name: str) -> tuple[str, ...]:
    candidates: list[str] = []
    if "conv" in conv_name:
        candidates.append(conv_name.replace("conv", "bn", 1))
    if conv_name.startswith("conv"):
        candidates.append(f"bn{conv_name[4:]}")

    deduplicated: list[str] = []
    for candidate in candidates:
        if candidate not in deduplicated:
            deduplicated.append(candidate)
    return tuple(deduplicated)


def _fold_batch_norms_recursive(module: nn.Module) -> None:
    for child in module.children():
        _fold_batch_norms_recursive(child)

    children_by_name = dict(module.named_children())
    consumed_bn_names: set[str] = set()

    for conv_name, child in list(children_by_name.items()):
        if not isinstance(child, nn.Conv2d):
            continue

        matched_bn_name: str | None = None
        matched_bn: nn.BatchNorm2d | None = None
        for bn_name in _candidate_bn_names(conv_name):
            if bn_name in consumed_bn_names:
                continue
            bn_candidate = children_by_name.get(bn_name)
            if isinstance(bn_candidate, nn.BatchNorm2d):
                matched_bn_name = bn_name
                matched_bn = bn_candidate
                break

        if matched_bn is None or matched_bn_name is None:
            continue

        setattr(module, conv_name, fold_conv_bn_pair(child, matched_bn))
        setattr(module, matched_bn_name, nn.Identity())
        consumed_bn_names.add(matched_bn_name)


def fold_batch_norms(model: nn.Module, *, inplace: bool = False) -> nn.Module:
    """Return a model with Conv+BatchNorm pairs folded into Conv layers.

    By default this exports a deep-copied model, leaving the source model
    untouched.
    """

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")

    folded_model = model if inplace else deepcopy(model)
    _fold_batch_norms_recursive(folded_model)
    return folded_model


def export_folded_model(model: nn.Module) -> nn.Module:
    """Export a BN-folded copy of a model for self-play inference."""

    return fold_batch_norms(model, inplace=False)


__all__ = [
    "export_folded_model",
    "fold_batch_norms",
    "fold_conv_bn_pair",
    "has_batch_norm_layers",
]
