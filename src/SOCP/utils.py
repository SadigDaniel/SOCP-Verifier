"""
Utility helpers for the SOCP module.

This file intentionally does not modify the original LP/CROWN/Wong--Kolter
code.  It only imports and reuses the existing model/data utilities from the
parent src package when available.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    """Make PyTorch/Numpy/Python randomness reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Use CUDA if PyTorch can see a GPU; otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Compute top-1 accuracy for a minibatch."""
    return (logits.argmax(dim=1) == y).float().mean().item()


@dataclass(frozen=True)
class DenseAffineLayer:
    """
    Dense representation of one affine layer.

    The SOCP solver uses CVXPY.  CVXPY is much easier to use with explicit
    affine matrices than with PyTorch convolution modules.  For MNIST-scale
    experiments this dense conversion is acceptable for small networks, but it
    will be expensive for larger CNNs.

    Attributes
    ----------
    weight:
        Numpy array with shape (out_dim, in_dim).
    bias:
        Numpy array with shape (out_dim,).
    name:
        Human-readable layer name.
    """

    weight: np.ndarray
    bias: np.ndarray
    name: str


def conv2d_to_dense_matrix(
    conv: nn.Conv2d,
    input_shape: Tuple[int, int, int],
    device: torch.device | None = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """
    Convert a Conv2d layer into an explicit dense matrix.

    If h is the flattened input image/activation, this returns A and b such that

        vec(conv(input)) = A @ h + b.

    This is only intended for small MNIST networks.  It is not the right way to
    build a scalable CIFAR/TinyImageNet SOCP verifier.
    """
    if device is None:
        device = next(conv.parameters()).device

    c, h, w = input_shape
    in_dim = c * h * w

    # Probe output shape.
    with torch.no_grad():
        dummy = torch.zeros(1, c, h, w, device=device)
        out = conv(dummy)
        out_shape = tuple(out.shape[1:])
        out_dim = int(np.prod(out_shape))

    # Build the matrix column-by-column by applying the convolution to basis
    # vectors.  This is slow but simple and acceptable for a one-time setup on
    # the small MNIST architecture.
    cols: List[np.ndarray] = []
    with torch.no_grad():
        eye = torch.eye(in_dim, device=device)
        # Process chunks to avoid a huge temporary tensor.
        chunk_size = 512
        for start in range(0, in_dim, chunk_size):
            chunk = eye[start : start + chunk_size].view(-1, c, h, w)
            # y = conv(chunk).view(chunk.size(0), -1)
            # # Remove bias contribution from every basis output so that the bias
            # # is stored separately.
            # if conv.bias is not None:
            #     b = conv.bias.view(1, -1, 1, 1).expand_as(conv(chunk)).contiguous().view(chunk.size(0), -1)
            #     y = y - b
            conv_out = conv(chunk)

            y = conv_out.view(chunk.size(0), -1)

            if conv.bias is not None:
                b = (
                    conv.bias.view(1, -1, 1, 1)
                    .expand_as(conv_out)
                    .contiguous()
                    .view(chunk.size(0), -1)
                )

                y = y - b
            cols.append(y.detach().cpu().numpy())

    # cols currently has rows = input basis vectors, columns = output coords.
    # We need A with shape (out_dim, in_dim).
    basis_outputs = np.concatenate(cols, axis=0)  # (in_dim, out_dim)
    A = basis_outputs.T.astype(np.float64)

    if conv.bias is None:
        bias = np.zeros(out_dim, dtype=np.float64)
    else:
        with torch.no_grad():
            zero = torch.zeros(1, c, h, w, device=device)
            bias = conv(zero).view(-1).detach().cpu().numpy().astype(np.float64)

    return A, bias, out_shape  # type: ignore[return-value]


def _conv_affine_from_weights(
    conv: nn.Conv2d, input_shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """Read zero-padded convolution coefficients without floating-point inference."""
    if conv.padding_mode != "zeros":
        raise ValueError("GPU SOCP supports Conv2d with zero padding only.")
    channels, height, width = input_shape
    weight = conv.weight.detach().to(device="cpu", dtype=torch.float64).numpy()
    out_channels, group_inputs, kernel_h, kernel_w = weight.shape
    stride_h, stride_w = conv.stride
    dilation_h, dilation_w = conv.dilation
    kernel_span_h = dilation_h * (kernel_h - 1)
    kernel_span_w = dilation_w * (kernel_w - 1)
    if conv.padding == "same":
        pad_h, pad_w = kernel_span_h // 2, kernel_span_w // 2
        total_h, total_w = kernel_span_h, kernel_span_w
    else:
        pad_h, pad_w = (0, 0) if conv.padding == "valid" else conv.padding
        total_h, total_w = 2 * pad_h, 2 * pad_w
    out_h = (height + total_h - kernel_span_h - 1) // stride_h + 1
    out_w = (width + total_w - kernel_span_w - 1) // stride_w + 1
    pixels = out_h * out_w
    matrix = np.zeros((out_channels * pixels, channels * height * width), dtype=np.float64)
    out_idx = np.arange(out_channels)
    in_channels = (out_idx // (out_channels // conv.groups))[:, None] * group_inputs
    in_channels = in_channels + np.arange(group_inputs)
    spatial = np.arange(pixels)
    base_h = spatial // out_w * stride_h - pad_h
    base_w = spatial % out_w * stride_w - pad_w
    for kh in range(kernel_h):
        input_h = base_h + kh * dilation_h
        for kw in range(kernel_w):
            input_w = base_w + kw * dilation_w
            valid = (0 <= input_h) & (input_h < height) & (0 <= input_w) & (input_w < width)
            rows = out_idx[:, None, None] * pixels + spatial[valid]
            cols = in_channels[:, :, None] * (height * width) + input_h[valid] * width + input_w[valid]
            matrix[rows, cols] = weight[:, :, kh, kw, None]
    bias = np.zeros(out_channels, dtype=np.float64)
    if conv.bias is not None:
        bias = conv.bias.detach().to(device="cpu", dtype=torch.float64).numpy()
    return matrix, np.repeat(bias, pixels), (out_channels, out_h, out_w)


def sequential_to_dense_affines(
    model: nn.Sequential,
    input_shape: Tuple[int, int, int] = (1, 28, 28),
    *,
    exact_coefficients: bool = False,
) -> List[DenseAffineLayer]:
    """
    Convert the supported MNIST Sequential model into affine matrices.

    Supported pattern:
        Conv2d, ReLU, Conv2d, ReLU, Flatten, Linear, ReLU, Linear

    ReLU is not converted here because the SOCP formulation represents it with
    explicit variables and convex constraints.
    """
    device = next(model.parameters()).device
    current_shape: Tuple[int, ...] = input_shape
    affines: List[DenseAffineLayer] = []

    if exact_coefficients:
        stages = [layer for layer in model if layer.__class__.__name__ != "Flatten"]
        if len(stages) % 2 == 0 or any(
            not isinstance(layer, nn.ReLU if i % 2 else (nn.Conv2d, nn.Linear))
            for i, layer in enumerate(stages)
        ):
            raise ValueError("GPU SOCP expects alternating affine and ReLU layers ending in logits.")

    for idx, layer in enumerate(model):
        if isinstance(layer, nn.Conv2d):
            if len(current_shape) != 3:
                raise ValueError(f"Conv2d expected 3D activation shape, got {current_shape}")
            if exact_coefficients:
                A, b, out_shape = _conv_affine_from_weights(layer, current_shape)
            else:
                A, b, out_shape = conv2d_to_dense_matrix(layer, current_shape, device=device)  # type: ignore[arg-type]
            affines.append(DenseAffineLayer(A, b, name=f"conv{idx}"))
            current_shape = out_shape
        elif isinstance(layer, nn.Linear):
            W = layer.weight.detach().cpu().numpy().astype(np.float64)
            b = layer.bias.detach().cpu().numpy().astype(np.float64) if layer.bias is not None else np.zeros(W.shape[0])
            affines.append(DenseAffineLayer(W, b, name=f"linear{idx}"))
            current_shape = (W.shape[0],)
        elif isinstance(layer, nn.ReLU):
            # ReLU is represented explicitly by constraints in the SOCP model.
            continue
        elif isinstance(layer, nn.Flatten) or layer.__class__.__name__ == "Flatten":
            current_shape = (int(np.prod(current_shape)),)
        else:
            raise TypeError(f"Unsupported layer in SOCP converter: {layer}")

    return affines


def torch_flatten_bounds(lower: torch.Tensor, upper: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a single-sample pair of PyTorch bounds to flat numpy arrays."""
    return (
        lower.detach().cpu().view(-1).numpy().astype(np.float64),
        upper.detach().cpu().view(-1).numpy().astype(np.float64),
    )
