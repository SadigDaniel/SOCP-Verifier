"""Adapt the trained CIFAR-10 ResNet-4b to the existing sequential verifier.

Save this file beside socp_certificate.py, as src/SOCP/resnet_adapter.py.
In eval_SOCP_robustness.py import:
    from SOCP.resnet_adapter import resnet4b_to_sequential
After model.load_state_dict(...) and model.eval(), add:
    if args.model_type == "resnet":
        model = resnet4b_to_sequential(model)

This supports the exact four-block, no-BatchNorm architecture from the supplied
training script. It is a sequential reformulation, not a general graph parser.
The shortcut is retained in extra channels; no checkpoint weights are renamed.
The transformed model has the same function in real arithmetic. Floating-point
execution can differ slightly because the convolution summation order changes.
It uses the existing Conv2d/ReLU/Flatten/Linear verifier backends, and increases
activation dimensions and the memory needed by the dense affine conversion.
For fixed 32x32 inputs, stride-2 3x3 kernels are extended with a zero final row
and column. This preserves their values while making the existing backward
convolution code (output_padding=0) recover the required even input sizes.
"""

from copy import deepcopy

import torch
from torch import nn


def _check_conv(conv, kernel, stride, padding):
    if not isinstance(conv, nn.Conv2d):
        raise TypeError("Expected a Conv2d in the ResNet-4b block.")
    if (conv.kernel_size != (kernel, kernel)
            or conv.stride != (stride, stride)
            or conv.padding != (padding, padding)
            or conv.dilation != (1, 1)
            or conv.groups != 1
            or conv.padding_mode != "zeros"):
        raise ValueError("Unsupported convolution geometry in ResNet-4b.")


@torch.no_grad()
def _convert_block(block, in_channels, out_channels, stride):
    if tuple(dict(block.named_children())) != ("conv1", "conv2", "shortcut"):
        raise TypeError("Expected a BasicBlock with conv1, conv2, and shortcut.")
    first, second = block.conv1, block.conv2
    _check_conv(first, 3, stride, 1)
    _check_conv(second, 3, 1, 1)
    if ((first.in_channels, first.out_channels) != (in_channels, out_channels)
            or (second.in_channels, second.out_channels) != (out_channels, out_channels)):
        raise ValueError("Block widths do not match the supplied ResNet-4b architecture.")
    if not isinstance(block.shortcut, nn.Sequential):
        raise TypeError("Expected a Sequential identity or 1x1 projection shortcut.")

    options = dict(device=first.weight.device, dtype=first.weight.dtype)
    shortcut_bias = None
    if len(block.shortcut) == 0:
        if stride != 1 or in_channels != out_channels:
            raise ValueError("An identity shortcut must preserve shape.")
        shortcut_weight = torch.eye(out_channels, in_channels, **options)
    elif len(block.shortcut) == 1:
        projection = block.shortcut[0]
        _check_conv(projection, 1, stride, 0)
        if (projection.in_channels, projection.out_channels) != (in_channels, out_channels):
            raise ValueError("Projection shortcut widths do not match the block.")
        shortcut_weight = projection.weight[:, :, 0, 0]
        shortcut_bias = projection.bias
    else:
        raise ValueError("BatchNorm and multi-layer shortcuts are not supported.")

    # The previous block ends in ReLU, so the shortcut input is nonnegative.
    # Carry only the spatial samples used by the 1x1 projection: h[:, :, ::s, ::s].
    # ReLU therefore leaves these additional channels unchanged.
    has_constant = shortcut_bias is not None
    width = out_channels + in_channels + int(has_constant)
    kernel = 4 if stride == 2 else 3
    expand = nn.Conv2d(in_channels, width, kernel, stride, 1, bias=True, **options)
    expand.weight.zero_()
    expand.bias.zero_()
    expand.weight[:out_channels, :, :3, :3].copy_(first.weight)
    if first.bias is not None:
        expand.bias[:out_channels].copy_(first.bias)
    indices = torch.arange(in_channels, device=first.weight.device)
    expand.weight[out_channels + indices, indices, 1, 1] = 1

    # Keep the two original biases separate. Adding them in float32 here would
    # round the coefficients and change the real-arithmetic network slightly.
    if has_constant:
        expand.bias[-1] = 1

    merge = nn.Conv2d(width, out_channels, 3, 1, 1, bias=True, **options)
    merge.weight.zero_()
    merge.bias.zero_()
    merge.weight[:, :out_channels].copy_(second.weight)
    merge.weight[:, out_channels:out_channels + in_channels, 1, 1].copy_(shortcut_weight)
    if second.bias is not None:
        merge.bias.copy_(second.bias)
    if has_constant:
        merge.weight[:, -1, 1, 1].copy_(shortcut_bias)

    return [expand, nn.ReLU(), merge, nn.ReLU()]


@torch.no_grad()
def resnet4b_to_sequential(model, *, check_forward=True):
    """Return a flat Sequential model that preserves the ResNet-4b function.

    Convert once, after loading the original checkpoint. Use this returned model
    throughout clean evaluation, bound collection, gamma computation, and SOCP.
    This must not be replaced with Sequential(*model.children()), which loses
    the residual arithmetic and the functional ReLUs.
    """
    expected = ("conv1", "layer1", "layer2", "linear1", "linear2")
    if model.__class__.__name__ != "ResNet4b" or tuple(dict(model.named_children())) != expected:
        raise TypeError("Expected the ResNet4b class from the supplied training script.")
    if model.training:
        raise ValueError("Call model.eval() before converting ResNet-4b.")
    _check_conv(model.conv1, 3, 2, 1)
    if (model.conv1.in_channels, model.conv1.out_channels) != (3, 16):
        raise ValueError("Expected a 3-to-16-channel stem.")
    if not all(isinstance(stage, nn.Sequential) and len(stage) == 2
               for stage in (model.layer1, model.layer2)):
        raise ValueError("Expected two stages containing two residual blocks each.")
    if (not isinstance(model.linear1, nn.Linear)
            or not isinstance(model.linear2, nn.Linear)
            or (model.linear1.in_features, model.linear1.out_features) != (512, 100)
            or (model.linear2.in_features, model.linear2.out_features) != (100, 10)):
        raise ValueError("Expected the CIFAR-10 dense head 512 -> 100 -> 10.")

    # On even spatial dimensions, appending zeros to a 3x3 stride-2 kernel
    # gives an equivalent 4x4 convolution with the same stride and padding.
    stem = nn.Conv2d(
        3, 16, 4, 2, 1, bias=model.conv1.bias is not None,
        device=model.conv1.weight.device, dtype=model.conv1.weight.dtype,
    )
    stem.weight.zero_()
    stem.weight[:, :, :3, :3].copy_(model.conv1.weight)
    if model.conv1.bias is not None:
        stem.bias.copy_(model.conv1.bias)
    layers = [stem, nn.ReLU()]
    layers.extend(_convert_block(model.layer1[0], 16, 32, 2))
    layers.extend(_convert_block(model.layer1[1], 32, 32, 1))
    layers.extend(_convert_block(model.layer2[0], 32, 32, 2))
    layers.extend(_convert_block(model.layer2[1], 32, 32, 1))
    layers.extend([
        nn.Flatten(), deepcopy(model.linear1), nn.ReLU(), deepcopy(model.linear2)
    ])
    converted = nn.Sequential(*layers).eval()

    if check_forward:
        parameter = next(model.parameters())
        generator = torch.Generator(device=parameter.device).manual_seed(0)
        probes = torch.rand(
            (4, 3, 32, 32), device=parameter.device,
            dtype=parameter.dtype, generator=generator,
        )
        probes[0].zero_()
        probes[1].fill_(1)
        reference = model(probes)
        actual = converted(probes)
        tolerance = 1e-10 if parameter.dtype == torch.float64 else 1e-4
        torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
    return converted
