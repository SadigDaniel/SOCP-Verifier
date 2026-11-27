import argparse
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

try:
    from src.model import build_cnn5_model, build_mnist_model, build_resnet4b_model
except ImportError:
    from model import build_cnn5_model, build_mnist_model, build_resnet4b_model


DATASET_ALIASES = {
    "mnist": "mnist",
    "cifar10": "cifar10",
    "cifar-10": "cifar10",
}
'''
python -m src.tools.export_to_onnx \
  --dataset CIFAR10 \
  --checkpoint checkpoints/cifar10/eps_0.0088889/cifar10_dual_cnn5_standard.pt \
  --output verifier/cifar10/eps_0.0089/vnnlib_dual/cifar10/cifar10_dual_cnn5_standard.onnx \
  --c1 24 \
  --c2 48 \
  --c3 96 \
  --linear-size 192
  
  
python -m src.tools.export_to_onnx \
  --dataset CIFAR10 \
  --checkpoint checkpoints/cifar10/resnet4b_last.pt \
  --output verifier/cifar10/eps_0.0089/vnnlib_ResNet/cifar10/resnet4b_last.onnx \
  --c1 24 \
  --c2 48 \
  --c3 96 \
  --linear-size 192
'''



def normalise_dataset_name(dataset: str) -> str:
    key = dataset.strip().lower()
    if key not in DATASET_ALIASES:
        supported = ", ".join(sorted(DATASET_ALIASES))
        raise ValueError(f"Unsupported dataset '{dataset}'. Choose one of: {supported}")
    return DATASET_ALIASES[key]



def build_model_and_input(
    dataset: str,
    n1: int = 16,
    n2: int = 32,
    c1: int = 16,
    c2: int = 32,
    c3: int = 64,
    linear_size: int = 100,
    num_classes: int = 10,
    model_type: Optional[str] = None,
) -> Tuple[nn.Module, torch.Tensor]:
    dataset = normalise_dataset_name(dataset)

    if model_type is None:
        model_type = "cnn" if dataset == "mnist" else "cnn5"

    if model_type == "resnet":
        if dataset != "cifar10":
            raise ValueError("--model-type resnet requires --dataset CIFAR10.")

        if num_classes != 10:
            raise ValueError("This ResNet-4b architecture has exactly 10 outputs.")

        model = build_resnet4b_model()
        dummy_input = torch.zeros(
            1, 3, 32, 32,
            dtype=torch.float32,
        )
        return model, dummy_input

    if model_type == "cnn":
        if dataset != "mnist":
            raise ValueError("--model-type cnn requires --dataset MNIST.")

        model = build_mnist_model(
            n1=n1,
            n2=n2,
            linear_size=linear_size,
        )
        dummy_input = torch.zeros(
            1, 1, 28, 28,
            dtype=torch.float32,
        )
        return model, dummy_input

    if model_type == "cnn5":
        if dataset != "cifar10":
            raise ValueError("--model-type cnn5 requires --dataset CIFAR10.")

        model = build_cnn5_model(
            c1=c1,
            c2=c2,
            c3=c3,
            linear_size=linear_size,
            num_classes=num_classes,
        )
        dummy_input = torch.zeros(
            1, 3, 32, 32,
            dtype=torch.float32,
        )
        return model, dummy_input

    raise ValueError(f"Unsupported model type: {model_type}")

def extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in ("model", "model_state_dict", "state_dict"):
        if key in checkpoint:
            return checkpoint[key]

    return checkpoint


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def export_model(
    checkpoint_path: str,
    output_path: str,
    dataset: str,
    n1: int,
    n2: int,
    c1: int,
    c2: int,
    c3: int,
    linear_size: int,
    num_classes: int,
    opset_version: int,
    model_type: Optional[str] = None,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)

    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state-dict dictionary, got {type(state_dict)}")

    state_dict = clean_state_dict(state_dict)
    model, dummy_input = build_model_and_input(
    dataset=dataset,
    n1=n1,
    n2=n2,
    c1=c1,
    c2=c2,
    c3=c3,
    linear_size=linear_size,
    num_classes=num_classes,
    model_type=model_type,
    )   

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint weights do not match the selected architecture. "
            "Check --dataset, --model-type, and any CNN width arguments.\n"
            f"Original error:\n{exc}"
        ) from exc

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    if output.shape != (1, num_classes):
        raise RuntimeError(
            f"Unexpected model output shape {tuple(output.shape)}; "
            f"expected (1, {num_classes})."
        )

    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
        dynamo=False,
    )

    print(f"Dataset: {normalise_dataset_name(dataset)}")
    print(f"Input shape: {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Successfully exported model to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export an MNIST CNN or CIFAR-10 CNN5 checkpoint to ONNX."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="MNIST",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--n1", type=int, default=16)
    parser.add_argument("--n2", type=int, default=32)
    parser.add_argument("--c1", type=int, default=16)
    parser.add_argument("--c2", type=int, default=32)
    parser.add_argument("--c3", type=int, default=64)
    parser.add_argument("--linear-size", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--opset-version", type=int, default=17)
    parser.add_argument(
        "--model-type",
        "--model_type",
        choices=["cnn", "cnn5", "resnet"],
        default=None,
        help="Defaults to cnn for MNIST and cnn5 for CIFAR10.",
    )

    args = parser.parse_args()
    dataset = normalise_dataset_name(args.dataset)
    linear_size = args.linear_size
    if linear_size is None:
        linear_size = 100 if dataset == "mnist" else 128

    export_model(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        dataset=dataset,
        n1=args.n1,
        n2=args.n2,
        c1=args.c1,
        c2=args.c2,
        c3=args.c3,
        linear_size=linear_size,
        num_classes=args.num_classes,
        opset_version=args.opset_version,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
