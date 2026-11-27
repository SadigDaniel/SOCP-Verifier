import argparse
import os

import torch
import torch.nn as nn


"""
Run:

python -m src.tools.export_to_onnx_old \
  --checkpoint checkpoints/REG/GLS/eps_0.3/crown_IBP_cnn_socp_gls.pt \
  --output verifier/REG/eps_0.1/vnnlib_Crown_Ibp/MNIST/crown_IBP_cnn_socp_gls.onnx
  
python -m src.tools.export_to_onnx_old \
  --checkpoint checkpoints/REG/GLS/eps_0.3/crown_IBP_cnn_socp_gls_bad.pt\
  --output /home/grad/dsadig/SOCP_old/lp_mnist_dual/verifier/REG/eps_0.3/vnnlib_Crown_Ibp_2/MNIST/crown_IBP_cnn_socp_gls_bad.onnx
"""

def build_model():
    return nn.Sequential(
        nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=4,
            stride=2,
            padding=1,
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1,
        ),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(
            in_features=1568,
            out_features=100,
        ),
        nn.ReLU(),
        nn.Linear(
            in_features=100,
            out_features=10,
        ),
    )


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint

    if "model" in checkpoint:
        return checkpoint["model"]

    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    return checkpoint


def export_model(checkpoint_path, output_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    state_dict = extract_state_dict(checkpoint)

    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Expected a state-dict dictionary, got {type(state_dict)}"
        )

    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    model = build_model()
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    dummy_input = torch.zeros(
        1,
        1,
        28,
        28,
        dtype=torch.float32,
    )

    with torch.no_grad():
        output = model(dummy_input)

    print("Model output shape:", tuple(output.shape))

    output_directory = os.path.dirname(output_path)

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13, #17
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )

    print(f"Successfully exported model to: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    export_model(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()