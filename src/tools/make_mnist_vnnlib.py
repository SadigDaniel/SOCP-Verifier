import argparse
import os
import torch
import csv
from fractions import Fraction
from torchvision import datasets, transforms


'''
LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6" \
python -m src.tools.make_mnist_vnnlib \
  --data-dir data \
  --data_set MNIST \
  --out-dir /home/grad/dsadig/SOCP_old/lp_mnist_dual/verifier/REG/eps_0.3/vnnlib_Crown_Ibp_2 \
  --epsilon 0.3 \
  --num-samples 5000 \
  --onnx-path MNIST/crown_IBP_cnn_socp_gls_bad.onnx
'''
'''
python -m src.tools.make_mnist_vnnlib \
  --data-dir data \
  --data_set cifar10 \
  --out-dir verifier/cifar10/eps_0.0089/vnnlib_dual/ \
  --epsilon 0.0088888889 \
  --num-samples 5000 \
  --onnx-path cifar10/cifar10_dual_cnn5_standard.onnx
  
  
python -m src.tools.make_mnist_vnnlib \
    --data-dir data \
    --data_set CIFAR10 \
    --out-dir verifier/cifar10/eps_1_255/vnnlib_resnet \
    --epsilon 1/255 \
    --num-samples 200 \
    --onnx-path verifier/cifar10/resnet4b_last.onnx \
    --timeout 120
  
'''


import argparse
import csv
import os
from fractions import Fraction

import torch
from torchvision import datasets, transforms


DATASET_ALIASES = {
    "mnist": "mnist",
    "cifar10": "cifar10",
    "cifar-10": "cifar10",
}


def normalise_dataset_name(value):
    key = value.strip().lower()

    if key not in DATASET_ALIASES:
        raise argparse.ArgumentTypeError(
            "Dataset must be MNIST, CIFAR10, or CIFAR-10."
        )

    return DATASET_ALIASES[key]


def parse_epsilon(value):
    try:
        epsilon = float(Fraction(value))
    except (ValueError, ZeroDivisionError, OverflowError):
        raise argparse.ArgumentTypeError(
            "Use a decimal or fraction, such as 0.3 or 1/255."
        )

    if not 0.0 <= epsilon <= 1.0:
        raise argparse.ArgumentTypeError(
            "Epsilon must be between 0 and 1."
        )

    return epsilon


def write_vnnlib(path, x, label, epsilon):
    # Flatten in channel-first order, matching the exported ONNX input.
    pixels = x.detach().cpu().to(torch.float64).reshape(-1)
    lower = torch.clamp(pixels - epsilon, 0.0, 1.0)
    upper = torch.clamp(pixels + epsilon, 0.0, 1.0)

    num_inputs = pixels.numel()
    num_outputs = 10
    label = int(label)

    if not 0 <= label < num_outputs:
        raise ValueError(f"Invalid class label: {label}")

    with open(path, "w", encoding="utf-8") as file:
        for i in range(num_inputs):
            file.write(f"(declare-const X_{i} Real)\n")

        for j in range(num_outputs):
            file.write(f"(declare-const Y_{j} Real)\n")

        file.write("\n")

        for i, (lo, hi) in enumerate(zip(lower.tolist(), upper.tolist())):
            file.write(f"(assert (>= X_{i} {lo:.17g}))\n")
            file.write(f"(assert (<= X_{i} {hi:.17g}))\n")

        # Unsafe condition: any competing logit reaches the true-class logit.
        file.write("\n(assert\n")
        file.write("  (or\n")

        for target in range(num_outputs):
            if target != label:
                file.write(
                    f"    (and (>= Y_{target} Y_{label}))\n"
                )

        file.write("  )\n")
        file.write(")\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate MNIST or CIFAR-10 robustness specifications "
            "for an exported ONNX model."
        )
    )

    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--dataset",
        "--data_set",
        dest="dataset",
        type=normalise_dataset_name,
        required=True,
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--epsilon",
        type=parse_epsilon,
        required=True,
        help="Raw-pixel Linf radius, e.g. 0.3 or 1/255.",
    )
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument(
        "--onnx-path",
        required=True,
        help="Existing ONNX file, relative to the working directory or absolute.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-image timeout in seconds.",
    )

    args = parser.parse_args()

    if args.num_samples < 1:
        parser.error("--num-samples must be positive.")

    if args.timeout < 1:
        parser.error("--timeout must be positive.")

    onnx_path = os.path.abspath(os.path.expanduser(args.onnx_path))
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))

    if not os.path.isfile(onnx_path):
        parser.error(f"ONNX model does not exist: {onnx_path}")

    dataset_class = (
        datasets.MNIST if args.dataset == "mnist"
        else datasets.CIFAR10
    )

    dataset = dataset_class(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    if args.num_samples > len(dataset):
        parser.error(
            f"Requested {args.num_samples} samples, "
            f"but the test dataset contains {len(dataset)}."
        )

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "instances.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")

        # Original test-set order; no shuffling or filtering.
        for idx in range(args.num_samples):
            x, label = dataset[idx]
            label = int(label)

            vnnlib_path = os.path.join(
                out_dir,
                f"sample_{idx}_label_{label}.vnnlib",
            )

            write_vnnlib(
                path=vnnlib_path,
                x=x,
                label=label,
                epsilon=args.epsilon,
            )

            writer.writerow([
                onnx_path,
                vnnlib_path,
                args.timeout,
            ])

    print(f"Dataset: {args.dataset}")
    print(f"Test indices: 0..{args.num_samples - 1}")
    print(f"Epsilon: {args.epsilon:.17g}")
    print(f"ONNX model: {onnx_path}")
    print(f"Saved VNNLIB files to: {out_dir}")
    print(f"Saved instance CSV to: {csv_path}")


if __name__ == "__main__":
    main()