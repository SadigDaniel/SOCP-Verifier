#!/usr/bin/env python3
"""Train CIFAR-10 ResNet-4b, then run L-infinity PGD on test indices 0..199.

Install dependencies in your PyTorch environment:
    pip install torch torchvision numpy

Place this file in src/ and run from the SOCP-Verifier repository root:
    nohup python -m src.train_resnet4b_cifar10\
        --device cuda\
        > train_resnet4b_cifar10.out 2>&1 &

Evaluate a saved checkpoint without training again:
    python -m src.train_resnet4b_cifar10 --device cuda --eval-only \
        --checkpoint checkpoints/cifar10/resnet4b_last.pt

Defaults copied from SadigDaniel/SOCP and SOCP-Verifier's CIFAR configuration:
    seed=0, epochs=40, batch_size=128, AdamW(lr=0.001, weight_decay=0.0005),
    gradient_clip=5, MultiStepLR(milestones=[20,27], gamma=0.5).
    CIFAR-10 uses ToTensor() only: no normalization or augmentation.

Training here is ordinary clean cross-entropy, without RGS. The repositories'
certified loss is not used: its sequential bound propagation does not support
this residual architecture. Epsilon=1/255 is the *evaluation* pixel-space
L-infinity radius, not the repositories' 2/225 training radius.

Architecture: alpha-beta-CROWN's CIFAR-10 resnet4b(), without batch norm:
    stem 3->16 (stride 2); two 32-channel residual stages, two blocks each;
    each stage starts with stride 2; flatten 32*4*4 -> 100 -> 10.
    State-dict names/shapes match that upstream resnet4b() implementation.

PGD defaults: untargeted cross-entropy, 100 steps, 10 random restarts,
step size epsilon/4.

Outputs (under checkpoints/cifar10/): resnet4b_last.pt, pgd_first200.csv.
The checkpoint is overwritten after every epoch. PGD evaluates the final epoch;
no checkpoint is selected on the test set.
Seeds are reproducible on a fixed software/hardware stack; different PyTorch
versions/devices need not yield bitwise-identical training results.
"""

import argparse
import csv
import math
import os
from pathlib import Path
import random

# Required by deterministic CUDA matrix multiplication, before CUDA starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


EVAL_IMAGES = 200


class BasicBlock(nn.Module):
    """Two 3x3 convolutions and an identity or projected residual shortcut."""

    def __init__(self, in_planes, planes=32, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=True)
            )

    def forward(self, x):
        out = self.conv2(F.relu(self.conv1(x)))
        return F.relu(out + self.shortcut(x))


class ResNet4b(nn.Module):
    """CIFAR-10 verification ResNet-4b: four residual blocks, no batch norm."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 2, 1, bias=True)
        self.layer1 = nn.Sequential(BasicBlock(16, 32, 2), BasicBlock(32))
        self.layer2 = nn.Sequential(BasicBlock(32, 32, 2), BasicBlock(32))
        self.linear1 = nn.Linear(32 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.layer2(self.layer1(x))
        return self.linear2(F.relu(self.linear1(torch.flatten(x, 1))))


def set_seed(seed):
    """Same four RNG seeds as SOCP's src/utils.py, plus deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loaders(args, device):
    """Same data representation/order as src/data.py; fully self-contained."""
    transform = transforms.ToTensor()
    test_set = datasets.CIFAR10(
        args.data_dir, train=False, download=True, transform=transform
    )
    options = dict(
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.test_batch_size, shuffle=False, **options
    )
    # Slice the original dataset, NOT a shuffled loader or first 200 correct cases.
    if len(test_set) < EVAL_IMAGES:
        raise ValueError("The test dataset must contain at least 200 images.")
    attack_loader = DataLoader(
        Subset(test_set, range(EVAL_IMAGES)),
        batch_size=args.pgd_batch_size, shuffle=False, **options
    )
    train_loader = None
    if not args.eval_only:
        train_set = datasets.CIFAR10(
            args.data_dir, train=True, download=True, transform=transform
        )
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, **options
        )
    return train_loader, test_loader, attack_loader


@torch.no_grad()
def evaluate_clean(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        correct += model(x).argmax(1).eq(y).sum().item()
        total += y.numel()
    return correct / total


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(model, train_loader, test_loader, args, device, output_dir):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma
    )
    checkpoint_path = output_dir / "resnet4b_last.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = correct = total = 0
        current_lr = optimizer.param_groups[0]["lr"]
        for batch, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss at epoch {epoch}, batch {batch}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            loss_sum += loss.item() * y.numel()
            correct += logits.detach().argmax(1).eq(y).sum().item()
            total += y.numel()
            if batch % 100 == 0:
                print(f"  epoch {epoch}: batch {batch}/{len(train_loader)} "
                      f"loss={loss_sum/total:.4f}", flush=True)
        scheduler.step()
        clean_accuracy = evaluate_clean(model, test_loader, device)
        # 'model' matches the checkpoint key used throughout the user's projects.
        torch.save(dict(
            model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), config=vars(args), epoch=epoch,
            dataset="cifar10", model_type="resnet4b", training_objective="clean_ce",
            preprocessing="ToTensor only; pixels in [0,1]",
            clean_acc=clean_accuracy, seed=args.seed,
        ), checkpoint_path)
        print(f"Epoch {epoch:02d}/{args.epochs}: loss={loss_sum/total:.4f} "
              f"train={100*correct/total:.2f}% "
              f"clean test={100*clean_accuracy:.2f}% lr={current_lr:.6g}", flush=True)
        print(f"Saved epoch {epoch} checkpoint: {checkpoint_path}", flush=True)
    return checkpoint_path


def pgd_linf(model, x, y, epsilon, step_size, steps, restarts, generator):
    """Return a per-example witness, any observed failure, and witness CE.

    Track successes across EVERY iterate/restart so later correct predictions
    cannot erase an adversarial example. Parameter .grad fields are untouched.
    Call with model.eval(); pixels and epsilon are in the original [0,1] space.
    """
    x = x.detach()
    lower = (x - epsilon).clamp(0, 1)
    upper = (x + epsilon).clamp(0, 1)
    with torch.no_grad():
        logits = model(x)
        best_loss = F.cross_entropy(logits, y, reduction="none")
        failed = logits.argmax(1).ne(y)
        best_adv = x.clone()

    def record(candidate, logits, losses):
        nonlocal best_adv, best_loss, failed
        with torch.no_grad():
            candidate_failed = logits.argmax(1).ne(y)
            # Prefer an actual failure over a higher-loss, correctly classified
            # iterate, then maximize CE among candidates with the same status.
            take = (candidate_failed & ~failed) | (
                candidate_failed.eq(failed) & (losses > best_loss)
            )
            best_adv[take] = candidate.detach()[take]
            best_loss = torch.where(take, losses.detach(), best_loss)
            failed = failed | candidate_failed

    with torch.enable_grad():
        for _restart in range(restarts):
            noise = torch.rand(x.shape, device=x.device, dtype=x.dtype,
                               generator=generator) * (2 * epsilon) - epsilon
            adv = torch.maximum(torch.minimum(x + noise, upper), lower).detach()
            # Also evaluate the FINAL updated point: steps+1 evaluated iterates.
            for step in range(steps + 1):
                adv.requires_grad_(True)
                logits = model(adv)
                losses = F.cross_entropy(logits, y, reduction="none")
                if not torch.isfinite(losses).all():
                    raise RuntimeError("Nonfinite logits/loss during PGD.")
                record(adv, logits, losses)
                if step == steps:
                    break
                gradient, = torch.autograd.grad(losses.sum(), adv)
                if not torch.isfinite(gradient).all():
                    raise RuntimeError("Nonfinite input gradient during PGD.")
                adv = adv.detach() + step_size * gradient.sign()
                adv = torch.maximum(torch.minimum(adv, upper), lower).detach()
    return best_adv.detach(), failed.detach(), best_loss.detach()


def evaluate_pgd(model, loader, args, device, output_dir):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    rows = []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.no_grad():
            clean_prediction = model(x).argmax(1)
        adv, failed, losses = pgd_linf(
            model, x, y, args.epsilon, args.pgd_step_size,
            args.pgd_steps, args.pgd_restarts, generator,
        )
        with torch.no_grad():
            adv_prediction = model(adv).argmax(1)
            linf = (adv - x).flatten(1).abs().amax(1)
            if (linf > args.epsilon + 1e-6).any() or (adv < 0).any() or (adv > 1).any():
                raise RuntimeError("PGD generated an out-of-budget candidate.")
            if not torch.equal(adv_prediction.ne(y), failed):
                raise RuntimeError("Stored PGD witnesses disagree with observed failures.")
        start_index = len(rows)
        for offset in range(y.numel()):
            rows.append(dict(
                test_index=start_index + offset, label=y[offset].item(),
                clean_prediction=clean_prediction[offset].item(),
                pgd_prediction=adv_prediction[offset].item(),
                clean_correct=clean_prediction[offset].eq(y[offset]).item(),
                pgd_robust=(~failed[offset]).item(),
                witness_ce=losses[offset].item(), witness_linf=linf[offset].item(),
            ))
        print(f"PGD {len(rows)}/{EVAL_IMAGES}: "
              f"clean={sum(r['clean_correct'] for r in rows)}/{len(rows)}, "
              f"PGD robust={sum(r['pgd_robust'] for r in rows)}/{len(rows)}", flush=True)
    if len(rows) != EVAL_IMAGES:
        raise RuntimeError(f"Expected exactly 200 test images, got {len(rows)}.")
    write_csv(output_dir / "pgd_first200.csv", rows)
    clean_correct = sum(row["clean_correct"] for row in rows)
    robust = sum(row["pgd_robust"] for row in rows)
    return clean_correct, robust


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CIFAR-10 ResNet-4b (seed 0); PGD-evaluate test indices 0..199.",
        epilog="Defaults: 40 epochs of clean CE; Linf PGD at 1/255, 100 steps, 10 restarts.",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="checkpoints/cifar10",
                        help="Directory for the checkpoint and PGD CSV, relative to the working directory.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--lr-milestones", type=int, nargs="*", default=[20, 27])
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=1.0 / 255.0,
                        help="Linf radius in raw [0,1] pixels (default: 1/255).")
    parser.add_argument("--pgd-steps", type=int, default=100)
    parser.add_argument("--pgd-restarts", type=int, default=10)
    parser.add_argument("--pgd-step-size", type=float, default=None,
                        help="Raw pixel step size (default: epsilon/4).")
    parser.add_argument("--pgd-batch-size", type=int, default=50)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", help="This script's checkpoint; requires --eval-only.")
    args = parser.parse_args()
    if args.pgd_step_size is None:
        args.pgd_step_size = args.epsilon / 4.0
    for name in ["epochs", "batch_size", "test_batch_size", "pgd_batch_size",
                 "pgd_steps", "pgd_restarts"]:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    for name in ["lr", "gradient_clip", "lr_gamma", "pgd_step_size"]:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive.")
    if not math.isfinite(args.epsilon) or not 0 < args.epsilon <= 1:
        parser.error("--epsilon must lie in (0,1].")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay must be finite and nonnegative.")
    if args.num_workers < 0 or not 0 <= args.seed < 2**32:
        parser.error("Use num-workers >= 0 and a seed in [0, 2**32).")
    if any(milestone <= 0 for milestone in args.lr_milestones):
        parser.error("Learning-rate milestones must be positive epochs.")
    if args.eval_only != bool(args.checkpoint):
        parser.error("Supply --eval-only and --checkpoint together.")
    return args


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but PyTorch cannot access a CUDA GPU.")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = ResNet4b().to(device)
    if args.eval_only:
        checkpoint_path = Path(args.checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("model_type") != "resnet4b":
            raise ValueError("Expected a ResNet-4b checkpoint produced by this script.")
        model.load_state_dict(checkpoint["model"], strict=True)
        print(f"Loaded {checkpoint_path} (epoch {checkpoint['epoch']}, "
              f"training seed {checkpoint['seed']})", flush=True)
    train_loader, test_loader, attack_loader = make_loaders(args, device)
    print(f"Device={device}; seed={args.seed}; model=ResNet-4b; "
          f"parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    if args.eval_only:
        full_clean_accuracy = evaluate_clean(model, test_loader, device)
        print(f"Clean accuracy (full test set): {100*full_clean_accuracy:.2f}%", flush=True)
    else:
        print("Training objective: clean cross-entropy", flush=True)
        train(
            model, train_loader, test_loader, args, device, output_dir
        )
    print(f"Attacking test indices 0..199: Linf epsilon={args.epsilon:.10f}, "
          f"steps={args.pgd_steps}, restarts={args.pgd_restarts}, "
          f"step size={args.pgd_step_size:.10f}", flush=True)
    clean_correct, robust = evaluate_pgd(
        model, attack_loader, args, device, output_dir
    )
    print(f"\nClean accuracy (first 200): {clean_correct}/200 "
          f"= {100*clean_correct/EVAL_IMAGES:.2f}%")
    print(f"PGD accuracy   (first 200): {robust}/200 "
          f"= {100*robust/EVAL_IMAGES:.2f}%")
    print(f"Results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()


# Upstream architecture license (BSD 3-Clause):
# Copyright (C) 2021-2026 The α,β-CROWN Team
# See CONTRIBUTORS for the list of all contributors and their affiliations.
#     Team leaders:
#          Faculty: Huan Zhang <huan@huan-zhang.com> (UIUC)
#          Student: Xiangru Zhong <xiangru4@illinois.edu> (UIUC)
#     Current developers:
#                   Duo Zhou <duozhou2@illinois.edu> (UIUC)
#                   Keyi Shen <keyis2@illinois.edu> (UIUC/Georgia Tech)
#                   Hesun Chen <hesunc2@illinois.edu> (UIUC)
#                   Haoyu Li <haoyuli5@illinois.edu> (UIUC)
#                   Ruize Gao <ruizeg2@illinois.edu> (UIUC)
#                   Hao Cheng <haoc539@illinois.edu> (UIUC)
#                   Zhouxing Shi <zhouxingshichn@gmail.com> (UCLA/UC Riverside)
#                   Lei Huang <leih5@illinois.edu> (UIUC)
#                   Taobo Liao <taobol2@illinois.edu> (UIUC)
#                   Jorge Chavez <jorgejc2@illinois.edu> (UIUC)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
