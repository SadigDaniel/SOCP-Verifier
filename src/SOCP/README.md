# SOCP implementation and evaluation

This package evaluates pretrained MNIST and CIFAR-10 ReLU classifiers with a
sparse SOCP relaxation. See the [main README](../../README.md) for the research
scope, method overview, environment setup, hardware, and reported results.

[MNIST command](#mnist-experiment-command) · [Other models](#other-model-examples) ·
[Options](#full-argument-reference) · [Metrics](#evaluation-behavior-and-metrics) ·
[Pipeline](#verification-pipeline) · [Backend](#legacy-backend-and-performance) ·
[Files](#implementation-inventory)

## Launch requirements

Activate the [environment](../../README.md#environment-setup). From the
repository root, change to `src/`:

```bash
cd src
```

The evaluator accepts a state dictionary directly or a dictionary containing it
under the `model` key. Architecture settings must match the checkpoint.
The loaders in [`../data.py`](../data.py) use `ToTensor()` without mean/std
normalization. Epsilon is an L-infinity radius in [0,1] pixel units, with
perturbations clipped to [0,1].

## MNIST experiment command

This experiment evaluates a CROWN-IBP-trained MNIST CNN with training epsilon
**0.3** and verification epsilon **0.2**. From `src/`:

```bash
nohup python -m SOCP.eval_SOCP_robustness \
  --checkpoint ../checkpoints/eps_0.3/crown_IBP_cnn_standard.pt \
  --model_type cnn --epsilon 0.2 \
  --linear_size 100 --max_E 16 --max_S 8 --max_T 8 \
  --prev_candidate_limit 32 --gamma_backend dual --max_targets 9 \
  --solver_max_iters 500 \
  > MNIST_crown_IBP_cnn_train0.3_Val0.2.out 2>&1 &
```

Monitor the output with:

```bash
tail -f MNIST_crown_IBP_cnn_train0.3_Val0.2.out
```

The command uses the included
[`eps_0.3/crown_IBP_cnn_standard.pt`](../../checkpoints/eps_0.3/crown_IBP_cnn_standard.pt)
checkpoint.

| Setting | Value |
| --- | --- |
| Architecture | `n1=16`, `n2=32` from CLI defaults; `linear_size=100`. |
| Training / verification epsilon | 0.3 / 0.2. |
| E/S/T budgets | `16 / 8 / 8` per hidden layer. |
| Candidate budgets | `prev_candidate_limit=32`; `nodes_per_layer=8` from the CLI default. |
| Influence backend | `dual`, the CROWN-style backward calculation. |
| Target coverage | `max_targets=9` checks all incorrect classes for a ten-class model. |
| Solver | `SCS` from the CLI default, with an explicit 500-iteration limit. |

## Other model examples

Use `--model_type tiny` for the fully connected MNIST model and set its two
hidden widths with `--n1` and `--n2`.

For CIFAR-10 CNN5, provide weights matching the architecture. From `src/`:

```bash
python -m SOCP.eval_SOCP_robustness \
  --checkpoint /path/to/cifar10_checkpoint.pt \
  --data_dir ../data --model_type cifar10 \
  --n1 24 --n2 48 --n3 96 --linear_size 192 --epsilon 0.0088889 \
  --max_E 8 --max_S 4 --max_T 4 --prev_candidate_limit 16 \
  --gamma_backend dual --all_classes \
  --solver SCS --solver_max_iters 500
```

These are Bash examples. In PowerShell, put the Python command on one line or
use PowerShell continuations.

| Model type | Dataset / input | Builder |
| --- | --- | --- |
| `tiny` | MNIST, `1 × 28 × 28` | `build_mnist_tiny_model(n1, n2, linear_size)`; `linear_size` is unused. |
| `cnn` | MNIST, `1 × 28 × 28` | `build_mnist_model(n1, n2, linear_size)`; two convolution/ReLU blocks and a hidden linear/ReLU layer. |
| `cifar10` | CIFAR-10, `3 × 32 × 32` | `build_cnn5_model(c1, c2, c3, linear_size)`; `n1/n2/n3` map to `c1/c2/c3`. |

## Full argument reference

The defaults below come from [`eval_SOCP_robustness.py`](eval_SOCP_robustness.py).

| Option | CLI default | Meaning |
| --- | --- | --- |
| `--checkpoint` | Required | Saved weights to evaluate. |
| `--data_dir` | `../data` | Dataset directory, relative to the working directory. |
| `--model_type` | `tiny` | `tiny` or `cnn` selects MNIST; `cifar10` selects CIFAR-10. |
| `--n1`, `--n2`, `--n3` | `16`, `32`, `96` | Architecture widths; `n3` is used only for CIFAR-10. |
| `--linear_size` | `100` | Hidden classifier width for CNN models. |
| `--epsilon` | Required | L-infinity radius in pixel units. |
| `--nodes_per_layer` | `8` | Retained neuron-candidate budget per hidden layer. |
| `--max_E`, `--max_S`, `--max_T` | `8`, `6`, `6` | Per-layer cross-layer, current-layer, and previous-layer product budgets. |
| `--prev_candidate_limit` | `16` | Previous-layer candidates considered when scoring pairs. |
| `--gamma_backend` | `dual` | CROWN-style `dual`, Wong–Kolter `dual_WK`, or clean-gradient `clean` influence. |
| `--solver` | `SCS` | Conic solver; `CLARABEL` is also supported. |
| `--solver_max_iters` | `200` | Solver iteration limit; the examples explicitly set 500. |
| `--solver_verbose` | Off | Print solver diagnostics. |
| `--all_classes` | Off | Check every incorrect class. |
| `--max_targets` | `4` | Number of incorrect classes checked without `--all_classes`. |
| `--test_batch_size` | `9` | Loader batch size; each SOCP problem handles one image and one target. |

Direct calls to [`CertificateConfig`](socp_certificate.py) have different defaults:

| Setting | Evaluator CLI | `CertificateConfig` |
| --- | --- | --- |
| Epsilon | Required | `0.3` |
| `nodes_per_layer` | `8` | `16` |
| `max_E / max_S / max_T` | `8 / 6 / 6` | `64 / 32 / 32` |
| `prev_candidate_limit` | `16` | `128` |
| Solver / iteration limit | `SCS / 200` | `CLARABEL / 100` |
| `all_classes` | `False` | `True` |
| `max_targets` | `4` | `4` |
| `gamma_backend` | `dual` | `dual` |

## Evaluation behavior and metrics

Cleanly misclassified inputs count as uncertified and skip SOCP construction.
For each cleanly correct input, the verifier maximizes
`f_target(x') - f_true(x')` over the relaxation.

| Metric | Definition |
| --- | --- |
| Clean accuracy | Cleanly correct inputs divided by evaluated inputs. |
| Printed certified accuracy | Inputs with the current `certified` flag divided by evaluated inputs. |
| Certification rate | Flagged inputs divided by cleanly correct inputs. |
| Average worst margin | Mean largest returned target margin for each cleanly correct input. |
| Average solve time | Mean full `certify_sample` time, including conversion, bounds, selection, and all target solves. |
| Variables / constraints | Averages over target problems; constraint count is the number of CVXPY constraint objects. |
| Solver statuses | Counts over individual target problems. |

Use `--all_classes`, or `--max_targets 9` for these ten-class models, for
complete target coverage. The default four targets give partial-target results.

The current `certified` property checks only whether the largest returned
margin is negative. It does not enforce target coverage or solver status and
can count `optimal_inaccurate` results. Check convergence and numerical validity
before treating the printed flag as a robustness guarantee.

On a solver exception, both solvers retry with SCS at 500 iterations. The
returned `solver` field retains the requested name. The evaluator has no PGD
pre-check.

## Verification pipeline

| Stage | Function / module |
| --- | --- |
| Dense affine conversion | `sequential_to_dense_affines` in [`utils.py`](utils.py). |
| Interval bounds | `collect_socp_bounds` in [`socp_bounds.py`](socp_bounds.py). |
| Target selection | `choose_target_classes` in [`socp_certificate.py`](socp_certificate.py). |
| Influence scoring | `compute_margin_gammas` in [`socp_influence.py`](socp_influence.py). |
| Sparse selection | `select_sparse_couplings` in [`socp_relaxation.py`](socp_relaxation.py). |
| SOCP construction and solve | `SparseSOCPVerifier.solve_margin` in [`socp_solver.py`](socp_solver.py). |

Couplings are selected separately for each input and target. Square lifts are
created only when needed by selected pairs. Previous-layer variables are the
perturbed image in the first hidden layer and ReLU activations in later layers.

## Legacy backend and performance

For slow runs, the legacy pair `socp_solver_old.py` and
`socp_relaxation_old.py` is available. There is no CLI backend selector.
From the repository root, back up the active files and copy the legacy pair:

```bash
mkdir -p .socp_backend_backup
cp -n src/SOCP/socp_solver.py .socp_backend_backup/socp_solver.py
cp -n src/SOCP/socp_relaxation.py .socp_backend_backup/socp_relaxation.py
cp src/SOCP/socp_solver_old.py src/SOCP/socp_solver.py
cp src/SOCP/socp_relaxation_old.py src/SOCP/socp_relaxation.py
```

Restart Python and evaluate from `src/`. Restore the active implementation
from the repository root with:

```bash
cp .socp_backend_backup/socp_solver.py src/SOCP/socp_solver.py
cp .socp_backend_backup/socp_relaxation.py src/SOCP/socp_relaxation.py
```

The legacy selector is currently identical to the active selector. The legacy
solver omits residual links between selected E products and activation-square
lifts, changing the relaxation and potentially weakening bounds. Record the
backend with each run; a speedup is not guaranteed.

Both backends use dense convolution matrices, rebuilt for each input by the
SOCP evaluator. Conversion and CVXPY construction can be costly. Smaller
coupling/candidate budgets can reduce work; fewer target classes give only
partial-target results.

## Implementation inventory

### Shared source files

| File | Role |
| --- | --- |
| [`model.py`](../model.py) | MNIST MLP/CNN and CIFAR-10 CNN5 builders. |
| [`data.py`](../data.py) | Dataset loaders and [0,1] preprocessing. |
| [`bound_layers.py`](../bound_layers.py) | Input and interval layer bounds. |
| [`dual_bounds.py`](../dual_bounds.py) | Bound records and CROWN/Wong–Kolter routines. |
| [`deeppoly_bounds.py`](../deeppoly_bounds.py) | Additional DeepPoly routines. |

### SOCP package

| File | Role |
| --- | --- |
| [`eval_SOCP_robustness.py`](eval_SOCP_robustness.py) | Evaluation and metrics. |
| [`socp_certificate.py`](socp_certificate.py) | Certificate orchestration. |
| [`socp_bounds.py`](socp_bounds.py) | Bound collection. |
| [`socp_influence.py`](socp_influence.py) | Backward influence scores. |
| [`socp_relaxation.py`](socp_relaxation.py) | Sparse E/S/T selection. |
| [`socp_solver.py`](socp_solver.py) | CVXPY conic formulation. |
| [`socp_solver_old.py`](socp_solver_old.py), [`socp_relaxation_old.py`](socp_relaxation_old.py) | Legacy backend pair. |
| [`utils.py`](utils.py) | Device/seed helpers and conversion. |
| [`__init__.py`](__init__.py) | Package marker. |
| [`README.md`](README.md) | Technical documentation. |
| [`FLOWCHART.md`](FLOWCHART.md) | Additional pipeline documentation. |

CORE's cascade, verifier interfaces, result types, and adapters are documented
in the [CORE README](../CORE/README.md).
