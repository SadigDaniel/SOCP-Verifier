"""Assemble the verifier's sparse ReLU relaxation for the PyTorch SOCP solver."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from SOCP.gpu_socp import Cones, SparseSOCP
from SOCP.socp_bounds import LayerBounds
from SOCP.socp_relaxation import SOCPPattern
from SOCP.utils import DenseAffineLayer


def _up(value):
    return np.nextafter(value, np.inf)


def _down(value):
    return np.nextafter(value, -np.inf)


def _sum_error(magnitude, terms):
    # Allow for products and reductions of float64 coefficients, including
    # gradual underflow. The extra factor covers rounding in the magnitude.
    info = np.finfo(np.float64)
    count = terms + 1
    return _up(_up((8 * info.eps * count) * magnitude) + 8 * info.tiny * count)


def _product_box(a_lower, a_upper, b_lower, b_upper):
    corners = np.stack(
        [a_lower * b_lower, a_lower * b_upper, a_upper * b_lower, a_upper * b_upper]
    )
    return _down(corners.min(axis=0)), _up(corners.max(axis=0))


def _affine_box(weight, bias, lower, upper):
    rows, cols = np.nonzero(weight)
    coeff = weight[rows, cols]
    left = coeff * lower[cols]
    right = coeff * upper[cols]
    size = bias.size
    low = np.bincount(rows, weights=np.minimum(left, right), minlength=size)
    high = np.bincount(rows, weights=np.maximum(left, right), minlength=size)
    mass = np.bincount(rows, weights=np.maximum(abs(left), abs(right)), minlength=size)
    error = _sum_error(mass, np.bincount(rows, minlength=size))
    return _down(_down(low - error) + bias), _up(_up(high + error) + bias)


def collect_cuda_bounds(
    affines: Sequence[DenseAffineLayer], x: torch.Tensor, epsilon: float,
) -> tuple[np.ndarray, np.ndarray, list[LayerBounds]]:
    """Enclose the real-valued affine/ReLU model using float64 intervals."""
    center = x.detach().to(device="cpu", dtype=torch.float64).reshape(-1).numpy()
    if not np.isfinite(epsilon) or epsilon < 0 or not np.isfinite(center).all():
        raise ValueError("GPU SOCP requires a finite input and a finite nonnegative epsilon.")
    input_lower = np.clip(_down(center - epsilon), 0.0, 1.0)
    input_upper = np.clip(_up(center + epsilon), 0.0, 1.0)
    lower, upper = input_lower, input_upper
    hidden_bounds = []
    for layer in affines[:-1]:
        s_lower, s_upper = _affine_box(layer.weight, layer.bias, lower, upper)
        lower, upper = np.maximum(s_lower, 0.0), np.maximum(s_upper, 0.0)
        hidden_bounds.append(LayerBounds(s_lower, s_upper, lower, upper, layer.name))
    return input_lower, input_upper, hidden_bounds


class _Rows:
    """Coordinate arrays for one group of cone rows."""

    def __init__(self):
        self.size = 0
        self.rhs = []
        self.indices = []
        self.values = []

    def add(self, rhs, rows, cols, values):
        rhs = np.atleast_1d(np.asarray(rhs, dtype=np.float64))
        self.rhs.append(rhs)
        self.indices.append(np.stack((np.asarray(rows, dtype=np.int64) + self.size, cols)))
        self.values.append(np.asarray(values, dtype=np.float64))
        self.size += rhs.size

    def row(self, rhs, *terms):
        self.add(rhs, [0] * len(terms), [col for col, _ in terms], [coef for _, coef in terms])

    def affine(self, output, inputs, weight, bias):
        rows, cols = np.nonzero(weight)
        diagonal = np.arange(output.size)
        self.add(
            bias, np.concatenate((rows, diagonal)), np.concatenate((inputs[cols], output)),
            np.concatenate((weight[rows, cols], -np.ones(output.size))),
        )


class _MarginProblem:
    def __init__(self):
        self.eq = _Rows()
        self.ineq = _Rows()
        self.soc = _Rows()
        self.lower = []
        self.upper = []
        self.squares = {}

    def variables(self, lower, upper):
        lower = np.atleast_1d(np.asarray(lower, dtype=np.float64))
        upper = np.atleast_1d(np.asarray(upper, dtype=np.float64))
        indices = np.arange(len(self.lower), len(self.lower) + lower.size)
        self.lower.extend(lower)
        self.upper.extend(upper)
        return indices

    def relu(self, pre, post, lower, upper):
        active = lower >= 0
        unstable = (lower < 0) & (upper > 0)
        active_rows = np.arange(np.count_nonzero(active))
        self.eq.add(
            np.zeros(active_rows.size), np.tile(active_rows, 2),
            np.concatenate((pre[active], post[active])),
            np.repeat([1.0, -1.0], active_rows.size),
        )
        # Inactive ReLUs already have [0, 0] as their activation box.
        rows = np.arange(np.count_nonzero(unstable))
        pre, post = pre[unstable], post[unstable]
        lower, upper = lower[unstable], upper[unstable]
        self.ineq.add(
            np.zeros(rows.size), np.tile(rows, 2), np.concatenate((post, pre)),
            np.repeat([1.0, -1.0], rows.size),
        )
        # (u - l) z <= u (s - l). Round in the relaxing direction since z >= 0.
        self.ineq.add(
            _up((-lower) * upper), np.tile(rows, 2), np.concatenate((pre, post)),
            np.concatenate((upper, -_down(upper - lower))),
        )

    def square(self, var):
        if var in self.squares:
            return self.squares[var]
        lower, upper = self.lower[var], self.upper[var]
        diagonal, = self.variables(0.0, _up(max(lower * lower, upper * upper)))
        slope = lower + upper
        # For the rounded slope, enclose a^2 - slope*a at both endpoints.
        intercept = max(_up(_up(a * a) - _down(slope * a)) for a in (lower, upper))
        self.ineq.row(intercept, (var, slope), (diagonal, -1.0))
        self.soc.row(1.0, (diagonal, 1.0))
        self.soc.row(0.0, (var, 2.0))
        self.soc.row(-1.0, (diagonal, 1.0))
        self.squares[var] = diagonal
        return diagonal

    def add_mccormick(self, a, b, lift):
        la, ua = self.lower[a], self.upper[a]
        lb, ub = self.lower[b], self.upper[b]
        self.ineq.row(_up(la * lb), (lift, 1.0), (a, -lb), (b, -la))
        self.ineq.row(_up(ua * ub), (lift, 1.0), (a, -ub), (b, -ua))
        self.ineq.row(_up(-ua * lb), (lift, -1.0), (a, lb), (b, ua))
        self.ineq.row(_up(-la * ub), (lift, -1.0), (a, ub), (b, la))

    def couple(self, a, b):
        if a == b:
            return self.square(a)
        low, high = _product_box(self.lower[a], self.upper[a], self.lower[b], self.upper[b])
        lift, = self.variables(low, high)
        self.add_mccormick(a, b, lift)
        da, db = self.square(a), self.square(b)
        self.soc.row(0.0, (da, 1.0), (db, 1.0))
        self.soc.row(0.0, (lift, 2.0))
        self.soc.row(0.0, (da, 1.0), (db, -1.0))
        return lift

    def residual_link(self, previous, current, weight, bias, selected):
        chosen, lifts = zip(*selected)
        omitted = weight != 0.0
        omitted[list(chosen)] = False
        terms = [(self.square(current), 1.0), (current, -bias)]
        terms.extend((lift, -weight[index]) for index, lift in zip(chosen, lifts))
        if not np.any(omitted):
            self.eq.row(0.0, *terms)
            return

        inputs = previous[omitted]
        weights = weight[omitted]
        previous_lower = np.asarray(self.lower)[inputs]
        previous_upper = np.asarray(self.upper)[inputs]

        # Keep the omitted affine sum explicit so its product with z_j is
        # linked to the previous activations by the McCormick envelope.
        affine_weight, affine_bias = weights[None, :], np.zeros(1)
        sum_lower, sum_upper = _affine_box(
            affine_weight, affine_bias, previous_lower, previous_upper,
        )
        omitted_sum = self.variables(sum_lower, sum_upper)
        self.eq.affine(omitted_sum, inputs, affine_weight, affine_bias)

        # Retain the previous interval as the box for the residual variable.
        low, high = _product_box(
            previous_lower, previous_upper, self.lower[current], self.upper[current],
        )
        low, high = _product_box(low, high, weights, weights)
        error = _sum_error(np.maximum(abs(low), abs(high)).sum(), low.size)
        residual, = self.variables(_down(low.sum() - error), _up(high.sum() + error))
        self.add_mccormick(omitted_sum[0], current, residual)
        self.eq.row(0.0, *terms, (residual, -1.0))

    def to_solver(self, device, *, allow_cpu=False):
        def float_tensor(data):
            return torch.as_tensor(data, dtype=torch.float64, device=device)

        indices, values, rhs = [], [], []
        offset = 0
        for group in (self.eq, self.ineq, self.soc):
            if group.rhs:
                coordinates = np.concatenate(group.indices, axis=1)
                coordinates[0] += offset
                indices.append(coordinates)
                values.extend(group.values)
                rhs.extend(group.rhs)
            offset += group.size
        matrix = torch.sparse_coo_tensor(
            torch.as_tensor(np.concatenate(indices, axis=1), dtype=torch.int64, device=device),
            float_tensor(np.concatenate(values)), size=(offset, len(self.lower)),
        ).coalesce()
        cones = Cones(self.eq.size, self.ineq.size, (3,) * (self.soc.size // 3))
        return SparseSOCP(
            matrix, float_tensor(np.concatenate(rhs)), float_tensor(self.lower),
            float_tensor(self.upper), cones, allow_cpu=allow_cpu,
        )


def build_margin_problem(
    affines: Sequence[DenseAffineLayer],
    input_lower: np.ndarray,
    input_upper: np.ndarray,
    hidden_bounds: Sequence[LayerBounds],
    pattern: SOCPPattern,
    true_class: int,
    target_class: int,
    device: torch.device | str,
    *,
    allow_cpu: bool = False,
) -> tuple[SparseSOCP, torch.Tensor]:
    """Return a conic problem and its exact target-minus-true objective."""
    if len(affines) != len(hidden_bounds) + 1:
        raise ValueError("Expected one affine layer per ReLU followed by the affine logits.")
    problem = _MarginProblem()
    activations = [problem.variables(input_lower, input_upper)]
    for layer, bounds in zip(affines[:-1], hidden_bounds):
        pre = problem.variables(bounds.s_lower, bounds.s_upper)
        post = problem.variables(bounds.z_lower, bounds.z_upper)
        problem.eq.affine(pre, activations[-1], layer.weight, layer.bias)
        problem.relu(pre, post, bounds.s_lower, bounds.s_upper)
        activations.append(post)

    # Keep two affine logits so the objective uses exact coefficients +1 and -1.
    final = affines[-1]
    classes = [target_class, true_class]
    weight, bias = final.weight[classes], final.bias[classes]
    last = activations[-1]
    lower, upper = _affine_box(
        weight, bias, np.asarray(problem.lower)[last], np.asarray(problem.upper)[last],
    )
    logits = problem.variables(lower, upper)
    problem.eq.affine(logits, last, weight, bias)

    for pairs in pattern.per_layer:
        k = pairs.layer_index
        layer = affines[k]
        previous, current = activations[k:k + 2]
        selected = {}
        for i, j in pairs.E:
            product = problem.couple(previous[i], current[j])
            selected.setdefault(j, []).append((i, product))
        for j, jp in pairs.S:
            problem.couple(current[j], current[jp])
        for p, q in pairs.T:
            problem.couple(previous[p], previous[q])
        for j, products in selected.items():
            problem.residual_link(previous, current[j], layer.weight[j], layer.bias[j], products)

    solver = problem.to_solver(device, allow_cpu=allow_cpu)
    coefficients = torch.zeros(len(problem.lower), dtype=torch.float64, device=device)
    coefficients[logits[0]] = 1.0
    coefficients[logits[1]] = -1.0
    return solver, coefficients
