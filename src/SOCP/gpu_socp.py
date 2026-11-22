"""PyTorch primal-dual backend for bounded sparse SOCP verification (Python 3.10+).

Solve max c @ x + offset subject to lower <= x <= upper and A @ x + b in K.
Rows are ordered as equalities, nonnegative entries, then SOC blocks. A cone
block is ordered (scalar, vector...), meaning scalar >= ||vector||_2.
A must be a coalesced 2-D torch sparse COO tensor. All data must be float64,
on the same device. For CUDA, pass tensors already on the chosen GPU.

Use result.upper_bound to certify a competitor-minus-reference logit margin:
a strictly negative finite bound proves the corresponding real SOCP property.
Bounds apply to the supplied floating-point coefficients interpreted as exact
reals. The caller must construct a sound model relaxation and valid finite
variable bounds, including bounds on lifts. The backend neither builds the
Qwen graph nor accounts for rounding in Qwen inference or in model assembly.
The primal and residual returned by the solver are diagnostics.

Run: python gpu_socp.py           # CUDA smoke test
     python gpu_socp.py --cpu     # same test on CPU
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Cones:
    equalities: int = 0
    nonnegative: int = 0
    soc: tuple[int, ...] = ()


@dataclass(frozen=True)
class Result:
    upper_bound: Tensor
    dual: Tensor
    primal: Tensor
    primal_objective: Tensor
    primal_cone_residual: Tensor


_EPS = torch.finfo(torch.float64).eps
_TINY = torch.finfo(torch.float64).tiny
_MAX_TERMS = 2**40


def _up(x: Tensor) -> Tensor:
    return torch.nextafter(x, torch.full_like(x, torch.inf))


def _down(x: Tensor) -> Tensor:
    return torch.nextafter(x, torch.full_like(x, -torch.inf))


def _roundoff(mass: Tensor, terms: Tensor | int) -> Tensor:
    # Assumes IEEE binary64 operations with gradual underflow. For at most n
    # products and n additions, the conventional gamma_(2n) bound is < 3n*eps.
    # Replacing exact absolute mass with its rounded reduction costs a factor
    # < 2 for n <= 2**40. The additive term covers subnormal rounding too.
    # Round both error calculations outwards.
    count = terms + 1
    return _up(_up((8 * _EPS * count) * mass) + (8 * _TINY * count))


def _sum_up(x: Tensor) -> Tensor:
    return _up(x.sum(dim=-1) + _roundoff(x.abs().sum(dim=-1), x.shape[-1]))


def _norm_up(v: Tensor) -> Tensor:
    """Upper enclosure of the row-wise 2-norm of binary64 vectors."""
    scale = v.abs().amax(dim=-1)
    scaled = _up(v.abs() / torch.where(scale == 0, 1, scale).unsqueeze(-1))
    squared = _sum_up(_up(scaled * scaled))
    root = _up(_up(squared.sqrt()))
    # Independently check that the proposed root is above the exact one.
    root = torch.where(_down(root * root) >= squared, root, squared.clamp_min(1))
    return torch.where(scale == 0, 0, _up(scale * root))


def _project_soc(block: Tensor) -> Tensor:
    scalar, vector = block[:, 0], block[:, 1:]
    # Scale the norm calculation to avoid spurious over/underflow.
    scale = vector.abs().amax(dim=-1)
    norm = scale * torch.linalg.vector_norm(
        vector / torch.where(scale == 0, 1, scale).unsqueeze(-1), dim=-1
    )
    new_scalar = scalar * 0.5 + norm * 0.5
    new_vector = vector * (new_scalar / torch.where(norm == 0, 1, norm)).unsqueeze(-1)
    projected = torch.cat((new_scalar.unsqueeze(-1), new_vector), dim=-1)
    return torch.where(
        (norm <= scalar).unsqueeze(-1),
        block,
        torch.where((norm <= -scalar).unsqueeze(-1), 0, projected),
    )


class SparseSOCP:
    """Reusable sparse matrix and cone layout for objectives on the same graph."""

    @torch.no_grad()
    def __init__(
        self,
        A: Tensor,
        b: Tensor,
        lower: Tensor,
        upper: Tensor,
        cones: Cones,
        *,
        allow_cpu: bool = False,
    ) -> None:
        if A.layout != torch.sparse_coo or A.ndim != 2 or A.sparse_dim() != 2:
            raise ValueError("A must be a two-dimensional sparse COO matrix")
        if not A.is_coalesced():
            raise ValueError("coalesce A and account for any rounding in duplicate entries")
        if A.dtype != torch.float64:
            raise ValueError("A must be float64")
        if A.device.type != "cuda" and not (allow_cpu and A.device.type == "cpu"):
            raise ValueError("use CUDA tensors, or set allow_cpu=True for CPU testing")
        m, n = A.shape
        if n == 0 or max(m, n, A._nnz()) > _MAX_TERMS:
            raise ValueError("unsupported problem dimensions")
        if (
            any(type(k) is not int or k < 0 for k in (cones.equalities, cones.nonnegative))
            or any(type(k) is not int or k < 2 for k in cones.soc)
            or cones.equalities + cones.nonnegative + sum(cones.soc) != m
        ):
            raise ValueError("cone dimensions do not match A")
        self.device, self.m, self.n = A.device, m, n
        self._same_vector(b, m, "b")
        self._same_vector(lower, n, "lower")
        self._same_vector(upper, n, "upper")
        if not bool(torch.isfinite(A.values()).all() & (lower <= upper).all()):
            raise ValueError("finite matrix coefficients and ordered box bounds required")
        self._b, self._lower, self._upper = b.clone(), lower.clone(), upper.clone()
        A = A.clone()
        self._row, self._col = A.indices().unbind(0)
        self._values = A.values()
        self._cone = cones
        self._build_groups()
        # Both CSR orientations are cached. Certificate evaluation uses explicit
        # binary64 multiplication and sums through scatter_add.
        self._A = A.to_sparse_csr() if self._values.numel() else None
        self._AT = A.transpose(0, 1).coalesce().to_sparse_csr() if self._A is not None else None

        zeros = self._values.new_zeros
        row_mass = zeros(m).scatter_add_(0, self._row, self._values.abs())
        col_mass = zeros(n).scatter_add_(0, self._col, self._values.abs())
        self._col_count = zeros(n).scatter_add_(
            0, self._col, torch.ones_like(self._values)
        )
        # Diagonal PDHG preconditioning. All rows of a single SOC get the
        # same dual step, so its Euclidean projection remains valid.
        for group in self._groups:
            row_mass[group] = row_mass[group].amax(dim=-1, keepdim=True).expand_as(group)
        self._sigma = 0.9 / row_mass.clamp_min(1)
        self._tau = 0.9 / col_mass.clamp_min(1)

    def _same_vector(self, tensor: Tensor, length: int, name: str) -> None:
        if (
            tensor.shape != (length,) or tensor.layout != torch.strided
            or tensor.device != self.device or tensor.dtype != torch.float64
        ):
            raise ValueError(f"{name} must be a float64 vector of length {length} on {self.device}")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite")

    def _build_groups(self) -> None:
        base = self._cone.equalities + self._cone.nonnegative
        starts: dict[int, list[int]] = {}
        for width in self._cone.soc:
            starts.setdefault(width, []).append(base)
            base += width
        self._groups = tuple(
            torch.tensor(group, device=self.device, dtype=torch.int64).unsqueeze(-1)
            + torch.arange(width, device=self.device)
            for width, group in starts.items()
        )

    def _matvec(self, matrix: Tensor | None, vector: Tensor, size: int) -> Tensor:
        if matrix is None:
            return vector.new_zeros(size)
        return torch.sparse.mm(matrix, vector.unsqueeze(-1)).squeeze(-1)

    def _project(self, y: Tensor) -> Tensor:
        """Project onto the dual of the product cone."""
        start = self._cone.equalities
        end = start + self._cone.nonnegative
        y[start:end].clamp_min_(0)
        for group in self._groups:
            y[group] = _project_soc(y[group])
        return y

    def _repair(self, y: Tensor) -> Tensor:
        """Enforce dual cone membership by enclosing each spatial 2-norm."""
        y = y.clone()
        start = self._cone.equalities
        y[start:start + self._cone.nonnegative].clamp_min_(0)
        for group in self._groups:
            y[group[:, 0]] = torch.maximum(y[group[:, 0]], _norm_up(y[group[:, 1:]]))
        return y

    def _bound(self, c: Tensor, offset: Tensor, y: Tensor) -> Tensor:
        # For y in K*, c'x + offset <= (c + A'y)'x + b'y + offset.
        # Maximize each interval-enclosed coefficient over its variable's box.
        products = self._values * y[self._row]
        summed = c.new_zeros(self.n).scatter_add_(0, self._col, products)
        mass = c.new_zeros(self.n).scatter_add_(0, self._col, products.abs())
        error = _roundoff(mass, self._col_count)
        lo = _down(c + _down(summed - error))
        hi = _up(c + _up(summed + error))
        support = torch.maximum(
            torch.maximum(_up(lo * self._lower), _up(lo * self._upper)),
            torch.maximum(_up(hi * self._lower), _up(hi * self._upper)),
        )
        # _sum_up includes error for both forming and accumulating b*y.
        bound = _up(_up(_sum_up(support) + _sum_up(self._b * y)) + offset)
        valid = torch.isfinite(bound) & torch.isfinite(y).all()
        return torch.where(valid, bound, torch.inf)

    def _keep_best(
        self, c: Tensor, offset: Tensor, y: Tensor, best_bound: Tensor, best_dual: Tensor,
    ) -> tuple[Tensor, Tensor]:
        candidate = self._repair(y)
        bound = self._bound(c, offset, candidate)
        better = bound < best_bound
        return torch.where(better, bound, best_bound), torch.where(better, candidate, best_dual)

    @torch.no_grad()
    def solve(
        self,
        c: Tensor,
        *,
        offset: float = 0.0,
        iterations: int = 2000,
        check_every: int = 25,
        initial_dual: Tensor | None = None,
    ) -> Result:
        """Run a fixed iteration budget with no per-iteration device sync.

        Bounds are checked for the last and mean dual iterates periodically.
        A failed numerical candidate is discarded. The initial box bound is
        always available if finite. There is no infeasibility/unboundedness
        status, and early termination requires a caller's own stopping policy.
        """
        self._same_vector(c, self.n, "c")
        if type(iterations) is not int or iterations < 0:
            raise ValueError("iterations must be a nonnegative integer")
        if type(check_every) is not int or check_every < 1:
            raise ValueError("check_every must be a positive integer")
        offset_tensor = c.new_tensor(offset)
        if offset_tensor.ndim != 0 or not bool(torch.isfinite(offset_tensor)):
            raise ValueError("offset must be a finite scalar")
        x = 0.5 * self._lower + 0.5 * self._upper
        x_bar = x
        y = c.new_zeros(self.m)
        best_dual = y
        best_bound = self._bound(c, offset_tensor, best_dual)
        if initial_dual is not None:
            self._same_vector(initial_dual, self.m, "initial_dual")
            y = self._project(initial_dual.clone())
            best_bound, best_dual = self._keep_best(c, offset_tensor, y, best_bound, best_dual)
        mean_dual = y

        for step in range(1, iterations + 1):
            y = self._project(y - self._sigma * (self._matvec(self._A, x_bar, self.m) + self._b))
            x_next = (x + self._tau * (c + self._matvec(self._AT, y, self.n))).clamp(
                self._lower, self._upper
            )
            x_bar = x_next + (x_next - x)
            x = x_next
            mean_dual = mean_dual + (y - mean_dual) / step
            if step % check_every == 0 or step == iterations:
                for iterate in (y, mean_dual):
                    best_bound, best_dual = self._keep_best(
                        c, offset_tensor, iterate, best_bound, best_dual
                    )

        slack = self._matvec(self._A, x, self.m) + self._b
        # Moreau decomposition: distance to K is ||Pi_{K*}(-slack)||.
        residual = torch.linalg.vector_norm(self._project(-slack))
        return Result(best_bound, best_dual, x, c.dot(x) + offset_tensor, residual)


def _sparse_rows(rows: list[tuple[tuple[int, float], ...]], n: int, device: str) -> Tensor:
    indices = [(i, j) for i, row in enumerate(rows) for j, _ in row]
    values = [value for row in rows for _, value in row]
    return torch.sparse_coo_tensor(
        torch.tensor(indices, device=device, dtype=torch.int64).reshape(-1, 2).T,
        torch.tensor(values, device=device, dtype=torch.float64),
        (len(rows), n),
        check_invariants=True,
    ).coalesce()


def _smoke_test(device: str) -> None:
    def vector(values: list[float]) -> Tensor:
        return torch.tensor(values, device=device, dtype=torch.float64)

    # q,k in [-1,1], w approximates q*k. McCormick alone admits w=1
    # under the diagonal energy budget dq+dk<=1. Adding the local cones
    # implies |w|<=sqrt(dq*dk)<=0.5, attained by a real q,k product.
    rows = [
        ((0, 1), (1, 1), (2, 1)),
        ((0, -1), (1, -1), (2, 1)),
        ((0, 1), (1, -1), (2, -1)),
        ((0, -1), (1, 1), (2, -1)),
        ((3, -1), (4, -1)),
        ((3, 1),), ((0, 2),), ((3, 1),),
        ((4, 1),), ((1, 2),), ((4, 1),),
        ((3, 1), (4, 1)), ((2, 2),), ((3, 1), (4, -1)),
    ]
    constants = [1, 1, 1, 1, 1, 1, 0, -1, 1, 0, -1, 0, 0, 0]
    lower, upper = vector([-1, -1, -1, 0, 0]), vector([1, 1, 1, 1, 1])
    c = vector([0, 0, 1, 0, 0])
    lp = SparseSOCP(
        _sparse_rows(rows[:5], 5, device), vector(constants[:5]), lower, upper,
        Cones(nonnegative=5), allow_cpu=device == "cpu",
    ).solve(c, iterations=250)
    problem = SparseSOCP(
        _sparse_rows(rows, 5, device), vector(constants), lower, upper,
        Cones(nonnegative=5, soc=(3, 3, 3)), allow_cpu=device == "cpu",
    )
    socp = problem.solve(c, iterations=750)
    margin = problem.solve(c, offset=-0.75, iterations=750, initial_dual=socp.dual)
    assert 1 <= lp.upper_bound.item() < 1.00001
    assert 0.5 <= socp.upper_bound.item() < 0.50001
    assert -0.25 <= margin.upper_bound.item() < -0.24999
    print(f"{device}: LP upper bound {lp.upper_bound.item():.10f}")
    print(f"{device}: SOCP upper bound {socp.upper_bound.item():.10f}")
    print(f"{device}: shifted margin upper bound {margin.upper_bound.item():.10f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the sparse SOCP smoke test")
    parser.add_argument("--cpu", action="store_true", help="use CPU for the test")
    args = parser.parse_args()
    if not args.cpu and not torch.cuda.is_available():
        parser.exit(1, "CUDA unavailable. Use CUDA-enabled PyTorch or run the CPU test with --cpu.\n")
    _smoke_test("cpu" if args.cpu else "cuda")
