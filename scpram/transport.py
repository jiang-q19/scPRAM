"""Checked, bounded optimal transport solvers used by scPRAM."""

from dataclasses import asdict, dataclass
import warnings

import numpy as np
import ot


@dataclass(frozen=True)
class OTConfig:
    """OT policy. Sinkhorn regularization is relative to mean transport cost.

    Exact EMD remains the default. No solver or regularization is changed
    silently. ``max_cost_entries`` bounds matrix size, not total process RAM.
    """

    method: str = "emd"
    max_iter: int | None = None
    retries: int = 1
    retry_multiplier: int = 5
    reg: float = 0.05
    tolerance: float = 1e-7
    max_cost_entries: int = 25_000_000

    def __post_init__(self):
        if self.max_iter is None:
            object.__setattr__(self, "max_iter", 1_000_000 if self.method == "emd" else 10_000)
        if self.method not in {"emd", "sinkhorn"}:
            raise ValueError("OT method must be 'emd' or 'sinkhorn'.")
        for name in ("max_iter", "retry_multiplier", "max_cost_entries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(self.retries, bool) or not isinstance(self.retries, int) or self.retries < 0:
            raise ValueError("retries must be a nonnegative integer.")
        for name in ("reg", "tolerance"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")


class OTConvergenceError(RuntimeError):
    """A transport plan failed solver-status or marginal checks."""

    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        last = diagnostics["attempts"][-1]
        super().__init__(
            f"{diagnostics['method']} did not converge after {len(diagnostics['attempts'])} "
            f"attempt(s); last iteration budget={last['max_iter']}, "
            f"marginal error={last['marginal_error']:.3g}. "
            "Increase OTConfig.max_iter; for Sinkhorn consider a larger reg. "
            "Switching from EMD to Sinkhorn requires an explicit method choice. "
            "The failed plan was not used."
        )


def solve_transport(source, target, config=None):
    """Return a checked float64 plan and serializable diagnostics.

    Uses Euclidean cost and uniform cell masses, matching the paper's EMD
    formulation. Entropic Sinkhorn is an explicit alternative, not equivalent
    to exact OT. Retry budgets are bounded and preserve the chosen objective.
    """
    config = OTConfig() if config is None else config
    if not isinstance(config, OTConfig):
        raise TypeError("config must be an OTConfig.")
    source, target = (np.asarray(x, dtype=np.float64) for x in (source, target))
    if any(x.ndim != 2 or not all(x.shape) for x in (source, target)):
        raise ValueError("OT inputs must be nonempty two-dimensional matrices.")
    if source.shape[1] != target.shape[1]:
        raise ValueError("OT inputs must have the same feature dimension.")
    if not all(np.isfinite(x).all() for x in (source, target)):
        raise ValueError("OT inputs contain NaN or infinity.")
    entries = len(source) * len(target)
    if entries > config.max_cost_entries:
        raise MemoryError(
            f"OT needs {entries:,} cost entries (cost + plan alone: "
            f"{entries * 16 / 2**20:.1f} MiB, solver workspace is additional). "
            "Explicitly raise max_cost_entries or reduce the reference cells."
        )
    cost = np.ascontiguousarray(ot.dist(source, target, metric="euclidean"))
    if not np.isfinite(cost).all():
        raise ValueError("OT distances overflowed; rescale latent coordinates.")
    scale = float(cost.mean())
    scale = scale if scale > 0 else 1.0
    cost /= scale
    a = np.full(len(source), 1.0 / len(source))
    b = np.full(len(target), 1.0 / len(target))
    diagnostics = {
        "method": config.method,
        "config": asdict(config),
        "cost_scale": scale,
        "cost_entries": entries,
        "attempts": [],
    }
    # POT checks the L2 target marginal residual; our public tolerance is L1.
    sinkhorn_tolerance = config.tolerance / (2 * np.sqrt(len(target)))
    for attempt in range(config.retries + 1):
        budget = config.max_iter * config.retry_multiplier**attempt
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if config.method == "emd":
                plan, log = ot.emd(a, b, cost, numItermax=budget, log=True)
                solver_ok = log.get("result_code") == 1 and not log.get("warning")
            else:
                plan, log = ot.bregman.sinkhorn_log(
                    a,
                    b,
                    cost,
                    config.reg,
                    numItermax=budget,
                    stopThr=sinkhorn_tolerance,
                    log=True,
                    warn=True,
                )
                solver_ok = bool(log.get("err")) and log["err"][-1] <= sinkhorn_tolerance
        finite = np.isfinite(plan).all() and (plan >= 0).all()
        error = (
            float(max(np.abs(plan.sum(axis=1) - a).sum(), np.abs(plan.sum(axis=0) - b).sum()))
            if finite
            else float("inf")
        )
        converged = bool(solver_ok and finite and error <= config.tolerance)
        diagnostics["attempts"].append(
            {
                "max_iter": budget,
                "marginal_error": error,
                "converged": converged,
                "warnings": [str(w.message) for w in caught],
            }
        )
        if converged:
            diagnostics["converged"] = True
            return plan, diagnostics
    diagnostics["converged"] = False
    raise OTConvergenceError(diagnostics)
