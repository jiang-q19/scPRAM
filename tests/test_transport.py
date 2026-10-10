from unittest.mock import patch

import numpy as np
import pytest

from scpram import OTConfig, OTConvergenceError
from scpram.transport import solve_transport


@pytest.mark.parametrize("method", ["emd", "sinkhorn"])
def test_unequal_populations_preserve_masses(method):
    source = np.array([[0.0, 0.0], [1.0, 1.0]])
    target = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 2.0]])
    plan, info = solve_transport(source, target, OTConfig(method=method))
    np.testing.assert_allclose(plan.sum(1), [0.5, 0.5], atol=1e-7)
    np.testing.assert_allclose(plan.sum(0), np.full(3, 1 / 3), atol=1e-7)
    assert info["converged"] and np.isfinite(plan).all()


def test_exact_plan_matches_known_optimum():
    plan, _ = solve_transport([[0.0], [2.0]], [[0.0], [2.0]])
    np.testing.assert_allclose(plan, np.eye(2) / 2)


@pytest.mark.parametrize("method", ["emd", "sinkhorn"])
def test_zero_cost(method):
    plan, _ = solve_transport(np.zeros((3, 2)), np.zeros((4, 2)), OTConfig(method=method))
    assert np.isfinite(plan).all()
    np.testing.assert_allclose(plan.sum(1), np.full(3, 1 / 3))


def test_status_failure_never_consumes_feasible_plan():
    feasible = np.eye(2) / 2
    failure = (feasible, {"result_code": 3, "warning": "iteration limit"})
    success = (feasible, {"result_code": 1, "warning": None})
    with patch("scpram.transport.ot.emd", side_effect=[failure, success]) as solver:
        _, info = solve_transport([[0.0], [1.0]], [[0.0], [1.0]], OTConfig(max_iter=1))
        assert [c.kwargs["numItermax"] for c in solver.call_args_list] == [1, 5]
        assert len(info["attempts"]) == 2
    with patch("scpram.transport.ot.emd", return_value=failure):
        with pytest.raises(OTConvergenceError) as exc:
            solve_transport([[0.0], [1.0]], [[0.0], [1.0]], OTConfig(max_iter=1))
        assert not exc.value.diagnostics["converged"]


def test_false_solver_success_with_invalid_marginals_is_rejected():
    with patch("scpram.transport.ot.emd", return_value=(np.zeros((2, 2)), {"result_code": 1})):
        with pytest.raises(OTConvergenceError):
            solve_transport([[0.0], [1.0]], [[0.0], [1.0]], OTConfig(retries=0))


@pytest.mark.parametrize("method", ["emd", "sinkhorn"])
def test_real_iteration_limit_raises(method):
    rng = np.random.default_rng(9)
    with pytest.raises(OTConvergenceError):
        solve_transport(
            rng.normal(size=(40, 3)),
            rng.normal(size=(50, 3)),
            OTConfig(method=method, max_iter=1, retries=0),
        )


def test_allocation_guard_runs_before_distance_computation():
    with patch("scpram.transport.ot.dist") as distance:
        with pytest.raises(MemoryError):
            solve_transport(np.ones((3, 2)), np.ones((3, 2)), OTConfig(max_cost_entries=8))
        distance.assert_not_called()


@pytest.mark.parametrize("bad", [[], [[float("nan")]], [[float("inf")]]])
def test_invalid_inputs(bad):
    with pytest.raises(ValueError):
        solve_transport(bad, [[1.0]])


@pytest.mark.parametrize(
    "options",
    [{"method": "auto"}, {"reg": 0}, {"max_iter": 0}, {"retries": -1}, {"tolerance": float("nan")}],
)
def test_config_validation(options):
    with pytest.raises(ValueError):
        OTConfig(**options)
