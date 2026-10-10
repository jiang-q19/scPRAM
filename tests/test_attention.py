import numpy as np
import pytest

from scpram.attention import attention_delta


def test_cellwise_neighbors_and_permutation_equivariance():
    query = reference = np.eye(2)
    delta = np.array([[1.0, 3.0], [8.0, 4.0]])
    actual = attention_delta(query, reference, delta, 1, batch_size=1)
    np.testing.assert_allclose(actual, delta)
    np.testing.assert_allclose(attention_delta(query[::-1], reference, delta, 1), actual[::-1])
    with pytest.warns(UserWarning, match="first target"):
        historical = attention_delta(query, reference, delta, 1, mode="legacy_first")
    np.testing.assert_allclose(historical, np.repeat(delta[:1], 2, axis=0))


def test_legacy_formula_for_nonzero_weights():
    query = np.array([[1.0, 1.0], [-1.0, 0.0]])
    reference = np.eye(2)
    delta = np.array([[1.0, 3.0], [8.0, 4.0]])
    with pytest.warns(UserWarning):
        result = attention_delta(query, reference, delta, 2, mode="legacy_first")
    np.testing.assert_allclose(result, np.repeat(delta.mean(0)[None], 2, axis=0))


def test_zero_similarity_and_signed_cancellation():
    with pytest.warns(RuntimeWarning, match="uniform"):
        result = attention_delta([[0.0, 0.0]], np.eye(2), [[1.0, 3.0], [3.0, 1.0]], 2)
    np.testing.assert_allclose(result, [[2.0, 2.0]])
    with pytest.warns(RuntimeWarning, match="uniform"):
        result = attention_delta(
            [[1.0, 0.0]], [[1.0, 0.0], [-1.0, 0.0]], [[1.0, 3.0], [3.0, 1.0]], 2
        )
    np.testing.assert_allclose(result, [[2.0, 2.0]])


def test_softmax_is_finite_and_batch_invariant():
    rng = np.random.default_rng(4)
    q, r, d = (rng.normal(size=(n, 3)) for n in (7, 10, 10))
    a = attention_delta(q, r, d, 4, weighting="softmax", batch_size=1)
    b = attention_delta(q, r, d, 4, weighting="softmax", batch_size=7)
    np.testing.assert_allclose(a, b)


def test_all_reference_l1_behavior():
    actual = attention_delta([[1.0, 0.0]], [[1.0, 0.0], [-1.0, 0.0]], [[4.0, 0.0], [2.0, 0.0]])
    np.testing.assert_allclose(actual, [[1.0, 0.0]])
