"""Batched attention with an explicit historical compatibility mode."""

import warnings

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def attention_delta(
    query,
    reference,
    delta,
    n_top=None,
    *,
    mode="cellwise",
    weighting="signed",
    temperature=1.0,
    batch_size=1024,
):
    """Transfer reference deltas using each query's own neighbors.

    ``signed`` retains historical cosine/sum weights for top-k attention.
    ``softmax`` is an opt-in alternative. A zero signed denominator uses
    uniform weights with a warning. With n_top=None, historical L1 weights
    are retained (zero similarity yields zero delta).
    """
    query, reference, delta = (np.asarray(x) for x in (query, reference, delta))
    if any(x.ndim != 2 or not all(x.shape) for x in (query, reference, delta)):
        raise ValueError("Attention inputs must be nonempty matrices.")
    if reference.shape != delta.shape or query.shape[1] != reference.shape[1]:
        raise ValueError("Attention input shapes do not agree.")
    if not all(np.isfinite(x).all() for x in (query, reference, delta)):
        raise ValueError("Attention inputs contain NaN or infinity.")
    if mode not in {"cellwise", "legacy_first"}:
        raise ValueError("attention mode must be 'cellwise' or 'legacy_first'.")
    if weighting not in {"signed", "softmax"}:
        raise ValueError("weighting must be 'signed' or 'softmax'.")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive.")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if n_top is not None and (
        not isinstance(n_top, (int, np.integer)) or not 1 <= n_top <= len(reference)
    ):
        raise ValueError("n_top must be between 1 and the number of reference cells.")
    if mode == "legacy_first":
        warnings.warn(
            "legacy_first uses the first target cell's attention for all targets.",
            UserWarning,
            stacklevel=2,
        )
    inputs = query[:1] if mode == "legacy_first" else query
    result = []
    for start in range(0, len(inputs), batch_size):
        similarity = cosine_similarity(inputs[start : start + batch_size], reference)
        if n_top is None:
            values = similarity
            selected_delta = delta
        else:
            # Stable sorting makes ties reproducible for a fixed reference order.
            indices = np.argsort(similarity, axis=1, kind="stable")[:, -n_top:]
            values = np.take_along_axis(similarity, indices, axis=1)
            selected_delta = delta[indices]
        if weighting == "softmax":
            weights = np.exp((values - values.max(axis=1, keepdims=True)) / temperature)
            weights /= weights.sum(axis=1, keepdims=True)
        elif n_top is None:
            denominator = np.abs(values).sum(axis=1, keepdims=True)
            weights = values / np.where(denominator > 0, denominator, 1)
        else:
            denominator = values.sum(axis=1, keepdims=True)
            bad = np.abs(denominator[:, 0]) <= 1e-8 * np.maximum(1, np.abs(values).sum(axis=1))
            weights = values / np.where(bad[:, None], 1, denominator)
            if bad.any():
                warnings.warn(
                    "Cosine weights cancel or are zero; using uniform top-k weights.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                weights[bad] = 1.0 / n_top
        result.append(
            weights @ selected_delta
            if n_top is None
            else np.einsum("ij,ijk->ik", weights, selected_delta)
        )
    result = np.concatenate(result)
    if mode == "legacy_first":
        result = np.repeat(result, len(query), axis=0)
    return result
