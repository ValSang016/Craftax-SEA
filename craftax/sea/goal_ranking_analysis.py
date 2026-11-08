"""Tie-aware diagnostics for frozen goal-controller evaluations."""

import numpy as np


def decision_metrics(static, residual, candidates, beta, temperature=1.0):
    """Compare candidate rankings without interpreting an arbitrary tie as a flip.

    Inputs have shape (decisions, goals). Argmax follows NumPy's first-index
    tie rule; ``strict_top_flip`` instead checks membership of the static top set.
    """
    static = np.asarray(static, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=bool)
    if static.shape != residual.shape or static.shape != candidates.shape:
        raise ValueError("static, residual and candidate shapes must match")
    if static.ndim != 2 or not candidates.any(axis=1).all():
        raise ValueError("each decision must have at least one candidate")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not np.isfinite(static[candidates]).all() or not np.isfinite(residual[candidates]).all():
        raise ValueError("candidate scores must be finite")

    augmented = static + beta * residual
    masked_static = np.where(candidates, static, -np.inf)
    masked_augmented = np.where(candidates, augmented, -np.inf)
    top_static = masked_static.argmax(axis=1)
    top_augmented = masked_augmented.argmax(axis=1)
    best_static = masked_static.max(axis=1)
    worst_static = np.where(candidates, static, np.inf).min(axis=1)
    rows = np.arange(len(static))

    def softmax(scores):
        logits = scores / temperature
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        return weights / weights.sum(axis=1, keepdims=True)

    p_static = softmax(masked_static)
    p_augmented = softmax(masked_augmented)
    count = candidates.sum(axis=1)
    uniform = candidates / count[:, None]
    strict_flip = static[rows, top_augmented] < best_static - 1e-7
    changed = top_static != top_augmented
    valid_pair = candidates[:, :, None] & candidates[:, None, :]
    ordered_pair = valid_pair & (static[:, :, None] > static[:, None, :])
    reversed_pair = ordered_pair & (augmented[:, :, None] < augmented[:, None, :])
    return {
        "candidate_count": count,
        "static_argmax": top_static,
        "augmented_argmax": top_augmented,
        "static_top_tie_count": (candidates & (static == best_static[:, None])).sum(axis=1),
        "all_static_tied": best_static == worst_static,
        "argmax_changed": changed,
        "strict_top_flip": strict_flip,
        "tie_only_argmax_change": changed & ~strict_flip,
        "ordered_pairs": ordered_pair.sum(axis=(1, 2)),
        "reversed_pairs": reversed_pair.sum(axis=(1, 2)),
        "probability_tv": 0.5 * np.abs(p_static - p_augmented).sum(axis=1),
        "static_uniform_tv": 0.5 * np.abs(p_static - uniform).sum(axis=1),
        "residual_range": np.where(candidates, residual, -np.inf).max(axis=1)
        - np.where(candidates, residual, np.inf).min(axis=1),
        "static_probabilities": p_static,
        "augmented_probabilities": p_augmented,
    }


def sequence_metrics(left, right):
    """Compare chronological goal IDs; repeated goals remain in the sequence."""
    prefix = 0
    for a, b in zip(left, right):
        if a != b:
            break
        prefix += 1
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return {
        "identical": list(left) == list(right),
        "common_prefix_length": prefix,
        "first_different_selection": None if list(left) == list(right) else prefix,
        "edit_distance": previous[-1],
        "normalized_edit_distance": previous[-1] / max(len(left), len(right), 1),
    }
