"""Crafter scores over 22 achievements and without eat_plant (21)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from craftax.sea.metrics import (
    ACHIEVEMENT_NAMES,
    HARD_ACHIEVEMENT_INDICES,
    rollout_episode_metrics,
)

NO_EAT_PLANT_INDICES = tuple(
    index for index, name in enumerate(ACHIEVEMENT_NAMES) if name != "eat_plant"
)
assert len(NO_EAT_PLANT_INDICES) == 21


def crafter_score(rates):
    """exp(mean(log(1 + 100 * rate))) - 1."""
    rates = np.asarray(rates, dtype=np.float64)
    return float(np.expm1(np.log1p(100.0 * rates).mean()))


def scores_from_counts(achievement_counts, episodes):
    rates = np.asarray(achievement_counts, dtype=np.float64) / max(float(episodes), 1.0)
    return {
        "episodes": int(episodes),
        "crafter_score22": crafter_score(rates),
        "crafter_score21": crafter_score(rates[list(NO_EAT_PLANT_INDICES)]),
        "hard4_success_rate": float(rates[list(HARD_ACHIEVEMENT_INDICES)].mean()),
        "achievement_success_rate": dict(zip(ACHIEVEMENT_NAMES, rates.tolist())),
    }


def episode_metrics(transitions):
    """``craftax.sea.metrics.rollout_episode_metrics`` plus Score21."""
    metrics = rollout_episode_metrics(transitions)
    denominator = jnp.maximum(metrics["episode_count"].astype(jnp.float32), 1.0)
    rates = metrics["achievement_success_count"] / denominator
    rates21 = rates[jnp.asarray(NO_EAT_PLANT_INDICES)]
    metrics["crafter_score21"] = jnp.expm1(jnp.log1p(100.0 * rates21).mean())
    return metrics


def summarize_last_updates(history, last_updates=100):
    """Score22/Score21 from episodes ending in the last ``last_updates`` updates."""
    counts = np.asarray(history["achievement_success_count"])[-last_updates:].sum(0)
    episodes = float(np.asarray(history["episode_count"])[-last_updates:].sum())
    summary = scores_from_counts(counts, episodes)
    summary["last_updates"] = int(min(last_updates, len(history["episode_count"])))
    return summary


__all__ = [
    "NO_EAT_PLANT_INDICES",
    "crafter_score",
    "episode_metrics",
    "scores_from_counts",
    "summarize_last_updates",
]
