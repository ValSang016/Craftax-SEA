"""Episode-level Craftax metrics for SEA training runs."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from craftax.craftax_classic.constants import Achievement

ACHIEVEMENT_NAMES = tuple(achievement.name.lower() for achievement in Achievement)
HARD_ACHIEVEMENT_NAMES = (
    "defeat_zombie",
    "collect_diamond",
    "make_iron_pickaxe",
    "make_iron_sword",
)
HARD_ACHIEVEMENT_INDICES = tuple(
    Achievement[name.upper()].value for name in HARD_ACHIEVEMENT_NAMES
)
EASY_ACHIEVEMENT_INDICES = tuple(
    index
    for index in range(len(ACHIEVEMENT_NAMES))
    if index not in HARD_ACHIEVEMENT_INDICES
)
EASY17_ACHIEVEMENT_INDICES = tuple(
    index
    for index, name in enumerate(ACHIEVEMENT_NAMES)
    if index not in HARD_ACHIEVEMENT_INDICES and name != "eat_plant"
)


def rollout_episode_metrics(transitions):
    """Return additive statistics for episodes ending in one PPO rollout.

    Keeping counts and sums (instead of only per-update averages) makes the
    final result exact even when PPO updates contain different episode counts.
    """

    ended = transitions.env_done
    episode_count = ended.astype(jnp.int32).sum()
    terminal_achievements = (
        transitions.achievements.astype(jnp.int32) * ended[..., None]
    )
    achievement_success_count = terminal_achievements.sum(axis=(0, 1))
    episode_return_sum = achievement_success_count.sum()
    episode_length_sum = (
        transitions.episode_length.astype(jnp.float32) * ended
    ).sum()
    denominator = jnp.maximum(episode_count.astype(jnp.float32), 1.0)
    rates = achievement_success_count / denominator
    crafter_score = jnp.exp(jnp.log1p(100.0 * rates).mean()) - 1.0
    easy_rate = rates[jnp.asarray(EASY_ACHIEVEMENT_INDICES)].mean()
    hard_rate = rates[jnp.asarray(HARD_ACHIEVEMENT_INDICES)].mean()
    return {
        "episode_count": episode_count,
        "episode_return_sum": episode_return_sum,
        "episode_length_sum": episode_length_sum,
        "mean_episode_return": episode_return_sum / denominator,
        "mean_episode_length": episode_length_sum / denominator,
        "achievement_success_count": achievement_success_count,
        "easy_success_rate": easy_rate,
        "hard_success_rate": hard_rate,
        "crafter_score": crafter_score,
    }


def summarize_training_metrics(metrics):
    """Convert scanned JAX metrics into an exact JSON-serializable summary."""

    arrays = {name: np.asarray(value) for name, value in metrics.items()}
    summary = {}
    for name, value in arrays.items():
        if name in {
            "achievement_success_count",
            "episode_count",
            "episode_return_sum",
            "episode_length_sum",
        }:
            continue
        if value.ndim == 1:
            summary[name] = float(value[-1])

    episode_count = float(arrays["episode_count"].sum())
    achievement_counts = arrays["achievement_success_count"].sum(axis=0)
    denominator = max(episode_count, 1.0)
    rates = achievement_counts / denominator
    summary.update(
        {
            "episode_count": int(episode_count),
            "mean_episode_return": float(
                arrays["episode_return_sum"].sum() / denominator
            ),
            "mean_episode_length": float(
                arrays["episode_length_sum"].sum() / denominator
            ),
            "easy_success_rate": float(rates[list(EASY_ACHIEVEMENT_INDICES)].mean()),
            "hard_success_rate": float(rates[list(HARD_ACHIEVEMENT_INDICES)].mean()),
            "crafter_score": float(np.exp(np.log1p(100.0 * rates).mean()) - 1.0),
            "achievement_success_rate": {
                name: float(rate) for name, rate in zip(ACHIEVEMENT_NAMES, rates)
            },
            "achievement_success_count": {
                name: int(count)
                for name, count in zip(ACHIEVEMENT_NAMES, achievement_counts)
            },
        }
    )
    return summary


__all__ = [
    "ACHIEVEMENT_NAMES",
    "EASY_ACHIEVEMENT_INDICES",
    "EASY17_ACHIEVEMENT_INDICES",
    "HARD_ACHIEVEMENT_INDICES",
    "rollout_episode_metrics",
    "summarize_training_metrics",
]
