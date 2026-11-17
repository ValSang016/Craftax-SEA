"""Configuration of the supervised option model and the K-hop planner."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class PlannerConfig:
    # Schedule: the existing SEA meta-controller picks goals for the first
    # ``warmup_steps`` env steps (the option model still learns), then only
    # the planner selects goals.
    warmup_steps: int = 0
    # High-level replay of options and the supervised option-model update.
    replay_capacity: int = 100_000
    batch_size: int = 256
    min_replay_size: int = 1024
    learning_rate: float = 3e-4
    replay_ratio: float = 4.0
    goal_timeout: int = 300
    # Goal eligibility after warmup.
    allow_unready_goals: bool = True
    # Reselection of completed goals: True, False or "half" (eat_cow,
    # collect_drink and eat_plant only).
    allow_completed_goals: bool | str = True
    # Planner.
    planner_horizon: int = 2
    planner_leaf: bool = True
    planner_beam_width: int = 4
    planner_max_decisions: int = 128
    planner_gamma: float = 0.99
    planner_temperature: float = 0.1
    # With probability epsilon a newly chosen goal is replaced by a uniform
    # draw from the candidate goals.  Epsilon decays linearly from
    # ``planner_epsilon_start`` to ``planner_epsilon`` over
    # ``planner_epsilon_decay_steps`` env steps after warmup, then stays.
    planner_epsilon: float = 0.2
    planner_epsilon_start: float = 1.0
    planner_epsilon_decay_steps: int = 50_000_000
    planner_prune: str = "none"
    planner_prune_width: int = 13
    # Option model F.
    model_hidden_size: int = 256
    model_latent_size: int = 128
    # Aux decoder latent -> inventory+intrinsics: 0 = linear (default),
    # 256 = the original MLP decoder (runs before 2026-11-17).
    model_decoder_hidden_size: int = 0
    # Completed vector after a hop: "learned" head of F, or the "rule"
    # c' = c | onehot(goal) (no completed head training).
    completed_prediction: str = "learned"
    model_target_ema: float = 0.995

    def epsilon_at(self, env_steps):
        """Exploration probability after ``env_steps`` total env steps."""
        since = jnp.maximum(jnp.asarray(env_steps, jnp.float32) - self.warmup_steps, 0.0)
        frac = jnp.clip(since / max(self.planner_epsilon_decay_steps, 1), 0.0, 1.0)
        if self.planner_epsilon_decay_steps == 0:
            frac = jnp.ones_like(frac)
        return self.planner_epsilon_start + frac * (self.planner_epsilon - self.planner_epsilon_start)

    @property
    def explores(self):
        return self.planner_epsilon > 0 or (
            self.planner_epsilon_start > 0 and self.planner_epsilon_decay_steps > 0
        )

    def validate(self):
        if self.warmup_steps < 0:
            raise ValueError("warmup steps cannot be negative")
        if self.replay_capacity < 1 or self.batch_size < 1:
            raise ValueError("replay capacity and batch size must be positive")
        if not self.batch_size <= self.min_replay_size <= self.replay_capacity:
            raise ValueError("min replay size must be between batch size and capacity")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("option-model learning rate must be positive")
        if not np.isfinite(self.replay_ratio) or self.replay_ratio < 0:
            raise ValueError("replay ratio must be non-negative")
        from craftax.sea_planner.goals import normalize_reselection

        normalize_reselection(self.allow_completed_goals)
        if self.goal_timeout < 1:
            raise ValueError("goal timeout must be positive")
        if self.planner_horizon < 1 or self.planner_beam_width < 1:
            raise ValueError("planner horizon and beam width must be positive")
        if self.planner_max_decisions < 1:
            raise ValueError("planner max decisions must be positive")
        if not 0 < self.planner_gamma <= 1:
            raise ValueError("planner gamma must be in (0, 1]")
        if not np.isfinite(self.planner_temperature) or self.planner_temperature <= 0:
            raise ValueError("planner temperature must be positive")
        if not (0 <= self.planner_epsilon <= 1 and 0 <= self.planner_epsilon_start <= 1):
            raise ValueError("planner epsilon (start and final) must be in [0, 1]")
        if self.planner_epsilon_decay_steps < 0:
            raise ValueError("planner epsilon decay steps cannot be negative")
        if self.planner_prune not in ("none", "leaf", "all"):
            raise ValueError("planner prune must be none, leaf, or all")
        if self.planner_prune_width < 1:
            raise ValueError("planner prune width must be positive")
        if self.completed_prediction not in ("learned", "rule"):
            raise ValueError("completed prediction must be learned or rule")
        if self.model_decoder_hidden_size < 0:
            raise ValueError("decoder hidden size cannot be negative")
        if self.model_hidden_size < 1 or self.model_latent_size < 1:
            raise ValueError("model sizes must be positive")
        if not 0 <= self.model_target_ema < 1:
            raise ValueError("model target EMA must be in [0, 1)")


__all__ = ["PlannerConfig"]
