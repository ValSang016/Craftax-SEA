"""Configuration of the supervised option model and the K-hop planner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlannerConfig:
    # Schedule: the existing SEA meta-controller picks goals for the first
    # ``warmup_steps`` env steps (the option model still learns), then only
    # the planner selects goals.
    warmup_steps: int = 50_000_000
    # High-level replay of options and the supervised option-model update.
    replay_capacity: int = 100_000
    batch_size: int = 256
    min_replay_size: int = 1024
    learning_rate: float = 3e-4
    replay_ratio: float = 4.0
    goal_timeout: int = 300
    # Goal eligibility after warmup.
    allow_unready_goals: bool = True
    allow_completed_goals: bool = True
    # Planner.
    planner_horizon: int = 2
    planner_leaf: bool = True
    planner_beam_width: int = 4
    planner_max_decisions: int = 128
    planner_gamma: float = 0.99
    planner_temperature: float = 0.1
    # With this probability a newly chosen goal is replaced by a uniform draw
    # from the candidate goals (exploration on top of the planner softmax).
    planner_epsilon: float = 0.0
    planner_prune: str = "none"
    planner_prune_width: int = 13
    # Option model F.
    model_hidden_size: int = 256
    model_latent_size: int = 128
    model_target_ema: float = 0.995

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
        if not 0 <= self.planner_epsilon <= 1:
            raise ValueError("planner epsilon must be in [0, 1]")
        if self.planner_prune not in ("none", "leaf", "all"):
            raise ValueError("planner prune must be none, leaf, or all")
        if self.planner_prune_width < 1:
            raise ValueError("planner prune width must be positive")
        if self.model_hidden_size < 1 or self.model_latent_size < 1:
            raise ValueError("model sizes must be positive")
        if not 0 <= self.model_target_ema < 1:
            raise ValueError("model target EMA must be in [0, 1)")


__all__ = ["PlannerConfig"]
