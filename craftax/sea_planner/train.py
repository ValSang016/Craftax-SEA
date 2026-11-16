"""Train the goal-conditioned worker with the supervised graph + K-hop planner.

    python -m craftax.sea_planner.train --output runs/planner/K2-leaf-r0/seed-0 --seed 0

Defaults are the experiment setting: symbolic, mortal, vanilla (cumulative)
combat, idle timeout 10000, oracle 22-achievement graph, 300M env steps with a
50M SEA meta-controller warmup.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from craftax.sea.checkpoint import save_params
from craftax.sea.env import COMBAT_MODES, make_sea_craftax_classic_env
from craftax.sea.goal import load_oracle_graph
from craftax.sea.metrics import ACHIEVEMENT_NAMES, summarize_training_metrics
from craftax.sea.ppo_rnn import PPOConfig
from craftax.sea_planner.config import PlannerConfig
from craftax.sea_planner.goals import PlannerGoalRuntime
from craftax.sea_planner.metrics import scores_from_counts, summarize_last_updates
from craftax.sea_planner.ppo import make_train

REPO = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = REPO / "craftax_classic_22.npz"
COUNT_KEYS = (
    "achievement_success_count",
    "episode_count",
    "episode_return_sum",
    "episode_length_sum",
)
RATIO_KEYS = (
    "mean_episode_return", "mean_episode_length", "easy_success_rate",
    "hard_success_rate", "crafter_score", "crafter_score21",
)


def parse_bool(value):
    lowered = str(value).lower()
    if lowered in ("1", "true", "yes", "y"):
        return True
    if lowered in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--graph-path", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--log-interval", type=int, default=10)
    # Environment (symbolic only).
    parser.add_argument("--symbolic", action="store_true", default=True,
                        help="accepted for compatibility; the planner is symbolic-only")
    parser.add_argument("--immortal", type=parse_bool, default=False, metavar="BOOL")
    parser.add_argument("--idle-timeout", type=int, default=10_000)
    parser.add_argument("--combat", choices=COMBAT_MODES, default="org",
                        help="org: vanilla cumulative damage and creature health")
    # Worker PPO.
    parser.add_argument("--goal-timesteps", type=int, default=300_000_000)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.8)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.001)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--reset-ratio", type=int, default=16)
    # Rewards.
    parser.add_argument("--goal-success-coefficient", type=float, default=3.0)
    parser.add_argument("--non-target-achievement-coefficient", type=float, default=1.0)
    parser.add_argument("--repeat-target-coefficient", type=float, default=0.0)
    # High-level controller.
    parser.add_argument("--high-level-controller", choices=("graph-planner",),
                        default="graph-planner")
    parser.add_argument("--dynamic-warmup-steps", type=int, default=50_000_000)
    parser.add_argument("--planner-horizon", type=int, default=2)
    parser.add_argument("--planner-leaf", type=parse_bool, default=True, metavar="BOOL")
    parser.add_argument("--planner-beam-width", type=int, default=4)
    parser.add_argument("--planner-max-decisions", type=int, default=128)
    parser.add_argument("--planner-gamma", type=float, default=0.99)
    parser.add_argument("--planner-temperature", type=float, default=0.1)
    parser.add_argument("--planner-prune", choices=("none", "leaf", "all"), default="none")
    parser.add_argument("--planner-prune-width", type=int, default=13)
    parser.add_argument("--model-hidden-size", type=int, default=256)
    parser.add_argument("--model-latent-size", type=int, default=128)
    parser.add_argument("--model-target-ema", type=float, default=0.995)
    parser.add_argument("--high-level-q-learning-rate", type=float, default=3e-4,
                        help="option-model (F) learning rate (name kept from the spec)")
    parser.add_argument("--high-level-q-replay-ratio", type=float, default=4.0)
    parser.add_argument("--high-level-goal-timeout", type=int, default=300)
    parser.add_argument("--high-level-replay-capacity", type=int, default=100_000)
    parser.add_argument("--high-level-batch-size", type=int, default=256)
    parser.add_argument("--high-level-min-replay-size", type=int, default=1024)
    parser.add_argument("--allow-unready-goals", type=parse_bool, default=True, metavar="BOOL")
    parser.add_argument("--allow-completed-goals", type=parse_bool, default=True, metavar="BOOL")
    # W&B.
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-job-type")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        default="online")
    parser.add_argument("--wandb-tags", nargs="*")
    return parser


def make_configs(args):
    ppo = PPOConfig(
        total_timesteps=args.goal_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coefficient=args.entropy_coefficient,
        value_coefficient=args.value_coefficient,
        max_grad_norm=args.max_grad_norm,
        hidden_size=args.hidden_size,
        reset_ratio=args.reset_ratio,
    )
    planner = PlannerConfig(
        warmup_steps=args.dynamic_warmup_steps,
        replay_capacity=args.high_level_replay_capacity,
        batch_size=args.high_level_batch_size,
        min_replay_size=args.high_level_min_replay_size,
        learning_rate=args.high_level_q_learning_rate,
        replay_ratio=args.high_level_q_replay_ratio,
        goal_timeout=args.high_level_goal_timeout,
        allow_unready_goals=args.allow_unready_goals,
        allow_completed_goals=args.allow_completed_goals,
        planner_horizon=args.planner_horizon,
        planner_leaf=args.planner_leaf,
        planner_beam_width=args.planner_beam_width,
        planner_max_decisions=args.planner_max_decisions,
        planner_gamma=args.planner_gamma,
        planner_temperature=args.planner_temperature,
        planner_prune=args.planner_prune,
        planner_prune_width=args.planner_prune_width,
        model_hidden_size=args.model_hidden_size,
        model_latent_size=args.model_latent_size,
        model_target_ema=args.model_target_ema,
    )
    ppo.validate()
    planner.validate()
    return ppo, planner


def make_env(args_dict):
    return make_sea_craftax_classic_env(
        symbolic=True,
        idle_timeout=args_dict["idle_timeout"],
        mortal=not args_dict["immortal"],
        combat=args_dict["combat"],
    )


def make_runtime(args_dict, graph_path):
    graph, achievement_ids = load_oracle_graph(graph_path)
    return PlannerGoalRuntime(
        graph=jnp.asarray(graph),
        achievement_ids=achievement_ids,
        goal_success_coefficient=args_dict["goal_success_coefficient"],
        non_target_achievement_coefficient=args_dict["non_target_achievement_coefficient"],
        repeat_target_coefficient=args_dict["repeat_target_coefficient"],
        allow_completed_goals=args_dict["allow_completed_goals"],
    )


class WandbLogger:
    """Windowed metrics: Score22/Score21 over episodes ending in the window."""

    def __init__(self, args, config):
        import wandb

        self.wandb = wandb
        self.run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.wandb_name, group=args.wandb_group, job_type=args.wandb_job_type,
            mode=args.wandb_mode, tags=args.wandb_tags, config=config,
            dir=str(Path(args.output).resolve()),
        )
        self.interval = args.log_interval
        self.wandb.define_metric("env_steps")
        self.wandb.define_metric("*", step_metric="env_steps")
        self.window = {}

    def __call__(self, update, metrics):
        update = int(update)
        for name, value in metrics.items():
            if name in RATIO_KEYS or name == "env_steps":
                continue
            value = np.asarray(value, dtype=np.float64)
            self.window[name] = self.window.get(name, 0.0) + value
        self.window["_updates"] = self.window.get("_updates", 0) + 1
        if update % self.interval:
            return
        updates = self.window.pop("_updates")
        counts = {key: self.window.pop(key) for key in COUNT_KEYS}
        record = {"env_steps": int(metrics["env_steps"]), "update": update}
        for name, value in self.window.items():
            if value.ndim == 0:
                record[name] = float(value / updates)
        distribution = self.window.get("selected_goal_distribution")
        if distribution is not None:
            for goal, share in zip(ACHIEVEMENT_NAMES, distribution / updates):
                record[f"selected_goal/{goal}"] = float(share)
        self.window.clear()
        episodes = float(counts["episode_count"])
        record["episodes_in_window"] = episodes
        if episodes > 0:
            scores = scores_from_counts(counts["achievement_success_count"], episodes)
            record["crafter_score22"] = scores["crafter_score22"]
            record["crafter_score21"] = scores["crafter_score21"]
            record["mean_episode_return"] = float(counts["episode_return_sum"] / episodes)
            record["mean_episode_length"] = float(counts["episode_length_sum"] / episodes)
            for name, rate in scores["achievement_success_rate"].items():
                record[f"achievements/{name}"] = 100.0 * rate
        self.wandb.log(record)

    def finish(self, summary):
        for name, value in summary.items():
            if not isinstance(value, dict):
                self.run.summary[f"final/{name}"] = value
        self.wandb.finish()


def progress_reporter(output, config):
    started = time.perf_counter()
    total = config.total_timesteps // (config.num_envs * config.num_steps)
    previous = [None, None]

    def report(update, metrics, checkpoint):
        update = int(update)
        steps = update * config.num_envs * config.num_steps
        now = time.perf_counter()
        scalars = {k: np.asarray(v).tolist() for k, v in metrics.items()}
        if not all(np.isfinite(np.asarray(v, dtype=np.float64)).all() for v in metrics.values()):
            raise FloatingPointError(f"non-finite metrics at update {update}")
        record = {"update": update, "steps": steps, "total_updates": total,
                  "elapsed_seconds": now - started, "last_update": scalars}
        if previous[0] is not None:
            record["steps_per_second"] = (steps - previous[0]) / (now - previous[1])
        previous[:] = [steps, now]
        for name, params in (
            ("goal_policy_latest.msgpack", checkpoint["policy"]),
            ("high_level_q_latest.msgpack", {
                "option_model": checkpoint["option_model"],
                "option_model_target": checkpoint["option_model_target"],
            }),
        ):
            temporary = output / f"{name}.tmp"
            save_params(temporary, params)
            temporary.replace(output / name)
        with (output / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        warmup = "warmup" if float(metrics["warmup_selection_fraction"]) > 0.5 else "planner"
        print(
            f"[planner] update={update}/{total} steps={steps} {warmup} "
            f"loss={float(metrics['loss']):.4f} graph_loss={float(metrics['graph_loss']):.4f} "
            f"return={float(metrics['mean_episode_return']):.2f} "
            f"sps={record.get('steps_per_second', 0):.0f}",
            flush=True,
        )

    return report


def run(args):
    output = Path(args.output)
    if (output / "config.json").exists():
        raise FileExistsError(f"run already exists: {output}")
    ppo_config, planner_config = make_configs(args)
    graph_path = Path(args.graph_path).expanduser().resolve()
    args_dict = vars(args) | {"graph_path": str(graph_path)}
    env = make_env(args_dict)
    runtime = make_runtime(args_dict, graph_path)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(graph_path, output / "oracle_graph.npz")
    config = {
        "method": "graph-planner",
        "args": args_dict,
        "env_name": env.name,
        "goal_ppo": asdict(ppo_config),
        "planner": asdict(planner_config),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    logger = WandbLogger(args, config) if args.wandb_project else None
    summary = {}
    try:
        started = time.perf_counter()
        result = jax.jit(make_train(
            ppo_config, env, runtime, planner_config,
            progress_callback=progress_reporter(output, ppo_config),
            progress_interval=args.log_interval,
            metrics_callback=logger,
        ))(jax.random.PRNGKey(args.seed))
        jax.block_until_ready(result["metrics"]["loss"])
        jax.effects_barrier()
        train_state = result["runner_state"][0]
        planner_state = result["runner_state"][7]
        save_params(output / "goal_policy.msgpack", train_state.params)
        save_params(output / "high_level_q.msgpack", {
            "option_model": planner_state.model.train_state.params,
            "option_model_target": planner_state.model.target_params,
        })
        history = {name: np.asarray(value) for name, value in result["metrics"].items()}
        np.savez_compressed(output / "goal_metrics_history.npz", **history)
        summary = summarize_training_metrics(history)
        last100 = summarize_last_updates(history, 100)
        summary["last100_crafter_score22"] = last100["crafter_score22"]
        summary["last100_crafter_score21"] = last100["crafter_score21"]
        summary["last100"] = last100
        (output / "goal_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"finished in {time.perf_counter() - started:.1f}s: "
              f"last100 Score22 {last100['crafter_score22']:.2f} "
              f"Score21 {last100['crafter_score21']:.2f}", flush=True)
    finally:
        if logger is not None:
            logger.finish(summary)


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
