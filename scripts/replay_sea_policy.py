#!/usr/bin/env python3
"""Replay a trained SEA/PPO-RNN policy with video and death diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Playback is lightweight and should not steal memory from training by default.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from PIL import Image, ImageDraw, ImageFont

from craftax.craftax_classic.constants import Action, Achievement
from craftax.craftax_classic.renderer import make_craftax_pixel_renderer
from craftax.sea.discovery import ClusterArtifacts
from craftax.sea.env import make_sea_craftax_classic_env
from craftax.sea.goal import (
    GoalRuntime,
    apply_goal_transition,
    initialize_goal_state,
    load_oracle_graph,
)
from craftax.sea.networks import ActorCriticRNN, ScannedGRU, TransitionEncoder


def _parse_optional_bool(value: str) -> bool:
    value = value.lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _load_font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _dominant_cluster_names(run_dir: Path, count: int) -> list[str]:
    path = run_dir / "cluster_achievements.json"
    names = [f"cluster_{index}" for index in range(count)]
    if not path.exists():
        return names
    report = json.loads(path.read_text(encoding="utf-8"))
    for row in report.get("clusters", []):
        achievements = row.get("achievements", {})
        if achievements:
            name = max(achievements.items(), key=lambda item: item[1]["percent"])[0]
            names[int(row["cluster_id"])] = name
    return names


def _load_policy(
    run_dir: Path, checkpoint: str, config: dict,
    oracle_graph_path: Path | None = None,
):
    args = config["args"]
    if checkpoint == "auto":
        checkpoint = "goal" if (run_dir / "goal_policy.msgpack").exists() else "base"
    policy_path = run_dir / f"{checkpoint}_policy.msgpack"
    if not policy_path.exists():
        raise FileNotFoundError(f"missing {policy_path}")
    policy_params = serialization.msgpack_restore(policy_path.read_bytes())
    hidden_size = int(config[f"{checkpoint}_ppo"]["hidden_size"])

    runtime = None
    cluster_names = []
    if checkpoint == "goal":
        oracle_graph = (
            load_oracle_graph(oracle_graph_path)
            if oracle_graph_path is not None
            else None
        )
        if oracle_graph is None:
            artifacts = ClusterArtifacts.load(run_dir / "clusters.npz")
            encoder_path = run_dir / "transition_encoder.msgpack"
            if not encoder_path.exists():
                raise FileNotFoundError(f"missing {encoder_path}")
            encoder_params = serialization.msgpack_restore(encoder_path.read_bytes())
            discovery = config["discovery"]
            encoder = TransitionEncoder(
                action_dim=17,
                hidden_size=int(discovery["hidden_size"]),
                embedding_size=int(discovery["embedding_size"]),
                pixel=bool(args["pixels"]),
            )
            centroids = jnp.asarray(artifacts.centroids)
            threshold = artifacts.threshold
            graph = jnp.asarray(artifacts.graph)
            oracle_ids = None
            cluster_names = _dominant_cluster_names(run_dir, len(artifacts.centroids))
        else:
            encoder = None
            encoder_params = None
            centroids = jnp.zeros(
                (len(oracle_graph.achievement_ids), 1), dtype=jnp.float32
            )
            threshold = 0.0
            graph = jnp.asarray(oracle_graph.graph)
            oracle_ids = oracle_graph.achievement_ids
            cluster_names = list(oracle_graph.achievement_names)

        include_new = not bool(args.get("no_new_tasks", False))
        runtime = GoalRuntime(
            transition_network=encoder,
            transition_params=encoder_params,
            centroids=centroids,
            threshold=threshold,
            graph=graph,
            exploration_reward_coefficient=float(
                args.get("exploration_reward_coefficient", 0.1)
            ),
            goal_success_coefficient=float(
                args.get("goal_success_coefficient", 5.0)
            ),
            non_target_achievement_coefficient=float(
                args.get("non_target_achievement_coefficient", 1.0)
            ),
            include_new_tasks=include_new,
            oracle_achievement_ids=oracle_ids,
        )
        if include_new:
            cluster_names.append("unknown/new task")

    return checkpoint, policy_params, hidden_size, runtime, cluster_names


def _combat_sources(old, new) -> list[str]:
    sources = []
    player = np.asarray(old.player_position)
    zombie_positions = np.asarray(old.zombies.position)
    zombie_mask = np.asarray(old.zombies.mask)
    zombie_cooldown = np.asarray(old.zombies.attack_cooldown)
    zombie_distance = np.abs(zombie_positions - player).sum(axis=1)
    if np.any(zombie_mask & (zombie_distance == 1) & (zombie_cooldown <= 0)):
        sources.append("zombie melee")

    arrow_positions = np.asarray(old.arrows.position)
    arrow_directions = np.asarray(old.arrow_directions)
    arrow_mask = np.asarray(old.arrows.mask)
    if np.any(arrow_mask & np.all(arrow_positions + arrow_directions == player, axis=1)):
        sources.append("skeleton arrow")
    new_arrow_positions = np.asarray(new.arrows.position)
    new_arrow_mask = np.asarray(new.arrows.mask)
    if np.any(new_arrow_mask & np.all(new_arrow_positions == player, axis=1)):
        if "skeleton arrow" not in sources:
            sources.append("skeleton arrow")
    return sources


def _damage_and_death(old, new, info: dict) -> tuple[list[str], str | None]:
    sources = _combat_sources(old, new)
    health_delta = int(np.asarray(new.player_health)) - int(np.asarray(old.player_health))
    if health_delta < 0 and not sources:
        depleted = []
        if int(np.asarray(old.player_food)) <= 0:
            depleted.append("food")
        if int(np.asarray(old.player_drink)) <= 0:
            depleted.append("drink")
        if int(np.asarray(old.player_energy)) <= 0 and not bool(np.asarray(old.is_sleeping)):
            depleted.append("energy")
        sources.append("vitals depletion" + (f" ({'/'.join(depleted)})" if depleted else ""))

    death = None
    if bool(np.asarray(info["health_done"])):
        death = "health: " + (", ".join(sources) if sources else "unknown damage")
    elif bool(np.asarray(info["idle_done"])):
        death = "idle timeout (no new achievement)"
    elif bool(np.asarray(info["time_done"])):
        death = "episode time limit"
    return sources, death


def _new_achievement_names(values) -> list[str]:
    return [
        Achievement(index).name.lower()
        for index in np.flatnonzero(np.asarray(values, dtype=bool))
    ]


def _make_frame(renderer, core_state, lines: list[str], scale: int = 2) -> np.ndarray:
    pixels = np.asarray(renderer(core_state), dtype=np.uint8)
    image = Image.fromarray(pixels)
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    panel_width = 390
    canvas = Image.new("RGB", (image.width + panel_width, image.height), "#f6f8fc")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(18)
    small = _load_font(15)
    x, y = image.width + 18, 18
    draw.text((x, y), "SEA Policy Replay", fill="#20345f", font=font)
    y += 34
    for line in lines:
        draw.text((x, y), line, fill="#263b66", font=small)
        y += 25
    return np.asarray(canvas)


def _show_frame(frame: np.ndarray, fps: int, state: dict) -> bool:
    import pygame

    if state.get("screen") is None:
        pygame.init()
        state["screen"] = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
        pygame.display.set_caption("Craftax-Classic SEA policy replay")
        state["clock"] = pygame.time.Clock()
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        ):
            return False
    surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
    state["screen"].blit(surface, (0, 0))
    pygame.display.flip()
    state["clock"].tick(fps)
    return True


def replay(args):
    run_dir = Path(args.run_dir).expanduser().resolve()
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    saved_args = config["args"]
    if not saved_args.get("pixels", False):
        raise ValueError("this viewer currently supports pixel policy checkpoints only")

    if args.oracle_graph is not None:
        oracle_path = Path(args.oracle_graph).expanduser().resolve()
    else:
        inherited_oracle = run_dir / "oracle_graph.npz"
        oracle_path = inherited_oracle if inherited_oracle.is_file() else None
    checkpoint, policy_params, hidden_size, runtime, cluster_names = _load_policy(
        run_dir, args.checkpoint, config,
        oracle_graph_path=oracle_path,
    )
    immortal = (
        bool(saved_args.get("immortal", True))
        if args.immortal is None
        else args.immortal
    )
    health_coef = (
        float(saved_args.get("health_reward_coef", 0.0))
        if args.health_reward_coef is None
        else args.health_reward_coef
    )
    idle_timeout = int(
        saved_args.get("idle_timeout", 100)
        if args.idle_timeout is None else args.idle_timeout
    )
    combat_damage_mode = (
        saved_args.get("combat_damage_mode", "lethal-only")
        if args.combat_damage_mode is None
        else args.combat_damage_mode
    )
    env = make_sea_craftax_classic_env(
        symbolic=False,
        idle_timeout=idle_timeout,
        immortal=immortal,
        health_reward_coef=health_coef,
        combat_damage_mode=combat_damage_mode,
    )
    params = env.default_params
    network = ActorCriticRNN(
        action_dim=env.action_space(params).n,
        hidden_size=hidden_size,
        pixel=True,
        num_objectives=0 if runtime is None else runtime.num_objectives,
    )
    renderer = jax.jit(make_craftax_pixel_renderer(args.block_size))
    step_fn = jax.jit(env.step)

    writer = None
    if args.video:
        video_path = Path(args.video).expanduser().resolve()
        video_path.parent.mkdir(parents=True, exist_ok=True)
        if video_path.suffix.lower() not in {".gif", ".mp4"}:
            raise ValueError("--video must end in .gif or .mp4")
        try:
            writer = imageio.get_writer(video_path, fps=args.fps, loop=0)
        except Exception as error:
            raise RuntimeError(
                "could not open video writer; GIF works without ffmpeg, while MP4 "
                "requires `pip install imageio-ffmpeg`"
            ) from error
    log_path = Path(args.log).expanduser().resolve() if args.log else run_dir / (
        f"replay_{checkpoint}_{'immortal' if immortal else 'mortal'}_seed{args.seed}.jsonl"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    pygame_state = {}
    key = jax.random.PRNGKey(args.seed)
    keep_running = True

    try:
        for episode in range(args.episodes):
            key, reset_key, goal_key = jax.random.split(key, 3)
            observation, state = env.reset(reset_key, params)
            hidden = ScannedGRU.initialize_carry(1, hidden_size)
            last_reset = jnp.ones((1,), dtype=bool)
            goal_state = (
                initialize_goal_state(
                    goal_key, 1, runtime.graph, runtime.include_new_tasks,
                )
                if runtime is not None
                else None
            )
            episode_return = 0.0
            for step in range(args.max_steps):
                key, action_key, env_key, goal_key = jax.random.split(key, 4)
                if runtime is None:
                    objective = jnp.zeros((1,), dtype=jnp.int32)
                    completed = jnp.zeros((1, 1), dtype=bool)
                    objective_before = None
                else:
                    objective = goal_state.objective
                    completed = goal_state.completed
                    objective_before = int(np.asarray(objective[0]))
                hidden, logits, _ = network.apply(
                    policy_params,
                    hidden,
                    observation[None, None],
                    last_reset[None],
                    objective[None],
                    completed[None],
                )
                if args.deterministic:
                    action = jnp.argmax(logits[0, 0]).astype(jnp.int32)
                else:
                    action = jax.random.categorical(action_key, logits[0, 0])
                old_core = state.env_state
                next_observation, next_state, reward, done, info = step_fn(
                    env_key, state, action, params
                )
                predicted_class = None
                goal_done = False
                if runtime is not None:
                    goal_state, _, goal_done_array, predicted, _ = apply_goal_transition(
                        runtime,
                        goal_state,
                        goal_key,
                        observation[None],
                        jnp.asarray([action]),
                        next_observation[None],
                        jnp.asarray([done]),
                        jnp.asarray([info["event_count"]]),
                        new_achievements=info["new_achievements"][None],
                    )
                    predicted_class = int(np.asarray(predicted[0]))
                    goal_done = bool(np.asarray(goal_done_array[0]))
                sources, death = _damage_and_death(old_core, next_state.env_state, info)
                achievements = _new_achievement_names(info["new_achievements"])
                action_name = Action(int(np.asarray(action))).name.lower()
                episode_return += float(np.asarray(reward))
                objective_name = (
                    "none"
                    if objective_before is None
                    else f"C{objective_before}: {cluster_names[objective_before]}"
                )
                record = {
                    "episode": episode,
                    "step": step + 1,
                    "checkpoint": checkpoint,
                    "environment": "immortal" if immortal else "mortal",
                    "idle_timeout": idle_timeout,
                    "combat_damage_mode": combat_damage_mode,
                    "action": action_name,
                    "objective": objective_before,
                    "objective_name": objective_name,
                    "predicted_event_cluster": predicted_class,
                    "goal_completed": goal_done,
                    "reward": float(np.asarray(reward)),
                    "health": int(np.asarray(next_state.env_state.player_health)),
                    "health_delta": int(np.asarray(next_state.env_state.player_health)) - int(np.asarray(old_core.player_health)),
                    "food": int(np.asarray(next_state.env_state.player_food)),
                    "drink": int(np.asarray(next_state.env_state.player_drink)),
                    "energy": int(np.asarray(next_state.env_state.player_energy)),
                    "damage_sources": sources,
                    "new_achievements": achievements,
                    "done": bool(np.asarray(done)),
                    "death_or_end_reason": death,
                }
                log_file.write(json.dumps(record) + "\n")
                if sources or achievements or death:
                    print(record, flush=True)

                if not args.no_window or writer is not None:
                    lines = [
                        f"Run: {run_dir.name} / {checkpoint}",
                        f"Env: {'IMMORTAL' if immortal else 'MORTAL'} / {combat_damage_mode}",
                        f"Episode {episode + 1}/{args.episodes}  Step {step + 1}",
                        f"Action: {action_name}",
                        f"Goal: {objective_name}",
                        f"HP {record['health']}  Food {record['food']}",
                        f"Drink {record['drink']}  Energy {record['energy']}",
                        f"Achievement: {', '.join(achievements) or '-'}",
                        f"Damage: {', '.join(sources) or '-'}",
                        f"End: {death or '-'}",
                    ]
                    frame = _make_frame(renderer, next_state.env_state, lines, args.scale)
                    if writer is not None:
                        writer.append_data(frame)
                    if not args.no_window:
                        keep_running = _show_frame(frame, args.fps, pygame_state)
                        if not keep_running:
                            break
                next_reset = bool(np.asarray(done)) or (
                    goal_done and runtime is not None and runtime.reset_rnn_on_goal
                )
                observation, state = next_observation, next_state
                last_reset = jnp.asarray([next_reset])
                if bool(np.asarray(done)):
                    print(
                        f"episode={episode} ended at step={step + 1}: {death}; "
                        f"return={episode_return:.2f}",
                        flush=True,
                    )
                    break
            if not keep_running:
                break
    finally:
        log_file.close()
        if writer is not None:
            writer.close()
        if pygame_state:
            import pygame
            pygame.quit()
    print(f"diagnostic log: {log_path}")
    if args.video:
        print(f"video: {Path(args.video).expanduser().resolve()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="run directory containing config.json and checkpoint")
    parser.add_argument("--checkpoint", choices=("auto", "base", "goal"), default="auto")
    parser.add_argument(
        "--oracle-graph",
        help="override with an achievement-indexed oracle graph .npz",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--deterministic", action="store_true", help="use argmax instead of training-style sampling")
    parser.add_argument("--immortal", type=_parse_optional_bool, default=None, help="override the run config")
    parser.add_argument("--health-reward-coef", type=float, default=None, help="override the run config")
    parser.add_argument("--idle-timeout", type=int, default=None, help="override the run config for evaluation")
    parser.add_argument(
        "--combat-damage-mode", choices=("lethal-only", "cumulative"),
        default=None, help="override combat damage; old runs default to lethal-only",
    )
    parser.add_argument("--video", help="optional .gif or .mp4 output")
    parser.add_argument("--log", help="diagnostic JSONL output path")
    parser.add_argument("--no-window", action="store_true", help="headless logging/video mode")
    parser.add_argument("--block-size", type=int, default=16, choices=(8, 16, 32, 64))
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args(argv)
    if (
        args.episodes < 1 or args.max_steps < 1 or args.fps < 1 or args.scale < 1
        or (args.idle_timeout is not None and args.idle_timeout < 1)
    ):
        parser.error("episodes, max-steps, fps and scale must be positive")
    replay(args)


if __name__ == "__main__":
    main()
