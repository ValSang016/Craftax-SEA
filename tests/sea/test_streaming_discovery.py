from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from craftax.sea.discovery import DiscoveryConfig, TransitionDataset, initialize_transition_encoder
from craftax.sea.networks import ActorCriticRNN, ScannedGRU
from craftax.sea.ppo_rnn import PPOConfig
from craftax.sea.rollout import iter_policy_transitions
from craftax.sea.streaming_discovery import collect_clustering_dataset, train_streaming_encoder


def batch(start, size):
    ids = np.arange(start, start+size)
    achievements = np.zeros((size, 22), dtype=bool)
    achievements[np.flatnonzero(ids % 2 == 0), (ids[ids % 2 == 0] % 4)//2] = True
    return TransitionDataset(
        observation=np.stack((ids, ids % 4), axis=1).astype(np.float32),
        action=np.full(size, 5, dtype=np.int16),
        next_observation=np.stack((ids+1, (ids+1) % 4), axis=1).astype(np.float32),
        terminal=ids % 4 == 3, event_count=(ids % 2 == 0).astype(np.float32),
        episode_id=(ids//4).astype(np.int32), episode_step=(ids%4).astype(np.int32),
        contrast_eligible=np.ones(size, dtype=bool), new_achievements=achievements,
    )


def test_reference_loss_is_invariant_to_prediction_batch_duplication():
    data = batch(0, 4)
    config = DiscoveryConfig(batch_size=4, hidden_size=8, embedding_size=8, contrast_groups=2)
    assert config.mean_contrast_coefficient == 1.0
    assert replace(config, batch_size=256, contrast_groups=64).mean_contrast_coefficient == 1.0
    prediction = tuple(getattr(data, f) for f in ('observation', 'action', 'next_observation', 'terminal', 'event_count'))
    indices = np.array([[0, 1, 2], [1, 2, 3]])
    contrast = tuple(x[indices] for x in prediction[:4])
    valid = np.ones((2, 3), dtype=bool)
    _, first_state, first_update = initialize_transition_encoder(data, config)
    _, second_state, second_update = initialize_transition_encoder(data, replace(config, batch_size=8))
    first_state, first = first_update(first_state, prediction, contrast, valid)
    second_state, second = second_update(second_state, tuple(np.concatenate([x,x]) for x in prediction), contrast, valid)
    np.testing.assert_allclose(float(first[0]), float(second[0]), rtol=1e-5)
    for a, b in zip(jax.tree_util.tree_leaves(first_state.params), jax.tree_util.tree_leaves(second_state.params)):
        np.testing.assert_allclose(a, b, atol=1e-6)


def test_stream_uses_each_fresh_example_once_and_freezes_completed_groups(monkeypatch):
    from craftax.sea import streaming_discovery
    seen = []
    original = streaming_discovery.initialize_transition_encoder

    def initialize(*args, **kwargs):
        network, state, update = original(*args, **kwargs)
        def recording_update(state, prediction, contrast, valid):
            seen.extend(prediction[0][:, 0].astype(int).tolist())
            return update(state, prediction, contrast, valid)
        return network, state, recording_update

    monkeypatch.setattr(streaming_discovery, 'initialize_transition_encoder', initialize)
    config = DiscoveryConfig(batch_size=8, train_steps=5, contrast_groups=2,
                             hidden_size=8, embedding_size=8, contrast_episode_capacity=6)
    # Cutoff falls inside episode 6, so its completion later in the same batch
    # must not admit the incomplete episode. A short final batch still trains.
    result = train_streaming_encoder((batch(start, min(8,34-start)) for start in range(0,34,8)),
                                     config, contrast_interactions=26)
    assert seen == list(range(34))
    assert result.summary['prediction_transitions_seen'] == 34
    assert result.summary['prediction_positive_count'] == 17
    assert result.summary['prediction_positive_fraction'] == 0.5
    assert result.summary['completed_episodes_seen'] == 6
    assert result.summary['contrast_episodes'] == 6
    assert result.summary['contrast_updates'] == 3
    assert set(result.contrast_dataset.episode_id) == set(range(6))
    assert int(result.state.step) == 5
    assert np.isfinite(float(result.metrics[0]))


def test_zero_window_and_budget_mismatch():
    config = DiscoveryConfig(batch_size=8, train_steps=1, contrast_groups=2, hidden_size=8, embedding_size=8)
    result = train_streaming_encoder([batch(0, 8)], config, contrast_interactions=0)
    assert result.contrast_dataset is None
    assert result.summary['contrast_updates'] == 0
    assert float(result.metrics[1][1]) == 0
    with pytest.raises(ValueError, match='stream ended'):
        train_streaming_encoder([], config)
    with pytest.raises(ValueError, match='refusing to drop'):
        train_streaming_encoder([batch(0,8),batch(8,8)], config)


def test_clustering_keeps_fresh_positive_prefix():
    dataset, consumed = collect_clustering_dataset((batch(i,8) for i in (100,108,116)), max_transitions=5)
    np.testing.assert_array_equal(dataset.observation[:,0], [100,102,104,106,108])
    assert consumed == 16
    assert (dataset.event_count > 0).all()
    with pytest.raises(ValueError, match='increase --clustering-interactions'):
        collect_clustering_dataset([batch(0,4)], max_transitions=3)


def test_easy_subset_masks_undiscovered_events_without_relabeling_them_negative():
    data = batch(0, 8)
    # Four positive rows: two discovered Easy events and two undiscovered events.
    data.new_achievements[:] = False
    data.new_achievements[[0, 4], 0] = True
    data.new_achievements[[2, 6], 8] = True
    config = DiscoveryConfig(
        batch_size=8, train_steps=1, contrast_groups=2,
        hidden_size=8, embedding_size=8,
    )
    result = train_streaming_encoder(
        [data], config, contrast_interactions=8, achievement_indices=(0,)
    )
    assert result.summary['prediction_transitions_seen'] == 8
    assert result.summary['prediction_examples_used'] == 6
    assert result.summary['prediction_excluded_event_transitions'] == 2
    assert result.summary['prediction_positive_count'] == 2
    assert result.summary['prediction_positive_fraction'] == pytest.approx(2/6)

    selected, consumed = collect_clustering_dataset(
        [data], max_transitions=2, achievement_indices=(0,)
    )
    assert consumed == 8
    np.testing.assert_array_equal(selected.observation[:, 0], [0, 4])
    assert selected.new_achievements[:, 0].all()
    assert not selected.new_achievements[:, 1:].any()


@struct.dataclass
class ToyInner:
    timestep: object


@struct.dataclass
class ToyState:
    env_state: ToyInner


class ToyEnv:
    default_params = None

    def action_space(self, params):
        from types import SimpleNamespace
        return SimpleNamespace(n=17)

    def reset(self, key, params):
        return jnp.zeros(2), ToyState(ToyInner(jnp.int32(0)))

    def step(self, key, state, action, params):
        t = state.env_state.timestep
        obs = jnp.array([t+1, action], dtype=jnp.float32)
        positive = t % 2 == 0
        achievements = jnp.zeros(22, dtype=bool).at[t//2].set(positive)
        return obs, ToyState(ToyInner(t+1)), positive.astype(jnp.float32), t == 3, {'new_achievements': achievements}


def test_policy_stream_carries_episodes_and_preserves_terminal_frames():
    env = ToyEnv()
    config = PPOConfig(num_envs=2, hidden_size=8, reset_ratio=1)
    network = ActorCriticRNN(action_dim=17, hidden_size=8, pixel=False, num_objectives=0)
    params = network.init(jax.random.PRNGKey(0), ScannedGRU.initialize_carry(2,8),
                          jnp.zeros((1,2,2)), jnp.zeros((1,2), dtype=bool),
                          jnp.zeros((1,2), dtype=jnp.int32), jnp.zeros((1,2,1), dtype=bool))
    chunks = list(iter_policy_transitions(env, params, config, interactions=13, chunk_steps=2))
    assert list(map(len,chunks)) == [4,4,4,1]
    ep = np.concatenate([c.episode_id for c in chunks])
    step = np.concatenate([c.episode_step for c in chunks])
    assert len(set(zip(ep,step))) == 13
    np.testing.assert_array_equal(ep, [0,1,0,1,0,1,0,1,2,3,2,3,2])
    after = np.concatenate([c.next_observation for c in chunks])
    terminal = np.concatenate([c.terminal for c in chunks])
    np.testing.assert_array_equal(after[terminal,0], [4,4])
    before = np.concatenate([c.observation for c in chunks])
    np.testing.assert_array_equal(before[8:10,0], [0,0])


def test_cli_rejects_offline_replay_and_inconsistent_update_budget(tmp_path):
    from craftax.sea.train import build_parser, run_pipeline
    parser = build_parser()
    args = parser.parse_args(['--output', str(tmp_path/'run'),
                              '--discovery-dataset-checkpoint', 'old.npz'])
    with pytest.raises(ValueError, match='cannot train from an old dataset'):
        run_pipeline(args)
    args = parser.parse_args(['--output', str(tmp_path/'run'),
                              '--discovery-interactions', '100',
                              '--discovery-batch-size', '32', '--discovery-num-envs', '2',
                              '--discovery-train-steps', '1'])
    with pytest.raises(ValueError, match='requires 4 updates'):
        run_pipeline(args)
    assert not (tmp_path/'run').exists()


def test_cli_exposes_contrast_weight_and_encoder_only_mode():
    from craftax.sea.train import build_parser
    args = build_parser().parse_args([
        '--contrast-weight', '2.0', '--stop-after-clustering'
    ])
    assert args.contrast_weight == 2.0
    assert args.stop_after_clustering


def test_easy17_definition_excludes_hard4_and_eat_plant():
    from craftax.sea.metrics import (
        ACHIEVEMENT_NAMES, EASY17_ACHIEVEMENT_INDICES, HARD_ACHIEVEMENT_INDICES,
    )
    assert len(EASY17_ACHIEVEMENT_INDICES) == 17
    assert not set(EASY17_ACHIEVEMENT_INDICES) & set(HARD_ACHIEVEMENT_INDICES)
    assert ACHIEVEMENT_NAMES.index('eat_plant') not in EASY17_ACHIEVEMENT_INDICES
