import numpy as np
from craftax.sea.episode_replay import EpisodeEventBuffer
from craftax.sea.discovery import DiscoveryConfig, _contrast_indices


def event(ep, step):
    achievements = np.zeros(22, dtype=bool)
    achievements[step % 22] = True
    return (np.array([step], dtype=np.float16), np.int16(5),
            np.array([step+1], dtype=np.float16), np.bool_(False),
            np.float32(1), np.int32(ep), np.int32(step), achievements)


def test_complete_episode_fifo_keeps_all_events_and_excludes_unfinished():
    buffer = EpisodeEventBuffer(2)
    # Interleaved episodes finish in a different order from their first events.
    for i in range(15):
        buffer.append(0, event(0, i), False)
        buffer.append(1, event(1, i), False)
    buffer.append(1, None, True)
    buffer.append(0, None, True)
    for i in range(12):
        buffer.append(2, event(2, i), False)
    buffer.append(2, None, True)
    buffer.append(3, event(3, 0), True)  # One-event episodes do not enter FIFO.
    for i in range(20):
        buffer.append(4, event(4, i), False)  # Incomplete at cutoff: exclude.
    dataset = buffer.dataset()
    np.testing.assert_array_equal(dataset.episode_id, [0]*15 + [2]*12)
    assert len(dataset) == 27
    assert buffer.completed_count == 3
    np.testing.assert_array_equal(dataset.new_achievements.sum(axis=1), dataset.event_count)
    indices, valid = _contrast_indices(dataset, np.random.default_rng(0), DiscoveryConfig())
    assert valid.all()  # Sample 8 from complete groups, not only 2 retained events.
    for group in indices:
        assert len(set(dataset.episode_id[group])) == 1
        assert len(set(group)) == 8


def test_encoder_uses_separate_complete_episode_dataset():
    from craftax.sea.discovery import TransitionDataset, train_transition_encoder
    buffer = EpisodeEventBuffer(2)
    for ep in range(2):
        for step in range(12):
            buffer.append(ep, event(ep, step), False)
        buffer.append(ep, None, True)
    contrast = buffer.dataset()
    prediction = TransitionDataset(
        observation=np.zeros((1, 1), dtype=np.float16),
        action=np.zeros(1, dtype=np.int16),
        next_observation=np.zeros((1, 1), dtype=np.float16),
        terminal=np.zeros(1, dtype=bool), event_count=np.zeros(1),
        episode_id=np.zeros(1, dtype=np.int32), episode_step=np.zeros(1, dtype=np.int32),
        contrast_eligible=np.zeros(1, dtype=bool),
    )
    _, state, metrics = train_transition_encoder(
        prediction, DiscoveryConfig(train_steps=1, batch_size=2, contrast_groups=2,
                                    hidden_size=8, embedding_size=8),
        contrast_dataset=contrast,
    )
    assert int(state.step) == 1
    assert np.isfinite(float(metrics[0]))
    assert float(metrics[1][1]) < 0
