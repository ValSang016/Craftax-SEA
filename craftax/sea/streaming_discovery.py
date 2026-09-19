"""SEA encoder learning from fresh frozen-policy rollouts, one pass only."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from craftax.sea.discovery import (
    TransitionDataset, _contrast_indices, initialize_transition_encoder,
)
from craftax.sea.episode_replay import EpisodeEventBuffer


@dataclass
class StreamingDiscoveryResult:
    network: object
    state: object
    metrics: object
    summary: dict
    contrast_dataset: TransitionDataset | None


def add_contrast_events(
    buffer, batch, *, offset, contrast_interactions, eligible_event=None,
    achievement_indices=None,
):
    """Admit complete episodes only, updating in stream order before the cutoff."""
    remaining = max(0, min(len(batch), contrast_interactions-offset))
    if eligible_event is None:
        eligible_event = batch.event_count > 0
    for i in np.flatnonzero(eligible_event[:remaining] | batch.terminal[:remaining]):
        event = None
        if eligible_event[i]:
            achievements = batch.new_achievements[i].copy()
            if achievement_indices is not None:
                selected = np.zeros_like(achievements)
                selected[list(achievement_indices)] = achievements[list(achievement_indices)]
                achievements = selected
            event = (
                batch.observation[i].copy(), batch.action[i].copy(),
                batch.next_observation[i].copy(), batch.terminal[i].copy(),
                np.float32(1.0), batch.episode_id[i].copy(),
                batch.episode_step[i].copy(), achievements,
            )
        buffer.append(int(batch.episode_id[i]), event, bool(batch.terminal[i]))
    if offset + len(batch) >= contrast_interactions:
        # Do not admit partially observed episodes after the window freezes.
        buffer.pending.clear()


def train_streaming_encoder(batches, config, *, contrast_interactions=1_000_000,
                            pixel=False, seed=0, progress_callback=None,
                            achievement_indices=None):
    """Use every incoming transition once for prediction, with no oversampling.

    Each batch is a newly generated rollout. The completed-episode FIFO evolves
    during the initial window and is frozen thereafter. As in iclr-SEA, contrast
    optimization starts after more than five eligible episodes have completed.
    All averages use the fixed reference loss coefficient, including a partial
    final prediction batch. Ground-truth achievement labels are diagnostics only.
    """
    if config.batch_size < 1 or config.train_steps < 1 or contrast_interactions < 0:
        raise ValueError("positive batch/update counts and a nonnegative contrast window are required")
    rng = np.random.default_rng(seed)
    buffer = EpisodeEventBuffer(config.contrast_episode_capacity)
    network = state = update = metrics = None
    contrast_dataset = None
    version = -1
    seen = positives = prediction_examples = excluded_events = updates = contrast_updates = 0
    for batch in batches:
        if updates >= config.train_steps:
            raise ValueError("stream contains more updates than configured; refusing to drop fresh data")
        if not 0 < len(batch) <= config.batch_size:
            raise ValueError("rollout batch must contain 1..batch_size transitions")
        if batch.new_achievements is None:
            raise ValueError("streaming collector must include achievement diagnostics")
        if achievement_indices is None:
            eligible_event = batch.event_count > 0
            prediction_valid = np.ones(len(batch), dtype=bool)
        else:
            eligible_event = np.asarray(
                batch.new_achievements[:, list(achievement_indices)].any(axis=1),
                dtype=bool,
            )
            any_event = np.asarray(batch.new_achievements.any(axis=1), dtype=bool)
            prediction_valid = ~(any_event & ~eligible_event)
        if update is None:
            network, state, update = initialize_transition_encoder(batch, config, pixel=pixel, seed=seed)
        add_contrast_events(
            buffer, batch, offset=seen,
            contrast_interactions=contrast_interactions,
            eligible_event=eligible_event,
            achievement_indices=achievement_indices,
        )
        if buffer.completed_count != version:
            contrast_dataset = buffer.dataset() if buffer.completed else None
            version = buffer.completed_count
        prediction = (batch.observation, batch.action, batch.next_observation,
                      batch.terminal, eligible_event.astype(np.float32),
                      prediction_valid)
        if len(buffer.completed) > 5:
            indices, valid = _contrast_indices(contrast_dataset, rng, config)
            contrast = tuple(getattr(contrast_dataset, field)[indices] for field in
                             ('observation', 'action', 'next_observation', 'terminal'))
            contrast_updates += 1
        else:
            shape = (config.contrast_groups, config.max_events_per_episode)
            contrast = tuple(np.zeros(shape+x.shape[1:], dtype=x.dtype) for x in prediction[:4])
            valid = np.zeros(shape, dtype=bool)
        state, metrics = update(state, prediction, contrast, valid)
        seen += len(batch)
        positives += int(eligible_event.sum())
        prediction_examples += int(prediction_valid.sum())
        excluded_events += int((~prediction_valid).sum())
        updates += 1
        if progress_callback is not None and (updates == 1 or updates % 500 == 0 or updates == config.train_steps):
            progress_callback(updates, metrics, {
                'prediction_transitions_seen': seen,
                'prediction_examples_used': prediction_examples,
                'prediction_excluded_event_transitions': excluded_events,
                'prediction_positive_fraction': positives/max(prediction_examples, 1),
                'prediction_positive_fraction_per_interaction': positives/seen,
                'contrast_episodes': len(buffer.completed),
                'completed_episodes_seen': buffer.completed_count,
                'contrast_updates': contrast_updates,
                'mean_contrast_coefficient': config.mean_contrast_coefficient,
            })
    if updates != config.train_steps:
        raise ValueError(f"stream ended after {updates} updates, expected {config.train_steps}")
    lengths = np.array([len(ep) for ep in buffer.completed])
    summary = {
        'updates': updates, 'prediction_transitions_seen': seen,
        'prediction_examples_used': prediction_examples,
        'prediction_excluded_event_transitions': excluded_events,
        'prediction_positive_count': positives,
        'prediction_positive_fraction': positives/max(prediction_examples, 1),
        'prediction_positive_fraction_per_interaction': positives/seen,
        'prediction_sampling': (
            'each eligible fresh transition once, no replay or positive oversampling; '
            'undiscovered achievement-event rows are masked, not relabeled negative'
        ),
        'mean_contrast_coefficient': config.mean_contrast_coefficient,
        'contrast_updates': contrast_updates, 'contrast_episodes': len(lengths),
        'completed_episodes_seen': buffer.completed_count,
        'mean_events_per_episode': float(lengths.mean()) if len(lengths) else 0.0,
        'two_event_fraction': float((lengths == 2).mean()) if len(lengths) else 0.0,
        'event_count_histogram': {str(int(n)): int((lengths == n).sum()) for n in np.unique(lengths)},
    }
    return StreamingDiscoveryResult(network, state, metrics, summary, contrast_dataset)


def collect_clustering_dataset(
    batches, max_transitions=10_000, achievement_indices=None
):
    """Keep the first N positive transitions from an independent fresh rollout."""
    if max_transitions < 2:
        raise ValueError("clustering needs at least two transitions")
    chunks = []
    count = consumed = 0
    for batch in batches:
        consumed += len(batch)
        eligible = batch.event_count > 0
        if achievement_indices is not None:
            eligible &= batch.new_achievements[:, list(achievement_indices)].any(axis=1)
        indices = np.flatnonzero(eligible)[:max_transitions-count]
        if len(indices):
            values = {name: (None if value is None else value[indices])
                      for name, value in vars(batch).items()}
            if achievement_indices is not None:
                masked = np.zeros_like(values['new_achievements'])
                masked[:, list(achievement_indices)] = values['new_achievements'][:, list(achievement_indices)]
                values['new_achievements'] = masked
                values['event_count'] = masked.sum(axis=1).astype(np.float32)
            chunks.append(values)
            count += len(indices)
        if count == max_transitions:
            break
    if count != max_transitions:
        raise ValueError(f"clustering rollout yielded {count}/{max_transitions} positive transitions; "
                         "increase --clustering-interactions")
    values = {name: (None if chunks[0][name] is None else
                     np.concatenate([chunk[name] for chunk in chunks])) for name in chunks[0]}
    return TransitionDataset(**values), consumed
