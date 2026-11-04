# 실행 및 코드 구조

## 기본 실험: SEA (fix123)

프로젝트 루트에서 고정 환경 `/opt/conda/envs/craftax`를 사용합니다.

```bash
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cuda,cpu /opt/conda/envs/craftax/bin/python scripts/run_sea_experiments.py --output runs/sea-new --gpus 0 1 --seeds 0 1
```

기본 pipeline: collection PPO 200M → fresh encoder discovery 50M → 독립
clustering 표본 10k events → goal PPO 300M. 원본 actor 수와 `num_envs`는
구현 구조가 다르므로 같은 이름의 숫자로 취급하지 않습니다.

## Symbolic state-conditioned dynamic graph

`--symbolic`은 graph recovery 이후 worker PPO와 high-level `Q_F`/`Q_B`를
하나의 online run에서 함께 학습합니다. 처음 30M environment steps에는 ready
candidate를 uniform하게 선택하며 goal 선택용 Q forward/residual 계산은 생략하지만,
replay 수집과 rollout 종료 후 Q update는 계속 진행합니다. 그 뒤에는
`A_SEA + beta * (Q_F - Q_B)`를 candidate-mask 후 softmax합니다. High-level option은
target goal 성공, 기본 300 primitive-step timeout, episode 종료에서만 끝납니다.
각 PPO rollout 뒤 새 high-level transition 수를 `N_H`, Q batch size를 `B`라 할 때
`ceil(high_level_q_replay_ratio * N_H / B)`회 batched Q update를 수행합니다.
기본 replay ratio는 4.0이며, replay가 minimum size에 도달하기 전에는 Q를
update하지 않습니다. 각 update에서 `Q_F`와 `Q_B`는 동일 minibatch를 사용합니다.

`--high-level-controller residual`은 state-conditioned Q residual을 사용하며,
`--high-level-controller sea-meta-controller`는 Q 없이 SEA의 기존
`80/10/10/5` graph meta-controller를 사용합니다.

Recovered graph:

```bash
python -m craftax.sea.train --symbolic --output runs/sea-symbolic
```

Precomputed achievement graph:

```bash
python -m craftax.sea.train --symbolic --graph-path graph.npz \
  --no-new-tasks --high-level-goal-timeout 300 \
  --output runs/sea-symbolic-oracle
```

Graph artifacts use the existing oracle format: `graph` and `achievement_ids`
arrays are required, while `achievement_names` is optional. Dynamic Q parameters
are saved in `high_level_q.msgpack`; per-update high-level metrics are included
in `goal_metrics_history.npz`.

## 모듈

| 모듈 | 역할 |
|---|---|
| `train.py` | 기본 fix123 pipeline 및 명시적 FF 선택 |
| `streaming_discovery.py`, `episode_replay.py` | fresh prediction + complete-episode contrast |
| `discovery.py` | encoder loss/optimizer, constrained KMeans, artifact |
| `graph.py` | learned/oracle 공통 temporal graph 생성 및 축약 |
| `goal.py` | 분류, 목표 선택, 보상 및 oracle 선택 분기 |
| `dynamic_graph.py` | symbolic state split, high-level replay, Q learning 및 dynamic edge |
| `ppo_rnn.py`, `networks.py` | 기본 PPO-GRU 및 픽셀 모델 |
| `env.py`, `vector_env.py`, `rollout.py` | 환경 규칙과 수집 |
| `metrics.py`, `checkpoint.py` | episode 지표, parameter 저장 |

`discovery.py`의 offline learner/collector는 과거 진단용으로 남겨 둡니다.
기본 CLI는 streaming 경로를 사용합니다.

## 결과 확인

각 seed 디렉터리의 `base_metrics.json`, `goal_metrics.json`,
`discovery_summary.json`, `cluster_achievements.json`을 확인합니다. 전체 PPO update
기록은 `base_metrics_history.npz`와 `goal_metrics_history.npz`에 저장됩니다.

- 정책 성능: seed별 완료 episode 합계에서 성공률 계산, 그 뒤 seed 동일 가중 평균
- last100: 마지막 100 update, 일반 full run에서 6,553,600 step
- score21: eat_plant 제외 geometric score
- hard4: zombie, diamond, iron pickaxe, iron sword
- unknown5: 위 4개 + eat_plant (oracle에서만 미발견 목표 집합)
- encoder 로그: 1/500/.../마지막 update의 단일 update 값; 구간 평균 아님
- 초기 original encoder는 최종 loss만 보존; 곡선을 복원하지 않음

실행 실패/중단·축소 smoke·재클러스터링만 한 결과를 완료된 정책 성능 평균에
섞지 않습니다. 정책 단계 x축은 사전 collection/discovery 비용을 제외하므로
PPO와 SEA를 동일 총 계산량의 비교라고 해석하지 않습니다.
