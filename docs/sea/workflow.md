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
`lambda * A_SEA + beta * signal`을 candidate-mask 후 softmax합니다.
기본값은 lambda=1, signal=Q_F−Q_B이므로 기존 식과 같습니다. High-level option은
target goal 성공, 기본 300 primitive-step timeout, episode 종료에서만 끝납니다.
각 PPO rollout 뒤 새 high-level transition 수를 `N_H`, Q batch size를 `B`라 할 때
`ceil(high_level_q_replay_ratio * N_H / B)`회 batched Q update를 수행합니다.
기본 replay ratio는 4.0이며, replay가 minimum size에 도달하기 전에는 Q를
update하지 않습니다. 각 update에서 `Q_F`와 `Q_B`는 동일 minibatch를 사용합니다.

선택 점수는 다음 CLI로 제어합니다.

- `--dynamic-lambda FLOAT` (기본 1): structural prior `A_SEA`의 계수입니다.
  0이면 선택 logit에서 A항을 제거합니다. 유한한 0 이상의 값만 허용합니다.
- `--dynamic-score-source residual` (기본): `signal=Q_F-Q_B`입니다.
  `base-q`이면 `signal=Q_B`, `full-q`이면 `signal=Q_F` 원값을 사용합니다.
  별도 정규화는 없습니다.
- `--dynamic-beta FLOAT` (`--beta` 별칭): 선택한 signal의 계수입니다.

이 옵션은 symbolic의 `--high-level-controller residual` 경로에서 사용합니다.
`base-q`와 `full-q`도 같은 controller 경로를 사용하되 선택 signal만 바뀝니다. 두 Q의 학습은
유지됩니다: Full Q는 기존 SMDP TD target, Base Q는 Full Q teacher 회귀입니다.
`c_t`와 last-completed-node 입력도 유지하며 `q_residual_*` metric은 계속 Q_F−Q_B입니다.
lambda=0은 선택 A항만 제거하므로 warmup의 prerequisite 제한은 유지됩니다.
`train_goal_only`도 같은 옵션을 지원하며 미지정 시 source 설정을 상속합니다.
새 필드가 없는 과거 config는 lambda=1, residual로 해석합니다.

Goal 후보 조건은 두 개의 명시적인 boolean CLI로 제어합니다.

- `--allow-unready-goals false` (기본): graph prerequisite가 모두 달성된
  ready goal만 허용합니다. `true`이면 **warmup 이후** unready도 허용합니다.
  Warmup의 uniform selector는 이 값과 관계없이 ready만 사용합니다.
- `--allow-completed-goals false` (기본): 같은 episode에서 이미 달성한
  achievement를 제외합니다. `true`이면 warmup부터 완료 goal도 재선택할 수 있고,
  실제 반복 수행 이벤트를 option 성공으로 인식하되 반복 achievement에는
  target 보상을 지급하지 않습니다. Symbolic residual controller와 achievement-indexed oracle graph가
  필요합니다. 단순히 cumulative achievement bit가 켜져 있다는 이유로 성공하지 않습니다.

Q bootstrap의 다음-goal 후보에도 같은 조건을 적용합니다. 반복 성공은
option을 종료할 뿐이며, worker target/non-target 보상과 환경 reward/idle timeout/episode achievement
집계는 모두 기존 최초 달성 기준을 유지합니다. 최초 target 달성은 +5입니다.
반복 성공과 동시에 새로운 non-target
achievement를 달성하면 해당 non-target 보상도 유지합니다. 기존 empty-set fallback은
유지되어, 모든 graph node를 달성한 퇴화 상태에서는 전체 node를 다시 허용합니다.

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

## Supervised state-conditioned graph + K-hop planner (실험 2)

`--high-level-controller graph-planner`는 그래프를 지도학습 option 모델
`F(s_t, c_t, g_i, g_j)`로 학습합니다. 출력은 성공 확률 p(=state-conditioned edge),
성공 시 소요 스텝 τ, option 중 사망 확률, 성공 후 completed vector, 다음 latent,
그리고 latent에서 복원한 inventory 12 + health/food/drink/energy 4입니다.
모든 head는 관측된 option 결과를 라벨로 하는 지도학습이며 TD/bootstrapping/
policy gradient를 쓰지 않습니다. 다음 latent는 EMA target encoder와의
consistency loss로 학습합니다. 학습 데이터는 같은 run의 high-level replay입니다.

Planner는 F를 K hop 굴려 경로를 평가합니다.

    J = Σ_k γ^{T_k} ∏_{i≤k} p_i · 1[g_k 미완료]
        (+ γ^{T_K} ∏p · Σ_j γ^{τ_j} p_j 1[g_j 미완료]   ← --planner-leaf true)

첫 hop은 22개 goal 전부, 이후는 첫 goal별 beam(`--planner-beam-width`)으로
탐색하고, 첫 goal별 최고 J에 softmax(`--planner-temperature`)로 goal을 뽑습니다.
한 step에 K-hop 계획은 최대 `--planner-max-decisions` env까지만 하고 나머지는
1-hop으로 대체합니다(`planner_overflow_fraction` 기록).

`--dynamic-warmup-steps` 동안은 실험 1과 같은 SEA meta-controller(어떤 새
achievement든 생기면 goal 재선택, timeout 없음, PPO bootstrap은 goal 성공/episode
종료)를 사용합니다. 이 기간에 다른 achievement로 잘린 option은 라벨이 불완전하므로
replay에 넣지 않습니다. 모델은 warmup 중에도 계속 학습됩니다.

주요 지표: `graph_*_loss`, `graph_brier`, `graph_edge_state_std`(같은 goal의 p가
state에 따라 얼마나 변하는지; 정적 그래프면 0), `selected_new_goal_fraction`,
`selected_goal_success_rate`.

전체 비교(실험 1 + K=2/3/4 × leaf/noleaf × seed 0/1/2, 300M, warmup 50M)는
GPU queue로 실행합니다. W&B project `sea-dynamic3`, group=조건, name=조건-seed.

```bash
python scripts/run_dynamic_graph_suite.py --output runs/sea-dynamic3 --dry-run
python scripts/run_dynamic_graph_suite.py --output runs/sea-dynamic3 --gpus 0 1 2 3 --jobs-per-gpu 1
```

같은 명령을 다시 실행하면 완료된 run은 건너뛰고 미완료 run만 다시 시작합니다.

## 모듈

| 모듈 | 역할 |
|---|---|
| `train.py` | 기본 fix123 pipeline 및 명시적 FF 선택 |
| `streaming_discovery.py`, `episode_replay.py` | fresh prediction + complete-episode contrast |
| `discovery.py` | encoder loss/optimizer, constrained KMeans, artifact |
| `graph.py` | learned/oracle 공통 temporal graph 생성 및 축약 |
| `goal.py` | 분류, 목표 선택, 보상 및 oracle 선택 분기 |
| `dynamic_graph.py` | symbolic state split, high-level replay, Q learning 및 dynamic edge |
| `graph_planner.py` | 지도학습 option 모델(state-conditioned edge)과 K-hop planner |
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
