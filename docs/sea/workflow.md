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

대표 조건 `ready-repeat`=(false,true), `all-repeat`=(true,true),
`all-once`=(true,false)에 대해 β=0/0.8/10, seed=0/1/2를 실행하려면:

```bash
/root/miniconda3/envs/craftax/bin/python scripts/run_goal_candidate_sweep.py \
  --output runs/oracle-goal-eligibility-new --gpus 0 1 2 3
```

Runner는 기존 β=0.8/seed0 설정을 기준으로 300M step씩 총 27개를 만들고,
GPU별 한 process를 실행합니다. 각 완료 직후 대기 중인 다음 run을 배정합니다.
실행 source와 graph를 output 아래에 복사해 고정하며, `manifest.json`에 전체 명령과
source hash를, `status.json`에 진행 상태를 저장합니다. W&B는 로컬 offline 모드입니다.
완료 후 `final-window.csv/json`에 마지막 458 update의 episode 집계 성능을 생성합니다.
기존 output을 덮어쓰지 않습니다. `--dry-run`은 학습 없이 실행 계획만 생성합니다.

Structural-prior/selector ablation은 아래처럼 먼저 계획만 생성합니다.

```bash
/root/miniconda3/envs/craftax/bin/python scripts/run_goal_candidate_sweep.py \
  --suite selector-ablations --dry-run \
  --output runs/oracle-selector-ablations-new --gpus 0 1 2 3
```

이 suite는 모두 unready=true/completed=false로 고정하고, lambda=0인 residual
beta=0.8/10 및 lambda=1인 base-q beta=0.5/0.3/0.2를 seed=0/1/2에 배정합니다.
총 15개이며 기존 run과 같은 300M step, warmup 30M step을 사용합니다.
`--execute`가 없으면 이 suite는 항상 준비만 하고 학습 process를 시작하지 않습니다.
검토 및 사용자 승인 후에는 준비한 source와 명령을 그대로 사용합니다.

```bash
/root/miniconda3/envs/craftax/bin/python scripts/run_goal_candidate_sweep.py \
  --output runs/oracle-selector-ablations-new --start-prepared --execute
```

준비한 계획은 모든 job이 queued이고 run output이 아직 없을 때만 시작할 수 있습니다.
source/graph hash를 다시 확인하며, 새 실험의 최종 집계에는 lambda와 signal도 기록합니다.
기존 27개 후보 실험용 beta 유의성 분석은 이 15개 suite에 적용하지 않습니다.

Full Q 선택 비교는 `--suite full-q`로 준비합니다. Unready 포함·완료 goal 재선택 금지,
lambda=1로 고정하고 beta=0.2/0.3/0.5 × seed=0/1/2, 총 9개를 만듭니다.
`--gpus 0 1`로 지정한 계획은 시작·재시작 시 GPU를 따로 지정하지 않아도 저장된
0·1번 GPU 목록을 상속합니다. 이 suite도 실제 시작에는 `--execute`가 필요합니다.

`evaluate_selector_checkpoints.py --training-sweep RUN_ROOT --gpus 0 1 --output EVAL_ROOT`는
해당 sweep의 완료 checkpoint 전부를 원래 고정 source로 평가합니다. Selector의
residual/base-q/full-q 모드를 각각 적용하며, 기존 평가와 같은 환경 seed 1,000,000부터
1,024개 episode를 끝까지 실행합니다. 학습 update와 optimistic reset은 없습니다.

완료된 run을 보존하면서 실패한 run만 다시 실행하려면 같은 output에
`--resume-failed`를 붙입니다. 원래 고정한 source/graph hash를 검증하고,
실패한 시도의 로그와 기존 manifest를 `retry-attempts/`에 보존한 뒤 실패한
run만 처음부터 실행합니다. `--resume-failed --dry-run`은 파일을 변경하지 않고
재실행할 대상을 검증합니다. GPU가 사라지면 새 run 배정을 GPU 복구까지 보류합니다.

각 run이 끝나면 `beta-comparison/`도 갱신합니다. β=10−0.8의 마지막 구간
Crafter score를 같은 training seed끼리 비교하며 paired t-test, 차이의 95% CI,
exact paired sign-flip test와 세 조건에 대한 Holm 보정을 기록합니다.
세 seed가 모두 끝나기 전에는 중간 집계로 표시합니다. Episode나 update를
독립 seed로 취급하지 않습니다. Seed 3쌍의 양측 exact 검정 최소 p-value는
0.25이며, 유의하지 않다는 결과만으로 성능 동등성을 주장할 수 없습니다.

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
