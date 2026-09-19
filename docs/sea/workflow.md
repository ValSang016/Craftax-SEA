# 실행 및 코드 구조

## 기본 실험: SEA (fix123)

프로젝트 루트에서 고정 환경 `/opt/conda/envs/craftax`를 사용합니다.

```bash
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cuda,cpu /opt/conda/envs/craftax/bin/python scripts/run_sea_experiments.py --output runs/sea-new --gpus 0 1 --seeds 0 1
```

기본 pipeline: collection PPO 200M → fresh encoder discovery 50M → 독립
clustering 표본 10k events → goal PPO 300M. 원본 actor 수와 `num_envs`는
구현 구조가 다르므로 같은 이름의 숫자로 취급하지 않습니다.

## 모듈

| 모듈 | 역할 |
|---|---|
| `train.py` | 기본 fix123 pipeline 및 명시적 FF 선택 |
| `streaming_discovery.py`, `episode_replay.py` | fresh prediction + complete-episode contrast |
| `discovery.py` | encoder loss/optimizer, constrained KMeans, artifact |
| `graph.py` | learned/oracle 공통 temporal graph 생성 및 축약 |
| `goal.py` | 분류, 목표 선택, 보상 및 oracle 선택 분기 |
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
