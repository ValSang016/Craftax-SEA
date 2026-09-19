# 원본 ICLR-SEA와 fix123의 대응

직접 비교 대상: `/workspace/iclr-23-sea-main/torchbeast/`와 현재 `craftax/sea/`.
fix123는 현재 수행한 비지도 SEA 변형 중 기준으로 삼을 버전입니다.
Oracle·GT 실험은 정답 라벨을 제공한 진단 실험이고, K=16과 FF는 별도 변형입니다.

| 요소 | ICLR-SEA 로컬 원본 | SEA fix123 |
|---|---|---|
| 환경 | Crafter fork | Craftax-Classic + SEA wrapper |
| 정책 학습 | TorchBeast IMPALA/V-trace, LSTM | JAX PPO/GAE, GRU |
| 정책 optimizer | RMSprop | Adam |
| prediction 데이터 | actor가 생성한 rollout | 고정 collection policy의 fresh transition, 1회 사용 |
| contrast 데이터 | episode event buffer, 128 groups, group당 최대 8 events | 완전한 episode FIFO 256개, 같은 group/event 크기 |
| prediction target | abs(reward) > .02 | event_count > 0 |
| prediction loss | 0.5 × squared error 합계 | 0.5 × squared error 평균 |
| contrast loss | 128 groups의 -det(K) 합계 | 유효 groups의 -det(K) 평균 |
| total loss | prediction_sum + 20 × contrast_sum | prediction_mean + 1 × contrast_mean |
| encoder optimizer | Adam lr=1e-4, eps=1e-5 | Optax Adam lr=1e-4, 기본 eps=1e-8 |
| encoder global norm clipping | 40 | 1 |
| 정책 global norm clipping | 40 | 1 |
| encoder 로깅 | prediction 합계, contrast 평균, total 합계 | prediction/contrast 평균, total 평균 스케일; clipping 전 norm |

## Loss 환산과 한계

기준 배치 2560 prediction terms, 128 contrast groups에서
`20 × 128 / 2560 = 1`. 두 항의 상대 가중치는 맞지만 전체 loss 스케일은
원본의 1/2560입니다. Gradient clipping과 Adam epsilon이 있으므로
동일한 optimizer update를 보장하지 않습니다. 배치가 작거나 contrast가
아직 활성화되지 않은 구간까지 모든 조건이 수치적으로 동일한 것도 아닙니다.

초기 Craftax 포팅(original, fix1)은 실제 prediction batch=256을 사용해
`20 × 128 / 256 = 10`을 적용했습니다. fix123는 기준 상대 가중치를 고정합니다.
이번 정리는 가중치를 새로 튜닝하거나 원본 clipping 40으로 변경하지 않습니다.

## 그래프와 oracle 실험

기본 경로는 GT 없이 embedding → constrained KMeans → temporal graph입니다.
`order[A,B]/count[B] > .97`, `order[B,A]/order[A,B] < .001` 후 transitive
reduction을 적용합니다. 관측된 순서는 필수조건이나 인과관계의 증명이 아닙니다.
Oracle은 실제 achievement ID를 이벤트 분류에 쓰므로 별도의 상한/진단 조건입니다.

## 원본 코드 근거

- `polybeast_learner.py:359–376`: prediction target 및 합계
- `polybeast_learner.py:442–499`: contrast 합계, 로깅 평균, coefficient 20, clipping
- `polybeast_learner.py:754–779`: 정책 RMSprop / encoder Adam
- `configs/default.yaml:60–74`: batch 32, unroll 80, clip 40
- 현재 `craftax/sea/discovery.py`: 평균 loss, 기준 coefficient, Optax chain

이 표는 확인한 주요 대응이며 완전한 구현 동등성 인증이 아닙니다. 특히 환경의
전투·인벤토리 동작까지 완전히 일치한다고 가정하지 않습니다.
