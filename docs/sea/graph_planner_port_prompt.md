# 작업: 상태 조건부 achievement 그래프(option model) + K-hop planner를 이 코드베이스에 이식

## 0. 배경과 절대 지켜야 할 원칙

Craftax-Classic(symbolic)에서 goal-conditioned PPO-GRU worker 위에 high-level goal 선택기를 얹는 연구다.
우리 contribution은 **그래프 학습**이다.

- 노드는 achievement 22개다(unknown 노드 없음).
- edge g_i → g_j는 **p(g_j 성공 | s_t, c_t, g_i)**다. c_t는 이번 에피소드에서 완료한 achievement 벡터이고, g_i는 현재 노드(마지막으로 달성한 achievement)다.
- 이 edge를 신경망(option model F)이 **순수 지도학습**으로 배운다. **그래프 쪽에는 RL을 쓰지 않는다.** return, bootstrapping, TD target, policy gradient, Q-learning 모두 금지다. RL은 low-level worker(PPO)에만 쓴다.
- F를 K hop 굴려서 경로 점수 J를 계산하고, 첫 goal을 softmax로 샘플링해 worker에게 준다.
- 생존 제약이 핵심 주제다. **체력(health) 관측을 빼거나 immortal로 바꾸지 마라.** 실험은 mortal이다.

원본 구현은 `Craftax-SEA` 저장소의 `dynamic-graph3` 브랜치에 있다. 아래 "가져올 것" 목록을 읽고 이 코드베이스 구조에 맞게 이식하라.

**가져오지 말 것.** 원본 SEA(pairlab/iclr-23-sea) 학습 방식을 재현하려고 넣은 코드다.

- `goal.py`의 `per_node_category_weights`와 `regenerate_on_any_event`, `--sea-category-weights`와 `--sea-regenerate` 플래그
- `--non-target-achievement-coefficient none`, `sea-orig` 조건(원본 SEA 보상 +1/+0.1)
- warmup이 SEA run과 비트 단위로 같도록 맞춘 RNG 소비 순서(`fold_in(goal_key, 1)`, `fold_in(key, 2)`)와 그 회귀 테스트
- `frozen_eval.py`의 legacy 규칙 fallback
- `run_dynamic_graph_suite.py`의 `sea` / `sea-orig` 조건

warmup에는 **이 코드베이스에 이미 있는 goal 선택기(meta-controller)를 그대로** 쓴다.

## 1. 가져올 것 (원본 파일 기준)

| 원본 | 내용 |
|---|---|
| `craftax/sea/graph_planner.py` | 전부: `OptionModel`, `GraphPlannerState`, loss/update, `plan_path_scores`, `sample_from_scores`, `choose_planner_goals`, `prerequisites_met`, `GRAPH_METRIC_NAMES` |
| `craftax/sea/dynamic_graph.py` | `DynamicGraphConfig`의 planner 필드와 검증, replay buffer의 `died` 필드, `get_goal_candidates`의 `allow_unready_goals` / `allow_completed_goals`, `high_level_option_done` |
| `craftax/sea/ppo_rnn.py` | `use_planner` 분기 전체: 초기화, goal 선택, option 종료와 replay 기록, F 업데이트, `frozen_params` 모드, 추가 metric |
| `craftax/sea/goal.py` | `repeat_target_coefficient`(반복 target 보상), **현재 노드 갱신 규칙**(아래 4.4) |
| `craftax/sea/env.py` | `track_achievement_events`: 이미 완료한 achievement를 다시 해도 이벤트로 잡는다(재선택 goal의 성공 판정에 필요) |
| `craftax/sea/train.py` | planner 관련 CLI 플래그(아래 6) |
| `craftax/sea/frozen_eval.py`, `scripts/evaluate_frozen.py` | 고정 checkpoint 평가 |
| `scripts/run_dynamic_graph_suite.py` | 다중 GPU 큐 runner. `K<h>-<leaf|noleaf>-r<x>[-p<none|leaf|all>][-nore]` 조건만 남긴다 |
| `tests/sea/test_graph_planner.py` | SEA 재현 테스트를 뺀 나머지 |

## 2. Option model F

입력: x = (s_t, c_t, g_i). s_t는 symbolic 관측 1339차원이고, 마지막 16차원(1323:1339)이 inventory 12 + intrinsic 4(health, food, drink, energy)다.

```
Encoder E:   [obs ‖ c_t(22) ‖ onehot(g_i, 23)] → Dense 512 → ReLU → 256 → ReLU → 128 → LayerNorm = z
Transition:  [z ‖ Embed(g_j, 32)] → Dense 256 → ReLU → 256 → ReLU = h
  success head    h → 1     p = σ(·)              # edge 확률
  duration head   h → 1     log τ                 # 성공까지 걸린 step
  death head      h → 1     σ(·)                  # option 도중 사망 확률
  completed head  h → 22    c' logits             # 성공 후 완료 벡터
  next latent     z' = LayerNorm(z + Dense128(h)) # 다음 hop 입력
Decoder:     z → Dense 256 → ReLU → 16            # inventory + intrinsic 복원
```

### Loss (전부 지도학습)

| 항목 | 손실 | 가중치 | 샘플 |
|---|---|---|---|
| success | BCE(p, success) | 1 | 전체 |
| death | BCE(d, died) | 1 | 전체 |
| duration | MSE(log τ, log max(duration, 1)) | 0.1 | 성공만 |
| completed | BCE(c', next_completed) 평균 | 1 | 성공만 |
| consistency | MSE(z', stop_grad(E_target(s', c', g_i'))) | 1 | 성공만 |
| aux | MSE(Dec(z), aux_t) + MSE(Dec(z'), aux_{t'}) | 10 | 앞 항은 전체, 뒤 항은 성공만 |

- E_target은 EMA(0.995) target 파라미터다.
- 최적화: Adam lr 3e-4, global norm clip 1.0.
- 성공 샘플에서만 학습하는 head가 있는 이유: planner는 각 hop을 "앞 goal이 성공했다"는 가정 위에서 이어 붙이기 때문이다.

## 3. 데이터 (high-level replay)

샘플 하나는 option 하나다. option은 goal을 받은 시점부터 끝날 때까지다.

- **저장 내용:** `(obs, completed, current_node, goal) → success, duration, died, next_obs, next_completed, next_current_node`
- **option 종료(50M 이후):** target 성공, duration ≥ 300(timeout), 에피소드 종료 중 하나.
- **라벨:**
  - timeout과 사망은 `success=False`로 **반드시 기록**한다(실패 데이터).
  - `died`는 env info의 `health_done`이다.
  - `success`는 `goal_done & duration ≤ 300`이다.
- **warmup:** 이 코드베이스의 기존 선택기가 goal을 바꾼 순간을 option 종료로 본다. 단, 성공·에피소드 종료·timeout이 아닌 다른 이유로 goal이 교체됐다면 그 option은 결과를 모르는(censored) 샘플이다. **기록하지 않는다.**
- **buffer:** 용량 100k, batch 256, 최소 1024개가 쌓인 뒤부터 학습한다.
- **업데이트 횟수:** PPO update마다 `round(이번 rollout에서 실제로 replay에 기록된 transition 수 × 4 / 256)`번이다. option이 끝난 횟수가 아니라 **기록된 수**로 센다. 이걸 틀리면 warmup에서 약 3.5배 과학습한다.

## 4. Planner

### 4.1 경로 점수 (K=2가 기본)

$$J=\sum_{k=1}^{K}\gamma^{T_k}\Big(\prod_{i\le k}p_i\Big)\mathbb 1[g_k\text{ 미완료}] \;+\; \underbrace{\gamma^{T_K}\Big(\prod p\Big)\sum_j\gamma^{\tau_j}p_j\,\mathbb 1[g_j\text{ 미완료}]\,\mathbb 1[g_j\in\text{후보}]}_{\text{leaf 항, planner\_leaf=True일 때만 (V=O)}}$$

- τ = clip(exp(log τ), 1, 300)이고, T_k는 누적 τ, γ = 0.99다.
- hop마다 예측 완료 벡터를 갱신한다: `path_completed |= onehot(g_k) | (σ(c') > 0.5)`. 다음 hop 입력은 z'다.
- **hop 1:** 후보 goal 전부를 펼친다.
- **hop 2 이상:** 첫 goal별로 beam(폭 4)을 유지하고 부분 점수로 top-k한다.
- 첫 goal마다 최고 경로 점수를 반환한다. shape은 [N, 22]다.
- **선택 옵션 prune:**
  - `leaf`: leaf 항을 static 선행조건을 만족하는 미완료 goal(최대 13개)에만 계산한다.
  - `all`: 모든 hop에서 선행조건 미충족 goal을 막는다.
  - 기본값은 `none`이다.

### 4.2 Goal 샘플링
`logits = J / max(max_candidate J, 1e-6) / T`, T = 0.1, 비후보는 -inf로 두고 categorical 샘플링한다. J는 대개 1보다 훨씬 작아서 raw J에 softmax를 쓰면 거의 uniform이 된다. 그래서 최댓값으로 정규화한다.

탐험은 이 softmax에서만 나온다. 50M 이후에는 meta-controller나 ε-greedy를 섞지 않는다.

### 4.3 후보 집합
- `allow_unready_goals=True`: 선행조건이 안 된 goal도 후보에 둔다. "선행조건이 안 되면 p가 낮다"는 것을 F가 배운다.
- `allow_completed_goals`:
  - True(재선택 O)면 이미 완료한 goal도 후보다. 물 마시기나 iron 다시 모으기가 가능해야 한다. **기본값으로 써라.**
  - False(재선택 X)면 iron pickaxe를 0%로 만든다. iron sword 제작에 iron을 쓴 뒤 다시 모으라는 goal을 줄 수 없기 때문이다.
- 재선택 O일 때는 env의 `track_achievement_events=True`가 필요하다. 반복 달성도 target 성공으로 잡아야 하기 때문이다.

### 4.4 현재 노드 g_i
현재 노드는 **처음 달성한 achievement 또는 target 성공**일 때만 갱신한다. 우연히 반복한 achievement(나무를 다시 캠 등)로는 바꾸지 않는다.

### 4.5 계산 예산
- 한 step에서 새 goal이 필요한 env 중 최대 128개만 K-hop으로 계산한다(무작위 우선순위).
- 넘친 env는 1-hop으로 처리한다. 1-hop fallback은 `lax.cond`로 감싸서 overflow가 있을 때만 실행한다.
- overflow 비율을 metric으로 남긴다.

## 5. 학습 루프에 끼울 때 반드시 지킬 것

1. **PPO transition의 goal은 그 step에 실제로 따른 goal이어야 한다.**
   - option 종료 step에서 다음 goal을 뽑더라도 `next_objective`에 따로 저장한다. transition을 만든 뒤에 반영한다.
   - 이전에 다음 goal로 덮어쓴 채 transition을 만드는 버그가 있었다. 보상을 받은 step의 credit이 엉뚱한 goal에 붙어서 성능이 크게 떨어졌다.
2. **PPO bootstrap:**
   - 50M 이후: `env_done | option_done`
   - warmup: 기존 선택기의 규칙을 따른다.
3. **일정:** 총 300M env step이고, warmup은 50M이다(`--dynamic-warmup-steps`). warmup 동안에도 F는 replay를 쌓고 학습한다. 50M 이후 goal 선택은 planner만 한다.
4. **frozen 모드:** `frozen_params={"policy", "option_model", "option_model_target"}`를 넘기면 같은 rollout과 goal 선택 코드를 돌리되 PPO epoch와 F 업데이트를 건너뛴다. 고정 평가에 쓴다.
5. **checkpoint:** `goal_policy.msgpack`, 그리고 `high_level_q.msgpack`에 `{option_model, option_model_target}`를 저장한다.
6. **metric:**
   - graph_*_loss, graph_brier, graph_edge_state_std, graph_mean_success_probability
   - selected_goal_distribution, selected_goal_success_rate, selected_new_goal_fraction
   - planner_overflow_fraction, high_level_recorded_count
   - 에피소드별 achievement 성공률, Crafter score(22개, eat_plant 제외 21개)

## 6. CLI와 기본값

```
--high-level-controller graph-planner
--planner-horizon 2  --planner-leaf {true,false}  --planner-beam-width 4
--planner-max-decisions 128  --planner-gamma 0.99  --planner-temperature 0.1
--planner-prune {none,leaf,all} (none)  --planner-prune-width 13
--model-hidden-size 256  --model-latent-size 128  --model-target-ema 0.995
--high-level-q-learning-rate 3e-4  --high-level-q-replay-ratio 4  --high-level-goal-timeout 300
--allow-unready-goals true  --allow-completed-goals {true,false}
--goal-success-coefficient 3  --non-target-achievement-coefficient 1
--repeat-target-coefficient 0  --exploration-reward-coefficient 0.1
```

환경 설정: `--symbolic --no-new-tasks --immortal false --idle-timeout 10000 --combat-damage-mode cumulative`. 이 코드베이스에 없는 옵션이면 동등한 설정을 찾아라.

## 7. 실험 조건과 평가

| 이름 | K | leaf(V) | 재선택 |
|---|---|---|---|
| K2-leaf-r0 | 2 | O | O |
| K2-leaf-r0-nore | 2 | O | X |
| K2-noleaf-r0 | 2 | X | O |

- seed는 0, 1, 2이고 seed 순서대로 GPU 4장에 큐로 돌린다.
- wandb: entity `zaezae01`, group은 지정한 이름, job_type은 조건 이름.
- **평가 1(주 지표):** 마지막 checkpoint를 고정하고, 학습에 쓰지 않은 env seed 84001–84005로 seed당 약 1M step(256 env × 62 update × 63 step)을 돌린다. Score22와 Score21의 평균 ± std를 낸다.
- **평가 2:** 학습 마지막 100 PPO update에서 끝난 에피소드로 계산한 Crafter score.
- **Crafter score:** `exp(mean(log(1 + 100·rate))) − 1`. Score22는 22개 전부, Score21은 eat_plant를 뺀 값이다. 에피소드가 약 300 step이라 eat_plant(600 step 숙성)는 사실상 불가능하다.

## 8. 검증 (테스트로 작성)

1. 손으로 만든 F 출력에서 K=2가 K=1보다 먼 미완료 goal의 가치를 반영하는지, leaf 항 on/off가 점수를 바꾸는지 확인한다.
2. overflow env가 1-hop으로 처리되고 overflow mask가 맞는지 확인한다.
3. 지도학습 update 몇 번으로 success loss가 줄어드는지 확인한다.
4. `sample_from_scores`가 J를 상수배해도 같은 분포를 내는지(scale-free) 확인한다.
5. prune `leaf`, `all`이 선행조건을 올바르게 쓰는지 확인한다.
6. frozen 모드에서 가중치가 바뀌지 않는지 확인한다.
7. option 종료 step의 PPO transition goal이 그 step에 따른 goal인지 확인한다.
8. timeout과 사망이 `success=False`로 replay에 들어가는지, warmup에서 censored option이 빠지는지 확인한다.
9. CLI 플래그가 config까지 전달되는지 확인한다.

## 9. 알려진 한계 (지금은 고치지 말고, 이식 후 다음 작업으로)

- **death head는 학습만 하고 J에는 쓰지 않는다.** 다음 단계로 `prod p_i (1 − d_i)`를 옵션으로 넣는 것을 검토 중이다.
- **diamond 성공률이 낮다.** 이전 결과에서 SEA baseline은 diamond 16–17%, planner는 0–3%였다. 쉬운 17개는 모든 조건이 95% 이상이다. 원인은 셋 중 하나로 보고 진단 예정이다.
  - planner가 diamond를 선택하지 않음
  - worker가 diamond goal을 수행하지 못함
  - 덜 시도된 edge의 p를 F가 과소평가함
