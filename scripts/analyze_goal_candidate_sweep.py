"""Compare beta 10 with beta 0.8 using matched training seeds, per condition."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from itertools import product
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy import stats

CONDITIONS = (
    ("ready-repeat", "① Unready 제외 · 완료 goal 허용"),
    ("all-repeat", "② Unready 포함 · 완료 goal 허용"),
    ("all-once", "③ Unready 포함 · 완료 goal 금지"),
)
EXPECTED_SEEDS = (0, 1, 2)


def paired_statistics(differences):
    differences = np.asarray(differences, dtype=float)
    n = differences.size
    result = dict(n_pairs=n, mean_difference=None, seed_sd=None,
                  ci95_low=None, ci95_high=None, paired_t_p=None,
                  exact_sign_flip_p=None)
    if not n:
        return result
    result["mean_difference"] = float(differences.mean())
    # Enumerate the paired label swaps; training updates are never replicates.
    signs = np.asarray(list(product((-1, 1), repeat=n)), dtype=float)
    permuted = (signs * differences).mean(axis=1)
    result["exact_sign_flip_p"] = float(np.mean(
        np.abs(permuted) >= abs(differences.mean()) - 1e-12
    ))
    if n < 2:
        return result
    sd = float(differences.std(ddof=1))
    result["seed_sd"] = sd
    half_width = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    result["ci95_low"] = float(differences.mean() - half_width)
    result["ci95_high"] = float(differences.mean() + half_width)
    if sd == 0:
        # A constant nonzero sample cannot validate a Gaussian variance model.
        result["paired_t_p"] = 1.0 if np.all(differences == 0) else None
    else:
        result["paired_t_p"] = float(stats.ttest_1samp(differences, 0).pvalue)
    return result


def holm_adjust(pvalues):
    """Keep the family size fixed, including any not-yet-testable conditions."""
    values = np.asarray([1.0 if p is None else p for p in pvalues])
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values))
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = previous
    return [None if p is None else float(adjusted[i]) for i, p in enumerate(pvalues)]


def collect_scores(root):
    manifest = json.loads((root / "manifest.json").read_text())
    rows = []
    for job in manifest["jobs"]:
        if job["state"] != "completed":
            continue
        with np.load(Path(job["output"]) / "goal_metrics_history.npz") as history:
            updates = len(history["episode_count"])
            if updates != 4577:
                raise ValueError(f"expected full 300M-step run, got {updates} updates: {job['output']}")
            count = history["achievement_success_count"][-458:].sum(axis=0)
            episodes = float(history["episode_count"][-458:].sum())
            if episodes <= 0 or not np.isfinite(count).all():
                raise ValueError(f"invalid episode statistics: {job['output']}")
            rates = count / episodes
            if rates.shape != (22,):
                raise ValueError("this comparison requires the 22-achievement oracle graph")
            rows.append(dict(condition=job["condition"], beta=job["beta"],
                seed=job["seed"], crafter_score=float(np.expm1(np.log1p(100 * rates).mean())),
                achievements_per_episode=float(rates.sum()), episodes=int(episodes)))
    return manifest, rows


def analyze(root):
    manifest, scores = collect_scores(root)
    lookup = {(r["condition"], r["beta"], r["seed"]): r for r in scores}
    pairs, comparisons = [], []
    for condition, label in CONDITIONS:
        paired = []
        for seed in EXPECTED_SEEDS:
            low = lookup.get((condition, 0.8, seed))
            high = lookup.get((condition, 10.0, seed))
            if low is None or high is None:
                continue
            difference = high["crafter_score"] - low["crafter_score"]
            paired.append(difference)
            pairs.append(dict(condition=condition, seed=seed,
                beta_0p8_score=low["crafter_score"], beta_10_score=high["crafter_score"],
                difference_10_minus_0p8=difference))
        comparisons.append(dict(condition=condition, label=label,
            status="full-three-seed-comparison" if len(paired) == 3 else "interim-incomplete",
            **paired_statistics(paired)))
    for field in ("paired_t_p", "exact_sign_flip_p"):
        adjusted = holm_adjust([row[field] for row in comparisons])
        for row, value in zip(comparisons, adjusted):
            row[field + "_holm"] = value
    output = root / "beta-comparison"
    output.mkdir(exist_ok=True)
    for name, rows in (("scores-by-seed.csv", scores), ("paired-differences.csv", pairs),
                       ("paired-tests.csv", comparisons)):
        if rows:
            with (output / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    result = dict(updated_at_kst=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
        completed_runs=len(scores), comparisons=comparisons,
        analysis_unit="one independently trained seed; matched between betas within condition",
        endpoint="22-achievement Crafter score, episodes ending in final 458 of 4577 training updates",
        alpha=0.05, multiplicity="Holm adjustment over three conditions, separately for each test family",
        stopping="descriptive interim updates; final inference only after all three paired seeds complete")
    (output / "statistics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    def fmt(value, digits=3):
        return "—" if value is None else f"{value:.{digits}f}"
    lines = ["# β=0.8과 β=10 비교", "",
        f"갱신: {result['updated_at_kst']}. 전체 {len(scores)}/{len(manifest['jobs'])}개 완료.", "",
        "차이는 β=10 − β=0.8의 Crafter score 차이다. 조건별 같은 seed를 짝지었다. "
        "학습 update나 episode를 독립 반복으로 취급하지 않았다. 조건들을 seed처럼 합치지 않았다.", "",
        "| 조건 | 짝지은 seed 수 | 평균 차이 | 차이의 95% CI | Paired t p | Holm p | Exact p | Exact Holm p |",
        "|---|---:|---:|---|---:|---:|---:|---:|"]
    for row in comparisons:
        interval = "—" if row["ci95_low"] is None else f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}]"
        lines.append(f"| {row['label']} | {row['n_pairs']} | {fmt(row['mean_difference'])} | {interval} | "
            f"{fmt(row['paired_t_p'],4)} | {fmt(row['paired_t_p_holm'],4)} | "
            f"{fmt(row['exact_sign_flip_p'],4)} | {fmt(row['exact_sign_flip_p_holm'],4)} |")
    lines += ["", "- Seed 한 쌍만 있을 때 paired t-test와 seed 변동에 대한 신뢰구간은 계산하지 않는다. "
        "한 쌍의 exact sign-flip p=1은 표본이 부족한 검정의 결과이며, 차이가 없다는 증거가 아니다.",
        "- Paired t-test는 seed별 차이의 정규성 가정에 의존한다. Seed 3개로는 이 가정을 평가하기 어렵다.",
        "- Exact paired sign-flip 검정은 귀무가설 아래 각 seed 내 beta label을 교환할 수 있다는 가정이다. "
        "3쌍의 양측 검정에서 가능한 최소 p-value는 0.25이므로, 이 표본 수로 exact p<0.05는 불가능하다.",
        "- 세 조건 비교에 Holm 보정을 적용했다. 표의 95% CI는 개별 비교의 paired t 구간이며 동시 신뢰구간은 아니다.",
        "- 중간 결과는 현황 확인용이다. 최종 판단은 세 seed가 모두 끝난 후에 한다. "
        "유의하지 않다는 결과는 성능이 동등하다는 결론과 다르다. 동등성 판단에는 사전에 정한 허용 차이가 필요하다.",
        "- 결과는 학습 rollout 기준이며 별도의 고정 evaluation seed에서 측정한 성능은 아니다.", ""]
    (output / "report.ko.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Beta comparison updated: {len(scores)} completed runs; pairs="
          + ",".join(str(row["n_pairs"]) for row in comparisons), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    analyze(parser.parse_args().root.resolve())
