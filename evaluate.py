# ============================================================
#  evaluate.py  -  STEP 3: check how well the LLM judge agrees
#  with the humans, for each dataset and each criterion.
#  Output: results/metrics_<dataset>.csv, results/disagreements_<dataset>.csv,
#          results/scatter_<dataset>.png, results/correlation_chart.png
#  Run:    python evaluate.py
# ============================================================
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

import config


def human_vs_human(human, criteria):
    """Average Spearman correlation between every pair of human raters.
    This tells us how much humans agree with EACH OTHER (a natural upper limit)."""
    raters = human["rater"].unique()
    result = {}
    for c in criteria:
        table = human.pivot_table(index="id", columns="rater", values=c)
        pairs = [spearmanr(table[a], table[b], nan_policy="omit")[0]
                 for a, b in itertools.combinations(raters, 2)]
        result[c] = sum(pairs) / len(pairs) if pairs else float("nan")
    return result


def evaluate(exp):
    s = config.EXPERIMENTS[exp]
    C = s["criteria"]
    judge_file = f"results/judge_{exp}.csv"
    if not (os.path.exists(judge_file) and os.path.exists(s["human"])):
        print(f"[{exp}] skipped: need {judge_file} and {s['human']}")
        return None
    judge = pd.read_csv(judge_file)
    human = pd.read_csv(s["human"])
    if human.empty:
        print(f"[{exp}] skipped: no human ratings yet (run python rate.py)")
        return None

    # Average the human raters for each sample -> one "human score" per sample.
    human_avg = human.groupby("id")[C].mean().reset_index()
    data = judge.merge(human_avg, on="id", suffixes=("_judge", "_human"))
    inter = human_vs_human(human, C)
    print(f"\n=== {s['name']}: {len(data)} samples, {human['rater'].nunique()} human rater(s) ===")

    rows = []
    for c in C:
        j, h = data[f"{c}_judge"], data[f"{c}_human"]
        rows.append({
            "dataset": s["name"], "criterion": c, "n": len(data),
            "pearson": pearsonr(j, h)[0],            # linear agreement (-1..1)
            "spearman": spearmanr(j, h)[0],          # ranking agreement (-1..1)
            "kappa": cohen_kappa_score(j, h.round().astype(int), weights="quadratic"),
            "mae": (j - h).abs().mean(),             # average gap in points
            "bias": (j - h).mean(),                  # + means judge is more generous than humans
            "within_1": ((j - h).abs() <= 1).mean() * 100,   # % of samples within 1 point
            "human_human_spearman": inter[c],
        })
    metrics = pd.DataFrame(rows).round(3)
    metrics.to_csv(f"results/metrics_{exp}.csv", index=False)
    print(metrics.drop(columns=["dataset"]).to_string(index=False))

    # Biggest disagreements -> useful examples for the report/presentation.
    samples = pd.read_json(s["samples"])
    data["total_gap"] = sum((data[f"{c}_judge"] - data[f"{c}_human"]).abs() for c in C)
    worst = data.sort_values("total_gap", ascending=False).head(5).merge(samples, on="id")
    worst.to_csv(f"results/disagreements_{exp}.csv", index=False)

    # Scatter plots (each dot is one response; small jitter so dots don't hide each other).
    fig, axes = plt.subplots(1, len(C), figsize=(3 * len(C), 3))
    for ax, c in zip(axes, C):
        jitter = (pd.Series(range(len(data))) % 5 - 2) * 0.04
        ax.scatter(data[f"{c}_human"] + jitter, data[f"{c}_judge"] - jitter, alpha=0.5)
        ax.plot([1, 5], [1, 5], "--", color="grey")   # perfect-agreement line
        ax.set_title(f"{c.capitalize()} ({s['name']})", fontsize=10)
        ax.set_xlabel("Human score")
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(0.5, 5.5)
    axes[0].set_ylabel("LLM judge score")
    plt.tight_layout()
    plt.savefig(f"results/scatter_{exp}.png", dpi=200)
    plt.close()
    return metrics


def main():
    os.makedirs("results", exist_ok=True)
    all_metrics = [m for m in (evaluate(e) for e in config.EXPERIMENTS) if m is not None]
    if not all_metrics:
        return
    m = pd.concat(all_metrics)
    m.to_csv("results/metrics_all.csv", index=False)

    # One summary chart: Spearman per criterion for each dataset (+ human-human reference).
    crit = [c for c in config.CRITERIA if c in set(m.criterion)]
    bars = [(f"Judge vs humans - {name}", m[m.dataset == name].set_index("criterion")["spearman"])
            for name in m.dataset.unique()]
    custom = m[m.dataset == config.EXPERIMENTS["custom"]["name"]]
    if not custom.empty and custom.human_human_spearman.notna().any():
        bars.append(("Human vs human - Our set",
                     custom.set_index("criterion")["human_human_spearman"]))
    width = 0.8 / len(bars)
    plt.figure(figsize=(6.5, 3.4))
    for k, (label, values) in enumerate(bars):
        xs = [i + (k - (len(bars) - 1) / 2) * width for i in range(len(crit))]
        plt.bar(xs, [values.get(c, float("nan")) for c in crit], width=width, label=label)
    plt.xticks(range(len(crit)), [c.capitalize() for c in crit])
    plt.axhline(0, color="black", linewidth=0.6)
    plt.ylabel("Spearman correlation")
    plt.ylim(min(0, m.spearman.min() - 0.1), 1)
    plt.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    plt.title("Agreement of the LLM judge with human ratings")
    plt.tight_layout()
    plt.savefig("results/correlation_chart.png", dpi=200)
    plt.close()
    print("\nSaved metrics, disagreements and charts in the results/ folder.")


if __name__ == "__main__":
    main()
