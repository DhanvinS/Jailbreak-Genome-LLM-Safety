"""
Visualization generator for Jailbreak Genome Lab.
Reads run_history.jsonl + result JSONs, saves timestamped PNGs to data/results/plots/.
Each plot also saves a latest_<name>.png for the dashboard to load.
Run standalone: python src/visualize.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"
LOG_PATH    = RESULTS_DIR / "run_history.jsonl"


def _savefig(fig, name: str) -> Path:
    """Save fig as timestamped + latest PNG. Returns timestamped path."""
    import matplotlib.pyplot as plt
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stamped = PLOTS_DIR / f"{name}_{ts}.png"
    latest  = PLOTS_DIR / f"latest_{name}.png"
    fig.savefig(stamped, dpi=150, bbox_inches="tight")
    fig.savefig(latest,  dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {stamped.name}")
    return stamped


def load_history() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    return [json.loads(l) for l in LOG_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


# ── Individual plot functions ──────────────────────────────────────────────────

def plot_auroc_history() -> Path | None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    entries = [e for e in load_history() if e["module"] == "module5_evaluate"
               and "logreg_auroc_unseen" in e.get("metrics", {})]
    if not entries:
        print("  No Module 5 runs yet — skipping AUROC history.")
        return None

    labels  = [f"Run {i+1}\n{e['timestamp'][:10]}" for i, e in enumerate(entries)]
    aurocs  = [e["metrics"]["logreg_auroc_unseen"] for e in entries]

    fig, ax = plt.subplots(figsize=(max(6, len(entries) * 1.5), 4))
    bars = ax.bar(labels, aurocs, color="#4C72B0", width=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.87, color="gray", linestyle="--", linewidth=1, label="Baseline (0.87)")
    ax.set_title("Open-Set AUROC Across Runs\n(held-out: encoding + multilingual)")
    ax.set_ylabel("AUROC")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _savefig(fig, "auroc_history")


def plot_f1_history() -> Path | None:
    import matplotlib.pyplot as plt
    import numpy as np

    entries = [e for e in load_history() if e["module"] in ("module4_classify", "module5_evaluate")]
    m4 = [e for e in entries if "logreg_f1" in e.get("metrics", {}) or "xgboost_f1" in e.get("metrics", {})]
    if not m4:
        print("  No Module 4 runs yet — skipping F1 history.")
        return None

    labels  = [f"Run {i+1}\n{e['timestamp'][:10]}" for i, e in enumerate(m4)]
    logreg  = [e["metrics"].get("logreg_f1") or e["metrics"].get("logreg_f1_seen") for e in m4]
    xgboost = [e["metrics"].get("xgboost_f1") for e in m4]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(m4) * 2), 4))
    b1 = ax.bar(x - width/2, logreg,  width, label="LogReg",  color="#4C72B0")
    b2 = ax.bar(x + width/2, [v if v is not None else 0 for v in xgboost],
                width, label="XGBoost", color="#DD8452")
    ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_title("Classifier F1 Across Runs\n(macro-averaged)")
    ax.set_ylabel("Macro F1")
    ax.legend()
    fig.tight_layout()
    return _savefig(fig, "f1_history")


def plot_corpus_growth() -> Path | None:
    import matplotlib.pyplot as plt

    entries = [e for e in load_history()
               if "db_total" in e.get("metrics", {})]
    if len(entries) < 2:
        print("  Not enough corpus data points — skipping growth chart.")
        return None

    labels = [f"{e['module'].replace('module','M')}\n{e['timestamp'][:10]}"
              for e in entries]
    totals = [e["metrics"]["db_total"] for e in entries]

    fig, ax = plt.subplots(figsize=(max(6, len(entries) * 1.4), 4))
    ax.plot(labels, totals, marker="o", color="#4C72B0", linewidth=2)
    for i, (l, v) in enumerate(zip(labels, totals)):
        ax.annotate(f"{v:,}", (i, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_title("Corpus Growth Over Time")
    ax.set_ylabel("Total Prompts in DB")
    ax.set_ylim(0, max(totals) * 1.15)
    fig.tight_layout()
    return _savefig(fig, "corpus_growth")


def plot_mutation_distance_curve() -> Path | None:
    import matplotlib.pyplot as plt

    report_path = RESULTS_DIR / "eval_report.json"
    if not report_path.exists():
        print("  eval_report.json not found — skipping mutation distance curve.")
        return None

    report = json.loads(report_path.read_text(encoding="utf-8"))
    curve  = report.get("mutation_distance_curve", [])
    if not curve:
        print("  No mutation distance curve data.")
        return None

    hops  = [p["hops"] for p in curve]
    rates = [p["detection_rate"] for p in curve]
    ns    = [p["n"] for p in curve]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hops, rates, marker="o", color="#4C72B0", linewidth=2)
    for h, r, n in zip(hops, rates, ns):
        ax.annotate(f"{r:.2f}\n(n={n})", (h, r),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8)
    ax.set_xticks(hops)
    ax.set_xlabel("Mutation Hops from Seed")
    ax.set_ylabel("Detection Rate")
    ax.set_ylim(0, 1.15)
    ax.set_title("Detection Rate vs Mutation Distance\n(how many hops before classifier misses it)")
    fig.tight_layout()
    return _savefig(fig, "mutation_distance_curve")


def plot_fitness_evolution() -> Path | None:
    import matplotlib.pyplot as plt

    report_path = RESULTS_DIR / "evolution_report.json"
    if not report_path.exists():
        print("  evolution_report.json not found — run Module 7 first.")
        return None

    report  = json.loads(report_path.read_text(encoding="utf-8"))
    history = report.get("generation_history", [])
    if not history:
        print("  No generation history in evolution report.")
        return None

    gens   = [h["generation"] for h in history]
    means  = [h["mean_fitness"] or 0 for h in history]
    maxes  = [h["max_fitness"] or 0 for h in history]
    rates  = [h["compliance_rate"] or 0 for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(gens, means,  marker="o", label="Mean fitness",   color="#4C72B0", linewidth=2)
    ax1.plot(gens, maxes,  marker="s", label="Max fitness",    color="#DD8452", linewidth=2, linestyle="--")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness (0-1)")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Fitness Across Generations\n(higher = more compliant / harder to refuse)")
    ax1.legend(fontsize=8)

    ax2.bar(gens, rates, color="#55A868", width=0.5)
    for g, r in zip(gens, rates):
        ax2.text(g, r + 0.01, f"{r:.2f}", ha="center", fontsize=9)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Compliance Rate (fitness ≥ 0.5)")
    ax2.set_ylim(0, 1.1)
    ax2.set_title("% Prompts with Partial/Full Compliance\nPer Generation")

    fig.tight_layout()
    return _savefig(fig, "fitness_evolution")


# ── Run all ────────────────────────────────────────────────────────────────────

def save_all() -> list[Path]:
    print("Generating visualizations...")
    paths = []
    for fn in [plot_auroc_history, plot_f1_history, plot_corpus_growth,
               plot_mutation_distance_curve, plot_fitness_evolution]:
        result = fn()
        if result:
            paths.append(result)
    print(f"\nDone. {len(paths)} plots saved to {PLOTS_DIR}")
    return paths


if __name__ == "__main__":
    save_all()
