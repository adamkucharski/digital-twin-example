"""Evaluate simulation accuracy — MAE and correlation."""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"


def main():
    global RESULTS_DIR
    if "--test" in sys.argv:
        RESULTS_DIR = BASE_DIR / "results_test"

    df = pd.read_csv(RESULTS_DIR / "all_results.csv")
    df = df.dropna(subset=["actual_pct", "simulated_pct"])

    summary_rows = []
    for sid, group in df.groupby("survey_id"):
        actual = group["actual_pct"].values
        simulated = group["simulated_pct"].values

        mae = np.mean(np.abs(actual - simulated))

        if len(actual) > 2 and np.std(actual) > 0 and np.std(simulated) > 0:
            corr = np.corrcoef(actual, simulated)[0, 1]
        else:
            corr = np.nan

        summary_rows.append({
            "survey_id": sid,
            "n_items": len(group),
            "mae_pp": round(mae, 1),
            "correlation": round(corr, 3) if not np.isnan(corr) else None
        })

    summary = pd.DataFrame(summary_rows).sort_values("mae_pp")

    overall_mae = np.mean(np.abs(df["actual_pct"].values - df["simulated_pct"].values))
    overall_actual = df["actual_pct"].values
    overall_sim = df["simulated_pct"].values
    if np.std(overall_actual) > 0 and np.std(overall_sim) > 0:
        overall_corr = np.corrcoef(overall_actual, overall_sim)[0, 1]
    else:
        overall_corr = np.nan

    print(f"\n{'Survey':<45} | {'Items':>5} | {'MAE (pp)':>8} | {'Corr':>6}")
    print("-" * 82)
    for _, row in summary.iterrows():
        corr_str = f"{row['correlation']:.3f}" if row['correlation'] is not None else "  N/A"
        print(f"{row['survey_id']:<45} | {row['n_items']:>5} | {row['mae_pp']:>8.1f} | {corr_str:>6}")
    print("-" * 82)
    print(f"{'OVERALL':<45} | {len(df):>5} | {overall_mae:>8.1f} | {overall_corr:>6.3f}")

    summary.to_csv(RESULTS_DIR / "evaluation_summary.csv", index=False)
    print(f"\nSaved evaluation_summary.csv")

    # Bar chart of per-survey MAE
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [s[:35] for s in summary["survey_id"]]
    ax.barh(labels, summary["mae_pp"], color="#4a90d9", edgecolor="white")
    ax.set_xlabel("Mean Absolute Error (percentage points)")
    ax.set_title("Simulation Accuracy by Survey: MAE vs Actual YouGov Results")
    ax.axvline(x=overall_mae, color="red", linestyle="--", label=f"Overall MAE: {overall_mae:.1f}pp")
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "mae_chart.png", dpi=150)
    print(f"Saved mae_chart.png")

    # Scatter plot: actual vs simulated
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.scatter(df["actual_pct"], df["simulated_pct"], alpha=0.5, s=20)
    max_val = max(df["actual_pct"].max(), df["simulated_pct"].max()) + 5
    ax2.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="Perfect agreement")
    ax2.set_xlabel("Actual (%)")
    ax2.set_ylabel("Simulated (%)")
    ax2.set_title(f"Actual vs Simulated Survey Responses (r={overall_corr:.3f})")
    ax2.legend()
    ax2.set_aspect("equal")
    plt.tight_layout()
    fig2.savefig(RESULTS_DIR / "scatter_plot.png", dpi=150)
    print(f"Saved scatter_plot.png")


if __name__ == "__main__":
    main()
