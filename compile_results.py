"""Compile simulated results — average probability distributions across personas."""

import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
EXTRACTED_DIR = BASE_DIR / "extracted"
RESULTS_DIR = BASE_DIR / "results"

SKIP_SURVEYS = set()


def get_primary_data(survey: dict) -> dict:
    """Get data for the primary breakdown. Flattens {"percentage": N} to N."""
    data = survey["data"]
    for key in ["All Britons", "All Respondents", "All counties", "Cornwall",
                "All British Parents", "All UK Adults (who borrowed)", "All Brits",
                "All GB Adults", "All"]:
        if key in data:
            raw = data[key]
            break
    else:
        raw = data[list(data.keys())[0]]
    result = {}
    for k, v in raw.items():
        if isinstance(v, dict) and list(v.keys()) == ["percentage"]:
            result[k] = v["percentage"]
        else:
            result[k] = v
    return result


def get_flat_value(val) -> float:
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, dict) and "percentage" in val:
        return val["percentage"]
    return 0


def compile_survey(survey: dict, raw_results: list) -> pd.DataFrame:
    """Compile raw per-persona results into averaged predictions vs actuals."""
    actual_data = get_primary_data(survey)
    valid = [r for r in raw_results if not r.get("error") and r.get("distribution")]
    if not valid:
        return pd.DataFrame()

    data_type = valid[0].get("data_type", "unknown")
    items = list(actual_data.keys())
    rows = []

    if data_type in ("single_pick", "single_item_options", "ranked_choice"):
        if data_type == "single_item_options":
            item = items[0]
            options = list(actual_data[item].keys()) if isinstance(actual_data[item], dict) else items
        else:
            options = items

        all_dists = [r["distribution"] for r in valid if r["distribution"]]
        n = len(all_dists)

        for opt in options:
            actual_pct = get_flat_value(actual_data.get(opt, actual_data[items[0]].get(opt, 0))
                                        if data_type == "single_item_options"
                                        else actual_data.get(opt, 0))
            sim_vals = [d.get(opt, 0) for d in all_dists]
            sim_pct = np.mean(sim_vals) if sim_vals else 0

            rows.append({
                "survey_id": survey["survey_id"],
                "question_item": items[0] if data_type == "single_item_options" else opt,
                "response_option": opt,
                "actual_pct": actual_pct,
                "simulated_pct": round(sim_pct, 1),
                "simulated_std": round(np.std(sim_vals), 1) if sim_vals else 0,
                "n_personas": n
            })

    elif data_type == "multi_pick_yesno":
        all_dists = [r["distribution"] for r in valid if r["distribution"]]
        n = len(all_dists)

        for item in items:
            actual_pct = get_flat_value(actual_data[item])
            sim_vals = [d.get(item, 50) for d in all_dists]
            sim_pct = np.mean(sim_vals) if sim_vals else 0

            rows.append({
                "survey_id": survey["survey_id"],
                "question_item": item,
                "response_option": "Yes %",
                "actual_pct": actual_pct,
                "simulated_pct": round(sim_pct, 1),
                "simulated_std": round(np.std(sim_vals), 1) if sim_vals else 0,
                "n_personas": n
            })

    elif data_type == "multi_item_options":
        all_dists = [r["distribution"] for r in valid if r["distribution"]]
        n = len(all_dists)

        for item in items:
            item_data = actual_data[item]
            if not isinstance(item_data, dict):
                continue
            for opt, actual_pct in item_data.items():
                sim_vals = []
                for d in all_dists:
                    item_dist = d.get(item, {})
                    if isinstance(item_dist, dict):
                        sim_vals.append(item_dist.get(opt, 0))
                sim_pct = np.mean(sim_vals) if sim_vals else 0

                rows.append({
                    "survey_id": survey["survey_id"],
                    "question_item": item,
                    "response_option": opt,
                    "actual_pct": actual_pct,
                    "simulated_pct": round(sim_pct, 1),
                    "simulated_std": round(np.std(sim_vals), 1) if sim_vals else 0,
                    "n_personas": n
                })

    elif data_type == "grouped_pick":
        all_dists = [r["distribution"] for r in valid if r["distribution"]]
        n = len(all_dists)

        groups = {}
        for item in items:
            if " - " in item:
                prefix, suffix = item.split(" - ", 1)
                groups.setdefault(prefix, []).append((item, suffix))

        for group_name, group_items in groups.items():
            for full_item, suffix in group_items:
                actual_pct = get_flat_value(actual_data.get(full_item, 0))
                sim_vals = []
                for d in all_dists:
                    group_dist = d.get(group_name, {})
                    if isinstance(group_dist, dict):
                        sim_vals.append(group_dist.get(suffix, 0))
                sim_pct = np.mean(sim_vals) if sim_vals else 0

                rows.append({
                    "survey_id": survey["survey_id"],
                    "question_item": full_item,
                    "response_option": suffix,
                    "actual_pct": actual_pct,
                    "simulated_pct": round(sim_pct, 1),
                    "simulated_std": round(np.std(sim_vals), 1) if sim_vals else 0,
                    "n_personas": n
                })

    return pd.DataFrame(rows)


def main():
    global EXTRACTED_DIR, RESULTS_DIR
    if "--test" in sys.argv:
        EXTRACTED_DIR = BASE_DIR / "extracted_test"
        RESULTS_DIR = BASE_DIR / "results_test"

    with open(EXTRACTED_DIR / "all_surveys.json") as f:
        all_surveys = json.load(f)

    all_dfs = []

    for survey in all_surveys:
        sid = survey["survey_id"]
        if sid in SKIP_SURVEYS:
            continue

        raw_path = RESULTS_DIR / f"{sid}_raw.json"
        if not raw_path.exists():
            print(f"Skipping {sid}: no raw results")
            continue

        with open(raw_path) as f:
            raw_results = json.load(f)

        df = compile_survey(survey, raw_results)
        if len(df) > 0:
            df.to_csv(RESULTS_DIR / f"{sid}_compiled.csv", index=False)
            print(f"{sid}: {len(df)} rows")
            all_dfs.append(df)
        else:
            print(f"{sid}: no data to compile")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(RESULTS_DIR / "all_results.csv", index=False)
        print(f"\nCombined: {len(combined)} rows -> all_results.csv")


if __name__ == "__main__":
    main()
