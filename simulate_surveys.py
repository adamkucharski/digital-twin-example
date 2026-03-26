"""Run LLM survey simulation with probability distribution approach.

For each survey question, generates N synthetic personas with correlated UK
demographics, then asks an LLM to estimate the response distribution for each
persona. Results are saved as raw JSON for later compilation and evaluation.
"""

import anthropic
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

from compile_demographics import sample_persona

BASE_DIR = Path(__file__).parent
EXTRACTED_DIR = BASE_DIR / "extracted"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MAX_CONCURRENT = 10
MODEL = "claude-haiku-4-5-20251001"
SKIP_SURVEYS = set()

BEHAVIOURAL_KEYWORDS = [
    "how many times", "how often", "have you", "do you use", "which method",
    "what types", "when was the last", "how far do you"
]
OPINION_KEYWORDS = [
    "support or oppose", "do you think", "would you say", "do you consider",
    "trust", "concerned"
]


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


def classify_data_structure(survey: dict) -> str:
    """Classify survey data structure into one of the supported types.

    Uses an explicit 'type' field if present; otherwise infers from data shape.
    """
    if "type" in survey:
        return survey["type"]

    data = get_primary_data(survey)
    items = list(data.keys())
    first_val = data[items[0]]
    sid = survey["survey_id"]

    if sid == "britains_favourite_romcom":
        return "ranked_choice"
    if sid == "gen_z_phone_myths":
        return "grouped_pick"

    if isinstance(first_val, dict):
        if len(items) == 1:
            return "single_item_options"
        else:
            return "multi_item_options"
    else:
        numeric_vals = [v for v in data.values() if isinstance(v, (int, float))]
        total = sum(numeric_vals)
        # If no numeric values (all null), default to single_pick
        if not numeric_vals or 70 <= total <= 115:
            return "single_pick"
        else:
            return "multi_pick_yesno"


def detect_distribution_axis(data: dict, items: list, opts: list) -> str:
    """Detect whether distribution runs across opts (per item) or across items (per opt).

    Returns 'per_item' if rows sum to ~100, 'per_opt' if columns sum to ~100.
    """
    row_sums = []
    for item in items:
        if isinstance(data.get(item), dict):
            row_sums.append(sum(v for v in data[item].values() if isinstance(v, (int, float))))

    col_sums = []
    for opt in opts:
        total = 0
        for item in items:
            if isinstance(data.get(item), dict):
                val = data[item].get(opt, 0)
                if isinstance(val, (int, float)):
                    total += val
        col_sums.append(total)

    avg_row = sum(row_sums) / len(row_sums) if row_sums else 0
    avg_col = sum(col_sums) / len(col_sums) if col_sums else 0

    return "per_item" if abs(avg_row - 100) <= abs(avg_col - 100) else "per_opt"


def get_temperature(survey: dict) -> float:
    """Choose temperature based on question type."""
    q = (survey.get("question") or "").lower()
    title = (survey.get("title") or "").lower()
    combined = q + " " + title

    if any(kw in combined for kw in BEHAVIOURAL_KEYWORDS):
        return 0.3
    if any(kw in combined for kw in OPINION_KEYWORDS):
        return 0.8
    return 0.5


def build_demographic_description(persona: dict) -> str:
    """Describe a demographic profile in third person for factual estimation."""
    return (
        f"a {persona['age_specific']}-year-old {persona['gender'].lower()} "
        f"living in a {persona['area_type'].lower()} area in {persona['region']}, UK, "
        f"whose highest education level is {persona['education']}, "
        f"who is {persona['employment_status'].lower()} with a household income of {persona['household_income']}, "
        f"whose religion is {persona['religion']}, "
        f"who mainly gets news from {persona['primary_news_source']}, "
        f"and who voted {persona['political_2024']} in the 2024 UK general election"
    )


def build_distribution_prompt(demographic_desc: str, question: str, options: list) -> str:
    """Build a prompt asking the model to estimate a % distribution across options."""
    options_json = json.dumps(options)
    return f"""You are a survey research analyst estimating how a UK demographic group would respond to a poll.

Demographic profile: {demographic_desc}

Question: {question}

Options: {options_json}

Estimate what percentage of people matching this profile would choose each option. Be realistic about uncertainty - many people genuinely don't know or lack strong opinions.

The percentages MUST sum to exactly 100.

Respond with ONLY a JSON object. Example: {{"{options[0]}": 45, "{options[1]}": 30, ...}}

JSON:"""


def build_yesno_distribution_prompt(demographic_desc: str, question: str, item: str) -> str:
    """Estimate % who would say yes to a specific item."""
    return f"""You are a survey research analyst estimating how a UK demographic group would respond to a poll.

Demographic profile: {demographic_desc}

Question: {question}
Specific item: {item}

What percentage of people matching this profile would say YES to this item?
Respond with ONLY a single number (0-100). Nothing else.

Percentage:"""


def group_items(items: list) -> dict:
    """Group items by prefix (split on ' - ')."""
    groups = {}
    for item in items:
        if " - " in item:
            prefix, suffix = item.split(" - ", 1)
            groups.setdefault(prefix, []).append(suffix)
        else:
            groups.setdefault("_default", []).append(item)
    return groups


def parse_json_response(text: str, options: list) -> dict:
    """Parse a JSON distribution response. Returns {option: percentage}."""
    text = text.strip()
    text = re.sub(r'^```json?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            total = sum(parsed.values())
            if total > 0:
                return {k: v / total * 100 for k, v in parsed.items()}
        return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: extract numbers from text
    result = {}
    for opt in options:
        pattern = re.escape(opt) + r'["\s:]+(\d+(?:\.\d+)?)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result[opt] = float(match.group(1))

    if result:
        total = sum(result.values())
        if total > 0:
            return {k: v / total * 100 for k, v in result.items()}

    return {}


def parse_number_response(text: str) -> float:
    """Parse a single number from response."""
    text = text.strip()
    match = re.search(r'(\d+(?:\.\d+)?)', text)
    if match:
        val = float(match.group(1))
        return min(max(val, 0), 100)
    return 50.0


async def api_call(client, semaphore, prompt, temperature):
    """Make a single API call with retry on rate limit."""
    async with semaphore:
        try:
            response = await asyncio.to_thread(
                client.messages.create,
                model=MODEL,
                max_tokens=512,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except anthropic.RateLimitError:
            await asyncio.sleep(2)
            return await api_call(client, semaphore, prompt, temperature)
        except Exception as e:
            return f"ERROR: {e}"


async def simulate_survey(client, survey, n):
    """Simulate one survey with N personas using factual estimation approach."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    rng = random.Random(41231242 + hash(survey["survey_id"]) % 10000)
    temperature = get_temperature(survey)
    question = survey.get("question") or survey.get("title") or ""

    primary_data = get_primary_data(survey)
    items = list(primary_data.keys())
    data_type = classify_data_structure(survey)

    personas = [sample_persona(rng) for _ in range(n)]
    results = []

    if data_type in ("single_pick", "single_item_options", "ranked_choice"):
        if data_type == "single_item_options":
            options = list(primary_data[items[0]].keys())
        else:
            options = items

        tasks = []
        for persona in personas:
            desc = build_demographic_description(persona)
            prompt = build_distribution_prompt(desc, question, options)
            tasks.append((persona, api_call(client, semaphore, prompt, temperature)))

        persona_list = [t[0] for t in tasks]
        coros = [t[1] for t in tasks]
        responses = await asyncio.gather(*coros)

        for persona, raw in zip(persona_list, responses):
            if raw.startswith("ERROR:"):
                results.append({"persona": persona, "distribution": {}, "error": True, "data_type": data_type})
            else:
                dist = parse_json_response(raw, options)
                results.append({"persona": persona, "distribution": dist, "raw_response": raw[:200], "data_type": data_type})

    elif data_type == "multi_pick_yesno":
        for persona in personas:
            desc = build_demographic_description(persona)
            item_tasks = []
            for item in items:
                prompt = build_yesno_distribution_prompt(desc, question, item)
                item_tasks.append(api_call(client, semaphore, prompt, temperature))

            item_responses = await asyncio.gather(*item_tasks)

            dist = {}
            for item, raw in zip(items, item_responses):
                if not raw.startswith("ERROR:"):
                    dist[item] = parse_number_response(raw)

            results.append({"persona": persona, "distribution": dist, "data_type": data_type})

    elif data_type == "multi_item_options":
        response_options = survey.get("response_options", [])
        axis = detect_distribution_axis(primary_data, items, response_options)

        if axis == "per_opt":
            for persona in personas:
                desc = build_demographic_description(persona)
                opt_tasks = []
                for opt in response_options:
                    sub_question = f"{question}\nSpecifically for: {opt}"
                    prompt = build_distribution_prompt(desc, sub_question, items)
                    opt_tasks.append((opt, api_call(client, semaphore, prompt, temperature)))

                opt_names = [t[0] for t in opt_tasks]
                opt_coros = [t[1] for t in opt_tasks]
                opt_responses = await asyncio.gather(*opt_coros)

                dist = {}
                for opt_name, raw in zip(opt_names, opt_responses):
                    if not raw.startswith("ERROR:"):
                        parsed = parse_json_response(raw, items)
                        for item in items:
                            if item not in dist:
                                dist[item] = {}
                            dist[item][opt_name] = parsed.get(item, 0)

                results.append({"persona": persona, "distribution": dist, "data_type": data_type})
        else:
            for persona in personas:
                desc = build_demographic_description(persona)
                item_tasks = []
                for item in items:
                    item_opts = primary_data[item]
                    if not isinstance(item_opts, dict):
                        continue
                    opts = list(item_opts.keys())
                    sub_question = f"{question}\nSpecifically about: {item}"
                    prompt = build_distribution_prompt(desc, sub_question, opts)
                    item_tasks.append((item, api_call(client, semaphore, prompt, temperature)))

                item_names = [t[0] for t in item_tasks]
                item_coros = [t[1] for t in item_tasks]
                item_responses = await asyncio.gather(*item_coros)

                dist = {}
                for item_name, raw in zip(item_names, item_responses):
                    if not raw.startswith("ERROR:"):
                        item_opts = list(primary_data[item_name].keys())
                        dist[item_name] = parse_json_response(raw, item_opts)

            results.append({"persona": persona, "distribution": dist, "data_type": data_type})

    elif data_type == "grouped_pick":
        groups = group_items(items)
        for persona in personas:
            desc = build_demographic_description(persona)
            group_tasks = []
            for group_name, group_opts in groups.items():
                sub_question = f"{question}\nScenario: {group_name}"
                prompt = build_distribution_prompt(desc, sub_question, group_opts)
                group_tasks.append((group_name, api_call(client, semaphore, prompt, temperature)))

            group_names = [t[0] for t in group_tasks]
            group_coros = [t[1] for t in group_tasks]
            group_responses = await asyncio.gather(*group_coros)

            dist = {}
            for gname, raw in zip(group_names, group_responses):
                if not raw.startswith("ERROR:"):
                    dist[gname] = parse_json_response(raw, groups[gname])

            results.append({"persona": persona, "distribution": dist, "data_type": data_type})

    return results


async def main():
    global EXTRACTED_DIR, RESULTS_DIR
    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser("~/.claude/.env"))

    args = [a for a in sys.argv[1:] if a != "--test"]
    use_test = "--test" in sys.argv
    if use_test:
        EXTRACTED_DIR = BASE_DIR / "extracted_test"
        RESULTS_DIR = BASE_DIR / "results_test"
        RESULTS_DIR.mkdir(exist_ok=True)

    n = int(args[0]) if len(args) > 0 else 50
    only_ids = set(args[1:]) if len(args) > 1 else None
    print(f"Simulating {n} personas per survey (distribution approach)")
    if use_test:
        print(f"  Using test data: {EXTRACTED_DIR} -> {RESULTS_DIR}")
    print()

    client = anthropic.Anthropic()

    with open(EXTRACTED_DIR / "all_surveys.json") as f:
        all_surveys = json.load(f)

    for survey in all_surveys:
        sid = survey["survey_id"]
        if only_ids and sid not in only_ids:
            continue
        if sid in SKIP_SURVEYS:
            print(f"Skipping {sid}")
            continue

        data_type = classify_data_structure(survey)
        temp = get_temperature(survey)
        print(f"Simulating: {sid} (type={data_type}, temp={temp})")

        results = await simulate_survey(client, survey, n)

        errors = sum(1 for r in results if r.get("error"))
        if errors:
            print(f"  {errors} errors out of {n}")

        sample = results[0]
        dist = sample.get("distribution", {})
        preview = str(dist)[:150]
        print(f"  Sample dist: {preview}")

        out_path = RESULTS_DIR / f"{sid}_raw.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {out_path.name}\n")

    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
