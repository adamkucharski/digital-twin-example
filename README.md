# Exploratory LLM survey simulation

This code simulates public opinion survey responses using LLM-generated probability distributions over demographically sampled personas, then compares predictions against actual results.

For background on this project, see this Substack post, which compares with publicly available survey results: [How to make 95% accurate digital twins (ish)](https://kucharski.substack.com/p/how-i-made-some-95-accurate-digital)

Note: this is a quick simulation model to support an exploratory commentary post, generated with the help of Claude Code, rather than production software. Code is provided only for illustration.

## How it works

1. **Demographic sampling** (`compile_demographics.py`) — draws synthetic personas from correlated UK demographic distributions (age, gender, region, education, income, employment, political vote, news source, religion).

2. **Survey simulation** (`simulate_surveys.py`) — for each persona, prompts an LLM to estimate the percentage distribution across survey response options. Supports single-pick, multi-pick yes/no, multi-item, grouped, and ranked-choice question formats.

3. **Result compilation** (`compile_results.py`) — averages per-persona distributions into a single predicted distribution per survey and compares against actual percentages.

4. **Evaluation** (`evaluate.py`) — computes per-survey MAE (mean absolute error in percentage points) and correlation. Outputs summary tables and charts.

## Quick start

```bash
pip install -r requirements.txt
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

The code uses Claude Haiku, so should cost a few cents per simulation with default settings.

### Prepare question data

Place your survey JSON files in an `extracted/` directory (see `examples/all_surveys.json` for some fictional illustrative examples). Each survey is an object with a question and response options:

```json
{
  "survey_id": "commute_method",
  "title": "How Britons get to work",
  "type": "single_pick",
  "question": "What is your main method of commuting to work?",
  "data": {
    "All Britons": {
      "Car": null,
      "Public transport": null,
      "Bicycle": null,
      "Walking": null,
      "Work from home": null,
      "Other": null
    }
  }
}
```

The `data` keys define the response options. Values can be `null` if you just want to run the simulation without ground-truth comparison, or actual percentages if you want to evaluate accuracy with `compile_results.py` and `evaluate.py`:

```json
{
  "survey_id": "favourite_hot_drink",
  "type": "single_pick",
  "question": "Which of the following is your favourite hot drink?",
  "data": {
    "All Britons": {
      "Tea": {"percentage": 50},
      "Coffee": {"percentage": 40},
      "Hot chocolate": {"percentage": 12},
      "Herbal tea": {"percentage": 5},
      "None of these": {"percentage": 3}
    }
  }
}
```

Create `extracted/all_surveys.json` containing a JSON array of all survey objects (one is already available in the repo).

### Run the pipeline

```bash
# Simulate with 50 personas per survey (default)
python simulate_surveys.py

# Or specify a custom number and specific survey IDs
python simulate_surveys.py 100 favourite_hot_drink

# Compile results
python compile_results.py

# Evaluate accuracy
python evaluate.py
```

Outputs are saved to `results/` (CSV summaries and PNG charts).

## Question data format

Each survey object requires:

| Field | Description |
|---|---|
| `survey_id` | Unique identifier |
| `question` | The survey question text |
| `data` | Nested dict: `{breakdown: {item: {option: percentage}}}` or `{breakdown: {item: percentage}}` |

Optional fields: `title`, `date`, `sample_info`, `breakdowns`, `questions_or_items`, `response_options`.

The `type` field specifies the question structure. Supported values: `single_pick`, `multi_pick_yesno`, `single_item_options`, `multi_item_options`, `ranked_choice`, `grouped_pick`. If omitted, the simulator infers the type from the data shape.
