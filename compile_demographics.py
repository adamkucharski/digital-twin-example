"""UK demographic distributions with correlated sampling for survey simulation."""

import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Marginal distributions (UK population baseline, approximated by Claude from ONS Census 2021)

MARGINALS = {
    "age": {
        "18-24": 0.10, "25-34": 0.17, "35-44": 0.16,
        "45-54": 0.16, "55-64": 0.16, "65+": 0.25
    },
    "gender": {"Male": 0.49, "Female": 0.51},
    "region": {
        "London": 0.13, "South East": 0.14, "North West": 0.11,
        "East of England": 0.09, "West Midlands": 0.09, "South West": 0.09,
        "Yorkshire and the Humber": 0.08, "East Midlands": 0.07,
        "North East": 0.04, "Scotland": 0.08, "Wales": 0.05, "Northern Ireland": 0.03
    },
    "area_type": {
        "Major urban city": 0.35, "Town or smaller city": 0.37,
        "Rural village or countryside": 0.28
    },
    "religion": {
        "No religion": 0.37, "Christian": 0.46, "Muslim": 0.07,
        "Hindu": 0.02, "Other religion": 0.03, "Prefer not to say": 0.05
    }
}

# Conditional distributions for correlated sampling
# Sources: approximated by Claude from ONS Census 2021 and other sources

# P(education | age)
EDUCATION_GIVEN_AGE = {
    "18-24": {"No qualifications": 0.08, "GCSEs or equivalent": 0.20, "A-levels or equivalent": 0.35, "University degree or higher": 0.37},
    "25-34": {"No qualifications": 0.08, "GCSEs or equivalent": 0.15, "A-levels or equivalent": 0.20, "University degree or higher": 0.57},
    "35-44": {"No qualifications": 0.10, "GCSEs or equivalent": 0.20, "A-levels or equivalent": 0.20, "University degree or higher": 0.50},
    "45-54": {"No qualifications": 0.15, "GCSEs or equivalent": 0.25, "A-levels or equivalent": 0.20, "University degree or higher": 0.40},
    "55-64": {"No qualifications": 0.22, "GCSEs or equivalent": 0.28, "A-levels or equivalent": 0.18, "University degree or higher": 0.32},
    "65+":   {"No qualifications": 0.30, "GCSEs or equivalent": 0.30, "A-levels or equivalent": 0.15, "University degree or higher": 0.25},
}

# P(income | education)
INCOME_GIVEN_EDUCATION = {
    "No qualifications":          {"Under £15,000": 0.35, "£15,000-£25,000": 0.30, "£25,000-£40,000": 0.20, "£40,000-£60,000": 0.10, "Over £60,000": 0.05},
    "GCSEs or equivalent":        {"Under £15,000": 0.18, "£15,000-£25,000": 0.28, "£25,000-£40,000": 0.30, "£40,000-£60,000": 0.16, "Over £60,000": 0.08},
    "A-levels or equivalent":     {"Under £15,000": 0.12, "£15,000-£25,000": 0.20, "£25,000-£40,000": 0.30, "£40,000-£60,000": 0.25, "Over £60,000": 0.13},
    "University degree or higher": {"Under £15,000": 0.06, "£15,000-£25,000": 0.10, "£25,000-£40,000": 0.22, "£40,000-£60,000": 0.32, "Over £60,000": 0.30},
}

# P(employment | age)
EMPLOYMENT_GIVEN_AGE = {
    "18-24": {"Employed full-time": 0.35, "Employed part-time": 0.20, "Self-employed": 0.03, "Retired": 0.00, "Student": 0.35, "Unemployed or not working": 0.07},
    "25-34": {"Employed full-time": 0.60, "Employed part-time": 0.12, "Self-employed": 0.08, "Retired": 0.00, "Student": 0.05, "Unemployed or not working": 0.15},
    "35-44": {"Employed full-time": 0.58, "Employed part-time": 0.15, "Self-employed": 0.12, "Retired": 0.01, "Student": 0.01, "Unemployed or not working": 0.13},
    "45-54": {"Employed full-time": 0.55, "Employed part-time": 0.14, "Self-employed": 0.12, "Retired": 0.05, "Student": 0.01, "Unemployed or not working": 0.13},
    "55-64": {"Employed full-time": 0.38, "Employed part-time": 0.12, "Self-employed": 0.10, "Retired": 0.28, "Student": 0.00, "Unemployed or not working": 0.12},
    "65+":   {"Employed full-time": 0.05, "Employed part-time": 0.05, "Self-employed": 0.03, "Retired": 0.80, "Student": 0.00, "Unemployed or not working": 0.07},
}

# P(politics | age) - 2024 UK general election
POLITICS_GIVEN_AGE = {
    "18-24": {"Labour": 0.45, "Conservative": 0.08, "Reform UK": 0.06, "Liberal Democrats": 0.12, "Green": 0.18, "Other": 0.04, "Did not vote": 0.07},
    "25-34": {"Labour": 0.42, "Conservative": 0.12, "Reform UK": 0.10, "Liberal Democrats": 0.12, "Green": 0.12, "Other": 0.05, "Did not vote": 0.07},
    "35-44": {"Labour": 0.38, "Conservative": 0.20, "Reform UK": 0.14, "Liberal Democrats": 0.12, "Green": 0.07, "Other": 0.04, "Did not vote": 0.05},
    "45-54": {"Labour": 0.32, "Conservative": 0.26, "Reform UK": 0.16, "Liberal Democrats": 0.12, "Green": 0.05, "Other": 0.04, "Did not vote": 0.05},
    "55-64": {"Labour": 0.30, "Conservative": 0.30, "Reform UK": 0.17, "Liberal Democrats": 0.12, "Green": 0.04, "Other": 0.03, "Did not vote": 0.04},
    "65+":   {"Labour": 0.25, "Conservative": 0.35, "Reform UK": 0.18, "Liberal Democrats": 0.12, "Green": 0.02, "Other": 0.03, "Did not vote": 0.05},
}

# P(news_source | age)
NEWS_GIVEN_AGE = {
    "18-24": {"BBC (TV, radio, or website)": 0.15, "Social media (Facebook, X/Twitter, TikTok, etc.)": 0.45, "ITV or Channel 4 news": 0.05, "Newspapers (print or online)": 0.08, "Radio (non-BBC)": 0.03, "Sky News or GB News": 0.06, "Word of mouth or don't follow news": 0.18},
    "25-34": {"BBC (TV, radio, or website)": 0.22, "Social media (Facebook, X/Twitter, TikTok, etc.)": 0.35, "ITV or Channel 4 news": 0.07, "Newspapers (print or online)": 0.12, "Radio (non-BBC)": 0.04, "Sky News or GB News": 0.08, "Word of mouth or don't follow news": 0.12},
    "35-44": {"BBC (TV, radio, or website)": 0.28, "Social media (Facebook, X/Twitter, TikTok, etc.)": 0.25, "ITV or Channel 4 news": 0.10, "Newspapers (print or online)": 0.14, "Radio (non-BBC)": 0.05, "Sky News or GB News": 0.10, "Word of mouth or don't follow news": 0.08},
    "45-54": {"BBC (TV, radio, or website)": 0.35, "Social media (Facebook, X/Twitter, TikTok, etc.)": 0.18, "ITV or Channel 4 news": 0.12, "Newspapers (print or online)": 0.15, "Radio (non-BBC)": 0.06, "Sky News or GB News": 0.08, "Word of mouth or don't follow news": 0.06},
    "55-64": {"BBC (TV, radio, or website)": 0.38, "Social media (Facebook, X/Twitter, TikTok, etc.)": 0.12, "ITV or Channel 4 news": 0.14, "Newspapers (print or online)": 0.16, "Radio (non-BBC)": 0.06, "Sky News or GB News": 0.08, "Word of mouth or don't follow news": 0.06},
    "65+":   {"BBC (TV, radio, or website)": 0.42, "Social media (Facebook, X/Twitter, TikTok, etc.)": 0.06, "ITV or Channel 4 news": 0.14, "Newspapers (print or online)": 0.18, "Radio (non-BBC)": 0.06, "Sky News or GB News": 0.10, "Word of mouth or don't follow news": 0.04},
}

def _pick(rng, dist):
    """Pick from a distribution dict."""
    cats = list(dist.keys())
    weights = list(dist.values())
    return rng.choices(cats, weights=weights, k=1)[0]


def sample_persona(rng=None):
    """Draw a single random persona with correlated UK demographics."""
    if rng is None:
        rng = random.Random()

    age = _pick(rng, MARGINALS["age"])
    gender = _pick(rng, MARGINALS["gender"])
    region = _pick(rng, MARGINALS["region"])
    area_type = _pick(rng, MARGINALS["area_type"])

    education = _pick(rng, EDUCATION_GIVEN_AGE[age])

    income = _pick(rng, INCOME_GIVEN_EDUCATION[education])
    employment = _pick(rng, EMPLOYMENT_GIVEN_AGE[age])

    political = _pick(rng, POLITICS_GIVEN_AGE[age])

    news = _pick(rng, NEWS_GIVEN_AGE[age])
    religion = _pick(rng, MARGINALS["religion"])

    lo, hi = age.replace("+", "-90").split("-")
    age_specific = rng.randint(int(lo), int(hi))

    return {
        "age": age,
        "age_specific": age_specific,
        "gender": gender,
        "region": region,
        "area_type": area_type,
        "education": education,
        "household_income": income,
        "employment_status": employment,
        "political_2024": political,
        "primary_news_source": news,
        "religion": religion,
    }


if __name__ == "__main__":
    # Validate conditional distributions
    for name, table in [
        ("edu|age", EDUCATION_GIVEN_AGE),
        ("income|edu", INCOME_GIVEN_EDUCATION),
        ("employ|age", EMPLOYMENT_GIVEN_AGE),
        ("politics|age", POLITICS_GIVEN_AGE),
        ("news|age", NEWS_GIVEN_AGE),
    ]:
        for key, dist in table.items():
            total = sum(dist.values())
            assert abs(total - 1.0) < 0.02, f"{name}[{key}] sums to {total}"
    print("All conditional distributions valid.\n")

    # Show sample personas
    print("Sample personas:")
    rng = random.Random(42)
    for i in range(5):
        p = sample_persona(rng)
        print(f"  {i+1}. Age {p['age_specific']}, {p['gender']}, {p['region']}, "
              f"{p['area_type']}")
        print(f"     {p['education']}, {p['employment_status']}, {p['household_income']}")
        print(f"     Voted {p['political_2024']}, news: {p['primary_news_source']}, "
              f"{p['religion']}")
        print()
