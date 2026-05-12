#!/usr/bin/env python3
"""Small evaluation helper for the Rxplain Colab demo.

Usage from a notebook after loading Gemma:

    import sys
    sys.path.insert(0, "/content/rxplain")
    from scripts.evaluate_drugs import evaluate_drugs
    results = evaluate_drugs(model, processor)

The script intentionally requires an already-loaded model and processor so it
does not hard-code a Gemma checkpoint or duplicate notebook setup.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dailymed import get_drug_leaflet
from extract import extract_drug_info_robust
from personalise import generate_personal_summary, personalise
from schema import UserProfile
from server import format_guide


DEFAULT_DRUGS = [
    "warfarin",
    "metformin",
    "ibuprofen",
    "amoxicillin",
    "atorvastatin",
    "lisinopril",
    "omeprazole",
    "albuterol",
    "levothyroxine",
    "acetaminophen",
]


def evaluate_drugs(
    model,
    processor,
    drug_names: Iterable[str] = DEFAULT_DRUGS,
    profile: UserProfile | None = None,
) -> list[dict]:
    """Run a simple end-to-end smoke evaluation over common drug names."""
    if model is None or processor is None:
        raise ValueError("evaluate_drugs requires a loaded Gemma model and processor.")

    profile = profile or UserProfile(age_group="adult")
    results = []

    for name in drug_names:
        started = time.time()
        row = {
            "drug_name": name,
            "dailymed_found": False,
            "json_valid": False,
            "guide_generated": False,
            "error": "",
            "seconds": 0.0,
        }

        try:
            leaflet = get_drug_leaflet(name)
            row["dailymed_found"] = bool(leaflet)
            if not leaflet:
                row["error"] = "DailyMed label not found"
                continue

            drug_info = extract_drug_info_robust(leaflet, model, processor)
            row["json_valid"] = True

            drug_info = personalise(drug_info, profile)
            summary = generate_personal_summary(drug_info, profile)
            guide_html = format_guide(drug_info, summary)
            row["guide_generated"] = bool(guide_html)
        except Exception as exc:
            row["error"] = str(exc)
        finally:
            row["seconds"] = round(time.time() - started, 2)
            results.append(row)
            print(json.dumps(row, ensure_ascii=False))

    return results


def print_markdown_table(results: list[dict]) -> None:
    """Print results in a format suitable for README or Kaggle writeup."""
    print("| Drug | DailyMed | JSON valid | Guide | Seconds | Error |")
    print("|---|---:|---:|---:|---:|---|")
    for row in results:
        print(
            f"| {row['drug_name']} | {row['dailymed_found']} | "
            f"{row['json_valid']} | {row['guide_generated']} | "
            f"{row['seconds']} | {row['error']} |"
        )


if __name__ == "__main__":
    raise SystemExit(
        "Load Gemma in a notebook, then call evaluate_drugs(model, processor)."
    )
