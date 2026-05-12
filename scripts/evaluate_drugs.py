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
    drug_names = list(drug_names)
    results = []
    total = len(drug_names)

    for i, name in enumerate(drug_names, 1):
        print(f"\n[{i}/{total}] Testing: {name} ...")
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
            row["error"] = str(exc)[:80]
        finally:
            row["seconds"] = round(time.time() - started, 2)
            results.append(row)
            status = "✅" if row["guide_generated"] else "❌"
            print(f"  {status} {row['seconds']}s{' — ' + row['error'] if row['error'] else ''}")

    succeeded = sum(1 for r in results if r["guide_generated"])
    print(f"\n{'='*40}")
    print(f"Result: {succeeded}/{total} guides generated successfully.")
    return results


def print_markdown_table(results: list[dict]) -> None:
    """Print results in a format suitable for README or Kaggle writeup."""
    def _bool(v: bool) -> str:
        return "✅" if v else "❌"

    print("| Drug | DailyMed | JSON valid | Guide | Seconds | Error |")
    print("|---|:---:|:---:|:---:|---:|---|")
    for row in results:
        error = (row["error"][:57] + "...") if len(row["error"]) > 60 else row["error"]
        print(
            f"| {row['drug_name']} | {_bool(row['dailymed_found'])} | "
            f"{_bool(row['json_valid'])} | {_bool(row['guide_generated'])} | "
            f"{row['seconds']} | {error} |"
        )


if __name__ == "__main__":
    raise SystemExit(
        "Load Gemma in a notebook, then call evaluate_drugs(model, processor)."
    )
