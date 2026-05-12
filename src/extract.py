# extract.py - Drug info extraction pipeline using local Gemma 4 inference

import json
import re
import torch
from schema import DrugInfo


def extract_drug_info_robust(leaflet_text: str, model, processor) -> DrugInfo:
    """
    Extract DrugInfo from leaflet text using local Gemma 4 model.
    Returns a validated DrugInfo pydantic object.
    """
    schema = json.dumps(DrugInfo.model_json_schema(), indent=2)

    prompt = f"""You are a clinical pharmacist AI. Extract medication information from the leaflet below.

STRICT OUTPUT RULES:
- Output ONLY valid JSON matching the schema. No markdown, no code blocks, no explanation.
- food_interactions action MUST be exactly one of: avoid, caution, ok
- side_effects severity MUST be exactly one of: HIGH, MEDIUM, LOW
- time_of_day MUST be exactly one of: morning, afternoon, evening, bedtime
- amount MUST be a clean dosage string like "5 mg", "1 tablet", "2-10 mg".
- warning text MUST be a complete readable sentence in plain English.
- Extract AT LEAST 3 side_effects with different severity levels.
- Extract AT LEAST 3 food_interactions with actual foods or drinks only.
- Extract AT LEAST 3 warnings as complete sentences.
- emergency_signs must be real medical emergencies only.

JSON SCHEMA:
{schema}

LEAFLET TEXT:
{leaflet_text}

JSON OUTPUT:"""

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=2048, do_sample=False)

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[-1]
    raw = processor.decode(output[0][input_len:], skip_special_tokens=True).strip()

    # Strip markdown code fences if present
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'^```\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    raw = raw.strip()

    data = json.loads(raw)

    # Auto-correct food_interactions action
    valid_actions = ["avoid", "caution", "ok"]
    for fi in data.get("food_interactions", []):
        if fi.get("action") not in valid_actions:
            action_lower = fi.get("action", "").lower()
            if any(w in action_lower for w in ["avoid", "do not", "never"]):
                fi["action"] = "avoid"
            elif any(w in action_lower for w in ["caution", "limit", "monitor"]):
                fi["action"] = "caution"
            else:
                fi["action"] = "ok"

    # Auto-correct side_effects severity
    for se in data.get("side_effects", []):
        if se.get("severity") not in ["HIGH", "MEDIUM", "LOW"]:
            se["severity"] = "MEDIUM"

    # Auto-correct time_of_day
    valid_times = ["morning", "afternoon", "evening", "bedtime"]
    for di in data.get("dosage_instructions", []):
        if di.get("time_of_day") not in valid_times:
            di["time_of_day"] = "morning"

    # Normalise ALL-CAPS warning text
    for w in data.get("warnings", []):
        text = w.get("text", "")
        if text == text.upper() and len(text) > 3:
            w["text"] = text.capitalize()

    # Remove duplicate warnings
    seen = set()
    unique_warnings = []
    for w in data.get("warnings", []):
        if w["text"] not in seen:
            seen.add(w["text"])
            unique_warnings.append(w)
    data["warnings"] = unique_warnings

    return DrugInfo(**data)
