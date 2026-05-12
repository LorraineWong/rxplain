# extract.py - Drug info extraction pipeline using local Gemma 4 inference

import json
import re
import torch
from schema import DrugInfo


def _load_json_object(raw: str) -> dict:
    """Load a JSON object from model output, tolerating small wrapper text."""
    cleaned = raw.strip()
    cleaned = re.sub(r'^```json\s*', '', cleaned)
    cleaned = re.sub(r'^```\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', cleaned):
        try:
            data, _ = decoder.raw_decode(cleaned[match.start():])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue

    raise ValueError("Gemma output did not contain valid JSON.")


def _trim_to_complete_sentence(value: str) -> str | None:
    """Return complete sentence text, or None if no complete sentence remains."""
    text = re.sub(r'\s+', ' ', value).strip()
    if not text:
        return None
    if text.endswith(("...", "…")):
        text = text.rstrip(".…").strip()
    elif text.endswith((".", "!", "?")):
        return text

    boundaries = [
        text.rfind(". "),
        text.rfind("! "),
        text.rfind("? "),
        text.rfind("."),
        text.rfind("!"),
        text.rfind("?"),
    ]
    boundary = max(boundaries)
    if boundary == -1:
        return None
    return text[:boundary + 1].strip() or None


def _clean_short_phrase(value: str) -> str | None:
    """Keep short clinical phrases unless they are visibly truncated."""
    text = re.sub(r'\s+', ' ', value).strip()
    if not text:
        return None
    if text.endswith(("...", "…")):
        return _trim_to_complete_sentence(text)
    return text


def _cleanup_generated_text(data: dict) -> None:
    """Drop or trim truncated patient-facing text before Pydantic validation."""
    prose_fields = {"text", "description", "reason", "personal_summary"}
    phrase_list_fields = {"contraindications", "emergency_signs"}

    def clean_item(item):
        if isinstance(item, str):
            return _clean_short_phrase(item)
        if isinstance(item, list):
            cleaned = []
            for child in item:
                fixed = clean_item(child)
                if fixed is not None:
                    cleaned.append(fixed)
            return cleaned
        if isinstance(item, dict):
            cleaned = {}
            for key, value in item.items():
                if key in prose_fields and isinstance(value, str):
                    fixed = _trim_to_complete_sentence(value)
                    if fixed is None:
                        return None
                    cleaned[key] = fixed
                elif key in {"warnings", "side_effects", "food_interactions"} and isinstance(value, list):
                    cleaned[key] = [fixed for child in value if (fixed := clean_item(child)) is not None]
                elif key in phrase_list_fields and isinstance(value, list):
                    cleaned[key] = [
                        fixed for child in value
                        if isinstance(child, str) and (fixed := _clean_short_phrase(child)) is not None
                    ]
                else:
                    cleaned[key] = value
            return cleaned
        return item

    cleaned = clean_item(data)
    if isinstance(cleaned, dict):
        data.clear()
        data.update(cleaned)


def extract_drug_info_robust(leaflet_text: str, model, processor) -> DrugInfo:
    """
    Extract DrugInfo from leaflet text using local Gemma 4 model.
    Returns a validated DrugInfo pydantic object.
    """
    schema = json.dumps(DrugInfo.model_json_schema(), indent=2)

    prompt = f"""You are a clinical pharmacist AI with expertise in drug safety. Extract medication information from the leaflet below.

STRICT OUTPUT RULES:
- Output ONLY valid JSON matching the schema. No markdown, no code blocks, no explanation.
- Extract SPECIFIC and CLINICALLY ACCURATE information only. Never use generic placeholder text.
- food_interactions action MUST be exactly one of: avoid, caution, ok
- side_effects severity MUST be exactly one of: HIGH, MEDIUM, LOW
- time_of_day MUST be exactly one of: morning, afternoon, evening, bedtime
- amount MUST be a clean dosage string like "5 mg", "1 tablet", "2-10 mg".
- drug_class MUST be the most specific pharmacologic class supported by the label, such as "Vitamin K antagonist" instead of only "Anticoagulant" for warfarin.
- Extract AT LEAST 3 side_effects with different severity levels:
  - At least one HIGH severity effect that can be life-threatening.
  - At least one MEDIUM severity effect that requires calling a doctor.
  - At least one LOW severity effect that can usually be monitored at home.
- Each side_effect description MUST describe what the patient may notice or experience, such as "You may notice unusual bruising or bleeding that does not stop"; do not write vague phrases like "may cause bleeding complications".
- Extract AT LEAST 3 food_interactions with actual foods or drinks only. Do NOT include drugs, medications, supplements, or medication classes.
- For warfarin-class anticoagulants, include clinically relevant food or drink examples when supported by the leaflet: green leafy vegetables such as spinach, kale, or broccoli; grapefruit; and alcohol. The action must reflect clinical guidance.
- warning text MUST be plain English for a patient with no medical background, written as complete readable sentences, with at least 2 sentences per warning. Do not use ALL CAPS.
- Extract AT LEAST 3 warnings as complete patient-facing warnings.
- emergency_signs MUST contain AT LEAST 3 real medical emergencies, written as specific observable symptoms a patient can recognise at home, such as "Coughing or vomiting blood", "Black or tarry stools", "Sudden severe headache", or "Unexplained bruising". Do not use generic phrases like "signs and symptoms of bleeding".

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
        output = model.generate(**inputs, max_new_tokens=3072, do_sample=False)

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[-1]
    raw = processor.decode(output[0][input_len:], skip_special_tokens=True).strip()

    data = _load_json_object(raw)

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

    _cleanup_generated_text(data)

    emergency_signs = [s for s in data.get("emergency_signs", []) if isinstance(s, str) and s.strip()]
    if len(emergency_signs) < 3:
        raise ValueError("Gemma extraction failed: emergency_signs must contain at least 3 specific symptoms.")
    data["emergency_signs"] = emergency_signs

    # Remove duplicate warnings
    seen = set()
    unique_warnings = []
    for w in data.get("warnings", []):
        if w["text"] not in seen:
            seen.add(w["text"])
            unique_warnings.append(w)
    data["warnings"] = unique_warnings

    return DrugInfo(**data)
