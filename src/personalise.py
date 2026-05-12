from schema import DrugInfo, UserProfile
import re


def personalise(drug_info: DrugInfo, profile: UserProfile) -> DrugInfo:
    """Re-prioritise warnings based on user health profile."""
    priority = []
    standard = []

    for warning in drug_info.warnings:
        text_lower = warning.text.lower()
        is_priority = False

        if profile.age_group == "elderly":
            if any(w in text_lower for w in [
                "fall", "bleed", "elderly", "older", "age", "inr", "monitor"
            ]):
                is_priority = True
            if "elderly" in warning.applies_to:
                is_priority = True

        if profile.pregnant or profile.breastfeeding:
            if any(w in text_lower for w in [
                "pregnan", "fetal", "birth", "breastfeed", "lactation"
            ]):
                is_priority = True

        if profile.kidney_issue:
            if any(w in text_lower for w in ["kidney", "renal"]):
                is_priority = True

        if profile.liver_issue:
            if any(w in text_lower for w in ["liver", "hepatic"]):
                is_priority = True

        if profile.heart_condition:
            if any(w in text_lower for w in [
                "heart", "cardiac", "cardiovascular", "arrhythmia"
            ]):
                is_priority = True

        if profile.diabetes:
            if any(w in text_lower for w in [
                "diabetes", "diabetic", "glucose", "blood sugar", "insulin"
            ]):
                is_priority = True

        if profile.hypertension:
            if any(w in text_lower for w in [
                "blood pressure", "hypertension", "hypotension"
            ]):
                is_priority = True

        if profile.asthma:
            if any(w in text_lower for w in [
                "asthma", "bronchospasm", "respiratory", "breathing"
            ]):
                is_priority = True

        if profile.other_medications:
            for med in profile.other_medications:
                if med.lower() in text_lower:
                    is_priority = True

        if is_priority:
            priority.append(warning)
        else:
            standard.append(warning)

    drug_info.warnings = priority + standard
    return drug_info


def _truncate(text: str, max_chars: int = 100) -> str:
    """Truncate warning text to max_chars, ending at a word boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return truncated.rstrip(".,;") + "…"


def _safe_amount(amount: str) -> str:
    if not amount:
        return ""
    if not re.search(r'\d|tablet|capsule|drop|patch|unit', amount, re.IGNORECASE):
        return ""
    if len(amount) > 20 or any(c in amount for c in [';', '/', '\\']):
        return ""
    return amount


def generate_personal_summary(drug_info: DrugInfo, profile: UserProfile) -> str:
    """
    2-sentence personalised summary. No AI call — deterministic.
    Sentence 1: safe dosage framing. Sentence 2: key risk for this profile.
    """
    lines = []

    # Sentence 1: dosage safety framing
    if drug_info.dosage_instructions:
        d = drug_info.dosage_instructions[0]
        amount = _safe_amount(d.amount)
        food_str = "with food" if d.with_food else "without food"
        if amount:
            lines.append(
                f"Follow your prescription label for {drug_info.drug_name}; "
                f"the official label mentions {amount} around {d.time_of_day}, {food_str}."
            )
        else:
            lines.append(
                f"Follow your prescription label for {drug_info.drug_name}; "
                f"the official label includes guidance around {d.time_of_day}, {food_str}."
            )

    # Sentence 2: top risk for this profile
    risk_parts = []
    if profile.pregnant:
        risk_parts.append("may not be safe during pregnancy — confirm with your doctor immediately")
    if profile.age_group == "elderly":
        risk_parts.append("fall-related bleeding is a serious concern for senior patients")
    if profile.kidney_issue:
        risk_parts.append("your kidney condition may affect how this drug is processed")
    if profile.liver_issue:
        risk_parts.append("your liver condition requires extra caution with this drug")
    if profile.heart_condition:
        risk_parts.append("monitor your heart condition carefully while taking this drug")
    if profile.diabetes:
        risk_parts.append("monitor your blood sugar closely while taking this drug")
    if profile.hypertension:
        risk_parts.append("this drug may affect your blood pressure")
    if profile.other_medications:
        meds = ", ".join(profile.other_medications[:2])
        risk_parts.append(f"taking {meds} alongside requires careful monitoring")

    if risk_parts:
        # Only use the top 2 most relevant risks
        lines.append("Important: " + "; ".join(risk_parts[:2]) + ".")
    elif drug_info.warnings:
        w_text = _truncate(drug_info.warnings[0].text, 100)
        if not w_text.endswith("."):
            w_text += "."
        lines.append(w_text)

    return " ".join(lines)
