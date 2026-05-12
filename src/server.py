"""
server.py — FastAPI backend for Legimed.
Replaces Gradio. Serves the HTML frontend and exposes two API endpoints.
"""

import base64
import gc
import json
import re
from io import BytesIO

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dailymed import get_drug_leaflet
from personalise import personalise, generate_personal_summary
from schema import UserProfile
from vision import image_to_drug_name

app = FastAPI()

# Global model references — set by launch()
_model = None
_tokenizer = None
_processor = None


# ── Request schemas ────────────────────────────────────────────────────────────

class ScanRequest(BaseModel):
    image_b64: str  # base64-encoded image (no data URL prefix)


class GenerateRequest(BaseModel):
    drug_name: str
    age_group: str = "adult"
    sex: str = "prefer_not_to_say"
    pregnant: bool = False
    breastfeeding: bool = False
    heart_condition: bool = False
    diabetes: bool = False
    hypertension: bool = False
    asthma: bool = False
    kidney_issue: bool = False
    liver_issue: bool = False
    other_conditions: str = ""
    allergies: str = ""
    other_medications: str = ""


# ── Helpers ────────────────────────────────────────────────────────────────────

DRUG_KEYWORDS = {
    "drug", "drugs", "medication", "medications", "medicine", "medicines",
    "maoi", "maois", "inhibitor", "inhibitors", "nsaid", "nsaids",
    "antibiotic", "antibiotics", "supplement", "supplements",
    "containing", "products", "anticoagulant", "anticoagulants",
    "antidepressant", "antidepressants", "sedative", "sedatives",
    "prescription", "otc", "tablet", "tablets", "capsule", "capsules",
}


def _is_food(substance: str) -> bool:
    words = set(substance.lower().replace("-", " ").split())
    return not bool(words & DRUG_KEYWORDS)


def _csv(v: str):
    return [x.strip() for x in v.split(",") if x.strip()] if v else []


def _truncate(text: str, max_chars: int = 120) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) <= max_chars:
        return text

    before_limit = text[:max_chars]
    boundaries = [before_limit.rfind("."), before_limit.rfind("!"), before_limit.rfind("?")]
    boundary = max(boundaries)
    if boundary != -1:
        return before_limit[:boundary + 1].strip()

    match = re.search(r'[.!?]', text[max_chars:])
    if match:
        return text[:max_chars + match.start() + 1].strip()

    return text


def format_guide(drug_info, personal_summary: str) -> str:
    """Render DrugInfo as a self-contained HTML string."""
    import html as h

    def e(v):
        return h.escape(str(v)) if v else ""

    sev_border = {"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#10B981"}
    sev_tag_bg = {"HIGH": "#FEE2E2", "MEDIUM": "#FEF3C7", "LOW": "#D1FAE5"}
    sev_text   = {"HIGH": "#991B1B", "MEDIUM": "#92400E", "LOW": "#065F46"}
    sev_label  = {"HIGH": "Emergency", "MEDIUM": "Call doctor", "LOW": "Monitor"}
    food_bg    = {"avoid": "#FEE2E2", "caution": "#FEF3C7", "ok": "#D1FAE5"}
    food_tc    = {"avoid": "#991B1B", "caution": "#92400E", "ok": "#065F46"}
    food_icon  = {"avoid": "🚫", "caution": "⚠️", "ok": "✅"}

    def card(content, bg="#fff", border="#E2E8F0"):
        return (f'<div style="background:{bg};border:1px solid {border};border-radius:16px;'
                f'padding:14px 16px;margin-bottom:12px;box-shadow:0 1px 6px rgba(0,0,0,0.06);">'
                f'{content}</div>')

    def slabel(t, color="#00A878"):
        return (f'<div style="font-size:10px;font-weight:800;color:{color};'
                f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;">{t}</div>')

    parts = ['<div style="font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',sans-serif;color:#1A202C;">']

    # Header
    parts.append(
        f'<div style="background:linear-gradient(135deg,#00A878,#047857);border-radius:16px;'
        f'padding:20px;margin-bottom:12px;color:#fff;">'
        f'<div style="font-size:10px;font-weight:800;opacity:0.8;text-transform:uppercase;'
        f'letter-spacing:0.08em;margin-bottom:6px;">Based on official drug label</div>'
        f'<div style="font-size:22px;font-weight:800;color:#fff;line-height:1.2;">{e(drug_info.drug_name)}</div>'
        f'<div style="font-size:12px;opacity:0.9;margin-top:4px;color:#fff;">{e(drug_info.active_ingredient)}</div>'
        f'<div style="display:inline-block;margin-top:8px;background:rgba(255,255,255,0.18);'
        f'border:1px solid rgba(255,255,255,0.25);padding:3px 10px;border-radius:999px;'
        f'font-size:11px;font-weight:600;color:#fff;">{e(drug_info.drug_class)}</div>'
        f'</div>'
    )

    # Summary
    if personal_summary:
        parts.append(card(
            slabel("Key Summary") +
            f'<div style="font-size:14px;font-weight:600;color:#1A202C;line-height:1.75;background:#ECFDF5;'
            f'border-radius:10px;padding:12px 13px;border-left:3px solid #00A878;">'
            f'{e(personal_summary)}</div>'
        ))

    # Label timing summary
    time_slots = {"morning": ("🌅", "Morning"), "afternoon": ("☀️", "Afternoon"),
                  "evening": ("🌆", "Evening"), "bedtime": ("🌙", "Bedtime")}
    dose_map = {d.time_of_day: d for d in (drug_info.dosage_instructions or [])}
    grid = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;">'
    for slot, (icon, label) in time_slots.items():
        d = dose_map.get(slot)
        if d:
            amt = e(d.amount or "—")
            food = "With food" if d.with_food else "Without food"
            grid += (f'<div style="background:#ECFDF5;border:1px solid #A7F3D0;'
                     f'border-radius:10px;padding:8px 4px;text-align:center;">'
                     f'<div style="font-size:16px;">{icon}</div>'
                     f'<div style="font-size:9px;color:#4A5568;margin-top:2px;">{label}</div>'
                     f'<div style="font-size:11px;font-weight:800;color:#065F46;margin-top:2px;">{amt}</div>'
                     f'<div style="font-size:9px;color:#6B7280;">{food}</div></div>')
        else:
            grid += (f'<div style="background:#F1F5F9;border-radius:10px;padding:8px 4px;text-align:center;">'
                     f'<div style="font-size:16px;opacity:0.2;">{icon}</div>'
                     f'<div style="font-size:9px;color:#94A3B8;margin-top:2px;">{label}</div>'
                     f'<div style="font-size:12px;color:#CBD5E1;margin-top:2px;">—</div></div>')
    grid += '</div>'
    parts.append(card(
        slabel("Label Timing") +
        grid +
        '<div style="font-size:10px;color:#64748B;line-height:1.5;margin-top:8px;">'
        'Verify dose and timing with your prescription label or pharmacist.</div>'
    ))

    # Side effects
    side_effects = sorted(
        (drug_info.side_effects or [])[:8],
        key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(x.severity, 3)
    )
    if side_effects:
        rows = ""
        for se in side_effects[:4]:
            sev = se.severity or "LOW"
            rows += (f'<div style="display:flex;align-items:flex-start;gap:8px;padding:9px 10px;'
                     f'border-radius:10px;border-left:3px solid {sev_border.get(sev,"#CBD5E1")};'
                     f'background:#F8FAFC;margin-bottom:6px;">'
                     f'<div style="flex:1;">'
                     f'<div style="font-size:12px;font-weight:700;color:#1A202C;">{e(se.name)}</div>'
                     f'<div style="font-size:11px;color:#475569;line-height:1.5;margin-top:1px;">'
                     f'{e(se.description)}</div></div>'
                     f'<span style="font-size:9px;padding:3px 7px;border-radius:999px;'
                     f'background:{sev_tag_bg.get(sev,"#F1F5F9")};'
                     f'color:{sev_text.get(sev,"#334155")};font-weight:700;white-space:nowrap;">'
                     f'{sev_label.get(sev,"Monitor")}</span></div>')
        parts.append(card(slabel("⚡ Side Effects") + rows))

    # Food & drink
    food_items = [fi for fi in (drug_info.food_interactions or []) if _is_food(fi.substance)]
    if food_items:
        chips = '<div style="display:flex;gap:6px;flex-wrap:wrap;">'
        for fi in food_items[:8]:
            action = fi.action or "caution"
            chips += (f'<span style="display:inline-flex;align-items:center;gap:4px;'
                      f'padding:6px 10px;border-radius:999px;'
                      f'background:{food_bg.get(action,"#F1F5F9")};'
                      f'color:{food_tc.get(action,"#334155")};'
                      f'font-size:12px;font-weight:600;">'
                      f'{food_icon.get(action,"⚠️")} {e(fi.substance)}</span>')
        chips += '</div>'
        parts.append(card(slabel("🍽 Food & Drink") + chips))
    else:
        parts.append(card(slabel("🍽 Food & Drink") +
                          '<p style="font-size:12px;color:#6B7280;margin:0;">'
                          'No specific food interactions found.</p>'))

    # Warnings
    warnings = drug_info.warnings or []
    if warnings:
        w_rows = ""
        for w in warnings[:3]:
            text = _truncate(w.text, 120)
            w_rows += (f'<div style="font-size:12px;color:#78350F;padding:5px 0;'
                       f'border-bottom:1px solid #FDE68A;line-height:1.6;">• {e(text)}</div>')
        parts.append(card(slabel("Warnings", "#B45309") + w_rows,
                          bg="#FFFBEB", border="#FDE68A"))

    # Emergency
    emergency = [s for s in (drug_info.emergency_signs or []) if str(s).strip()]
    if emergency:
        e_rows = "".join(
            f'<div style="font-size:12px;color:#7F1D1D;padding:3px 0;line-height:1.6;">• {e(s)}</div>'
            for s in emergency[:3]
        )
        parts.append(card(slabel("Emergency Signs", "#B91C1C") + e_rows,
                          bg="#FEF2F2", border="#FCA5A5"))

    return "".join(parts)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("/content/legimed/src/index.html", "r") as f:
        return f.read()


@app.post("/scan")
async def scan(req: ScanRequest):
    try:
        from PIL import Image
        img_bytes = base64.b64decode(req.image_b64)
        pil_image = Image.open(BytesIO(img_bytes)).convert("RGB")
        drug_name, method = image_to_drug_name(pil_image, model=_model, processor=_processor)
        if drug_name:
            return {"ok": True, "drug_name": drug_name, "method": method}
        return {"ok": False, "drug_name": "", "method": method}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})


@app.post("/generate")
async def generate(req: GenerateRequest):
    from extract import extract_drug_info_robust
    try:
        name = req.drug_name.strip()
        if not name:
            return JSONResponse(status_code=400, content={"ok": False, "error": "No drug name provided."})
        if _model is None or _processor is None:
            return JSONResponse(status_code=503, content={"ok": False, "error": "Model not loaded."})

        leaflet = get_drug_leaflet(name)
        if not leaflet:
            return {"ok": False, "error": f"'{name}' not found in DailyMed. Try the generic name."}

        is_female = req.sex == "female"
        profile = UserProfile(
            age_group=req.age_group,
            sex=req.sex,
            pregnant=req.pregnant and is_female,
            breastfeeding=req.breastfeeding and is_female,
            heart_condition=req.heart_condition,
            diabetes=req.diabetes,
            hypertension=req.hypertension,
            asthma=req.asthma,
            kidney_issue=req.kidney_issue,
            liver_issue=req.liver_issue,
            other_conditions=req.other_conditions,
            allergies=_csv(req.allergies),
            other_medications=_csv(req.other_medications),
        )

        drug_info = extract_drug_info_robust(leaflet, _model, _processor)
        drug_info = personalise(drug_info, profile)
        summary = generate_personal_summary(drug_info, profile)
        guide_html = format_guide(drug_info, summary)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"ok": True, "html": guide_html}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})


# ── Launch ─────────────────────────────────────────────────────────────────────

def launch(model, processor, port=7860):
    import asyncio
    global _model, _tokenizer, _processor
    _model = model
    _tokenizer = processor   # Gemma 4 tokenizer == processor
    _processor = processor

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    asyncio.get_event_loop().run_until_complete(server.serve())

if __name__ == "__main__":
    launch()
