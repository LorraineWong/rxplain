# Rxplain — Your Medication, Made Legible

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Hackathon](https://img.shields.io/badge/Kaggle-Gemma%204%20Good%20Hackathon-orange)](https://www.kaggle.com/competitions/gemma-4-good-hackathon)
[![Track](https://img.shields.io/badge/Track-Health%20%26%20Sciences-green)]()
[![Track](https://img.shields.io/badge/Track-Digital%20Equity-blueviolet)]()

> Turn any medicine box or drug name into a plain-language, personalized patient guide — powered by Gemma 4 and NIH DailyMed.

---

## The Problem

Medication leaflets are written for pharmacists, not patients. The information needed to take a drug safely — which side effects matter for *you*, what foods to avoid, when to call a doctor — is buried in pages of clinical text most patients cannot parse.

This is not an information gap. It is a comprehension gap. And it disproportionately affects people with low health literacy, older adults, non-native speakers, and anyone managing a complex condition without reliable access to a clinician.

---

## What Rxplain Does

1. User fills in a temporary health profile (age, sex, conditions, allergies, current medications)
2. User photographs a medicine box with their camera, or types a drug name
3. Gemma 4 reads the photo and identifies the medicine
4. NIH DailyMed returns the official drug label
5. Gemma 4 extracts structured medication data from the label
6. Deterministic personalisation rules elevate warnings relevant to the user's profile
7. The app returns a mobile-friendly guide with:
   - A plain-English personalised summary
   - Morning / afternoon / evening / bedtime dosage timeline
   - Side effects ranked by severity (Emergency / Call doctor / Monitor)
   - Food and drink interactions
   - Warnings prioritised for the user's conditions and allergies
   - Emergency signs to watch for

---

## Gemma 4 in Action

Rxplain uses Gemma 4 for two distinct tasks:

### 1. Vision — Medicine Box Recognition (`src/vision.py`)

```python
messages = [{"role": "user", "content": [
    {"type": "image", "image": pil_image},
    {"type": "text",  "text": "What is the drug name shown on this box? Reply with ONLY the drug name."}
]}]
output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
```

The model reads a photo of a medicine box and returns a clean drug name, stripping dosage numbers and brand taglines. This lets users scan a box instead of typing.

### 2. Structured Extraction — Label → Patient Guide (`src/extract.py`)

```python
prompt = f"""You are a clinical pharmacist AI. Extract medication information from the leaflet below.
Output ONLY valid JSON matching this schema: {schema}
LEAFLET TEXT: {leaflet_text}"""
output = model.generate(**inputs, max_new_tokens=3072, do_sample=False)
```

Gemma 4 reads up to 6,000 characters of the official NIH DailyMed label and returns a validated `DrugInfo` JSON object — side effects with severity levels, food interactions, patient-facing warnings, dosage instructions, and emergency signs.

The personalisation step (warning reordering, summary generation) is deliberately **not** an AI call — it is deterministic Python logic, making it auditable and safe.

---

## Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JCgQPB0JUsRntVpPzwljnRXo01ABTuyF?usp=sharing)

```bash
# In Colab
git clone https://github.com/LorraineWong/rxplain.git /content/rxplain
cd /content/rxplain
pip install -r requirements.txt
pip install torch transformers accelerate
```

```python
import sys
sys.path.insert(0, "/content/rxplain/src")

# Load your Gemma 4 checkpoint
model = ...
processor = ...

from server import launch
launch(model, processor, port=7860)
# Expose port 7860 via Colab tunnel to open the UI
```

---

## Hackathon Track Fit

| Track | How Rxplain qualifies |
|---|---|
| **Health & Sciences** | Democratises official drug-label knowledge for patients and caregivers who cannot interpret clinical text |
| **Digital Equity & Inclusivity** | Targets users with low health literacy, older adults, non-native speakers, and those without easy clinician access |
| **Safety & Trust** | Grounds all output in NIH DailyMed labels; validates with Pydantic; personalisation is rule-based and inspectable; explicit medical disclaimers throughout |

---

## Project Structure

```
rxplain/
├── src/
│   ├── server.py       # FastAPI backend + HTML guide renderer
│   ├── index.html      # Mobile-first single-file frontend
│   ├── vision.py       # Gemma 4 vision — medicine box → drug name
│   ├── dailymed.py     # NIH DailyMed label retrieval + preprocessing
│   ├── extract.py      # Gemma 4 text — label → structured DrugInfo JSON
│   ├── personalise.py  # Deterministic profile-aware warning prioritisation
│   └── schema.py       # Pydantic data models
├── scripts/
│   └── evaluate_drugs.py  # Smoke-test helper for submission writeup
├── requirements.txt
└── LICENSE
```

---

## Safety Design

- All content is grounded in official NIH DailyMed drug labels — no free-form generation
- Model output is validated against strict Pydantic schemas before rendering
- Fixed enums for severity, food interaction type, and dosage timing prevent hallucinated categories
- Personalisation logic is deterministic Python, not a second model call
- Explicit "for reference only — consult your doctor or pharmacist" disclaimer throughout the UI
- No user data is stored or transmitted beyond the current session

**Known limitations:** DailyMed search returns the closest label match, which may differ in formulation or strength from the user's specific product. Dosage shown is from the label, not the user's prescription. English only in this prototype.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model inference | Gemma 4 (local, via `transformers`) |
| Vision input | Gemma 4 multimodal chat template |
| Structured output | Pydantic v2 |
| Drug data source | NIH DailyMed API |
| Backend | FastAPI + Uvicorn |
| Frontend | Single-file mobile HTML/CSS/JS |
| Camera input | `getUserMedia` API (desktop + mobile) |
| Demo environment | Google Colab (A100 GPU) |

---

## Roadmap

- [x] Drug name input + medicine-box camera capture
- [x] NIH DailyMed label retrieval
- [x] Gemma 4 structured extraction (vision + text)
- [x] Profile-aware warning prioritisation (conditions, allergies, medications)
- [x] Mobile-first patient guide UI
- [x] Temporary health profile with save/collapse UX
- [ ] Multilingual output (Mandarin, Malay, Spanish)
- [ ] Source citations linking each warning to its label section
- [ ] Multi-drug interaction checks
- [ ] Offline label cache for low-connectivity settings

---

## Author & License

**Lorraine Wong** · Apache 2.0

*Rxplain is a hackathon prototype, not a certified medical device.*
