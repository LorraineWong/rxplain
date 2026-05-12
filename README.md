# Legimed - Your Medication, Made Legible

Legimed turns a medication name or medicine-box photo into a clear, personalized patient guide using Gemma 4 local inference and official NIH DailyMed drug labels.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Hackathon](https://img.shields.io/badge/Kaggle-Gemma%204%20Good%20Hackathon-orange)](https://www.kaggle.com/competitions/gemma-4-good-hackathon)
[![Track](https://img.shields.io/badge/Track-Health%20%26%20Sciences-green)]()
[![Focus](https://img.shields.io/badge/Focus-Digital%20Equity%20%2B%20Safety-blueviolet)]()

## Summary

Medication labels and leaflets are dense, small, and difficult to act on. Patients often know when to take a pill, but not what side effects matter, what food or drink to avoid, or which warnings apply to their own health profile.

Legimed is a prototype for the Kaggle Gemma 4 Good Hackathon. It uses Gemma 4 as a local multimodal and structured-extraction model to convert official medication label text into a mobile-friendly guide.

The output is designed for patients and caregivers:

- A plain-English summary for the user's health profile
- A morning / afternoon / evening / bedtime dosage timeline
- Side effects grouped by severity and action
- Food and drink interaction chips
- Personalized warnings elevated for pregnancy, age, kidney/liver issues, heart disease, diabetes, asthma, hypertension, allergies, and other medications
- Emergency signs that should trigger urgent medical help

Legimed is not a diagnostic or prescribing system. It is an accessibility layer over official drug-label information and should always be checked against a clinician's advice and the user's prescription label.

## Hackathon Fit

Gemma 4 Good emphasizes useful AI for real-world constraints: local intelligence, multimodal understanding, edge deployment, privacy, safety, and social impact. Legimed targets these directly.

| Hackathon theme | How Legimed addresses it |
|---|---|
| Health & Sciences | Makes official medication information easier for patients to understand and act on. |
| Digital Equity | Helps people with low health literacy, limited English fluency, visual difficulty, or caregiver needs. |
| Safety & Trust | Grounds extraction in NIH DailyMed labels, validates output with Pydantic, and keeps medical-disclaimer boundaries explicit. |
| Multimodal Gemma 4 | Uses Gemma 4 vision to read a medicine-box photo and Gemma 4 text inference to extract structured medication guidance. |
| Local / edge inference | Runs model inference in a Colab GPU demo environment and is designed for future pharmacy-workstation or clinic-local deployment. |

## Why This Matters

Medication errors are often caused by misunderstanding, not lack of information. The information exists, but it is buried in long leaflets and written for professionals.

Legimed focuses on one workflow:

1. Identify the medicine.
2. Retrieve the official label.
3. Extract the parts a patient needs.
4. Personalize warning priority.
5. Render a guide that can be read quickly on a phone.

This is intentionally not a general medical chatbot. It is a constrained, grounded pipeline for one high-impact medication-literacy task.

## How It Works

```text
User fills in a temporary health profile (age, sex, conditions, allergies, medications)
        |
        v
User takes a photo with device camera or uploads an image, or types a medicine name
        |
        v
Gemma 4 vision extracts medicine name from image
        |
        v
DailyMed API retrieves official NIH drug-label text
        |
        v
Gemma 4 extracts validated DrugInfo JSON
        |
        v
Python personalization rules reorder relevant warnings
        |
        v
FastAPI returns a mobile-friendly HTML patient guide
```

## Architecture

```text
legimed/
├── src/
│   ├── server.py       # FastAPI backend and HTML guide renderer
│   ├── index.html      # Mobile-first frontend
│   ├── vision.py       # Gemma 4 vision drug-name extraction
│   ├── dailymed.py     # NIH DailyMed label retrieval
│   ├── extract.py      # Gemma 4 structured JSON extraction
│   ├── personalise.py  # Deterministic profile-based warning priority
│   └── schema.py       # Pydantic data models
├── requirements.txt
├── LICENSE
└── README.md
```

## Gemma 4 Usage

Legimed uses Gemma 4 in two places:

1. Image input: `src/vision.py`
   - Input: medication-box photo
   - Output: a clean medicine name
   - Method: local model inference with `processor.apply_chat_template()` and `model.generate()`

2. Label extraction: `src/extract.py`
   - Input: DailyMed label text plus the Pydantic JSON schema
   - Output: validated `DrugInfo`
   - Method: constrained JSON extraction, followed by validation and small enum cleanup

The personalization summary is deliberately deterministic Python logic, not another model call. This keeps the medical personalization step auditable and easier to debug.

## Demo

The current demo is built for Google Colab because Gemma 4 local inference needs GPU memory that may not be available on a typical laptop.

[Open the Colab demo](https://colab.research.google.com/drive/1JCgQPB0JUsRntVpPzwljnRXo01ABTuyF?usp=sharing)

Recommended Colab flow:

```bash
git clone https://github.com/LorraineWong/legimed.git /content/legimed
cd /content/legimed
pip install -r requirements.txt
pip install torch transformers accelerate
```

Then load the Gemma 4 model and processor in the notebook, and launch the FastAPI app:

```python
import sys
sys.path.insert(0, "/content/legimed/src")

# Load your approved Gemma 4 checkpoint here.
# The exact model class and model id should match the checkpoint used in the Kaggle submission notebook.
model = ...
processor = ...

from server import launch
launch(model, processor, port=7860)
```

If running in Colab, expose port `7860` using the tunnel method in the notebook so judges can open the demo UI.

## API Endpoints

`GET /`

Serves the mobile web UI from `src/index.html`.

`POST /scan`

Input:

```json
{
  "image_b64": "base64-encoded-image"
}
```

Output:

```json
{
  "ok": true,
  "drug_name": "Metformin",
  "method": "gemma4"
}
```

`POST /generate`

Input:

```json
{
  "drug_name": "warfarin",
  "age_group": "elderly",
  "sex": "female",
  "pregnant": false,
  "breastfeeding": false,
  "kidney_issue": false,
  "liver_issue": false,
  "heart_condition": true,
  "diabetes": false,
  "hypertension": true,
  "asthma": false,
  "allergies": "aspirin",
  "other_medications": "metformin, lisinopril"
}
```

Output:

```json
{
  "ok": true,
  "html": "<div>...</div>"
}
```

## Safety Design

Legimed includes several safety boundaries:

- It uses official DailyMed label text as the source document.
- It validates model output against Pydantic models before rendering.
- It uses fixed enums for severity, food interaction action, and dosage timing.
- It keeps personalization rule-based and inspectable.
- It avoids diagnosis and prescribing claims.
- It displays clear "for reference only" medical disclaimers.

Known limitations:

- DailyMed search currently uses the first matching result, which may not always be the exact brand, strength, or formulation.
- Dosage text is extracted from labels by the model and must be checked against the patient's actual prescription label.
- The prototype currently focuses on English output.
- The demo depends on internet access for DailyMed retrieval, although Gemma inference itself is local.
- This is a hackathon prototype, not a certified medical device.

## Evaluation Helper

The repo includes a small smoke-test helper for the Kaggle writeup:

```python
import sys
sys.path.insert(0, "/content/legimed")

from scripts.evaluate_drugs import evaluate_drugs, print_markdown_table

results = evaluate_drugs(model, processor)
print_markdown_table(results)
```

It runs common drug names through DailyMed retrieval, Gemma JSON extraction, Pydantic validation, personalization, and guide rendering. The output can be pasted into the Kaggle submission writeup as a simple reproducibility table.

## Tech Stack

| Layer | Technology |
|---|---|
| Model inference | Gemma 4 local inference |
| Vision input | Gemma 4 multimodal chat template |
| Structured output | Pydantic v2 |
| Drug source | NIH DailyMed API |
| Backend | FastAPI + Uvicorn |
| Frontend | Single-file mobile HTML/CSS/JS |
| Demo environment | Google Colab GPU |

## Roadmap

- [x] Drug name input
- [x] Medicine-box image input (upload or live camera via `getUserMedia`)
- [x] DailyMed label retrieval
- [x] Gemma-based structured extraction
- [x] Profile-aware warning prioritization
- [x] Mobile-first visual guide
- [x] Temporary health profile with save/collapse UX
- [x] UI lock during guide generation to prevent mid-flight data changes
- [ ] Stronger DailyMed result selection and formulation confirmation
- [ ] More robust JSON repair and extraction diagnostics
- [ ] Multilingual output for Mandarin, Malay, and Spanish
- [ ] Source citations linking each warning back to the label section
- [ ] Medication interaction checks across multiple drugs
- [ ] Offline packaged label cache for low-connectivity clinics

## Project Status

Legimed is an open-source hackathon prototype built for the Kaggle Gemma 4 Good Hackathon.

Primary target track: Health & Sciences.

Additional fit: Digital Equity, Safety & Trust, and edge/local inference.

Author: Lorraine Wong

License: Apache 2.0
