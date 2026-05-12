import requests
import re


def search_drug(drug_name: str) -> dict:
    """Search DailyMed for a drug by name, return the best matching result."""
    url = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
    params = {"drug_name": drug_name, "pagesize": 10}
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    results = data.get("data") or []
    if not results:
        return None

    query = re.sub(r'[^a-z0-9]+', ' ', drug_name.lower()).strip()
    query_tokens = [t for t in query.split() if t]

    def score_result(item: dict) -> int:
        title = str(item.get("title", "")).lower()
        title_clean = re.sub(r'\[[^\]]+\]', ' ', title)
        title_clean = re.sub(r'[^a-z0-9]+', ' ', title_clean).strip()
        title_tokens = set(title_clean.split())
        score = 0

        if query and query == title_clean:
            score += 100
        if query and query in title_clean:
            score += 40
        if query_tokens and all(token in title_tokens for token in query_tokens):
            score += 30
        if query_tokens and title_clean.startswith(query_tokens[0]):
            score += 15

        repackager_markers = ["repack", "repackager", "packager"]
        if any(marker in title for marker in repackager_markers):
            score -= 20

        if "table" in title_clean or "tablet" in title_clean or "capsule" in title_clean or "oral" in title_clean:
            score += 5

        return score

    return max(results, key=score_result)


def fetch_leaflet_text(set_id: str) -> str:
    """Fetch leaflet text from DailyMed XML format."""
    url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{set_id}.xml"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    text = re.sub(r'<[^>]+>', ' ', response.text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:12000]


def preprocess_leaflet(text: str) -> str:
    """Extract the most relevant safety and dosing sections from leaflet text."""
    keywords = [
        "warnings",
        "adverse reactions",
        "side effects",
        "drug interactions",
        "food",
        "contraindications",
        "dosage",
        "overdose",
        "precautions",
        "emergency",
    ]
    lower_text = text.lower()
    spans = []

    for keyword in keywords:
        for match in re.finditer(re.escape(keyword), lower_text, flags=re.IGNORECASE):
            start = match.start()
            end = min(len(text), start + 800)
            spans.append((start, end))

    if not spans:
        return text[:6000]

    spans.sort()
    merged = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    sections = [text[start:end].strip() for start, end in merged if text[start:end].strip()]
    return "\n\n".join(sections)[:6000]


def get_drug_leaflet(drug_name: str) -> str:
    """Main function: drug name -> full leaflet text."""
    print(f"Searching DailyMed for: {drug_name}...")
    result = search_drug(drug_name)
    if not result:
        print(f"'{drug_name}' not found.")
        return None
    print(f"Found DailyMed label: {result.get('title')} [{result.get('setid')}]")
    leaflet_text = fetch_leaflet_text(result.get("setid"))
    processed_text = preprocess_leaflet(leaflet_text)
    print(f"Retrieved {len(leaflet_text)} characters, using {len(processed_text)} relevant characters.")
    return processed_text
