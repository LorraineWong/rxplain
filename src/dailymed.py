import requests
import re


def search_drug(drug_name: str) -> dict:
    """Search DailyMed for a drug by name, return first result."""
    url = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
    params = {"drug_name": drug_name, "pagesize": 1}
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    if not data.get("data"):
        return None
    return data["data"][0]


def fetch_leaflet_text(set_id: str) -> str:
    """Fetch leaflet text from DailyMed XML format."""
    url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{set_id}.xml"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    text = re.sub(r'<[^>]+>', ' ', response.text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:12000]


def get_drug_leaflet(drug_name: str) -> str:
    """Main function: drug name -> full leaflet text."""
    print(f"Searching DailyMed for: {drug_name}...")
    result = search_drug(drug_name)
    if not result:
        print(f"'{drug_name}' not found.")
        return None
    print(f"Found DailyMed label: {result.get('title')} [{result.get('setid')}]")
    leaflet_text = fetch_leaflet_text(result.get("setid"))
    print(f"Retrieved {len(leaflet_text)} characters.")
    return leaflet_text
