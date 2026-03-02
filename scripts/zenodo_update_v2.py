"""Zenodo: add publisher + publish draft 18833371."""
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv(os.path.join("d:", os.sep, "00.test", "PAPER", "EthicaAI", ".env"))
token = os.getenv("ZENODO_ACCESS_TOKEN")
h = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}
new_id = "18833371"

# Add publisher field
metadata = {
    "metadata": {
        "title": "From Situational to Unconditional: The Spectrum of Moral Commitment Required for Multi-Agent Survival in Non-linear Social Dilemmas",
        "resource_type": {"id": "publication-preprint"},
        "publication_date": "2026-03-02",
        "publisher": "Zenodo",
        "description": (
            "NeurIPS 2026 submission. We establish the Moral Commitment Spectrum: "
            "a systematic relationship between environmental severity and the minimum "
            "moral commitment required for multi-agent system survival. "
            "Code: https://github.com/neogenesislab/EthicaAI-NeurIPS2026"
        ),
        "creators": [
            {"person_or_org": {"type": "personal", "given_name": "Anonymous", "family_name": "Author"}}
        ],
    }
}

print("=== Updating metadata with publisher ===")
rm = requests.put(
    "https://zenodo.org/api/records/" + new_id + "/draft",
    data=json.dumps(metadata),
    headers=h,
)
print("  Status:", rm.status_code)
if rm.status_code != 200:
    print("  Error:", rm.text[:500])

print("\n=== Publishing ===")
rp = requests.post(
    "https://zenodo.org/api/records/" + new_id + "/draft/actions/publish",
    headers={"Authorization": "Bearer " + token},
)
print("  Status:", rp.status_code)
if rp.ok:
    pub = rp.json()
    doi = pub.get("doi", "check zenodo")
    pid = pub.get("pids", {}).get("doi", {}).get("identifier", doi)
    print("  DOI:", pid)
    print("  URL: https://zenodo.org/records/" + new_id)
else:
    print("  Error:", rp.text[:500])

print("\nDone!")
