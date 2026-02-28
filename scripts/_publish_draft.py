"""Retry publishing Zenodo draft 18795926."""
from dotenv import load_dotenv
import os, requests

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
token = os.getenv("ZENODO_ACCESS_TOKEN")
headers = {"Authorization": f"Bearer {token}"}

# First check current state
r = requests.get(
    "https://zenodo.org/api/deposit/depositions/18795926",
    headers=headers,
    timeout=30,
)
print(f"GET status: {r.status_code}")
if r.status_code == 200:
    d = r.json()
    print(f"  State: {d.get('state')}")
    print(f"  Notes: {d.get('metadata', {}).get('notes', '(empty)')[:100]}")

    if d.get("state") == "unsubmitted":
        print("\nPublishing...")
        pub = requests.post(
            "https://zenodo.org/api/deposit/depositions/18795926/actions/publish",
            headers=headers,
            timeout=120,
        )
        print(f"  Publish status: {pub.status_code}")
        if pub.status_code == 202:
            result = pub.json()
            print(f"  SUCCESS! Record ID: {result['id']}")
            print(f"  URL: https://zenodo.org/records/{result['id']}")
        else:
            print(f"  Response: {pub.text[:500]}")
    elif d.get("state") == "done":
        print("  Already published!")
else:
    print(f"  Error: {r.text[:300]}")
