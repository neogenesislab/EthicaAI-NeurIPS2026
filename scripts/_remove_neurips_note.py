"""Remove NeurIPS submission note from Zenodo metadata."""
from dotenv import load_dotenv
import os, requests, json

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
token = os.getenv("ZENODO_ACCESS_TOKEN")
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

# 1. List all depositions
print("=== Listing depositions ===")
resp = requests.get(
    "https://zenodo.org/api/deposit/depositions",
    headers=headers,
    params={"size": 10}
)
if resp.status_code != 200:
    print(f"ERROR listing: {resp.status_code}: {resp.text[:300]}")
    exit(1)

for d in resp.json():
    meta = d.get("metadata", {})
    notes = meta.get("notes", "(none)")
    state = d.get("state", "?")
    vid = meta.get("version", "?")
    print(f"  ID: {d['id']} | v{vid} | state: {state} | notes: {notes[:80]}")

# 2. Find records with NeurIPS in notes and clear them
for record_id in [18795185, 18795081, 18728438]:
    print(f"\n=== Trying record {record_id} ===")
    resp = requests.get(
        f"https://zenodo.org/api/deposit/depositions/{record_id}",
        headers=headers,
    )
    if resp.status_code != 200:
        print(f"  Cannot access {record_id}: {resp.status_code}")
        continue

    data = resp.json()
    state = data.get("state")
    current_notes = data.get("metadata", {}).get("notes", "")
    print(f"  State: {state}")
    print(f"  Current notes: {current_notes}")

    if "neurips" not in current_notes.lower():
        print("  No NeurIPS mention - skipping")
        continue

    if state == "done":
        print("  Record is published - creating new version draft...")
        resp2 = requests.post(
            f"https://zenodo.org/api/deposit/depositions/{record_id}/actions/newversion",
            headers=headers,
        )
        if resp2.status_code == 201:
            new_draft_url = resp2.json().get("links", {}).get("latest_draft", "")
            print(f"  New draft URL: {new_draft_url}")
            resp3 = requests.get(new_draft_url, headers=headers)
            if resp3.status_code == 200:
                draft_data = resp3.json()
                draft_id = draft_data["id"]
                print(f"  Draft ID: {draft_id}")
                new_meta = draft_data["metadata"].copy()
                new_meta["notes"] = ""
                update_resp = requests.put(
                    f"https://zenodo.org/api/deposit/depositions/{draft_id}",
                    data=json.dumps({"metadata": new_meta}),
                    headers=headers,
                )
                if update_resp.status_code == 200:
                    print("  Notes cleared!")
                    pub_resp = requests.post(
                        f"https://zenodo.org/api/deposit/depositions/{draft_id}/actions/publish",
                        headers=headers,
                    )
                    if pub_resp.status_code == 202:
                        result = pub_resp.json()
                        print(f"  PUBLISHED! ID: {result['id']}")
                        print(f"  URL: https://zenodo.org/records/{result['id']}")
                    else:
                        print(f"  Publish error: {pub_resp.status_code}: {pub_resp.text[:300]}")
                else:
                    print(f"  Update error: {update_resp.status_code}: {update_resp.text[:300]}")
            else:
                print(f"  Draft fetch error: {resp3.status_code}")
        else:
            print(f"  New version error: {resp2.status_code}: {resp2.text[:300]}")
        break
    elif state == "unsubmitted":
        meta = data["metadata"].copy()
        meta["notes"] = ""
        update_resp = requests.put(
            f"https://zenodo.org/api/deposit/depositions/{record_id}",
            data=json.dumps({"metadata": meta}),
            headers=headers,
        )
        if update_resp.status_code == 200:
            print(f"  Notes cleared on draft {record_id}!")
        else:
            print(f"  Update error: {update_resp.status_code}: {update_resp.text[:300]}")
        break
