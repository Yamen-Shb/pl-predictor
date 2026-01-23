import json
import os

METADATA_PATH = "data/metadata/fetch_state.json"

def load_metadata():
    if not os.path.exists(METADATA_PATH):
        return {}
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


def save_metadata(metadata):
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)


def update_last_fetched_date(competition_key, date_str):
    metadata = load_metadata()
    metadata.setdefault(competition_key, {})
    metadata[competition_key]["last_fetched_date"] = date_str
    save_metadata(metadata)
