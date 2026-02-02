import json
import os
from pathlib import Path

def get_metadata_path():
    # Prefer project root relative to this file, so it works from any cwd
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "data" / "metadata" / "fetch_state.json"

def load_metadata():
    path = get_metadata_path()
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_metadata(metadata):
    path = get_metadata_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def update_last_fetched_date(competition_key, date_str):
    metadata = load_metadata()
    metadata.setdefault(competition_key, {})
    metadata[competition_key]["last_fetched_date"] = date_str
    save_metadata(metadata)


def get_last_upcoming_match_date(competition_key):
    metadata = load_metadata()
    return metadata.get(competition_key, {}).get("last_upcoming_match_date")


def update_last_upcoming_match_date(competition_key, datetime_str):
    metadata = load_metadata()
    metadata.setdefault(competition_key, {})
    metadata[competition_key]["last_upcoming_match_date"] = datetime_str
    save_metadata(metadata)