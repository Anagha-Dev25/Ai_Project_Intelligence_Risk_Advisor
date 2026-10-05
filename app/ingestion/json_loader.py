import json


def load_json(file_path: str) -> str:
    """Extract and format JSON project documents."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return json.dumps(data, indent=2)
