import json
import os
from pathlib import Path
def get_params(name: str, prompt_path: str, video_path: str) -> str:
    params = {
    "name": name,
    "prompt_path": prompt_path,
    "video_path": video_path,
    "guidance": 3,
    "edge": {
        "control_weight": 0.5
    },
    "vis": {
        "control_weight": 0.2
    }
}

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    file_path = project_root / "params.json"
    with open(file_path, "w") as f:
        json.dump(params, f)
    return file_path