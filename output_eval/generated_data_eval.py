import os
import json
import statistics
import base64
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------
# LOAD ENVIRONMENT VARIABLES
# ---------------------------
load_dotenv()

API_KEY = os.getenv("DATAEVAL_API_KEY")
API_ENDPOINT = os.getenv("DATAEVAL_ENDPOINT", "https://api.openai.com/v1")
API_MODEL = os.getenv("DATAEVAL_MODEL", "chat4o")
APP_VERSION = os.getenv("DATAEVAL_APP_VERSION", "1.0")

if not API_KEY:
    raise RuntimeError("Missing DATAEVAL_API_KEY in .env file")

client = OpenAI(
    api_key=API_KEY,
    base_url=API_ENDPOINT
)

# ---------------------------
# CONFIGURABLE PARAMETERS
# ---------------------------

DATA_ROOT = "../data/path_video_frames_random_humans_33w"

MAX_FRAMES_PER_VIDEO = 12
OVERALL_CUTOFF = 0

PROMPT_PATH = "eval_prompt.txt"


def load_eval_prompt():
    if not os.path.exists(PROMPT_PATH):
        raise FileNotFoundError(f"Evaluation prompt file not found: {PROMPT_PATH}")
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


EVAL_PROMPT = load_eval_prompt()


# ---------------------------
# FRAME SAMPLING
# ---------------------------

def sample_video_frames(video_path, max_frames=MAX_FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        cap.release()
        return []

    if frame_count <= max_frames:
        indices = list(range(frame_count))
    else:
        indices = np.linspace(0, frame_count - 1, max_frames).astype(int)

    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        _, buffer = cv2.imencode(".jpg", frame)
        b64 = base64.b64encode(buffer).decode("utf-8")
        frames.append(b64)

    cap.release()
    return frames


# ---------------------------
# VIDEO EVALUATION
# ---------------------------

def evaluate_video(video_path):
    frames_b64 = sample_video_frames(video_path)

    content_blocks = []
    for b64 in frames_b64:
        content_blocks.append({
            "type": "input_image",
            "image_base64": b64,
            "image_type": "jpg"
        })

    response = client.chat.completions.create(
        model=API_MODEL,
        messages=[
            {"role": "system", "content": EVAL_PROMPT},
            {"role": "user", "content": content_blocks},
        ],
    )

    text = response.choices[0].message["content"]
    lines = [x.strip() for x in text.split("\n") if x.strip()]
    out = {}

    for line in lines:
        if line.startswith("Quality:"):
            out["quality"] = float(line.split(":")[1].strip())
        elif line.startswith("Realism:"):
            out["realism"] = float(line.split(":")[1].strip())
        elif line.startswith("Overall:"):
            out["overall"] = float(line.split(":")[1].strip())
        elif line.startswith("Reason:"):
            out["reason"] = line.split(":", 1)[1].strip()

    required = ["quality", "realism", "overall", "reason"]
    for k in required:
        if k not in out:
            raise ValueError(f"Invalid agent output for {video_path}: {text}")

    out["pass"] = (out["overall"] >= OVERALL_CUTOFF)
    return out


# ---------------------------
# AGGREGATION & REPORTING
# ---------------------------

def summary_stats(values):
    if len(values) == 0:
        return {}
    return {
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / len(values),
        "median": statistics.median(values),
        "count": len(values)
    }


def create_graph(scores):
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        [scores["quality"], scores["realism"], scores["overall"]],
        labels=["Quality", "Realism", "Overall"]
    )
    plt.title("NavDP 3DGS Video Evaluation Score Distribution")
    plt.ylabel("Score (0–10)")
    plt.grid(True, alpha=0.4)
    plt.savefig("evaluation_distribution.png", dpi=200)
    plt.close()


def iterate_data(curr_path):
    results = {}
    global_scores = {"quality": [], "realism": [], "overall": []}

    if not curr_path.is_dir():
        raise ValueError(f"Provided path is not a directory: {curr_path}")

    for scene in curr_path.iterdir():
        if not scene.is_dir():
            continue

        scene_name = scene.name
        results.setdefault(scene_name, {})

        video_files = list(scene.glob("*.mp4"))
        if not video_files:
            continue

        for file in video_files:
            print(f"Evaluating {scene_name}/{file.name} ...")

            eval_result = evaluate_video(str(file))
            path_id = file.stem
            results[scene_name][path_id] = eval_result

            global_scores["quality"].append(eval_result["quality"])
            global_scores["realism"].append(eval_result["realism"])
            global_scores["overall"].append(eval_result["overall"])

        with open(f"{scene_name}_report.json", "w") as f:
            json.dump(results[scene_name], f, indent=2)

    summary = {
        "quality": summary_stats(global_scores["quality"]),
        "realism": summary_stats(global_scores["realism"]),
        "overall": summary_stats(global_scores["overall"]),
    }

    with open("global_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    create_graph(global_scores)

    return results, summary


# ---------------------------
# ENTRY POINT
# ---------------------------

if __name__ == "__main__":
    datapath = Path(DATA_ROOT)
    results, summary = iterate_data(datapath)
    print("Evaluation complete. Reports generated.")
