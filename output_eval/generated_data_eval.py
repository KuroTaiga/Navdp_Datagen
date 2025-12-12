from __future__ import annotations

import os
import json
import statistics
import base64
from typing import Any, List, TYPE_CHECKING
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

if TYPE_CHECKING:
    from openai.types.chat import (
        ChatCompletionContentPartParam,
        ChatCompletionMessageParam,
    )

# ---------------------------
# LOAD ENVIRONMENT VARIABLES
# ---------------------------
load_dotenv()

API_KEY = os.getenv("DATAEVAL_API_KEY")
API_ENDPOINT = os.getenv("DATAEVAL_ENDPOINT", "https://api.openai.com/v1")
API_MODEL = os.getenv("DATAEVAL_MODEL", "chat4o")
APP_VERSION = os.getenv("DATAEVAL_APP_VERSION", "1.0")

QWEN_MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2-VL-7B-Instruct")
QWEN_MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "256"))
QWEN_DEVICE = os.getenv("QWEN_DEVICE", "auto")
PROVIDER = os.getenv("DATAEVAL_PROVIDER", "openai").strip().lower()

VALID_PROVIDERS = {"openai", "qwen_local"}
if PROVIDER not in VALID_PROVIDERS:
    raise RuntimeError(f"Unsupported DATAEVAL_PROVIDER '{PROVIDER}'. Valid options: {sorted(VALID_PROVIDERS)}")

if PROVIDER == "openai" and not API_KEY:
    raise RuntimeError("Missing DATAEVAL_API_KEY in .env file")


def create_openai_client():
    from openai import OpenAI, AzureOpenAI  # Local import to keep dependency optional for Qwen

    endpoint = (API_ENDPOINT or "").rstrip("/")
    if endpoint and ("azure.com" in endpoint or "cognitiveservices" in endpoint):
        if not APP_VERSION:
            raise RuntimeError("DATAEVAL_APP_VERSION must be set for Azure OpenAI usage")
        return AzureOpenAI(
            api_key=API_KEY,
            azure_endpoint=endpoint,
            api_version=APP_VERSION
        )
    # Default to the public OpenAI API
    return OpenAI(
        api_key=API_KEY,
        base_url=endpoint or None
    )


client = create_openai_client() if PROVIDER == "openai" else None

_QWEN_MODEL = None
_QWEN_PROCESSOR = None
_QWEN_DEVICE_CACHE = None

# ---------------------------
# CONFIGURABLE PARAMETERS
# ---------------------------

DATA_ROOT = "../data/path_video_frames_random_humans_65k"

MAX_FRAMES_PER_VIDEO = 12
OVERALL_CUTOFF = 0

PROMPT_PATH = Path(__file__).with_name("eval_prompt.txt")


def load_eval_prompt():
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Evaluation prompt file not found: {PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8")


EVAL_PROMPT = load_eval_prompt()


def load_qwen_model():
    """Lazy-load the local Qwen-VL model and processor."""
    global _QWEN_MODEL, _QWEN_PROCESSOR, _QWEN_DEVICE_CACHE

    if PROVIDER != "qwen_local":
        raise RuntimeError("Qwen model requested but DATAEVAL_PROVIDER is not 'qwen_local'")

    if _QWEN_MODEL is not None and _QWEN_PROCESSOR is not None:
        return _QWEN_MODEL, _QWEN_PROCESSOR, _QWEN_DEVICE_CACHE

    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "transformers and torch are required for DATAEVAL_PROVIDER=qwen_local"
        ) from exc

    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForVision2Seq.from_pretrained(
        QWEN_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if QWEN_DEVICE == "auto" else None,
    )

    target_device: Any
    if QWEN_DEVICE == "auto":
        target_device = getattr(model, "device", torch.device("cpu"))
    else:
        target_device = torch.device(QWEN_DEVICE)
        model.to(target_device)

    _QWEN_MODEL = model
    _QWEN_PROCESSOR = processor
    _QWEN_DEVICE_CACHE = target_device
    return _QWEN_MODEL, _QWEN_PROCESSOR, _QWEN_DEVICE_CACHE


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
    if not frames_b64:
        raise ValueError(f"Could not extract frames from video: {video_path}")

    if PROVIDER == "openai":
        return evaluate_with_openai(video_path, frames_b64)
    if PROVIDER == "qwen_local":
        return evaluate_with_qwen(video_path, frames_b64)
    raise RuntimeError(f"Unsupported provider: {PROVIDER}")


def build_user_instructions(video_name: str) -> str:
    return (
        f"Video file: {video_name}. Review the evenly sampled frames provided below. "
        "Assign 0-10 scores for Quality, Realism, and Overall fidelity, then explain your judgment. "
        "Respond in four lines starting with 'Quality:', 'Realism:', 'Overall:', and 'Reason:'."
    )


def evaluate_with_openai(video_path: str, frames_b64: List[str]):
    if client is None:
        raise RuntimeError("OpenAI client is not initialized")

    video_name = Path(video_path).name
    user_instructions = build_user_instructions(video_name)

    content_blocks: List["ChatCompletionContentPartParam"] = [
        {"type": "text", "text": user_instructions}
    ]
    for b64 in frames_b64:
        content_blocks.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
            }
        })

    messages: List["ChatCompletionMessageParam"] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": EVAL_PROMPT}],
        },
        {
            "role": "user",
            "content": content_blocks,
        },
    ]

    response = client.chat.completions.create(
        model=API_MODEL,
        messages=messages,
    )

    message = response.choices[0].message
    text = _coerce_message_text(message)
    if not text:
        raise ValueError(f"LLM returned empty content for {video_path}")
    return _parse_score_block(text, video_path)


def evaluate_with_qwen(video_path: str, frames_b64: List[str]):
    model, processor, target_device = load_qwen_model()
    video_name = Path(video_path).name

    from io import BytesIO
    from PIL import Image

    images = []
    for b64 in frames_b64:
        img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        images.append(img)

    text_prompt = f"{EVAL_PROMPT.strip()}\n\n{build_user_instructions(video_name)}"
    user_content = [{"type": "text", "text": text_prompt}]
    user_content.extend({"type": "image"} for _ in images)

    messages = [{"role": "user", "content": user_content}]
    inputs = processor(messages=messages, images=images, return_tensors="pt")
    if hasattr(inputs, "to") and target_device is not None:
        inputs = inputs.to(target_device)

    generated_ids = model.generate(**inputs, max_new_tokens=QWEN_MAX_NEW_TOKENS)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    if not text:
        raise ValueError(f"Qwen returned empty content for {video_path}")
    return _parse_score_block(text, video_path)


def _coerce_message_text(message: Any) -> str:
    content = getattr(message, "content", "")

    def _extract_text(part: Any) -> str:
        if isinstance(part, dict):
            if part.get("type") == "text":
                return part.get("text", "")
            return ""
        if getattr(part, "type", None) == "text":
            return getattr(part, "text", "")
        return ""

    if isinstance(content, list):
        return "\n".join(filter(None, (_extract_text(p) for p in content))).strip()
    return (content or "").strip()


def _parse_score_block(text: str, video_path: str):
    lines = [x.strip() for x in text.split("\n") if x.strip()]
    out: dict[str, Any] = {}

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
