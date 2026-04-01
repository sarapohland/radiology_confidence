"""
Run MedGemma inference on chest X-ray studies via SGLang and record per-token
logprobs and entropies for downstream confidence estimation.

Prerequisites: SGLang server running with MedGemma, e.g.:
    python -m sglang.launch_server \
        --model-path google/medgemma-1.5-4b-it \
        --port 30000 \
        --enable-multimodal \
        --trust-remote-code

Usage:
    python scripts/infer.py \
        --model_url http://127.0.0.1:30000/v1 \
        --data_dir data/xray_samples/studies/ \
        --ids_path data/xray_samples/selected_study_ids.json \
        --output_path outputs/predictions.jsonl
"""

import argparse
import json
import math
import sys
from pathlib import Path

from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from prompt import PROMPT
from utils import discover_series, extract_field, frames_to_data_uris, load_study_ids

MODEL_ID = "google/medgemma-1.5-4b-it"
TOP_LOGPROBS = 20


# ---------------------------------------------------------------------------
# Entropy approximation
# ---------------------------------------------------------------------------

def entropy_from_top_logprobs(top_logprobs: list) -> float:
    """Approximate token entropy from a top-k log-probability distribution.

    Normalizes the top-k probabilities to sum to 1, then computes Shannon
    entropy. This underestimates true entropy when k < vocabulary size but is
    a reasonable approximation without the full logit vector.
    """
    log_probs = [lp.logprob for lp in top_logprobs]
    probs = [math.exp(lp) for lp in log_probs]
    total = sum(probs)
    if total == 0:
        return 0.0
    norm = [p / total for p in probs]
    return -sum(p * math.log(p) for p in norm if p > 0)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    client: OpenAI,
    model_id: str,
    video_paths: list[Path],
    indication: str,
) -> dict:
    """Call MedGemma for one study and return tokens, log_probs, and entropies."""
    formatted_prompt = PROMPT.format(context_info=indication)

    image_content = [
        {"type": "image_url", "image_url": {"url": uri}}
        for uri in frames_to_data_uris(video_paths)
    ]

    messages = [
        {"role": "system", "content": "You are an expert radiologist."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": formatted_prompt},
                *image_content,
            ],
        },
    ]

    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=2000,
        temperature=0,
        logprobs=True,
        top_logprobs=TOP_LOGPROBS,
    )

    choice = response.choices[0]
    response_text = choice.message.content

    tokens, log_probs, entropies = [], [], []
    if choice.logprobs and choice.logprobs.content:
        for token_lp in choice.logprobs.content:
            tokens.append(token_lp.token)
            log_probs.append(token_lp.logprob)
            entropies.append(entropy_from_top_logprobs(token_lp.top_logprobs))

    return {
        "response": response_text,
        "tokens": tokens,
        "log_probs": log_probs,
        "entropies": entropies,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run MedGemma inference and record logprobs.")
    parser.add_argument("--model_url", default="http://127.0.0.1:30000/v1",
                        help="SGLang server base URL.")
    parser.add_argument("--model_id", default=MODEL_ID,
                        help="Model ID served by SGLang.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/xray_samples/studies"),
                        help="Path to studies directory (contains <study_id>/ subdirs).")
    parser.add_argument("--ids_path", type=Path, default=Path("data/xray_samples/selected_study_ids.json"),
                        help="JSON file containing a flat list of study IDs to process.")
    parser.add_argument("--output_path", type=Path, default=Path("outputs/predictions.jsonl"),
                        help="Output JSONL path.")
    args = parser.parse_args()

    study_ids = load_study_ids(args.ids_path)
    print(f"Loaded {len(study_ids)} study IDs from {args.ids_path}")

    client = OpenAI(base_url=args.model_url, api_key="EMPTY")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok, n_fail = 0, 0
    with open(args.output_path, "w") as out_f:
        for idx, study_id in enumerate(study_ids):
            print(f"[{idx + 1}/{len(study_ids)}] {study_id} ...", end=" ", flush=True)

            video_paths = discover_series(args.data_dir, study_id)
            if not video_paths:
                print(f"SKIPPED (no volume.mp4 under {args.data_dir / study_id})")
                n_fail += 1
                continue

            report_path = args.data_dir / study_id / "report.txt"
            indication = ""
            if report_path.exists():
                report_text = report_path.read_text(encoding="utf-8")
                indication = extract_field(report_text, "Indication")

            print(f"({len(video_paths)} series) ", end="", flush=True)

            try:
                result = run_inference(client, args.model_id, video_paths, indication)
            except Exception as exc:
                print(f"FAILED ({exc})")
                n_fail += 1
                continue

            record = {"id": study_id, **result}
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            print("OK")
            n_ok += 1

    print(f"\nDone. Succeeded: {n_ok}, Failed/skipped: {n_fail}")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
