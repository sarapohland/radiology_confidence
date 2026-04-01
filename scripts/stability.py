"""
Measure how much MedGemma's output varies under temperature sampling.

For each study, generates --n_samples responses at --temperature > 0 and
computes mean pairwise ROUGE-1 and ROUGE-L similarity across all sample 
pairs. Additionally computes ROUGE similarity of each sample against the 
original greedy response from predictions.jsonl as a reference-anchored 
stability metric.

Metrics (higher = more stable = higher confidence)
------------------------------------------------------------
  mean_pairwise_rouge1    mean ROUGE-1 F1 across all pairwise sample comparisons
  mean_pairwise_rougeL    mean ROUGE-L F1 across all pairwise sample comparisons
  mean_vs_greedy_rouge1   mean ROUGE-1 F1 of each sample vs. the greedy response
  mean_vs_greedy_rougeL   mean ROUGE-L F1 of each sample vs. the greedy response

Usage
-----
    python scripts/stability.py \\
        --predictions_path outputs/predictions.jsonl \\
        --judge_scores_path outputs/judge_scores.jsonl \\
        --model_url http://127.0.0.1:30000/v1 \\
        --data_dir data/xray_samples/studies/ \\
        --ids_path data/xray_samples/selected_study_ids.json \\
        --output_path outputs/stability_scores.jsonl \\
        [--n_samples 5] \\
        [--temperature 0.7] \\
        [--metrics mean_pairwise_rouge1 mean_pairwise_rougeL mean_vs_greedy_rouge1 mean_vs_greedy_rougeL]
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from prompt import PROMPT
from utils import (
    discover_series, extract_field, extract_findings_text,
    frames_to_data_uris, load_jsonl_by_id, load_study_ids,
    mean_pairwise_rouge, rouge1_f1, rougeL_f1,
)

MODEL_ID = "google/medgemma-1.5-4b-it"

ALL_METRICS = [
    "mean_pairwise_rouge1", "mean_pairwise_rougeL",
    "mean_vs_greedy_rouge1", "mean_vs_greedy_rougeL",
]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate_response(
    client: OpenAI,
    model_id: str,
    data_uris: list,
    indication: str,
    temperature: float,
) -> str:
    """Call MedGemma at the given temperature and return the response text."""
    formatted_prompt = PROMPT.format(context_info=indication)
    image_content = [{"type": "image_url", "image_url": {"url": u}} for u in data_uris]
    messages = [
        {"role": "system", "content": "You are an expert radiologist."},
        {"role": "user", "content": [
            {"type": "text", "text": formatted_prompt},
            *image_content,
        ]},
    ]
    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=2000,
        temperature=temperature,
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stability analysis via temperature sampling and pairwise ROUGE similarity."
    )
    parser.add_argument("--predictions_path", type=Path, required=True,
                        help="predictions.jsonl from infer.py.")
    parser.add_argument("--judge_scores_path", type=Path, required=True,
                        help="judge_scores.jsonl from evaluate.py.")
    parser.add_argument("--model_url", default="http://127.0.0.1:30000/v1",
                        help="SGLang server base URL.")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--data_dir", type=Path, default=Path("data/xray_samples/studies"),
                        help="Path to studies directory.")
    parser.add_argument("--ids_path", type=Path, default=Path("data/xray_samples/selected_study_ids.json"),
                        help="JSON file containing study IDs to process.")
    parser.add_argument("--output_path", type=Path,
                        default=Path("outputs/stability_scores.jsonl"))
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of temperature-sampled responses per study (default: 5).")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7). Must be > 0.")
    parser.add_argument("--metrics", nargs="+", default=ALL_METRICS,
                        choices=ALL_METRICS, metavar="METRIC",
                        help=(
                            "Metrics to write to output. Default: all. "
                            "Choices: " + " ".join(ALL_METRICS)
                        ))
    parser.add_argument("--findings_only", action=argparse.BooleanOptionalAction, default=True,
                        help="Restrict ROUGE comparisons to the FINDINGS section of each response "
                             "(default: True; use --no-findings_only to compare full responses).")
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Number of parallel inference threads per study "
                             "(default: n_samples — fully parallel).")
    args = parser.parse_args()
    if args.n_workers is None:
        args.n_workers = args.n_samples
    if args.temperature <= 0:
        parser.error("--temperature must be > 0 for stochastic sampling.")

    predictions  = load_jsonl_by_id(args.predictions_path)
    judge_scores = load_jsonl_by_id(args.judge_scores_path)
    study_ids    = load_study_ids(args.ids_path)

    client = OpenAI(base_url=args.model_url, api_key="EMPTY")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok, n_fail = 0, 0
    with open(args.output_path, "w") as out_f:
        for idx, study_id in enumerate(study_ids):
            print(f"[{idx + 1}/{len(study_ids)}] {study_id} ...", end=" ", flush=True)

            if study_id not in predictions:
                print("SKIPPED (no prediction)")
                n_fail += 1
                continue

            greedy_response = predictions[study_id]["response"]
            if args.findings_only:
                greedy_response = extract_findings_text(greedy_response)
            video_paths = discover_series(args.data_dir, study_id)
            if not video_paths:
                print("SKIPPED (no volume.mp4)")
                n_fail += 1
                continue

            report_path = args.data_dir / study_id / "report.txt"
            indication = ""
            if report_path.exists():
                indication = extract_field(report_path.read_text(encoding="utf-8"), "Indication")

            try:
                # Encode frames once (same images for all samples)
                uris = frames_to_data_uris(video_paths)

                # Generate n_samples responses in parallel
                with ThreadPoolExecutor(max_workers=args.n_workers) as pool:
                    samples = list(pool.map(
                        lambda _: generate_response(
                            client, args.model_id, uris, indication, args.temperature
                        ),
                        range(args.n_samples),
                    ))
            except Exception as exc:
                print(f"FAILED ({exc})")
                n_fail += 1
                continue

            if args.findings_only:
                samples = [extract_findings_text(s) for s in samples]

            # Mean pairwise ROUGE across all sample pairs
            pw_r1, pw_rL = mean_pairwise_rouge(samples)

            # Mean ROUGE of each sample vs. the original greedy response
            vg_r1 = float(np.mean([rouge1_f1(greedy_response, s) for s in samples]))
            vg_rL = float(np.mean([rougeL_f1(greedy_response, s) for s in samples]))

            computed = {
                "mean_pairwise_rouge1":  round(pw_r1, 6),
                "mean_pairwise_rougeL":  round(pw_rL, 6),
                "mean_vs_greedy_rouge1": round(vg_r1, 6),
                "mean_vs_greedy_rougeL": round(vg_rL, 6),
            }
            out = {"id": study_id, "score": judge_scores.get(study_id, {}).get("score")}
            out.update({k: v for k, v in computed.items() if k in args.metrics})
            out_f.write(json.dumps(out) + "\n")
            out_f.flush()
            parts = []
            if "mean_pairwise_rouge1" in args.metrics:
                parts.append(f"pw_r1={pw_r1:.3f}")
            if "mean_pairwise_rougeL" in args.metrics:
                parts.append(f"pw_rL={pw_rL:.3f}")
            if "mean_vs_greedy_rouge1" in args.metrics:
                parts.append(f"vg_r1={vg_r1:.3f}")
            if "mean_vs_greedy_rougeL" in args.metrics:
                parts.append(f"vg_rL={vg_rL:.3f}")
            print(f"OK  ({', '.join(parts)})")
            n_ok += 1

    print(f"\nDone.  Succeeded: {n_ok},  Failed/skipped: {n_fail}")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
