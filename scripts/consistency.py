"""
Measure how much MedGemma's output changes under controlled input 
perturbations.

For each study, generates --n_samples additional responses per 
perturbation type and computes ROUGE-1 and ROUGE-L similarity bewteen all
generated responses.

Perturbation types
------------------
  noise   Gaussian noise added to decoded frames (sigma=--noise_sigma).
          Each of the n_samples uses a different random seed.
  blur    Gaussian blur applied to decoded frames. Blur radius is varied evenly
          from blur_radius * 0.5 to blur_radius * 1.5 across the n_samples.

Metrics (higher = more consistent = higher confidence)
---------------------------------------------------------------
  blur_rouge1       mean ROUGE-1 F1 for blur perturbations (if selected)
  blur_rougeL       mean ROUGE-L F1 for blur perturbations (if selected)
  noise_rouge1      mean ROUGE-1 F1 for noise perturbations (if selected)
  noise_rougeL      mean ROUGE-L F1 for noise perturbations (if selected)
  mean_rouge1       mean ROUGE-1 F1 across all samples and perturbation types
  mean_rougeL       mean ROUGE-L F1 across all samples and perturbation types

Usage
-----
    python scripts/consistency.py \\
        --predictions_path outputs/predictions.jsonl \\
        --judge_scores_path outputs/judge_scores.jsonl \\
        --model_url http://127.0.0.1:30000/v1 \\
        --data_dir data/xray_samples/studies/ \\
        --ids_path data/xray_samples/selected_study_ids.json \\
        --output_path outputs/consistency_scores.jsonl \\
        [--n_samples 5] \\
        [--perturbations noise blur] \\
        [--noise_sigma 25] \\
        [--blur_radius 3] \\
        [--metrics noise_rouge1 noise_rougeL blur_rouge1 blur_rougeL mean_rouge1 mean_rougeL]
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from prompt import PROMPT
from utils import (
    discover_series, extract_field, extract_findings_text,
    frames_to_data_uris, load_jsonl_by_id, load_study_ids,
    rouge1_f1, rougeL_f1,
)

MODEL_ID = "google/medgemma-1.5-4b-it"

SUPPORTED_PERTURBATIONS = ["noise", "blur"]

ALL_METRICS = [
    "noise_rouge1", "noise_rougeL",
    "blur_rouge1",  "blur_rougeL",
    "mean_rouge1",  "mean_rougeL",
]


# ---------------------------------------------------------------------------
# Perturbation helpers
# ---------------------------------------------------------------------------

def apply_noise(img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    arr = np.array(img, dtype=float)
    arr = np.clip(arr + rng.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate_response(
    client: OpenAI,
    model_id: str,
    data_uris: list,
    indication: str,
) -> str:
    """Call MedGemma with temperature=0 and return the response text."""
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
        temperature=0,
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Self-consistency analysis via frame perturbations and ROUGE similarity."
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
                        default=Path("outputs/consistency_scores.jsonl"))
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of perturbed responses per perturbation type (default: 5).")
    parser.add_argument("--perturbations", nargs="+", default=["noise", "blur"],
                        choices=SUPPORTED_PERTURBATIONS,
                        help="Perturbation types to apply (default: noise blur).")
    parser.add_argument("--noise_sigma", type=float, default=25.0,
                        help="Gaussian noise standard deviation in pixel units (default: 25).")
    parser.add_argument("--blur_radius", type=float, default=3.0,
                        help="Centre Gaussian blur radius in pixels (default: 3). "
                             "n_samples radii are spread from 0.5× to 1.5× this value.")
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
                        help="Number of parallel inference threads per perturbation batch "
                             "(default: n_samples — fully parallel).")
    args = parser.parse_args()
    if args.n_workers is None:
        args.n_workers = args.n_samples

    predictions  = load_jsonl_by_id(args.predictions_path)
    judge_scores = load_jsonl_by_id(args.judge_scores_path)
    study_ids    = load_study_ids(args.ids_path)

    client = OpenAI(base_url=args.model_url, api_key="EMPTY")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-compute per-sample perturbation parameters for each type
    # noise: n_samples different random seeds at a fixed sigma
    # blur:  n_samples radii evenly spaced from 0.5× to 1.5× the target radius
    noise_rngs = [np.random.default_rng(seed=i) for i in range(args.n_samples)]
    blur_radii = np.linspace(
        args.blur_radius * 0.5, args.blur_radius * 1.5, args.n_samples
    ).tolist()

    n_ok, n_fail = 0, 0
    with open(args.output_path, "w") as out_f:
        for idx, study_id in enumerate(study_ids):
            print(f"[{idx + 1}/{len(study_ids)}] {study_id} ...", end=" ", flush=True)

            if study_id not in predictions:
                print("SKIPPED (no prediction)")
                n_fail += 1
                continue

            original_response = predictions[study_id]["response"]
            if args.findings_only:
                original_response = extract_findings_text(original_response)
            video_paths = discover_series(args.data_dir, study_id)
            if not video_paths:
                print("SKIPPED (no volume.mp4)")
                n_fail += 1
                continue

            report_path = args.data_dir / study_id / "report.txt"
            indication = ""
            if report_path.exists():
                indication = extract_field(report_path.read_text(encoding="utf-8"), "Indication")

            def run_sample(i, ptype):
                if ptype == "noise":
                    rng = noise_rngs[i]
                    perturb_fn = lambda img, rng=rng: apply_noise(img, args.noise_sigma, rng)
                else:  # blur
                    radius = blur_radii[i]
                    perturb_fn = lambda img, r=radius: apply_blur(img, r)
                uris = frames_to_data_uris(video_paths, perturb_fn)
                resp = generate_response(client, args.model_id, uris, indication)
                if args.findings_only:
                    resp = extract_findings_text(resp)
                return resp

            scores_per_type = {}
            try:
                for ptype in args.perturbations:
                    with ThreadPoolExecutor(max_workers=args.n_workers) as pool:
                        responses = list(pool.map(
                            lambda i: run_sample(i, ptype),
                            range(args.n_samples),
                        ))
                    r1_vals = [rouge1_f1(original_response, r) for r in responses]
                    rL_vals = [rougeL_f1(original_response, r) for r in responses]
                    scores_per_type[ptype] = {
                        "rouge1": float(np.mean(r1_vals)),
                        "rougeL": float(np.mean(rL_vals)),
                    }
            except Exception as exc:
                print(f"FAILED ({exc})")
                n_fail += 1
                continue

            out = {
                "id": study_id,
                "score": judge_scores.get(study_id, {}).get("score"),
            }
            all_r1, all_rL = [], []
            for ptype, s in scores_per_type.items():
                if f"{ptype}_rouge1" in args.metrics:
                    out[f"{ptype}_rouge1"] = round(s["rouge1"], 6)
                if f"{ptype}_rougeL" in args.metrics:
                    out[f"{ptype}_rougeL"] = round(s["rougeL"], 6)
                all_r1.append(s["rouge1"])
                all_rL.append(s["rougeL"])
            if "mean_rouge1" in args.metrics:
                out["mean_rouge1"] = round(float(np.mean(all_r1)), 6) if all_r1 else None
            if "mean_rougeL" in args.metrics:
                out["mean_rougeL"] = round(float(np.mean(all_rL)), 6) if all_rL else None

            out_f.write(json.dumps(out) + "\n")
            out_f.flush()
            r1_str = f"{out['mean_rouge1']:.3f}" if "mean_rouge1" in out else "n/a"
            rL_str = f"{out['mean_rougeL']:.3f}" if "mean_rougeL" in out else "n/a"
            print(f"OK  (r1={r1_str}, rL={rL_str})")
            n_ok += 1

    print(f"\nDone.  Succeeded: {n_ok},  Failed/skipped: {n_fail}")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
