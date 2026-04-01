#!/usr/bin/env bash
# run_pipeline.sh — end-to-end confidence estimation pipeline
#
# Run from the radiology_confidence/ root directory:
#   bash run_pipeline.sh [options]
#
# Steps:
#   1. MedGemma inference          (outputs/predictions.jsonl)
#   2. CRIMSON evaluation          (outputs/judge_scores.jsonl)
#   3. Plot CRIMSON scores         (plots/judge/, results/judge_summary.csv)
#   4. Logit-based metrics         (outputs/logit_scores.jsonl)
#   5. Self-consistency metrics    (outputs/consistency_scores.jsonl)
#   6. Stability metrics           (outputs/stability_scores.jsonl)
#   7. Calibration and analysis    (runs on all available score files)
#
# Skip flags (resume from checkpoint):
#   --skip-infer        Skip step 1 (outputs/predictions.jsonl must exist)
#   --skip-evaluate     Skip step 2 (outputs/judge_scores.jsonl must exist)
#   --skip-logits       Skip step 4
#   --skip-consistency  Skip step 5
#   --skip-stability    Skip step 6
#
# Example (skip inference and evaluation, rerun from step 3):
#   bash run_pipeline.sh --skip-infer --skip-evaluate

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_URL="http://127.0.0.1:30000/v1"
MODEL_ID="google/medgemma-1.5-4b-it"
DATA_DIR="data/xray_samples/studies"
IDS_PATH="data/xray_samples/selected_study_ids.json"
LEXICON_PATH="lexicon/RadLex.owl"
N_SAMPLES=5
TEMPERATURE=0.7
SGLANG_PORT=30000
SGLANG_READY_TIMEOUT=300   # seconds to wait for server to become ready

# MedGemma is a gated model — set HF_TOKEN before running this script:
#   export HF_TOKEN=<your_huggingface_token>
# or pass it inline: HF_TOKEN=<token> bash run_pipeline.sh
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN is not set. The SGLang server will fail if the model"
    echo "         is not already cached locally. Set HF_TOKEN to your HuggingFace"
    echo "         access token for gated-model access."
fi

# ---------------------------------------------------------------------------
# Parse skip flags
# ---------------------------------------------------------------------------

SKIP_INFER=0
SKIP_EVALUATE=0
SKIP_LOGITS=0
SKIP_CONSISTENCY=0
SKIP_STABILITY=0

for arg in "$@"; do
    case $arg in
        --skip-infer)       SKIP_INFER=1 ;;
        --skip-evaluate)    SKIP_EVALUATE=1 ;;
        --skip-logits)      SKIP_LOGITS=1 ;;
        --skip-consistency) SKIP_CONSISTENCY=1 ;;
        --skip-stability)   SKIP_STABILITY=1 ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SGLANG_PID=""

header() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

require_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: Required file not found: $1"
        echo "       Run the preceding step first or remove the --skip flag."
        exit 1
    fi
}

start_sglang_server() {
    header "Starting SGLang server"
    conda run -n sglang python -m sglang.launch_server \
        --model-path "$MODEL_ID" \
        --port "$SGLANG_PORT" \
        --enable-multimodal \
        --trust-remote-code &
    SGLANG_PID=$!
    echo "SGLang server PID: $SGLANG_PID"

    echo "Waiting for server to become ready (timeout: ${SGLANG_READY_TIMEOUT}s) ..."
    local elapsed=0
    until curl -sf "${MODEL_URL}/models" > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $elapsed -ge $SGLANG_READY_TIMEOUT ]; then
            echo "ERROR: SGLang server did not become ready within ${SGLANG_READY_TIMEOUT}s."
            stop_sglang_server
            exit 1
        fi
        echo "  ... still waiting (${elapsed}s elapsed)"
    done
    echo "Server is ready."
}

stop_sglang_server() {
    if [ -n "$SGLANG_PID" ] && kill -0 "$SGLANG_PID" 2>/dev/null; then
        echo "Stopping SGLang server (PID $SGLANG_PID) ..."
        kill "$SGLANG_PID"
        wait "$SGLANG_PID" 2>/dev/null || true
        SGLANG_PID=""
        echo "Server stopped."
    fi
}

# Ensure server is stopped on exit even if the script fails
trap stop_sglang_server EXIT

# ---------------------------------------------------------------------------
# Step 1: Inference
# ---------------------------------------------------------------------------

if [ $SKIP_INFER -eq 0 ]; then
    start_sglang_server
    header "Step 1/7: MedGemma inference (infer.py)"
    conda run -n medgemma python scripts/infer.py \
        --model_url "$MODEL_URL" \
        --model_id  "$MODEL_ID" \
        --data_dir  "$DATA_DIR" \
        --ids_path  "$IDS_PATH" \
        --output_path outputs/predictions.jsonl
    stop_sglang_server
else
    header "Step 1/7: Skipping inference"
    require_file outputs/predictions.jsonl
fi

# ---------------------------------------------------------------------------
# Step 2: CRIMSON evaluation
# ---------------------------------------------------------------------------

if [ $SKIP_EVALUATE -eq 0 ]; then
    header "Step 2/7: CRIMSON evaluation (evaluate.py)"
    conda run -n crimson python scripts/evaluate.py \
        --predictions_path outputs/predictions.jsonl \
        --data_dir "$DATA_DIR" \
        --ids_path "$IDS_PATH" \
        --output_path outputs/judge_scores.jsonl
else
    header "Step 2/7: Skipping CRIMSON evaluation"
    require_file outputs/judge_scores.jsonl
fi

# ---------------------------------------------------------------------------
# Step 3: Plot CRIMSON score distribution
# ---------------------------------------------------------------------------

header "Step 3/7: Plotting CRIMSON scores (plot_scores.py)"
require_file outputs/judge_scores.jsonl
conda run -n medgemma python scripts/plot_scores.py \
    --judge_scores_path outputs/judge_scores.jsonl \
    --plots_dir plots/judge/ \
    --output_path results/judge_summary.csv

# ---------------------------------------------------------------------------
# Step 4: Logit-based metrics
# ---------------------------------------------------------------------------

if [ $SKIP_LOGITS -eq 0 ]; then
    header "Step 4/7: Logit-based metrics (logits.py)"
    require_file outputs/predictions.jsonl
    require_file outputs/judge_scores.jsonl
    conda run -n medgemma python scripts/logits.py \
        --predictions_path  outputs/predictions.jsonl \
        --judge_scores_path outputs/judge_scores.jsonl \
        --output_path       outputs/logit_scores.jsonl \
        --lexicon_path      "$LEXICON_PATH"
else
    header "Step 4/7: Skipping logit metrics"
fi

# ---------------------------------------------------------------------------
# Step 5: Self-consistency metrics
# ---------------------------------------------------------------------------

if [ $SKIP_CONSISTENCY -eq 0 ]; then
    start_sglang_server
    header "Step 5/7: Self-consistency analysis (consistency.py)"
    require_file outputs/predictions.jsonl
    require_file outputs/judge_scores.jsonl
    conda run -n medgemma python scripts/consistency.py \
        --predictions_path  outputs/predictions.jsonl \
        --judge_scores_path outputs/judge_scores.jsonl \
        --model_url "$MODEL_URL" \
        --model_id  "$MODEL_ID" \
        --data_dir  "$DATA_DIR" \
        --ids_path  "$IDS_PATH" \
        --output_path outputs/consistency_scores.jsonl \
        --n_samples "$N_SAMPLES"
    stop_sglang_server
else
    header "Step 5/7: Skipping self-consistency analysis"
fi

# ---------------------------------------------------------------------------
# Step 6: Stability metrics
# ---------------------------------------------------------------------------

if [ $SKIP_STABILITY -eq 0 ]; then
    start_sglang_server
    header "Step 6/7: Stability analysis (stability.py)"
    require_file outputs/predictions.jsonl
    require_file outputs/judge_scores.jsonl
    conda run -n medgemma python scripts/stability.py \
        --predictions_path  outputs/predictions.jsonl \
        --judge_scores_path outputs/judge_scores.jsonl \
        --model_url "$MODEL_URL" \
        --model_id  "$MODEL_ID" \
        --data_dir  "$DATA_DIR" \
        --ids_path  "$IDS_PATH" \
        --output_path outputs/stability_scores.jsonl \
        --n_samples   "$N_SAMPLES" \
        --temperature "$TEMPERATURE"
    stop_sglang_server
else
    header "Step 6/7: Skipping stability analysis"
fi

# ---------------------------------------------------------------------------
# Step 7: Calibration and analysis
# ---------------------------------------------------------------------------

header "Step 7/7: Calibration and analysis"

for scores_path in outputs/logit_scores.jsonl outputs/consistency_scores.jsonl outputs/stability_scores.jsonl; do
    if [ ! -f "$scores_path" ]; then
        continue
    fi
    base=$(basename "$scores_path" _scores.jsonl)
    conda run -n medgemma python scripts/calibrate.py \
        --input_path  "$scores_path" \
        --output_path "$scores_path" \
        --calibrators_path "outputs/${base}_calibrator.pkl"
    conda run -n medgemma python scripts/analyze.py \
        --input_path "$scores_path" \
        --plots_dir  plots/
done

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

header "Pipeline complete"
echo "Outputs:  outputs/"
echo "Results:  results/"
echo "Plots:    plots/"
