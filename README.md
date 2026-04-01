# Confidence Estimation for Radiology Report Generation

This repository implements a post-training confidence estimation pipeline for a radiology report generation model. Given a chest X-ray, MedGemma generates a free-text radiology report. Confidence estimation determines how likely that report is to be judged correct by CRIMSON--a clinically-grounded LLM-based radiology report evaluator--without requiring CRIMSON at inference time.

The pipeline covers three baseline approaches to confidence estimation:

- **Logit-based metrics** ([Section 2.1](#21-logit-based-confidence)): functions of token log-probabilities and entropies recorded during generation
- **Self-consistency** ([Section 2.2](#22-self-consistency-analysis)): response consistency under controlled input perturbations
- **Stability analysis** ([Section 2.3](#23-stability-analysis)): response variance under temperature sampling

Each method includes a calibration step that fits a multivariate logistic regression over all of that method's metrics to produce a single P(incorrect) score.

See [Section 3](#3-future-work) for planned extensions including vision-text grounding, hidden-state probing, and more in-depth calibration combining signals across various methods.

The goal of this repository is to get a sense for which metrics provide some signal about report accuracy and create baselines for more involved metrics. The main effort of future work would be to explore more interesting metrics and decide how to use all these signals.

## Full pipeline

The `run_pipeline.sh` script runs all steps end-to-end, managing conda environments and the SGLang server automatically:

```bash
bash run_pipeline.sh
```

Use `--skip-*` flags to bypass steps and resume from a checkpoint:

```bash
bash run_pipeline.sh [--skip-infer] [--skip-evaluate] [--skip-logits] [--skip-consistency] [--skip-stability]
```

See the sections below for details on each step and its outputs. Configurable parameters (model URL, data paths, number of samples, temperature) are set at the top of `run_pipeline.sh`.

## Table of Contents

1. [Setup](#1-setup)
   - [1.1 Initial Setup](#11-initial-setup)
   - [1.2 Generate Confidence Estimation Dataset](#12-generate-confidence-estimation-dataset)
2. [Confidence Estimation](#2-confidence-estimation)
   - [2.1 Logit-Based Confidence](#21-logit-based-confidence)
   - [2.2 Self-Consistency Analysis](#22-self-consistency-analysis)
   - [2.3 Stability Analysis](#23-stability-analysis)
3. [Future Work](#3-future-work)
   - [3.1 Vision-Text Grounding](#31-vision-text-grounding)
   - [3.2 Hidden-State Probing](#32-hidden-state-probing)
   - [3.3 Confidence Calibration](#33-confidence-calibration)
   - [3.4 Other Avenues](#34-other-avenues-for-future-work)
4. [Repository Structure](#4-repository-structure)

---

## 1. Setup

### 1.1 Initial Setup

#### 1.1.1 Set up conda environments:

This pipeline uses three conda environments due to dependency conflicts:

| Environment | Requirements file          | Used by                                                                                    |
|-------------|----------------------------|--------------------------------------------------------------------------------------------|
| `medgemma`  | `requirements_medgemma.txt` | `infer.py`, `logits.py`, `consistency.py`, `stability.py`, `analyze.py`, `calibrate.py`, `plot_scores.py` |
| `crimson`   | `requirements_crimson.txt`  | `evaluate.py`                                                                              |
| `sglang`    | `requirements_sglang.txt`   | SGLang server (MedGemma inference backend)                                                 |

Create each environment (Python 3.12):

```bash
conda create -n medgemma python=3.12 -y && conda run -n medgemma pip install -r requirements_medgemma.txt
```
```bash
conda create -n crimson  python=3.12 -y && conda run -n crimson  pip install -r requirements_crimson.txt
```
```bash
conda create -n sglang   python=3.12 -y && conda run -n sglang   pip install -r requirements_sglang.txt
```

#### 1.1.2 Request access to MedGemma model:

MedGemma is a gated model on HuggingFace. Before running the pipeline, accept their Terms \& Conditions, and export your HuggingFace access token:

```bash
export HF_TOKEN=<your_huggingface_token>
```

#### 1.1.3 Download the X-ray samples dataset:

Place (or symlink) the X-ray study data under `data/xray_samples/`:

```bash
mkdir -p data
ln -s /path/to/xray_samples data/xray_samples
```

The expected structure is:
```
data/xray_samples/
├── selected_study_ids.json       # flat JSON list of study IDs
└── studies/
    └── <study_id>/
        ├── report.txt            # radiologist report with Findings / Indication fields
        └── <series_id>/
            └── volume.mp4        # HEVC-encoded X-ray frames
```

### 1.2 Generate confidence estimation dataset

#### 1.2.1 Run inference with MedGemma:

```bash
python scripts/infer.py \
    --model_url http://127.0.0.1:30000/v1 \
    --data_dir data/xray_samples/studies/ \
    --ids_path data/xray_samples/selected_study_ids.json \
    --output_path outputs/predictions.jsonl
```

This script runs inference using MedGemma served via SGLang. For each study, it decodes all series MP4s into frames using `av`, encodes each frame as a base64 PNG data URI, and passes them to MedGemma along with the patient indication extracted from `report.txt`. Logprobs are requested via `logprobs=True, top_logprobs=20`; per-token entropy is approximated from the top-20 distribution. The script stores the following in `outputs/predictions.jsonl`:
- `id`: study identifier (DICOM study UID)
- `response`: full generated text response
- `tokens`: list of generated token strings
- `log_probs`: per-token log probability of predicted token (one float per token)
- `entropies`: per-token predictive entropy approximated from top-20 logprobs (one float per token)

#### 1.2.2 Check correctness of output responses:

```bash
python scripts/evaluate.py \
    --predictions_path outputs/predictions.jsonl \
    --data_dir data/xray_samples/studies/ \
    --ids_path data/xray_samples/selected_study_ids.json \
    --output_path outputs/judge_scores.jsonl \
    [--crimson_api hf|vllm] \
    [--batch_size 4]
```

This script runs CRIMSON scoring for each prediction against its study's ground-truth findings, and stores the following in `outputs/judge_scores.jsonl`:
- `id`: study identifier
- `score`: CRIMSON raw score ∈ (−1, 1]
- `question_scores`: CRIMSON error breakdown -- counts of false findings, missing findings, and attribute errors by type (location, severity, descriptor, measurement, certainty, unspecific, overinterpretation, temporal)

#### 1.2.3 Plot distribution of report scores:

```bash
python scripts/plot_scores.py \
    --judge_scores_path outputs/judge_scores.jsonl \
    --plots_dir plots/judge/ \
    --output_path results/judge_summary.csv
```

This script produces the following:
- **Score distribution**: histogram and KDE of CRIMSON `score` values, with percentile markers (10th, 25th, 50th, 75th, 90th) annotated
- **Threshold sensitivity**: a table showing what fraction of studies would be labelled correct at a range of score thresholds, printed to stdout and saved to `results/judge_summary.csv`
- **Error type breakdown**: bar chart of mean per-study counts for each field in `question_scores` (false findings, missing findings, and each attribute error type), showing which error categories are most common across the dataset


---

## 2. Confidence Estimation

### 2.1 Logit-Based Confidence

During generation, the model outputs a probability distribution over its vocabulary at each step, which reflects how certain it is about each token. Low log-probabilities and high entropy indicate that the model was uncertain in its predictions, which tends to correlate with lower-quality or incorrect responses. These token-level signals are cheap to collect during inference and require no additional forward passes.

#### 2.1.1 Estimate confidence from raw logits:

```bash
python scripts/logits.py \
    --predictions_path outputs/predictions.jsonl \
    --judge_scores_path outputs/judge_scores.jsonl \
    --output_path outputs/logit_scores.jsonl \
    --lexicon_path lexicon/RadLex.owl \
    [--k 5] \
    [--metrics min_lp mean_lp var_lp p5_lp p10_lp botk_lp \
               order_lp semantic_lp domain_lp \
               max_ent mean_ent var_ent p90_ent p95_ent topk_ent \
               order_ent semantic_ent domain_ent]
```

For each response, this script computes the following metrics over the per-token log probabilities and entropies and stores them in `outputs/logit_scores.jsonl`. The `--metrics` flag can be used to select the metrics to be collected. By default, all of the following are computed:

*Log-probability metrics (higher = more confident):*
- `min_lp`: Minimum token log probability
- `mean_lp`: Average token log probability
- `var_lp`: Variance of token log probabilities (higher = more erratic)
- `p5_lp`: 5th-percentile token log probability (robust alternative to `min_lp`)
- `p10_lp`: 10th-percentile token log probability
- `botk_lp`: Mean of the k lowest token log probabilities (can set k with `--k`)
- `order_lp`: Average token log probability weighted by word order (earlier tokens weighted higher, as they set diagnostic context)
- `semantic_lp`: Average token log probability weighted by semantic significance (content words weighted over filler)
- `domain_lp`: Average token log probability weighted by RadLex ontology importance (requires `--lexicon_path`; falls back to a small hardcoded term list)

*Entropy metrics (lower = more confident):*
- `max_ent`: Maximum token entropy
- `mean_ent`: Average token entropy
- `var_ent`: Variance of token entropy
- `p90_ent`: 90th-percentile token entropy (robust alternative to `max_ent`)
- `p95_ent`: 95th-percentile token entropy
- `topk_ent`: Mean of the k highest token entropies (can set k with `--k`)
- `order_ent`: Average token entropy weighted by word order
- `semantic_ent`: Average token entropy weighted by semantic significance
- `domain_ent`: Average token entropy weighted by RadLex ontology importance

#### 2.1.2 Calibrate logit-based confidence scores:

```bash
python scripts/calibrate.py \
    --input_path outputs/logit_scores.jsonl \
    [--output_path outputs/logit_scores.jsonl] \
    [--n_folds 5] \
    [--calibrators_path results/logit_calibrator.pkl]
```

This script fits a multivariate logistic regression model over all logit metrics to produce a single `prob_incorrect` field per record. By default, this result is written to the same input JSONL if `--output_path` is not set. See [Section 3.3](#33-confidence-calibration) for limitations and what a more complete calibration setup would require.

#### 2.1.3 Evaluate predictive power of logit-based metrics:

```bash
python scripts/analyze.py \
    --input_path outputs/logit_scores.jsonl \
    --plots_dir plots/
```

`analyze.py` auto-detects the score type from the input filename and routes all output to the corresponding subdirectory.

This command produces the following:
- **Distribution plots** (`plots/logit/distributions/`): KDE for three populations -- perfect reports (score=1.0), partially correct (0≤score<1), and incorrect (score<0)
- **Precision-recall curves** (`plots/logit/pr_curves/`): PR curve per metric treating incorrect (score<0) as the positive class; saves AUPRC and Precision @ 90/95/99% Recall
- **ROC curves** (`plots/logit/roc_curves/`): ROC curve per metric; saves AUROC and Sensitivity @ 90/95/99% Specificity
- **Summary CSV** (`results/logit_summary.csv`): all of the above metrics per field; also printed to stdout in two tables (PR and ROC)

#### Initial findings:

- The log probability of the generated token and the per-token entropy are (unsurprisingly) tightly coupled and a sort of inverse of one another. Neither is clearly better to use than the other and may be used in combination.
- Ordered weighting is not at all helpful. Semantic and domain weighting provide a small benefit over the unweighted average with domain weighting preferred between the two.
- It is generally preferred to look at bottom percentiles or bottom k samples over the absolute minimum log probability. A similar conclusion holds for entropy.

#### Limitations:

- Log probabilities at temperature=0 are systematically overconfident without appropriate calibration during training.
- Entropy is approximated from the top-20 token distribution, excluding the long tail of the vocabulary; for tokens where probability mass is spread broadly, this underestimates uncertainty.
- The weighting schemes (order, semantic, domain) are hand-designed heuristics; the weights are not learned and may not be optimal for this task or this model.

#### Future work:

- Test different values of k for top/bottom k sample metrics.
- Consider the use of other lexicons (beyond RadLex) for domain weighting.
- Learn the weighting function (e.g. train a small probe to predict per-token importance from the token and its position in the report structure).
- Conduct more formal calibration with a larger dataset.
- Investigate whether per-sentence or per-finding aggregation outperforms per-token aggregation.


### 2.2 Self-Consistency Analysis

A confident model should produce similar responses when the input is slightly perturbed, while an uncertain model is more sensitive to small changes. By generating multiple responses under controlled input perturbations and measuring how much the output changes, we can estimate model uncertainty without requiring additional model components or external supervision.

#### 2.2.1 Determine extent to which output changes when perturbing input:

```bash
python scripts/consistency.py \
    --predictions_path outputs/predictions.jsonl \
    --judge_scores_path outputs/judge_scores.jsonl \
    --model_url http://127.0.0.1:30000/v1 \
    --data_dir data/xray_samples/studies/ \
    --ids_path data/xray_samples/selected_study_ids.json \
    --output_path outputs/consistency_scores.jsonl \
    [--n_samples 5] \
    [--perturbations noise blur] \
    [--metrics noise_rouge1 noise_rougeL blur_rouge1 blur_rougeL mean_rouge1 mean_rougeL]
```

For each study, this script generates N additional responses under input perturbations and measures how much the response changes. All N inference calls within a study are issued in parallel by default; use `--n_workers` to cap concurrency. The currently supported perturbations are:
- **Noise**: Gaussian noise added to input images before passing to MedGemma
- **Blur**: Gaussian blur applied to input images before passing to MedGemma

<!-- TODO: Create script to visualize these perturbations. -->

The `--metrics` flag can be used to select the metrics to be collected. By default, all of the following are computed:

*Per-perturbation-type metrics:*
- `noise_rouge1`: Mean ROUGE-1 F1 between each noise-perturbed response and the original (requires `noise` in `--perturbations`)
- `noise_rougeL`: Mean ROUGE-L F1 for noise perturbations
- `blur_rouge1`: Mean ROUGE-1 F1 between each blur-perturbed response and the original (requires `blur` in `--perturbations`)
- `blur_rougeL`: Mean ROUGE-L F1 for blur perturbations

*Aggregate metrics:*
- `mean_rouge1`: Mean ROUGE-1 F1 averaged across all perturbation types and samples
- `mean_rougeL`: Mean ROUGE-L F1 averaged across all perturbation types and samples

Results are stored in `outputs/consistency_scores.jsonl`.

#### 2.2.2 Calibrate self-consistency scores:

```bash
python scripts/calibrate.py \
    --input_path outputs/consistency_scores.jsonl \
    [--calibrators_path results/consistency_calibrator.pkl]
```

This script follows the procedure described in Section 2.1.2, applied to the self-consistency metric features.

#### 2.2.3 Evaluate predictive power of self-consistency metrics:

```bash
python scripts/analyze.py \
    --input_path outputs/consistency_scores.jsonl \
    --plots_dir plots/
```

This command produces distribution plots, precision-recall curves, and ROC curves for each self-consistency metric, following the same format as Section 2.1.3. Results are saved to `plots/consistency/` and `results/consistency_summary.csv`.

#### Initial findings:

- Perturbing the input using blur provides a more useful signal than adding noise, but taking the average across both methods performs the best.
- Mean ROUGE-1 outperforms ROUGE-L in this setting, suggesting unigram overlap is more informative than sequence order preservation for this task.
- The self-consistency metrics are less useful than the logit-based ones on their own and take longer to compute. However, there are many ways to improve these metrics, and they may be used in combination with logit-based metrics.

#### Limitations:

- The ROUGE similarity scores only provide lexical similarity. Two correct reports about the same case can have very different wording and still be equally right.
- Perturbations are applied uniformly across all frames rather than targeting diagnostically relevant regions. Gaussian noise and blur also do not necessarily reflect natural variations in X-Ray images.
- With temperature=0, the only stochasticity comes from the image perturbation; small perturbations that don't change the visual content enough to shift the response will produce trivially high consistency.

#### Future work:

- Consider other metrics to measure extent of response change (e.g., score of LLM judge), while considering the potential increase in inference time.
- Apply more natural variations to input frames that reflect realistic scanner variability (e.g., different viewing angles, scanner noise profiles).
- Selectively mask or occlude anatomical regions and measure whether the report changes for that finding specifically.
- Apply noise to the vision encoding directly if available.
- Increase the number of samples and test if performance is affected. Also evaluate how much this affects computation time.


### 2.3 Stability Analysis

Under temperature sampling, a model draws responses stochastically rather than greedily, producing different outputs on each forward pass. A confident model will produce responses with low variance across samples, while an uncertain model will produce inconsistent or contradictory outputs. Unlike self-consistency analysis (Section 2.2), this approach perturbs the model's decoding rather than its input.

#### 2.3.1 Evaluate stability of model under temperature sampling:

```bash
python scripts/stability.py \
    --predictions_path outputs/predictions.jsonl \
    --judge_scores_path outputs/judge_scores.jsonl \
    --model_url http://127.0.0.1:30000/v1 \
    --data_dir data/xray_samples/studies/ \
    --ids_path data/xray_samples/selected_study_ids.json \
    --output_path outputs/stability_scores.jsonl \
    [--n_samples 5] \
    [--temperature 0.7] \
    [--metrics mean_pairwise_rouge1 mean_pairwise_rougeL mean_vs_greedy_rouge1 mean_vs_greedy_rougeL]
```

For each study, this script generates N responses by sampling from MedGemma at a given temperature and measures variance across the resulting responses. All N inference calls are issued in parallel by default; use `--n_workers` to cap concurrency. The `--metrics` flag can be used to select the metrics to be collected. By default, all of the following are computed:

*Pairwise stability metrics:*
- `mean_pairwise_rouge1`: Mean ROUGE-1 F1 across all unique pairs of sampled responses -- measures intrinsic output variance under stochastic decoding
- `mean_pairwise_rougeL`: Mean ROUGE-L F1 across all unique pairs of sampled responses

*Greedy-anchored stability metrics:*
- `mean_vs_greedy_rouge1`: Mean ROUGE-1 F1 of each sampled response against the original greedy response -- measures how far stochastic decoding drifts from the deterministic baseline
- `mean_vs_greedy_rougeL`: Mean ROUGE-L F1 of each sampled response against the original greedy response

Results are stored in `outputs/stability_scores.jsonl`.

#### 2.3.2 Calibrate stability scores:

```bash
python scripts/calibrate.py \
    --input_path outputs/stability_scores.jsonl \
    [--calibrators_path results/stability_calibrator.pkl]
```

This script follows the procedure described in Section 2.1.2, applied to the stability metric features.

#### 2.3.3 Evaluate predictive power of stability metrics:

```bash
python scripts/analyze.py \
    --input_path outputs/stability_scores.jsonl \
    --plots_dir plots/
```

This command produces distribution plots, precision-recall curves, and ROC curves for each stability metric, following the same format as Section 2.1.3. Results are saved to `plots/stability/` and `results/stability_summary.csv`.

#### Initial findings:

- Greedy-anchored similarity (`mean_vs_greedy`) substantially outperforms pairwise similarity (`mean_pairwise`). It matters more how much samples drift from the model's modal prediction than how much they vary relative to one another.
- For greedy-anchored similarity (`mean_vs_greedy`), ROUGE1 appears to be more informative than ROUGEL.
- Like the self-consistency metrics, stability metrics are less useful than the logit-based ones on their own and take longer to compute. However, there are many ways to improve these metrics, and they may be used in combination with logit-based metrics (and the self-consistency ones).


#### Limitations:

- The ROUGE similarity scores only provide lexical similarity. Two correct reports about the same case can have very different wording and still be equally right.
- The choice of temperature is a free hyperparameter -- higher temperatures amplify variance for all studies equally, potentially changing which studies rank as uncertain.

#### Future work:

- Consider other metrics to measure extent of response change (e.g., score of LLM judge), while considering the potential increase in inference time.
- Evaluate whether sampling variance correlates with correctness across different temperature values. We may also combine the scores produced at multiple temperature values.
- Increase the number of samples and test if performance is affected. Also evaluate how much this affects computation time.
- Implement MC Dropout as an alternative to temperature sampling (requires dropout layers to remain active at inference).


---

## 3. Future Work

Below are some other methods for confidence estimation that cannot be easily implemented with the currently available model/data, but would be interesting to explore with a model that exposes intermediate representations and additional data. 

### 3.1 Vision-Text Grounding

**Overview**: A well-grounded response should describe content that is present in the input image. If the model's generated text is semantically distant from what it actually saw, the response is likely hallucinated or otherwise incorrect. Vision-text grounding measures the cosine similarity between the vision encoding of the input and the text encoding of the generated response in a shared embedding space -- a direct signal of how well the response is anchored to the visual input, with no ground-truth report required.

**What would be needed**:
- Access to intermediate model representations
- A shared text embedding space
- Storage for vision and response encodings

### 3.2 Hidden-State Probing

**Overview**: Hidden-state probing trains a lightweight classifier (logistic regression or small MLP) on the model's internal representations to predict whether a given response will be judged correct by CRIMSON. Unlike the currently implemented approaches, this can capture uncertainty signals encoded in the model's internal state rather than derived solely from its outputs.

**What would be needed**:
- Access to hidden states
- A labelled training split
- Probe training and serialization

### 3.3 Confidence Calibration

**Overview**: A simple per-method calibration is implemented in Sections 2.1.2, 2.2.2, and 2.3.2 (multivariate logistic regression over each method's own metrics). It would be useful to implement multi-signal calibration, combining logit-based, grounding, consistency, stability, and probe features into a single unified P(incorrect) score, and evaluating it on a completely unseen test split.

We could consider calibration methods beyond logistic regression, such as non-parametric isotonic regression and a learned scalar temperature applied to an aggregated confidence score. We should also evaluate the calibrated score not only in terms of discrimination ability (i.e., AUPRC, Precision @ 90/95/99% Recall, AUROC as currently implemented) but also how well the score is actually calibrated (in terms of ECE, reliability diagrams, coverage-accuracy curves, etc.).

**What would still be needed**:
- Multiple confidence signals
- A held-out test split


### 3.4 Other Avenues for Future Work

- Combine the three already implemented methods to create new **hybrid methods**.
- Estimate **confidence per-finding**, rather than for the report as a whole.
- Utilize the **error breakdown** provided by CRIMSON (or another LLM judge). Can we detect false findings, or predict particular types of errors without access to the ground-truth reports?
- Explore **human-centered** confidence estimation. A starting point for this line of work: ["From Calibration to Collaboration: LLM Uncertainty Quantification Should Be More Human-Centered"](https://arxiv.org/abs/2506.07461).
- Explore confidence estimation with **statistical guarantees**. A starting point for this line of work: ["Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners"](https://arxiv.org/abs/2307.01928).
- Where relevant, better calibrate token probabilities during model training.
- Evaluate how confidence estimation methods generalize across data modalities and different generative models.
- Think about the utility of **out-of-distribution (OOD) detection** for the datasets of interest.
- Explore **multi-step** confidence estimation methods that move from coarse/simple/fast confidence metrics to more accurate/involved metrics depending on some decision criteria.
- Evaluate ways to **incorporate** confidence scores into the platform or to **improve** the model performance.

---

## 4. Repository Structure

```
radiology_confidence/
├── data/                               # gitignored — symlink or copy xray_samples here
│   └── xray_samples/
│       ├── selected_study_ids.json
│       └── studies/<study_id>/...
│
├── outputs/
│   ├── predictions.jsonl              # inference outputs (1.1.1)
│   ├── judge_scores.jsonl             # CRIMSON scores (1.1.2)
│   ├── logit_scores.jsonl             # logit-based metrics + prob_incorrect (2.1.1, 2.1.2)
│   ├── consistency_scores.jsonl       # self-consistency metrics + prob_incorrect (2.2.1, 2.2.2)
│   └── stability_scores.jsonl         # stability metrics + prob_incorrect (2.3.1, 2.3.2)
│
├── results/
│   ├── judge_summary.csv              # CRIMSON score distribution + threshold table (1.1.3)
│   ├── logit_summary.csv              # logit metric AUPRC / AUROC table (2.1.3)
│   ├── consistency_summary.csv        # consistency metric AUPRC / AUROC table (2.2.3)
│   ├── stability_summary.csv          # stability metric AUPRC / AUROC table (2.3.3)
│
├── plots/
│   ├── judge/                         # CRIMSON score distribution + error breakdown (1.1.3)
│   ├── logit/
│   │   ├── distributions/             # three-class KDE plots per metric (2.1.3)
│   │   ├── pr_curves/                 # precision-recall curves per metric (2.1.3)
│   │   └── roc_curves/                # ROC curves per metric (2.1.3)
│   ├── consistency/
│   │   ├── distributions/             # consistency score distributions (2.2.3)
│   │   ├── pr_curves/                 # consistency PR curves (2.2.3)
│   │   └── roc_curves/                # consistency ROC curves (2.2.3)
│   └── stability/
│       ├── distributions/             # stability score distributions (2.3.3)
│       ├── pr_curves/                 # stability PR curves (2.3.3)
│       └── roc_curves/                # stability ROC curves (2.3.3)
│
├── lexicon/
│   └── RadLex.owl                     # RadLex ontology for domain lexical weighting (2.1.1)
│
├── CRIMSON/                           # CRIMSON scoring package
│   ├── __init__.py
│   ├── generate_score.py
│   ├── prompt_parts.py
│   └── utils.py
│
├── scripts/
│   ├── prompt.py                      # MedGemma report generation prompt template
│   ├── infer.py                       # 1.1.1 -- run MedGemma inference, record logits
│   ├── evaluate.py                    # 1.1.2 -- CRIMSON judge scoring
│   ├── plot_scores.py                 # 1.1.3 -- plot CRIMSON score distribution
│   ├── logits.py                      # 2.1.1 -- compute logit-based metrics
│   ├── consistency.py                 # 2.2.1 -- self-consistency under perturbation
│   ├── stability.py                   # 2.3.1 -- stability under temperature sampling
│   ├── calibrate.py                   # 2.1.2, 2.2.2, 2.3.2 -- multivariate logistic regression calibration
│   ├── analyze.py                     # 2.1.3, 2.2.3, 2.3.3 -- distributions, PR + ROC curves
│   ├── text_utils.py                  # shared text preprocessing utilities
│   ├── utils.py                       # shared utilities (I/O, ROUGE, image helpers, metrics)
│   └── inspect_lexicon.py             # browse and filter the RadLex lexicon
│
├── requirements_medgemma.txt          # medgemma environment dependencies
├── requirements_crimson.txt           # crimson environment dependencies
├── requirements_sglang.txt            # sglang environment dependencies
├── run_pipeline.sh                    # end-to-end pipeline runner
└── README.md
```