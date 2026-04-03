# Confidence Estimation for Radiology Report Generation

This repository implements a post-training confidence estimation pipeline for a radiology report generation model. Given a chest X-ray, MedGemma generates a free-text radiology report. Confidence estimation determines how likely that report is to be judged correct by CRIMSON -- a clinically-grounded LLM-based radiology report evaluator -- without requiring CRIMSON at inference time.

The repository is organized into four phases: confidence signal extraction ([Section 2](#2-confidence-signals)), calibration and evaluation ([Section 3](#3-confidence-calibration-and-evaluation)), and integration into an assistive agent ([Section 4](#4-confidence-integration-and-usability)). We have partially implemented three signal extraction methods:

- **Logit-based metrics** ([Section 2.1](#21-logit-based-confidence)): functions of token log-probabilities and entropies recorded during generation
- **Consistency under input perturbation** ([Section 2.2](#22-consistency-under-input-perturbation)): response consistency under controlled input perturbations
- **Consistency during sampling** ([Section 2.3](#23-consistency-during-sampling)): response variance under temperature sampling

We have not yet implemented vision-text grounding ([Section 2.4](#24-future-vision-text-grounding)), hidden-state probing ([Section 2.5](#25-future-hidden-state-probing)), explicit human feedback ([Section 2.6](#26-future-explicit-human-feedback)), or implicit human feedback ([Section 2.7](#27-future-implicit-human-feedback)). Some initial efforts have been made towards confidence calibration and evaluation ([Section 3](#3-confidence-calibration-and-evaluation)), but there is much work to be done in this direction. We also discuss some initial ideas surrounding integration into an assistive agent ([Section 4](#4-confidence-integration-and-usability)).

## Full Pipeline

The `run_pipeline.sh` script runs all currently implemented steps end-to-end, managing conda environments and the SGLang server automatically:

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
2. [Confidence Signals](#2-confidence-signals)
   - [2.1 Logit-Based Confidence](#21-logit-based-confidence)
   - [2.2 Consistency under Input Perturbation](#22-consistency-under-input-perturbation)
   - [2.3 Consistency during Sampling](#23-consistency-during-sampling)
   - [2.4 (Future) Vision-Text Grounding](#24-future-vision-text-grounding)
   - [2.5 (Future) Hidden-State Probing](#25-future-hidden-state-probing)
   - [2.6 (Future) Explicit Human Feedback](#26-future-explicit-human-feedback)
   - [2.7 (Future) Implicit Human Feedback](#27-future-implicit-human-feedback)
3. [Confidence Calibration and Evaluation](#3-confidence-calibration-and-evaluation)
   - [3.1 Confidence Calibration](#31-confidence-calibration)
   - [3.2 (Future) Combining Multiple Signals](#32-future-combining-multiple-signals)
   - [3.3 Confidence Evaluation](#33-confidence-evaluation)
4. [Confidence Integration and Usability](#4-confidence-integration-and-usability)
5. [Repository Structure](#5-repository-structure)

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

---

### 1.2 Generate Confidence Estimation Dataset

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

## 2. Confidence Signals

This section covers methods for extracting a confidence signal from the model and its outputs. Sections 2.1–2.3 describe implemented methods; each includes initial findings, limitations, and directions for future work. Sections 2.4–2.7 describe future directions that require additional model access or platform infrastructure not currently available.

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
- Investigate whether per-sentence or per-finding aggregation outperforms per-token aggregation.

---

### 2.2 Consistency under Input Perturbation

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

#### Initial findings:

- Perturbing the input using blur provides a more useful signal than adding noise, but taking the average across both methods performs the best.
- Mean ROUGE-1 outperforms ROUGE-L in this setting, suggesting unigram overlap is more informative than sequence order preservation for this task.
- The consistency metrics are less useful than the logit-based ones on their own and take longer to compute. However, there are many ways to improve these metrics, and they may be used in combination with logit-based metrics.

#### Limitations:

- The ROUGE similarity scores only provide lexical similarity. Two correct reports about the same case can have very different wording and still be equally right.
- Perturbations are applied uniformly across all frames rather than targeting diagnostically relevant regions. Gaussian noise and blur also do not necessarily reflect natural variations in X-ray images.
- With temperature=0, the only stochasticity comes from the image perturbation; small perturbations that don't change the visual content enough to shift the response will produce trivially high consistency.

#### Future work:

- Consider other metrics to measure extent of response change (e.g., score of LLM judge), while considering the potential increase in inference time.
- Apply more natural variations to input frames that reflect realistic scanner variability (e.g., different viewing angles, scanner noise profiles).
- Selectively mask or occlude anatomical regions and measure whether the report changes for that finding specifically.
- Apply noise to the vision encoding directly if available.
- Increase the number of samples and test if performance is affected. Also evaluate how much this affects computation time.

---

### 2.3 Consistency during Sampling

Under temperature sampling, a model draws responses stochastically rather than greedily, producing different outputs on each forward pass. A confident model will produce responses with low variance across samples, while an uncertain model will produce inconsistent or contradictory outputs. Unlike Section 2.2, this approach perturbs the model's decoding rather than its input.

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

#### Initial findings:

- Greedy-anchored similarity (`mean_vs_greedy`) substantially outperforms pairwise similarity (`mean_pairwise`). It matters more how much samples drift from the model's modal prediction than how much they vary relative to one another.
- For greedy-anchored similarity (`mean_vs_greedy`), ROUGE1 appears to be more informative than ROUGEL.
- Like the consistency metrics, stability metrics are less useful than the logit-based ones on their own and take longer to compute. However, there are many ways to improve these metrics, and they may be used in combination with logit-based metrics (and the consistency ones).

#### Limitations:

- The ROUGE similarity scores only provide lexical similarity. Two correct reports about the same case can have very different wording and still be equally right.
- The choice of temperature is a free hyperparameter -- higher temperatures amplify variance for all studies equally, potentially changing which studies rank as uncertain.

#### Future work:

- Consider other metrics to measure extent of response change (e.g., score of LLM judge), while considering the potential increase in inference time.
- Evaluate whether sampling variance correlates with correctness across different temperature values. We may also combine the scores produced at multiple temperature values.
- Increase the number of samples and test if performance is affected. Also evaluate how much this affects computation time.
- Implement MC Dropout as an alternative to temperature sampling (requires dropout layers to remain active at inference).

---

### 2.4 (Future) Vision-Text Grounding

**Overview**: A well-grounded response should describe content that is present in the input image. If the model's generated text is semantically distant from what it actually saw, the response is likely hallucinated or otherwise incorrect. Vision-text grounding measures the cosine similarity between the vision encoding of the input and the text encoding of the generated response in a shared embedding space -- a direct signal of how well the response is anchored to the visual input, with no ground-truth report required.

**What would be needed**:
- Access to intermediate model representations
- A shared vision-text embedding space
- Storage for vision and response encodings

---

### 2.5 (Future) Hidden-State Probing

**Overview**: Hidden-state probing trains a lightweight classifier (logistic regression or small MLP) on the model's internal representations to predict whether a given response will be judged correct by CRIMSON. Unlike the currently implemented approaches, this can capture uncertainty signals encoded in the model's internal state rather than derived solely from its outputs.

**What would be needed**:
- Access to hidden states
- A labelled training split
- Probe training and serialization

---

### 2.6 (Future) Explicit Human Feedback

**Overview**: When a radiologist reviews a generated report, their accept/reject decisions and any corrections they make are a rich source of supervisory signal. A rejected suggestion paired with the radiologist's replacement text reveals potential errors, as well as individual preferences. This direction connects to research on [Reinforcement Learning from Human Feedback (RLHF)](https://arxiv.org/abs/2405.20677), as well as [Recommender Systems (RS)](https://arxiv.org/html/2407.13699v1).

**Additional confidence signal**: Rejection rates are a direct real-world signal of model failure. Over time, per-study and per-finding-type rejection rates can be aggregated to recalibrate confidence scores and identify where the model is systematically overconfident.

Furthermore, the delta between a rejected suggestion and the radiologist's replacement can reveal systematic error patterns -- e.g., the model consistently misses a particular finding type or uses incorrect severity language. These corrections could also serve as a fine-tuning signal to improve the underlying model. This has some connection to work on [Expert Intervention Learning (EIL)](https://roboticsconference.org/2020/program/papers/55.html), whereby at every iteration, we execute the learner, collect intervention data, aggregate it, map to constraints on the learner's action-value function, and update the learner.

**Individual preferences**: Radiologists vary in their reporting style -- preferred terminology, level of detail, hedging language, and how they describe ambiguous findings. Some of these differences reflect genuine clinical disagreement rather than model error. Modeling per-user correction patterns could allow the system to distinguish idiosyncratic style preferences from consensus errors, and to personalize suggestions to individual radiologists over time.

This also has implications for confidence estimation: a rejection driven by style preference carries different weight than one driven by a factual error, and the two should ideally be distinguished.

**What would still be needed**:
- A platform that captures accept/reject decisions and correction text
- Sufficient user interaction volume to learn meaningful patterns
- A privacy and consent framework for capturing and using clinician feedback

---

### 2.7 (Future) Implicit Human Feedback

**Overview**: Beyond explicit accept/reject decisions, users reveal information about their uncertainty and cognitive load through behavioral signals. Signs of frustration or confusion -- such as erratic cursor movements, extended review time, or facial expressions -- could serve as a proxy for cases where the model output was difficult to verify or likely incorrect, without requiring any deliberate action from the user. This connects to a relatively small body of work on [Learning from Implicit Human Feedback (LIHF)](https://proceedings.mlr.press/v155/cui21a/cui21a.pdf).

**Additional confidence signal**: Cursor movement frequency and patterns (e.g., erratic or hesitant movement), time spent reviewing a report, scroll and re-read behavior, and hesitation before accepting a suggestion could all correlate with model errors or uncertain outputs. Facial expression recognition (e.g., signs of frustration, concentration, or confusion) could provide an additional signal, though this raises significant privacy concerns and would require explicit consent.

**What would still be needed**:
- Platform integration with behavioral tracking
- (Optionally) Consent-based physiological signal capture
- Sufficient interaction data and ground-truth labels to validate whether implicit signals correlate with model errors
- A privacy and consent framework


---

## 3. Confidence Calibration and Evaluation

The confidence signals from Section 2 are calibrated to produce a single P(incorrect) score per record and evaluated for their discriminative power. Sections 3.1 and 3.3 describe the current implementation; each includes a future work subsection covering more rigorous extensions. Section 3.2 covers multi-signal fusion, which requires multiple signals to be implemented first.

### 3.1 Confidence Calibration

Each signal computed in Sections 2.1–2.3 is calibrated independently using a multivariate logistic regression over that method's metrics, producing a single `prob_incorrect` field per record. The calibrated scores are written back to the same JSONL files as the raw signals.

#### 3.1.1 Logit-based signals:

```bash
python scripts/calibrate.py \
    --input_path outputs/logit_scores.jsonl \
    [--output_path outputs/logit_scores.jsonl] \
    [--n_folds 5] \
    [--calibrators_path results/logit_calibrator.pkl]
```

This script fits a multivariate logistic regression model over all logit metrics to produce a single `prob_incorrect` field per record. By default, this result is written to the same input JSONL if `--output_path` is not set.

#### 3.1.2 Consistency under input perturbation:

```bash
python scripts/calibrate.py \
    --input_path outputs/consistency_scores.jsonl \
    [--calibrators_path results/consistency_calibrator.pkl]
```

#### 3.1.3 Consistency during sampling:

```bash
python scripts/calibrate.py \
    --input_path outputs/stability_scores.jsonl \
    [--calibrators_path results/stability_calibrator.pkl]
```

#### Future work:

- Implement and compare calibration methods beyond logistic regression: temperature scaling, Platt scaling, and non-parametric isotonic regression.
- As an alternative to probability calibration, conformal prediction provides distribution-free coverage guarantees -- e.g., a set of candidate findings that contains the correct finding with at least 95% probability. ["Conformal Language Modeling"](https://arxiv.org/pdf/2306.10193) demonstrates this in the context of language model generation. In the radiology setting this could produce a set of plausible findings for each study rather than a single report.
- Where relevant, better calibrate token probabilities during model training.

---

### 3.2 (Future) Combining Multiple Signals

**Overview**: Each method in Sections 2.1–2.3 is calibrated independently. A more complete approach would combine logit-based, grounding, consistency, stability, and probe features into a single unified P(incorrect) score, calibrated on a training split and evaluated on a completely unseen test split.

**Ensembling and hybrid methods**: We could also consider model ensembling and mixture-of-experts approaches for confidence estimation, and combine categories of methods to create new hybrid methods that are not directly derivable from any single signal source.

**What would still be needed**:
- Multiple confidence signals implemented (Sections 2.4–2.7)
- A labelled training split
- A held-out test split

---

### 3.3 Confidence Evaluation

`analyze.py` evaluates the discriminative power of each confidence signal by computing distribution plots, precision-recall curves, and ROC curves. It auto-detects the score type from the input filename and routes all output to the corresponding subdirectory.

#### 3.3.1 Logit-based signals:

```bash
python scripts/analyze.py \
    --input_path outputs/logit_scores.jsonl \
    --plots_dir plots/
```

This command produces the following:
- **Distribution plots** (`plots/logit/distributions/`): KDE for three populations -- perfect reports (score=1.0), partially correct (0≤score<1), and incorrect (score<0)
- **Precision-recall curves** (`plots/logit/pr_curves/`): PR curve per metric treating incorrect (score<0) as the positive class; saves AUPRC and Precision @ 90/95/99% Recall
- **ROC curves** (`plots/logit/roc_curves/`): ROC curve per metric; saves AUROC and Sensitivity @ 90/95/99% Specificity
- **Summary CSV** (`results/logit_summary.csv`): all of the above metrics per field; also printed to stdout in two tables (PR and ROC)

#### 3.3.2 Consistency under input perturbation:

```bash
python scripts/analyze.py \
    --input_path outputs/consistency_scores.jsonl \
    --plots_dir plots/
```

This command produces distribution plots, precision-recall curves, and ROC curves for each consistency metric, following the same format as Section 3.3.1. Results are saved to `plots/consistency/` and `results/consistency_summary.csv`.

#### 3.3.3 Consistency during sampling:

```bash
python scripts/analyze.py \
    --input_path outputs/stability_scores.jsonl \
    --plots_dir plots/
```

This command produces distribution plots, precision-recall curves, and ROC curves for each stability metric, following the same format as Section 3.3.1. Results are saved to `plots/stability/` and `results/stability_summary.csv`.

#### Future work:

- **Calibration quality**: Evaluate the calibrated score not only in terms of discrimination ability (AUPRC, AUROC) but also how well the score is actually calibrated: Brier score, ECE, reliability diagrams, coverage-accuracy curves.
- **Trade-off analysis**: Confidence estimation methods vary in practical utility -- both in terms of their discriminative power plus how well they are calibrated and in terms of their computational complexity. We should evaluate all methods in terms of their performance and computational cost to better understand this trade-off.
- **Distribution shifts**: It is possible that models trained on one set of inputs will perform poorly on another set drawn from a different distribution (e.g., inputs from a different patient population or collected with a different scanner). It is important to understand how model performance varies across different distributions of data and whether confidence estimates continue to provide a useful signal.
  - **OOD detection**: If model performance varies across distributions in a way that is not fully captured by the confidence signal, we should design out-of-distribution (OOD) detection methods. We could explore existing methods for visual OOD detection that utilize the vision embeddings of the generative radiology model, considering first distance-based and reconstruction-based methods.
  - **What would still be needed**: access to vision embeddings and training data (preferably), and OOD dataset(s).
- **Error type prediction**: Utilize the error breakdown provided by CRIMSON (or another LLM judge). Can we detect false findings, or predict particular types of errors without access to the ground-truth reports?
- **Generalization**: Evaluate how confidence estimation methods generalize across data modalities and different generative models.


---

## 4. Confidence Integration and Usability

Confidence estimation is most valuable when embedded in an assistive agent that can act on it -- deciding whether to surface a suggestion, present alternatives, request user input, or abstain entirely. The question shifts from *how certain is the model?* to *what should the agent do, given that certainty?* This section covers how confidence signals feed into agent behavior and how that behavior should be evaluated in terms of its utility to the radiologist.

**Human uplift**: ["From Calibration to Collaboration: LLM Uncertainty Quantification Should Be More Human-Centered"](https://arxiv.org/abs/2506.07461) argues for a focus on metrics that correlate with human uplift on real-world decision tasks. In particular, do humans with access to algorithmic advice with quantified uncertainty perform better on real-world tasks than humans with (i) algorithmic advice but no quantified uncertainty; and (ii) no algorithmic advice at all? This framing motivates evaluating the entire pipeline -- not just calibration metrics -- against team-level outcomes. [Bansal et al. (2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17359) make a related argument: the most accurate AI is not necessarily the best teammate, and AI systems should be trained to optimize human-AI team utility rather than standalone accuracy.

**Agent actions under uncertainty**: Given a confidence estimate, the agent can take a range of actions. Taking inspiration from ["Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners"](https://arxiv.org/abs/2307.01928), we may have the agent present the user with multiple options when it is not confident in a particular response. ["Conformal Language Modeling"](https://arxiv.org/pdf/2306.10193) presents a method to generate a complete set of candidate options with coverage guarantees (already done in the context of radiology report generation). Alternatively, we may choose to simply not show low-confidence suggestions below some threshold (selective prediction / abstention), and evaluate the accuracy of retained reports as a function of the threshold.

**Multi-step estimation**: Depending on the trade-offs identified in Section 3.3, we could design multi-step confidence estimation schemes that move from coarse/simple/fast metrics to more accurate/involved ones depending on some decision criteria, gating expensive methods behind a cheap initial screen.

**Per-finding confidence**: To improve the utility of confidence information, it would likely be helpful to estimate confidence per finding rather than for the report as a whole (potentially using a simple sentence splitter to extract individual findings).

**Improving the model**: Confidence scores can also be used to improve the underlying model. A low-confidence prediction could trigger re-generation with a different prompt or sampling strategy. We may also better characterize situations where the model does poorly and use this to guide re-training.

**What would still be needed**:
- (Eventually) Platform integration and human evaluators


---

## 5. Repository Structure

```
radiology_confidence/
├── data/                               # gitignored — symlink or copy xray_samples here
│   └── xray_samples/
│       ├── selected_study_ids.json
│       └── studies/<study_id>/...
│
├── outputs/
│   ├── predictions.jsonl              # inference outputs (1.2.1)
│   ├── judge_scores.jsonl             # CRIMSON scores (1.2.2)
│   ├── logit_scores.jsonl             # logit-based metrics + prob_incorrect (2.1.1, 3.1)
│   ├── consistency_scores.jsonl       # consistency metrics + prob_incorrect (2.2.1, 3.1)
│   └── stability_scores.jsonl         # sampling stability metrics + prob_incorrect (2.3.1, 3.1)
│
├── results/
│   ├── judge_summary.csv              # CRIMSON score distribution + threshold table (1.2.3)
│   ├── logit_summary.csv              # logit metric AUPRC / AUROC table (3.3)
│   ├── consistency_summary.csv        # consistency metric AUPRC / AUROC table (3.3)
│   ├── stability_summary.csv          # stability metric AUPRC / AUROC table (3.3)
│
├── plots/
│   ├── judge/                         # CRIMSON score distribution + error breakdown (1.2.3)
│   ├── logit/
│   │   ├── distributions/             # three-class KDE plots per metric (3.3)
│   │   ├── pr_curves/                 # precision-recall curves per metric (3.3)
│   │   └── roc_curves/                # ROC curves per metric (3.3)
│   ├── consistency/
│   │   ├── distributions/             # consistency score distributions (3.3)
│   │   ├── pr_curves/                 # consistency PR curves (3.3)
│   │   └── roc_curves/                # consistency ROC curves (3.3)
│   └── stability/
│       ├── distributions/             # stability score distributions (3.3)
│       ├── pr_curves/                 # stability PR curves (3.3)
│       └── roc_curves/                # stability ROC curves (3.3)
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
│   ├── infer.py                       # 1.2.1 -- run MedGemma inference, record logits
│   ├── evaluate.py                    # 1.2.2 -- CRIMSON judge scoring
│   ├── plot_scores.py                 # 1.2.3 -- plot CRIMSON score distribution
│   ├── logits.py                      # 2.1.1 -- compute logit-based metrics
│   ├── consistency.py                 # 2.2.1 -- consistency under input perturbation
│   ├── stability.py                   # 2.3.1 -- consistency during sampling
│   ├── calibrate.py                   # 3.1 -- multivariate logistic regression calibration
│   ├── analyze.py                     # 3.3 -- distributions, PR + ROC curves
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
