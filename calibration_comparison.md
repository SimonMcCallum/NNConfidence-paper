# Calibration Baseline Comparison Plan

Systematic comparison of HLCC Norm-Shift confidence against established calibration methods.

## Methods

| Method | Category | Key Paper | Implementation | Cost |
|--------|----------|-----------|----------------|------|
| Temperature Scaling | Post-hoc | Guo et al. 2017 | Learn T on val set, apply to softmax | 1 pass + fit |
| Platt Scaling | Post-hoc | Platt 1999 | Logistic regression on logits | 1 pass + fit |
| Softmax Entropy | Token distribution | — | `1 - H(p)/log(V)` | 1 pass |
| Top-k Probability Mass | Token distribution | — | Sum of top-k softmax probs | 1 pass |
| Self-Consistency | Multi-sample | Wang et al. 2023 | N samples, majority vote fraction | N passes |
| Semantic Entropy | Multi-sample | Kuhn et al. 2023 | Cluster by meaning, entropy over clusters | N passes + NLI |
| Verbalized Confidence | Prompting | Xiong et al. 2024 | Ask model to state confidence 0-100 | 2 passes |
| P(True) | Prompting | Kadavath et al. 2022 | P("True") for correctness check prompt | 2 passes |
| Conformal Prediction | Statistical | Angelopoulos et al. 2021 | Prediction sets with coverage guarantee | 1 pass + cal set |
| **HLCC Norm-Shift (ours)** | Trained head | This work | Per-layer norm signals + HLCC loss | 1 pass (trained) |

## Implementation Status

### Phase 1: Single Forward Pass (Implemented)

- **Softmax Entropy** (`softmax_entropy_confidence`): Normalized Shannon entropy of the next-token distribution. Zero-cost baseline — purely a function of the logits already computed.

- **Top-k Probability Mass** (`topk_probability_confidence`): Sum of the k=5 highest token probabilities. Captures whether the model's probability mass is concentrated.

- **Verbalized Confidence** (`verbalized_confidence`): Prompt the model to self-assess confidence on a 0-100 scale. Requires a second generation pass but uses no external tools.

### Phase 2: Multiple Passes (Implemented)

- **Self-Consistency** (`self_consistency_confidence`): Sample N=10 answers at temperature 0.7, return majority vote fraction as confidence. N forward passes per example.

- **Temperature Scaling** (`fit_temperature_scaling` + `temperature_scaled_confidence`): Fit a single scalar T that minimizes NLL on the choice logits. Standard post-hoc calibration from Guo et al. 2017.

### Phase 3: Advanced (Not Yet Implemented)

- **Semantic Entropy**: Requires NLI model to cluster semantically equivalent answers. More complex infrastructure.

- **P(True)**: Requires two-stage prompting (answer, then P("True" | "Is this correct?")).

- **Conformal Prediction**: Requires calibration dataset for quantile estimation.

- **Platt Scaling**: Logistic regression variant of temperature scaling. Straightforward to add.

## Evaluation Plan

### Datasets

| Dataset | Examples | Purpose |
|---------|----------|---------|
| TruthfulQA | 300 | Tests overconfidence on misleading questions |
| ARC-Challenge | 300 | Tests reasoning under uncertainty |

### Models

| Model | Parameters | Rationale |
|-------|-----------|-----------|
| Qwen2.5-7B | 7.6B | Primary evaluation model |
| Llama-3.1-8B | 8.0B | Cross-architecture validation |

### Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| ECE | Expected Calibration Error (10 bins) | Lower is better |
| Brier Score | Mean squared error of confidence vs correctness | Lower is better |
| AUROC | Confidence as a binary classifier for correctness | Higher is better |
| HLCC Score | Mean HLCC score across examples | Higher is better |

### Execution

```bash
# Run all Phase 1+2 baselines
python calibration_baselines.py --model qwen2.5-7b --dataset truthfulqa --max-examples 300

# Include trained HLCC norm-shift head
python calibration_baselines.py --model qwen2.5-7b --dataset truthfulqa --max-examples 300 \
    --norm-shift-checkpoint data/checkpoints/best_norm_shift_combined.pt

# Run on ARC-Challenge
python calibration_baselines.py --model qwen2.5-7b --dataset arc-challenge --max-examples 300
```

## Expected Results

### Hypotheses

1. **Entropy and top-k** will have poor calibration (high ECE) because they measure distributional uncertainty, not correctness probability.

2. **Temperature scaling** will improve ECE over raw softmax but may not improve AUROC (it doesn't change ranking, only scaling).

3. **Self-consistency** will have moderate calibration but high computational cost (10x forward passes).

4. **Verbalized confidence** will be poorly calibrated for smaller models but may improve for larger instruction-tuned models.

5. **HLCC Norm-Shift** should achieve the best ECE and HLCC score among single-pass methods because:
   - It is directly trained to optimize HLCC (calibration-aware loss)
   - It has access to internal model dynamics (norm-shift signals) not visible to other methods
   - The training objective asymmetrically penalizes overconfidence

### Result Tables (To Be Filled)

**Table 1: Single-Pass Methods**

| Method | ECE | Brier | AUROC | HLCC Score |
|--------|-----|-------|-------|------------|
| Softmax Entropy | | | | |
| Top-k Probability | | | | |
| Temperature Scaling | | | | |
| Verbalized Confidence | | | | |
| **HLCC Norm-Shift** | | | | |

**Table 2: Multi-Sample Methods**

| Method | ECE | Brier | AUROC | HLCC Score | Cost (passes) |
|--------|-----|-------|-------|------------|---------------|
| Self-Consistency (N=10) | | | | | 10 |
| **HLCC Norm-Shift** | | | | | 1 |

## References

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. ICML.
- Platt, J. (1999). Probabilistic outputs for support vector machines. Advances in Large Margin Classifiers.
- Wang, X., et al. (2023). Self-consistency improves chain of thought reasoning in language models. ICLR.
- Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic entropy probes the linguistic confidence of large language models. Nature.
- Xiong, M., et al. (2024). Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs. ICLR.
- Kadavath, S., et al. (2022). Language models (mostly) know what they know. arXiv:2207.05221.
- Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction. arXiv:2107.07511.
