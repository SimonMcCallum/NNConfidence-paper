# Kumaran et al. (2026) versus the NormShift / activation-forensics programme

Reference: Kumaran, Daw, Osindero, Veličković, Patraucean.
*Causal evidence that language models use confidence to drive behaviour.*
Nature Machine Intelligence, published 7 September 2026.
DOI 10.1038/s42256-026-01293-x. Local copy: `docs/references/s42256-026-01293-x.pdf`.
Code and data (OSF): https://osf.io/resw4/

Written 10 September 2026 as a working note for both papers
(`docs/hlcc_paper.tex` here and `D:/git/activation-paper/paper.tex`).

---

## 1. What they did

Four-phase paradigm on SimpleQA rewritten as four-option MCQ (1,000 calibration
questions, 1,000 test questions; foils are similar, dissimilar and unrelated).
Models: GPT-4o (API), Gemma 3 27B (JAX, only model with activation access),
DeepSeek 671B and Qwen 80B (Together.ai).

| Phase | Manipulation | Finding |
|---|---|---|
| 1 | MCQ, no abstain option. Record temperature-scaled logprob confidence (T = 4.1, ECE 0.046, AUROC 0.90) and a separate-pass 10-class verbal confidence. | Calibrated confidence predicts error rate, r = -0.97. |
| 2 | Same questions, "none of the above" allowed. | Abstention ~ calibrated confidence with standardised beta 0.99, about 10x difficulty, RAG score, or sentence-embedding PCs. Implicit threshold T50 = 77 %. |
| 3 | Activation steering in Gemma 3 27B: difference-of-means of high-margin vs low-margin trials, injected at 3 % of residual norm, layers 30 to 40 of 60 (peak layer 31). | Abstention swings 66.5 % (low-conf steer) to 7.0 % (high-conf steer). Parallel mediation: 67 % via confidence redistribution, 26 % via policy shift. |
| 4 | Instructed threshold T in 0 to 100 %. | Models obey; fitted policy over-weights own confidence (scale 1.80) with a conservative bias (shift -97.6). Phase 1 confidence stays predictive across all T. |
| Decoding | Ridge probes at the "answer colon" (AC) token. | AC activations explain verbal confidence (R2 0.39, layer 37) and calibrated confidence (R2 0.58, layer 61), each adding over the other (dR2 +0.27 and +0.45). AC activations only weakly predict correctness (AUROC 0.59 vs 0.62 for calibrated confidence). |

Their Fig. 6 architecture: one multidimensional internal confidence
representation at the AC token, comprising a generative component (X_gen, read
out lossily as calibrated logprob confidence) and an evaluative component
(X_eval, read out lossily as verbal confidence, r = 0.3 to 0.4 between them), feeding
a threshold policy sigma((T - C)/tau).

## 2. Where our work overlaps

| Theirs | Ours | Notes |
|---|---|---|
| "Both observable measures are lossy read-outs of a richer internal representation." | Robins and McCallum (2004) energy-profile thesis; `paper.tex` Section 1. | Same central claim. They arrive at it via ridge probes on the residual; we arrive via structural statistics. Cite them as independent confirmation. |
| Pre-answer "answer colon" token as the probe site. | Last prompt token in `popqa_familiarity_probe.py`, `extract_layer_signals.py`; last token of the assertion in FamGate. | Identical position. Their entity-token control (near-zero R2 at last question token before options) matches our `ent:` position features. |
| Two-stage architecture: confidence formation then threshold policy. | FamGate pipeline (gate on familiarity, then Kelly-size on MSP); NormShift-gated retry; multi-pass evaluator thresholds. | Our pipeline is a two-stage policy; HLCC is the scoring rule that fixes where the threshold should be. |
| Confidence-gated abstention on MCQ. | `multi_pass_evaluator.py`, `eval_cross_dataset.py --retry`. | They abstain; we retry with a trick-question prompt. Rescue/damage rates are our extension of their coverage-accuracy trade-off. |
| MSP-style calibrated confidence is the strongest observable predictor. | HLCC paper Discussion: "MSP is underrated"; `paper.tex` Section 10.4 MSP steelman. | Agreement. |
| Difficulty (multi-seed accuracy), RAG score and sentence-embedding PCs as controls. | PopQA popularity (external support proxy), relation type, subject token count, prompt length as controls in `analyse_popqa_familiarity.py`. | We have an *external* knowledge-support variable they lack. |
| Gemma 3 27B, instruction-tuned. | `model_loader.py` has gemma2-27b (4-bit). Gemma 3 27B is not yet registered. | Gemma uses GeGLU, so gate statistics apply directly. |

## 3. Where we differ, and why the difference is the paper

### 3.1 What is being probed

Kumaran decode a *direction*. Ridge regression on the full residual vector at
the AC token is a linear functional of the latent; it lives in the same regime
as difference-of-means, linear probes, and dictionary learning (sparse
autoencoders). Anything such a probe reads is, by construction, a linear
combination of residual-stream coordinates, and is therefore also readable by
the model's own downstream attention and MLP weights.

We measure *moments of the computation*:

| Statistic | Tensor | Basis dependence | Linear in the residual? |
|---|---|---|---|
| NormShift `1 - std(h)` and pre-norm L2 | residual h, pre-RMSNorm | rotation-invariant | No (second moment) |
| Pre-norm CV `std/mean|h|` | residual h | rotation-invariant | No |
| Gate near-zero ratio, gate std | gate pre-activation g = W_gate h, dimension d_g | privileged neuron basis | No; and not a residual tensor at all |
| Gate kurtosis | g | privileged basis (fourth moment) | No |
| SwiGLU sparsity | sigma(g) * (W_up h) | privileged basis | No |

Three consequences:

1. **A ridge probe on the residual cannot represent them.** A second or fourth
   moment of the coordinates is not a linear functional. This is provable, not
   empirical, for the pre-norm statistics; for the gate statistics the tensor
   itself is not in the residual stream.
2. **The model's own next sublayer cannot read the magnitude either.** RMSNorm
   divides by the RMS before any attention or MLP block reads the state, and
   the final norm does the same before `lm_head`. The pre-norm magnitude only
   survives implicitly, as the ratio of skip-path to sublayer contribution. So
   the verbal-confidence pass, which is a second forward pass over tokens the
   model itself emitted, has no direct access to the quantity NormShift
   measures. This is the precise sense in which "the LLM cannot access it".
3. **A dictionary (SAE) built on the residual cannot recover them without
   loss.** SAE features are sparse linear directions with the reconstruction
   error absorbed as noise. Kurtosis and near-zero ratio are exactly the kind of
   structure that reconstruction error discards.

Caveat to state honestly in the paper: the model *could* have learned a
residual direction that correlates with its own norm or with routing sparsity.
Lindsey (2025) reports partial introspective access to injected concepts. So
"not linearly decodable" is an empirical claim to be tested, not assumed. The
test is in Section 4.1.

### 3.2 What is being predicted

Kumaran predict the model's *abstention behaviour* and its *confidence
read-outs*. Correctness is secondary; the AC probe's correctness AUROC of 0.59
is close to the calibrated-confidence AUROC of 0.62, and they conclude the model
has no privileged access to correctness beyond its confidence.

We predict *correctness* and *answerability* (the HK-/HK+ split). Our matched
format result (MSP with isotonic calibration below chance on answerability,
`paper.tex` Table matched_format) is the same "lossy read-out" claim on a
different axis: the axis the model does not act on.

This sets up the dissociation that would be the headline of an update:

> The model abstains on what it can read (X_gen, X_eval), not on what would
> make it correct. Structural statistics predict correctness and answerability
> but should *not* predict the model's own abstention decision once calibrated
> confidence is controlled, because the policy has no read access to them.

If that dissociation holds, it is a direct extension of Kumaran's Fig. 6: a third
component, X_struct, that is causally upstream of correctness but sits outside
the read-out channels the threshold policy consumes.

### 3.3 Causality

They have steering and mediation. We have none. This is the largest gap and
the most valuable update (Section 4.3). It also offers a different lever:
their intervention adds a direction; ours can rescale the pre-norm residual or
clamp gate sparsity, which is an intervention on the structure rather than the
content.

### 3.4 Scoring rule

They use ECE, AUROC and logistic modelling. There is no asymmetric scoring
rule, so they describe the policy the model *has* (scale 1.80, shift -97.6,
T50 = 46 % at 80 % confidence) but not the policy it *should* have. HLCC
supplies the normative side: c* = p / [4(1 - p)], the abstain zone under the
unbounded variant, and Kelly-sized wagers. Their Phase 4 fitted policy can be
compared directly to the HLCC-optimal policy (Section 4.5).

### 3.5 Scale and access

They work at 27B to 671B, mostly through APIs, with activation access only on
one model. We work at 1.5B to 32B with full activation access on all of them.
The complementary framing is that they show the phenomenon in frontier models
and we show its mechanism in open ones.

## 4. Concrete updates, in order of cost

### 4.1 Incremental-variance test (their methodology, our features)

Cheapest and most decisive. On one open model at the AC token:

1. Ridge probe on the full residual at the best layer, 5-fold CV, predicting
   (a) calibrated confidence, (b) verbal confidence, (c) correctness,
   (d) abstention in a Phase 2 replica.
2. Add our structural scalars (7 statistics x chosen layers, or the 12
   mid-layer familiarity features) to the same ridge/logistic model.
   Report dR2 and dAUROC.
3. Reverse direction: ridge-predict each structural statistic *from* the
   residual. Low R2 is the direct demonstration that the statistic is not
   linearly decodable from the latent.
4. If a Gemma Scope SAE is available for the chosen Gemma model, recompute the
   pre-norm statistics on the SAE reconstruction and compare predictive power
   with the original. Loss of predictive power shows the dictionary does not
   carry the signal.

Existing code: `popqa_familiarity_probe.py` already records every feature
needed except the residual vector itself and verbal confidence. Add
`--save-residual` (store the AC-token hidden state at a few layers) and
`--verbal-confidence` (second pass with their 10-class prompt from the Methods).

### 4.2 Phase 2 replication on open models

Add a `--abstain-option` flag to `hlcc_mcq_system.py` that appends "None of
the above" using their exact Phase 2 prompt, on their exact SimpleQA-MCQ
1,000-question set from OSF. Then fit their Eq. 1 and Eq. 4 with our structural
statistics as additional predictors. The prediction from Section 3.2 is that
structural statistics add little to the abstention model but a lot to the
correctness model. Use the same models as the calibration baselines
(mistral-7b, qwen2.5-7b, llama3.1-8b) plus Gemma 3 27B for a direct anchor to
their Phase 3 model.

### 4.3 Steering cross-check and structural intervention

1. Reproduce their steering vectors (difference of means of 25 high-margin
   and 25 low-margin correct trials, 3 % of residual norm, mid-depth layers)
   on an open model. Measure whether steering moves gate near-zero ratio,
   kurtosis and NormShift. If abstention and confidence move but the
   structural statistics do not, the two are dissociable.
2. Structural intervention: multiply the residual stream by alpha in
   {0.5, 0.8, 1.25, 2.0} before layer l. RMSNorm cancels the scale for the
   sublayer read, so any behavioural change is due to the skip-to-sublayer
   ratio. Measure abstention, calibrated confidence and correctness. This is a
   lever they do not have.
3. Mediation as in their Fig. 4 with X_struct as a third mediator.

### 4.4 PopQA as the external-support chain

We already have mistral-7b n = 300 results (`data/results/popqa/`): pre-norm CV
at layer 19 tracks popularity at rho +0.50 and predicts correctness at AUROC
0.80, and MSP + familiarity beats MSP + 5-sample agreement on HLCC. Extend
with the abstain option so the chain popularity -> structural familiarity ->
abstention can be tested. Kumaran have no external support variable; their
difficulty score is the model's own multi-seed accuracy, which is
circular for this purpose.

### 4.5 Normative versus observed policy

Pure analysis on existing data. Their Phase 4 gives the observed policy
parameters. HLCC gives the optimal threshold as a function of p. Plot the two
and report where the observed policy is over- or under-conservative relative
to HLCC. This slots into the HLCC paper's Discussion and needs no new runs.

### 4.6 Code (written 10 September 2026)

- `data/kumaran2026/`: their OSF release (`phase1_questions.csv` = the 1,000
  main-experiment questions, `phase0_questions.csv` calibration set, GPT-4o
  difficulty, RAG scores, embedding PCs, and the notebook). CC-BY-4.0.
- `readout_comparison.py`: Phases 1, 2, 4 and verbal confidence with their
  prompts and seed-1042 arrangement, plus structural statistics and the
  residual at every layer at the answer-colon token.
- `analyse_readout_comparison.py`: the report (abstention and correctness
  models, the dissociation, ridge probes with and without structural
  features, reverse decodability, HLCC scoring, Phase 4 policy fit).
- `steer_abstention.py`: their difference-of-means steering plus the
  residual-rescale intervention, with structural statistics recorded per
  condition.
- `run_readout_sweep.sh`: all locally cached models. `jobs/idun_readout_large.sh`
  and `docs/idun-readout-plan.md`: Gemma 3 27B, Gemma 2 27B, Qwen2.5-32B/72B,
  Llama 3.1 70B at FP16 on Idun.
- `model_loader.py`: registered `gemma3-12b`, `gemma3-27b`, `qwen2.5-72b`.
  Gemma is gated; the licence must be accepted on Hugging Face first.
- `popqa_familiarity_probe.py`: `LayerHooks` now reads the gate activation
  from the model config, so GeGLU (Gemma) works.

Original list of changes needed, kept for the record:

- `model_loader.py`: register `gemma3-27b` (google/gemma-3-27b-it), 4-bit.
- `popqa_familiarity_probe.py:152` hard-codes SiLU. Gemma uses GeGLU
  (`gelu_pytorch_tanh`); read the activation function from `config.hidden_act`.
- `hlcc_mcq_system.py`: `--abstain-option`, `--verbal-confidence`.
- New `steer.py`: forward hook that adds alpha * v to the residual at layer l,
  or rescales it; reuse the hook plumbing in `popqa_familiarity_probe.py`.
- New `readout_comparison.py`: the ridge dR2 analysis of Section 4.1.

## 5. Text changes to the papers

### Activation paper (`D:/git/activation-paper/paper.tex`)

- Introduction: cite Kumaran as independent confirmation that observable
  confidence is a lossy read-out, then state the distinction of Section 3.1 in
  one paragraph.
- Related Work, "Truth and Correctness in Hidden States": add Kumaran (this
  paper), Kumaran et al. 2026 "Reported confidence tracks commitment more than
  correctness" (arXiv 2606.29490), Kumaran et al. 2026 "How do LLMs compute
  verbal confidence" (ICML 2026), and Lindsey 2025 on introspection.
- Related Work, "Abstention and Selective Answering": add Kumaran, Wen et al.
  2025 (Know Your Limits survey, TACL), Madhusudhan et al. 2025, Arditi et al.
  2024 (refusal is a single direction; contrast with our claim that
  answerability is not a direction), Rimsky et al. 2024 and Turner et al. 2023
  for steering.
- Discussion: the X_struct extension of their Fig. 6.
- Fix: both AbstentionBench entries in `references.bib` list "Feng" as first
  author. The paper is Kirichenko, Ibrahim, Chaudhuri and Bell (Meta FAIR,
  arXiv 2506.09038, NeurIPS 2026). Corrected in
  `shared-references/activation-references.bib`; re-run `sync-bib.sh`.

### HLCC paper (`docs/hlcc_paper.tex`)

- Discussion, "Conservative Bias as a Feature": their Phase 4 gives empirical
  policy parameters to compare against c*.
- Related Work, "Selective Prediction": cite Kumaran for the two-stage
  confidence-decision framework and Yadkori et al. 2024, Tomani et al. 2024 for
  conformal abstention baselines.
- Related Work, "Uncertainty Estimation in LLMs": Steyvers and Peters 2026
  (Curr. Dir. Psych. Sci.) and Geng et al. 2023 survey.

## 6. Citation alignment: Kumaran's 50 references against ours

Checked against `activation-paper/references.bib`,
`NNConfidence/docs/references.bib`, `shared-references/activation-references.bib`
and `shared-references/cbm-references.bib`.

**Already cited by us (5):** Xiong et al. 2023 [6], Tian et al. 2023 [7],
Steyvers et al. 2025 NMI [8], Guo et al. 2017 [25], Kadavath et al. 2022 [45].

**Not cited by us but directly relevant, grouped by where they land:**

| Ref | Paper | Lands in |
|---|---|---|
| 1, 2, 9, 10, 11, 12, 23, 24 | Pouget 2016; Kepecs and Mainen 2012; Fleming and Daw 2017; Kepecs 2008; Foote and Crystal 2007; Kiani and Shadlen 2009; Gold and Shadlen 2007; Yeung and Summerfield 2012 | Neuroscience of confidence. The energy-profile framing in `paper.tex` Section 1.1 currently reaches back only to Hopfield work; Fleming and Daw's first-order/second-order distinction is the right frame for X_gen/X_eval/X_struct, and the CBM papers already cite Fleming and Dolan. |
| 3, 36 | Steyvers and Peters 2026; Steyvers, Belem, Smyth 2025 | Metacognition and uncertainty communication. HLCC Related Work. |
| 4, 39 | Stone et al. 2022 (changes of mind); Kumaran 2026 (confidence tracks commitment) | Multi-pass evaluator: our rescue/damage analysis is a change-of-mind study. |
| 13, 14, 15 | Wen et al. 2025 survey; Kirichenko et al. AbstentionBench; Madhusudhan et al. 2025 | Abstention. We already use AbstentionBench data; the survey classifies us as in-processing inference-stage. |
| 16 | Arditi et al. 2024, refusal mediated by a single direction | Contrast case: refusal is a direction; we argue answerability is not. |
| 17, 18, 19, 21, 22 | Chuang 2024 (Self-REF); Tjandra 2024; Zhang 2024 (R-tuning); Yadkori 2024; Tomani 2024 | Training-time and conformal abstention baselines. HLCC Related Work, Selective Prediction. |
| 20, 44 | Plaut et al. 2024; Tao et al. 2025 | Calibration of chat LLMs on MCQ. HLCC benchmark section. |
| 26, 27, 28 | Lewis 2020 (RAG); Bommasani 2021; Reimers and Gurevych 2019 | Their control predictors. Only needed if we replicate their Eq. 4. |
| 29, 30, 40, 43, 48 | Turner 2023; Stolfo 2025; Rimsky 2024 (CAA); Venhoff 2025; Hua 2025 | Activation steering. Needed for Section 4.3. |
| 31 | VanderWeele 2015 | Mediation analysis. Needed for Section 4.3. |
| 33 | Sclar et al. 2024, prompt-format sensitivity | Supports our chat-template finding in FamGate Section 5.5. |
| 34, 35 | Geng et al. 2023 survey; Yoon et al. 2026 (reasoning models express confidence better) | Uncertainty surveys. |
| 37, 41 | Kumaran 2026 ICML (verbal confidence computation); Kumaran 2026 (error detection from internal signals) | Closest competitors. Read both before submitting. |
| 38 | Niculescu-Mizil and Caruana 2005 | Isotonic calibration; we use it in the MSP steelman without citing it. |
| 42 | Gandhi et al. 2025 (four habits of STaRs) | Process-level metacognition; future work only. |
| 46 | Lindsey 2025, emergent introspection | The caveat in Section 3.1. |
| 47 | Wei et al. 2024, SimpleQA | Their dataset; cite if we use their MCQ conversion. |
| 50 | Webb et al. 2023, natural statistics of confidence biases | Supports the conservative-bias argument in HLCC. |

**Cited by us and not by them, which is our differentiation:** Robins and
McCallum 2004, McAlister et al. 2025, Ramsauer et al. 2021 (Hopfield-attention
link), Africa et al. 2025, Simhi et al. 2024 (HK-/HK+), Kossen et al. 2024
(semantic entropy probes), Kuhn et al. 2023, Fadeeva et al. 2025
(LM-Polygraph), Gardner-Medwin CBM, Kelly 1956. They have no scoring-rule or
associative-memory lineage at all.

BibTeX for the new entries is in `shared-references/activation-references.bib`
under the heading "Kumaran 2026 and related".

## 7. Pilot results (10 September 2026, local sweep in progress)

Full reports: `data/results/readout/<model>_phase1_n1000_report.md`.
Kumaran's own 1,000 questions, prompts and option arrangement; chat template on.

| | qwen2-1.5b | mistral-7b | Kumaran GPT-4o |
|---|---|---|---|
| Phase 1 accuracy | 0.344 | 0.478 | 0.637 |
| Phase 2 abstention | 0.829 | 0.156 | 0.566 |
| calibrated conf AUROC(correct) | 0.53 | 0.56 | 0.90 |
| fitted temperature | 3.8 | 9.9 | 4.1 |
| verbal conf AUROC(correct) | 0.50 | 0.51 | lower than calibrated |
| abstain ~ conf, AUROC | 0.67 | 0.71 | beta_std 0.99 |
| abstain ~ structural stats, AUROC | 0.86 | 0.89 | not measured |
| abstain ~ residual ridge (best layer), AUROC | 0.91 | 0.93 | not measured |
| correct ~ anything, best AUROC | 0.62 | 0.59 | 0.90 |

Three things stand out, one of them against the framing in Section 3.

**The "not linearly decodable" claim is false for these statistics.** Ridge
from the residual at layer l to each structural statistic at layer l reaches
out-of-fold R2 of 0.9 to 1.0 for pre-norm L2/std/CV and 0.7 to 0.95 for the
gate statistics on both models, and the *final*-layer residual still recovers
late-layer statistics at R2 above 0.9. The reason is presumably that residual
norm is dominated by a few massive-activation coordinates, so the norm is close
to linear in the residual. So a linear probe, and by extension a dictionary,
can read these statistics; Section 3.1's argument holds for the maths of a
fourth moment but not for the empirical residual geometry. The paper text must
say so. What survives: the statistics are a training-free, 7-scalars-per-layer
summary that carries most of what a 4096-dimensional supervised probe finds for
abstention (0.89 vs 0.93), transfers across models without fitting, and needs
no labelled data.

**The model's abstention decision is driven by internal state that its output
confidence does not carry.** On mistral-7b, calibrated logprob confidence
predicts abstention at AUROC 0.71; the structural statistics reach 0.89 and a
ridge on the residual 0.93, with surface features (question length, Kumaran's
embedding PCs) contributing little (struct beyond conf + difficulty + surface:
+0.14). The single strongest univariate predictor is the late-layer pre-norm
residual norm (layers 28 to 30, AUROC 0.81): smaller late residual, more
abstention. That is the NormShift quantity. This is Kumaran's "richer internal
representation" made concrete, and it is exactly what activation steering at
mid-to-late layers would move.

**Correctness is unpredictable for open 7B models on SimpleQA-MCQ.** Nothing
reaches AUROC 0.6 for correctness on mistral-7b: not calibrated confidence
(0.56), not GPT-4o difficulty (0.59), not the residual (0.55), not the
structural statistics (0.52). The foils were generated to be plausible and the
facts are obscure; a 7B model is choosing between two plausible options it
cannot tell apart, and MSP averages 0.95 while accuracy is 0.48. So the
correctness half of the dissociation cannot be tested on this dataset at this
scale. Two consequences: the abstention decision on mistral-7b is being driven
by a signal that does not track correctness on this dataset, which is itself a
finding about what the policy consumes; and the correctness test has to be run
where the model has knowledge. An ARC-Challenge sweep on the 7B-8B models is
queued for that, and the Idun runs at 27B-70B should recover Kumaran's regime
(their GPT-4o calibrated-confidence AUROC of 0.90 is a frontier-model result).

Smaller notes: verbal confidence collapses to "Likely" on 96 % of mistral-7b
trials and is uncorrelated with calibrated confidence (r = 0.07, Kumaran
0.3 to 0.4); Phase 4 thresholds barely move mistral-7b (abstention 0.15 at
T = 20 to 0.27 at T = 80, fitted scale 5.2 vs 1.80); and the Kumaran
difficulty score predicts the 7B model's correctness at only 0.59, so
GPT-4o's difficulty is a weak proxy for a 7B model's knowledge.
