# Citation Suggestions

Sentences below have at least one bib entry whose abstract/tldr scores **>= 0.55** semantic similarity, but the top match isn't currently cited in the sentence.

**Total flagged sentences:** 23

## hlcc_paper.tex

### Sentence 1 (score 0.563)

> Unlike proper scoring
rules such as the Brier score or log-loss, where the optimal confidence
equals the true accuracy ( ), \hlcc{} induces a conservative
mapping   that naturally suppresses overconfidence
below 80\% accuracy.

**No existing citation in this sentence.**

**Top suggestions:**
- `nixon2019measuring` — 0.563 — Measuring Calibration in Deep Learning
- `mukhoti2020calibrating` — 0.404 — Calibrating Deep Neural Networks Using Focal Loss
- `guo2017calibration` — 0.399 — On Calibration of Modern Neural Networks

### Sentence 2 (score 0.582)

> We implement \hlcc{} as a trainable objective for
language-model confidence heads, proposing a dual-path
\emph{NormShift} architecture that routes per-layer normalisation
dynamics alongside final hidden states to estimate per-token confidence.

**No existing citation in this sentence.**

**Top suggestions:**
- `mccallum2026hlcc` — 0.582 — Hybrid Linear-Convex Confidence Loss for Calibrated Language Models
- `vashurin2025benchmarking` — 0.508 — Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph
- `xiong2024llms` — 0.476 — Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs

### Sentence 3 (score 0.556)

> We benchmark against 11 baseline confidence methods---including maximum
softmax probability, entropy, self-consistency, semantic entropy,
chain-of-thought verbalisation, temperature scaling, isotonic
regression, and Platt scaling---across 9 datasets and 7 model scales
from 1.1B to 8B parameters.

**No existing citation in this sentence.**

**Top suggestions:**
- `xiong2024llms` — 0.556 — Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs
- `guo2017calibration` — 0.552 — On Calibration of Modern Neural Networks
- `kuhn2023semantic` — 0.516 — Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation

### Sentence 4 (score 0.709)

> Results are evaluated on discrimination
(AUROC, AUPRC), calibration (ECE, Brier), and selective prediction
(PRR, accuracy at coverage thresholds), providing a comprehensive
assessment of confidence estimation in modern language models.

**No existing citation in this sentence.**

**Top suggestions:**
- `mccallum2026hlcc` — 0.709 — Hybrid Linear-Convex Confidence Loss for Calibrated Language Models
- `vashurin2025benchmarking` — 0.551 — Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph
- `tian2023just` — 0.540 — Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback

### Sentence 5 (score 0.673)

> \end{abstract}


\section{Introduction}
\label{sec:introduction}


Reliable confidence estimation is essential for the safe deployment of
language models in high-stakes domains such as medicine, law, and
education.

**No existing citation in this sentence.**

**Top suggestions:**
- `mccallum2026hlcc` — 0.673 — Hybrid Linear-Convex Confidence Loss for Calibrated Language Models
- `xiong2024llms` — 0.641 — Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs
- `vashurin2025benchmarking` — 0.585 — Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph

### Sentence 37 (score 0.634)

> \section{Architecture}
\label{sec:architecture}


\subsection{NormShift Confidence Head}

We propose the \emph{NormShift} confidence head, a lightweight module
trained on top of a frozen language model to predict per-token
confidence.

**No existing citation in this sentence.**

**Top suggestions:**
- `mccallum2026hlcc` — 0.634 — Hybrid Linear-Convex Confidence Loss for Calibrated Language Models
- `xiong2024llms` — 0.621 — Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs
- `lin2024generating` — 0.568 — Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models

### Sentence 66 (score 0.592)

> \item[CoT Confidence] Two-stage prompting: generate chain-of-thought
        reasoning, then extract answer and confidence from a structured
        JSON response.

**No existing citation in this sentence.**

**Top suggestions:**
- `wang2023selfconsistency` — 0.592 — Self-Consistency Improves Chain of Thought Reasoning in Language Models
- `xiong2024llms` — 0.518 — Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs
- `kadavath2022language` — 0.515 — Language Models (Mostly) Know What They Know

### Sentence 71 (score 0.584)

> \begin{itemize}[nosep]
  \item \textbf{AUROC}: Area under the ROC curve, treating
          as a binary classifier for correctness.

**No existing citation in this sentence.**

**Top suggestions:**
- `saito2015precision` — 0.584 — The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets
- `zadrozny2002transforming` — 0.509 — Transforming Classifier Scores into Accurate Multiclass Probability Estimates
- `nixon2019measuring` — 0.496 — Measuring Calibration in Deep Learning

### Sentence 129 (score 0.599)

> \end{enumerate}



\section{LM-Polygraph Integration}
\label{sec:polygraph}


To enable direct comparison with the broader uncertainty estimation
literature, we implement the NormShift confidence head as a custom
estimator within the
LM-Polygraph~\cite{fadeeva2024lmpolygraph} benchmarking framework.

**Already cites:** fadeeva2024lmpolygraph

**Top suggestions:**
- `vashurin2025benchmarking` — 0.599 — Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph
- `malinin2017predictive` — 0.495 — Predictive Uncertainty Estimation via Prior Networks
- `xiong2024llms` — 0.458 — Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs

### Sentence 130 (score 0.640)

> LM-Polygraph provides over 40 uncertainty estimation methods and
standardised evaluation protocols, making it the primary benchmark
for comparing UE methods across models and tasks.

**No existing citation in this sentence.**

**Top suggestions:**
- `vashurin2025benchmarking` — 0.640 — Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph
- `malinin2017predictive` — 0.417 — Predictive Uncertainty Estimation via Prior Networks
- `xiong2024llms` — 0.391 — Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs

### Sentence 136 (score 0.577)

> Five uncertainty estimation methods
are evaluated: four logit-based baselines available through
LM-Polygraph (entropy, MSP, top- , perplexity) and our trained
NormShift head.

**No existing citation in this sentence.**

**Top suggestions:**
- `malinin2017predictive` — 0.577 — Predictive Uncertainty Estimation via Prior Networks
- `vashurin2025benchmarking` — 0.571 — Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph
- `xiong2024llms` — 0.457 — Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs

### Sentence 160 (score 0.782)

> \section{Conclusion}
\label{sec:conclusion}


We have presented the \hlcc{} scoring function and the NormShift
confidence head as a method for training calibrated confidence estimates
in language models.

**No existing citation in this sentence.**

**Top suggestions:**
- `mccallum2026hlcc` — 0.782 — Hybrid Linear-Convex Confidence Loss for Calibrated Language Models
- `vashurin2025benchmarking` — 0.573 — Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph
- `xiong2024llms` — 0.565 — Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs

### Sentence 167 (score 0.593)

> Our
LM-Polygraph integration enables direct comparison with the broader
uncertainty estimation literature, and our benchmarking framework
provides a systematic evaluation methodology for future confidence
estimation research.

**No existing citation in this sentence.**

**Top suggestions:**
- `vashurin2025benchmarking` — 0.593 — Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph
- `xiong2024llms` — 0.473 — Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs
- `malinin2017predictive` — 0.472 — Predictive Uncertainty Estimation via Prior Networks

## small_network_paper.tex

### Sentence 0 (score 0.676)

> \documentclass[11pt,a4paper]{article}


\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{cleveref}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{multirow}
\usepackage{subcaption}
\usepackage[margin=2.5cm]{geometry}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{float}


\newtheorem{theorem}{Theorem}
\newtheorem{proposition}{Proposition}
\newtheorem{corollary}{Corollary}
\newtheorem{definition}{Definition}
\newtheorem{remark}{Remark}


\newcommand{\hlcc}{\te

**No existing citation in this sentence.**

**Top suggestions:**
- `mccallum2026hlcc` — 0.676 — Hybrid Linear-Convex Confidence Loss for Calibrated Language Models
- `vashurin2025benchmarking` — 0.445 — Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph
- `xiong2024llms` — 0.411 — Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs

### Sentence 5 (score 0.615)

> \end{abstract}


\section{Introduction}
\label{sec:intro}

Confidence calibration --- ensuring that a model's stated certainty
matches its empirical accuracy --- is a long-standing problem in
machine learning~\cite{guo2017calibration, naeini2015obtaining}.

**Already cites:** guo2017calibration, naeini2015obtaining

**Top suggestions:**
- `nixon2019measuring` — 0.615 — Measuring Calibration in Deep Learning
- `guo2017calibration` — 0.545 — On Calibration of Modern Neural Networks (already cited here)
- `kumar2019verified` — 0.503 — Verified Uncertainty Calibration

### Sentence 6 (score 0.636)

> Recent work has extended calibration to LLMs through methods such as
maximum softmax probability, verbalized confidence, semantic entropy,
and self-consistency~\cite{kadavath2022language, xiong2024llms,
kuhn2023semantic, wang2023selfconsistency}.

**Already cites:** kadavath2022language, xiong2024llms, kuhn2023semantic, wang2023selfconsistency

**Top suggestions:**
- `shen2024thermometer` — 0.636 — Thermometer: Towards Universal Calibration of Language Models
- `xiong2024llms` — 0.578 — Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs (already cited here)
- `mccallum2026hlcc` — 0.533 — Hybrid Linear-Convex Confidence Loss for Calibrated Language Models

### Sentence 26 (score 0.567)

> Here
we deliberately avoid softmax and instead derive confidence from
raw output logits:
 
where   are the output logits,  ,   is the
number of classes, and   is the sigmoid function.

**No existing citation in this sentence.**

**Top suggestions:**
- `hendrycks2017baseline` — 0.567 — A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks
- `corbiere2022confidence` — 0.487 — Confidence Estimation via Auxiliary Models
- `platt1999probabilistic` — 0.481 — Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods

### Sentence 46 (score 0.565)

> \subsection{Calibration Baselines}
\label{sec:baselines}

We compare the \hlcc{} confidence head against four standard
calibration methods:

\begin{enumerate}
  \item \textbf{Temperature scaling}~\cite{guo2017calibration}: A
        single scalar   is fit on the validation set to minimise
        NLL of  .

**Already cites:** guo2017calibration

**Top suggestions:**
- `mccallum2026hlcc` — 0.565 — Hybrid Linear-Convex Confidence Loss for Calibrated Language Models
- `nixon2019measuring` — 0.518 — Measuring Calibration in Deep Learning
- `guo2017calibration` — 0.507 — On Calibration of Modern Neural Networks (already cited here)

### Sentence 65 (score 0.575)

> \subsection{Metrics}

We evaluate on the following metrics:
\begin{itemize}
  \item \textbf{ECE} (Expected Calibration Error)~\cite{naeini2015obtaining}:
        average gap between confidence and accuracy across 10 bins.

**Already cites:** naeini2015obtaining

**Top suggestions:**
- `nixon2019measuring` — 0.575 — Measuring Calibration in Deep Learning
- `kumar2019verified` — 0.507 — Verified Uncertainty Calibration
- `zhu2024calibration` — 0.429 — A Benchmark Study on Calibration

### Sentence 71 (score 0.601)

> \item \textbf{AUROC}: area under the ROC curve treating confidence as a
        discriminator for correct/incorrect predictions.

**No existing citation in this sentence.**

**Top suggestions:**
- `saito2015precision` — 0.601 — The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets
- `nixon2019measuring` — 0.535 — Measuring Calibration in Deep Learning
- `zadrozny2002transforming` — 0.458 — Transforming Classifier Scores into Accurate Multiclass Probability Estimates

### Sentence 147 (score 0.558)

> \paragraph{Deep ensemble comparison.}
Deep ensembles represent the gold standard for small-network
uncertainty.

**No existing citation in this sentence.**

**Top suggestions:**
- `liu2020simple` — 0.558 — Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness
- `gal2016dropout` — 0.513 — Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
- `mukhoti2020calibrating` — 0.494 — Calibrating Deep Neural Networks Using Focal Loss

### Sentence 162 (score 0.625)

> Focal loss~\cite{lin2017focal} provides an alternative asymmetric
training signal that down-weights easy examples, improving implicit
calibration.

**Already cites:** lin2017focal

**Top suggestions:**
- `mukhoti2020calibrating` — 0.625 — Calibrating Deep Neural Networks Using Focal Loss
- `hebbalaguppe2022stitch` — 0.514 — A Stitch in Time Saves Nine: A Train-Time Regularizing Loss for Improved Neural Network Calibration
- `lin2017focal` — 0.500 — Focal Loss for Dense Object Detection (already cited here)

### Sentence 175 (score 0.554)

> Across 13 training
configurations, 6 datasets, 4 calibration baselines, and 5 random
seeds on a 4-layer feedforward network, we find:

\begin{enumerate}
  \item \textbf{\hlcc{} loss enables calibration across all datasets.}
        Trainable confidence heads trained with CE alone show no
        calibration improvement (ECE 0.42--0.48), while
        \hlcc{}-trained variants achieve ECE as low as 0.020
        (two\_moons), 0.023 (breast\_cancer), 0.034 (rings), and
        0.068 (blobs).

**No existing citation in this sentence.**

**Top suggestions:**
- `guo2017calibration` — 0.554 — On Calibration of Modern Neural Networks
- `zhu2024calibration` — 0.541 — A Benchmark Study on Calibration
- `mukhoti2020calibrating` — 0.538 — Calibrating Deep Neural Networks Using Focal Loss
