# Citation Suggestions

Sentences below have at least one bib entry whose abstract/tldr scores **>= 0.55** semantic similarity, but the top match isn't currently cited in the sentence.

**Total flagged sentences:** 33

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
- `corbiere2019addressing` — 0.388 — Addressing Failure Prediction by Learning Model Confidence
- `gal2016dropout` — 0.387 — Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning

### Sentence 2 (score 0.582)

> We implement \hlcc{} as a trainable objective for
language-model confidence heads, proposing a dual-path
\emph{NormShift} architecture that routes per-layer normalisation
dynamics alongside final hidden states to estimate per-token confidence.

**No existing citation in this sentence.**

**Top suggestions:**
- `mccallum2026hlcc` — 0.582 — Hybrid Linear-Convex Confidence Loss for Calibrated Language Models
- `vashurin2025benchmarking` — 0.577 — Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph
- `corbiere2022confidence` — 0.478 — Confidence Estimation via Auxiliary Models

### Sentence 3 (score 0.575)

> We benchmark against 11 baseline confidence methods---including maximum
softmax probability, entropy, self-consistency, semantic entropy,
chain-of-thought verbalisation, temperature scaling, isotonic
regression, and Platt scaling---across 9 datasets and 7 model scales
from 1.1B to 8B parameters.

**No existing citation in this sentence.**

**Top suggestions:**
- `vashurin2025benchmarking` — 0.575 — Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph
- `xiong2024llms` — 0.556 — Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs
- `guo2017calibration` — 0.528 — On Calibration of Modern Neural Networks

### Sentence 4 (score 0.709)

> Results are evaluated on discrimination
(AUROC, AUPRC), calibration (ECE, Brier), and selective prediction
(PRR, accuracy at coverage thresholds), providing a comprehensive
assessment of confidence estimation in modern language models.

**No existing citation in this sentence.**

**Top suggestions:**
- `mccallum2026hlcc` — 0.709 — Hybrid Linear-Convex Confidence Loss for Calibrated Language Models
- `shen2024thermometer` — 0.586 — Thermometer: Towards Universal Calibration of Language Models
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
- `corbiere2022confidence` — 0.560 — Confidence Estimation via Auxiliary Models

### Sentence 15 (score 0.551)

> \paragraph{Contributions.}
\begin{enumerate}[nosep]
  \item We derive the theoretical properties of \hlcc{}, showing that
        the optimal confidence   is strictly below
        the true accuracy   for  , providing a built-in
        conservative bias (\Cref{sec:theory}).

**No existing citation in this sentence.**

**Top suggestions:**
- `corbiere2022confidence` — 0.551 — Confidence Estimation via Auxiliary Models
- `mccallum2026hlcc` — 0.440 — Hybrid Linear-Convex Confidence Loss for Calibrated Language Models
- `kumar2019verified` — 0.420 — Verified Uncertainty Calibration

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
- `shen2024thermometer` — 0.533 — Thermometer: Towards Universal Calibration of Language Models

### Sentence 38 (score 0.565)

> The architecture exploits an observation about transformer
normalisation layers: the magnitude of the correction applied by
LayerNorm encodes information about the model's internal uncertainty.

**No existing citation in this sentence.**

**Top suggestions:**
- `ba2016layer` — 0.565 — Layer Normalization
- `lakshminarayanan2017simple` — 0.382 — Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles
- `liu2020simple` — 0.381 — Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness

### Sentence 46 (score 0.591)

> \subsection{Uncertainty-Aware LayerNorm}

To capture the doubt signals, we extend standard LayerNorm to record
pre-normalisation statistics:

 

These three signals per layer---pre-norm magnitude, correction scale,
and variance---form the input to the norm-shift path.

**No existing citation in this sentence.**

**Top suggestions:**
- `ba2016layer` — 0.591 — Layer Normalization
- `ioffe2015batch` — 0.438 — Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
- `vashurin2025benchmarking` — 0.435 — Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph

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

### Sentence 108 (score 0.601)

> \subsection{Theoretical Analysis}

\Cref{fig:theory} shows the theoretical optimal confidence curves
derived in \Cref{sec:theory}.

**No existing citation in this sentence.**

**Top suggestions:**
- `corbiere2022confidence` — 0.601 — Confidence Estimation via Auxiliary Models
- `mccallum2026hlcc` — 0.415 — Hybrid Linear-Convex Confidence Loss for Calibrated Language Models
- `kumar2019verified` — 0.378 — Verified Uncertainty Calibration

### Sentence 114 (score 0.551)

> On Llama-3.1-8B, NormShift achieves the best
calibration of any method (ECE\,=\,0.105 vs.\ 0.322 for
self-consistency) and competitive discrimination (AUROC\,=\,0.870
vs.\ 0.858 for entropy), using only a single forward pass.

**No existing citation in this sentence.**

**Top suggestions:**
- `zhu2024calibration` — 0.551 — A Benchmark Study on Calibration
- `kumar2019verified` — 0.499 — Verified Uncertainty Calibration
- `hebbalaguppe2022stitch` — 0.433 — A Stitch in Time Saves Nine: A Train-Time Regularizing Loss for Improved Neural Network Calibration

### Sentence 136 (score 0.577)

> Five uncertainty estimation methods
are evaluated: four logit-based baselines available through
LM-Polygraph (entropy, MSP, top- , perplexity) and our trained
NormShift head.

**No existing citation in this sentence.**

**Top suggestions:**
- `malinin2017predictive` — 0.577 — Predictive Uncertainty Estimation via Prior Networks
- `liu2020simple` — 0.577 — Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness
- `lakshminarayanan2017simple` — 0.548 — Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles

### Sentence 138 (score 0.554)

> The logit-based methods perform
substantially better, with top-  achieving the best discrimination
(AUROC~0.634, PRR~0.177) and entropy achieving the best calibration
(ECE~0.145).

**No existing citation in this sentence.**

**Top suggestions:**
- `zhu2024calibration` — 0.554 — A Benchmark Study on Calibration
- `nixon2019measuring` — 0.477 — Measuring Calibration in Deep Learning
- `naeini2015obtaining` — 0.432 — Obtaining Well Calibrated Probabilities Using Bayesian Binning into Quantiles

### Sentence 150 (score 0.558)

> \citet{nixon2019measuring} showed that ECE is sensitive to
binning choices and proposed adaptive calibration error.

**Already cites:** nixon2019measuring

**Top suggestions:**
- `kumar2019verified` — 0.558 — Verified Uncertainty Calibration
- `zhu2024calibration` — 0.510 — A Benchmark Study on Calibration
- `naeini2015obtaining` — 0.446 — Obtaining Well Calibrated Probabilities Using Bayesian Binning into Quantiles

### Sentence 160 (score 0.782)

> \section{Conclusion}
\label{sec:conclusion}


We have presented the \hlcc{} scoring function and the NormShift
confidence head as a method for training calibrated confidence estimates
in language models.

**No existing citation in this sentence.**

**Top suggestions:**
- `mccallum2026hlcc` — 0.782 — Hybrid Linear-Convex Confidence Loss for Calibrated Language Models
- `shen2024thermometer` — 0.581 — Thermometer: Towards Universal Calibration of Language Models
- `xiong2024llms` — 0.565 — Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs

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
- `corbiere2022confidence` — 0.488 — Confidence Estimation via Auxiliary Models
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
- `kumar2019verified` — 0.576 — Verified Uncertainty Calibration
- `corbiere2019addressing` — 0.555 — Addressing Failure Prediction by Learning Model Confidence

### Sentence 16 (score 0.654)

> \item Does layer normalisation provide useful doubt signals at
        small scale?

**No existing citation in this sentence.**

**Top suggestions:**
- `ba2016layer` — 0.654 — Layer Normalization
- `mccallum2026smallnet` — 0.426 — HLCC Loss on Small Networks: Analysing Normalisation-Aware Confidence in a Controlled Setting
- `hebbalaguppe2022stitch` — 0.404 — A Stitch in Time Saves Nine: A Train-Time Regularizing Loss for Improved Neural Network Calibration

### Sentence 23 (score 0.589)

> \item \textbf{Correction scale}:  , measuring how
        much the normalisation rescaled the activations.

**No existing citation in this sentence.**

**Top suggestions:**
- `ba2016layer` — 0.589 — Layer Normalization
- `ioffe2015batch` — 0.510 — Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
- `hebbalaguppe2022stitch` — 0.408 — A Stitch in Time Saves Nine: A Train-Time Regularizing Loss for Improved Neural Network Calibration

### Sentence 24 (score 0.572)

> \item \textbf{Pre-normalisation L2 norm}: the overall magnitude of
        activations before normalisation.

**No existing citation in this sentence.**

**Top suggestions:**
- `ba2016layer` — 0.572 — Layer Normalization
- `ioffe2015batch` — 0.506 — Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
- `mccallum2026smallnet` — 0.349 — HLCC Loss on Small Networks: Analysing Normalisation-Aware Confidence in a Controlled Setting

### Sentence 25 (score 0.586)

> \end{enumerate}

\subsection{Confidence Without Softmax}

Standard calibration methods derive confidence from probability
distributions (softmax outputs, logit-based entropy, etc.).

**No existing citation in this sentence.**

**Top suggestions:**
- `kumar2019verified` — 0.586 — Verified Uncertainty Calibration
- `zhu2024calibration` — 0.534 — A Benchmark Study on Calibration
- `nixon2019measuring` — 0.525 — Measuring Calibration in Deep Learning

### Sentence 26 (score 0.566)

> Here
we deliberately avoid softmax and instead derive confidence from
raw output logits:
 
where   are the output logits,  ,   is the
number of classes, and   is the sigmoid function.

**No existing citation in this sentence.**

**Top suggestions:**
- `hendrycks2017baseline` — 0.566 — A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks
- `platt1999probabilistic` — 0.481 — Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods
- `corbiere2022confidence` — 0.461 — Confidence Estimation via Auxiliary Models

### Sentence 46 (score 0.589)

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
- `zhu2024calibration` — 0.589 — A Benchmark Study on Calibration
- `mccallum2026hlcc` — 0.565 — Hybrid Linear-Convex Confidence Loss for Calibrated Language Models
- `kumar2019verified` — 0.554 — Verified Uncertainty Calibration

### Sentence 56 (score 0.591)

> This addresses a common weakness in
calibration papers where single-run results may reflect
initialisation luck rather than method quality.

**No existing citation in this sentence.**

**Top suggestions:**
- `zhu2024calibration` — 0.591 — A Benchmark Study on Calibration
- `kumar2019verified` — 0.490 — Verified Uncertainty Calibration
- `nixon2019measuring` — 0.439 — Measuring Calibration in Deep Learning

### Sentence 65 (score 0.626)

> \subsection{Metrics}

We evaluate on the following metrics:
\begin{itemize}
  \item \textbf{ECE} (Expected Calibration Error)~\cite{naeini2015obtaining}:
        average gap between confidence and accuracy across 10 bins.

**Already cites:** naeini2015obtaining

**Top suggestions:**
- `kumar2019verified` — 0.626 — Verified Uncertainty Calibration
- `nixon2019measuring` — 0.575 — Measuring Calibration in Deep Learning
- `zhu2024calibration` — 0.565 — A Benchmark Study on Calibration

### Sentence 71 (score 0.601)

> \item \textbf{AUROC}: area under the ROC curve treating confidence as a
        discriminator for correct/incorrect predictions.

**No existing citation in this sentence.**

**Top suggestions:**
- `saito2015precision` — 0.601 — The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets
- `nixon2019measuring` — 0.535 — Measuring Calibration in Deep Learning
- `zadrozny2002transforming` — 0.458 — Transforming Classifier Scores into Accurate Multiclass Probability Estimates

### Sentence 79 (score 0.681)

> This confirms that calibration
requires a \emph{trainable} confidence signal.

**No existing citation in this sentence.**

**Top suggestions:**
- `kumar2019verified` — 0.681 — Verified Uncertainty Calibration
- `zhu2024calibration` — 0.570 — A Benchmark Study on Calibration
- `nixon2019measuring` — 0.464 — Measuring Calibration in Deep Learning

### Sentence 146 (score 0.580)

> At LLM scale, where 28--32 layers provide richer normalisation dynamics,
this limitation disappears.

**No existing citation in this sentence.**

**Top suggestions:**
- `ba2016layer` — 0.580 — Layer Normalization
- `ioffe2015batch` — 0.412 — Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
- `mukhoti2020calibrating` — 0.294 — Calibrating Deep Neural Networks Using Focal Loss

### Sentence 147 (score 0.713)

> \paragraph{Deep ensemble comparison.}
Deep ensembles represent the gold standard for small-network
uncertainty.

**No existing citation in this sentence.**

**Top suggestions:**
- `lakshminarayanan2017simple` — 0.713 — Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles
- `gal2016dropout` — 0.611 — Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
- `liu2020simple` — 0.576 — Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness

### Sentence 162 (score 0.693)

> Focal loss~\cite{lin2017focal} provides an alternative asymmetric
training signal that down-weights easy examples, improving implicit
calibration.

**Already cites:** lin2017focal

**Top suggestions:**
- `mukhoti2020calibrating` — 0.693 — Calibrating Deep Neural Networks Using Focal Loss
- `hebbalaguppe2022stitch` — 0.559 — A Stitch in Time Saves Nine: A Train-Time Regularizing Loss for Improved Neural Network Calibration
- `lin2017focal` — 0.558 — Focal Loss for Dense Object Detection (already cited here)

### Sentence 175 (score 0.585)

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
- `hebbalaguppe2022stitch` — 0.585 — A Stitch in Time Saves Nine: A Train-Time Regularizing Loss for Improved Neural Network Calibration
- `mukhoti2020calibrating` — 0.550 — Calibrating Deep Neural Networks Using Focal Loss
- `guo2017calibration` — 0.506 — On Calibration of Modern Neural Networks
