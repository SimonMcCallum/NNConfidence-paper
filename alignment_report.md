# Citation Alignment Report

- **Paper:** G:/git/NNConfidence/docs
- **Bibliography:** G:\git\NNConfidence\docs
- **Entries checked:** 43

## Summary

| Key | Keyword | TF-IDF | Embed | NLI Δ | Status | Note |
|---|---|---|---|---|---|---|
| guo2017calibration | 0.058 | — | 0.579 | +0.00 | OK |  |
| hendrycks2017baseline | 0.033 | — | 0.560 | +0.00 | OK |  |
| gardnermedwin1995 | 0.018 | — | 0.434 | -0.00 | OK |  |
| mccallum2010gamedesign | 0.074 | — | 0.543 | -0.00 | OK |  |
| budescu2012optimal | 0.026 | — | 0.280 | +0.00 | CHECK | borderline similarity (embedding=0.280, 0.2–0.3) |
| dettmers2023qlora | 0.032 | — | 0.552 | +0.00 | OK |  |
| lin2022truthfulqa | 0.013 | — | 0.249 | -0.00 | CHECK | borderline similarity (embedding=0.249, 0.2–0.3) |
| clark2018arc | 0.016 | — | 0.476 | -0.00 | OK |  |
| hendrycks2021mmlu | 0.017 | — | 0.350 | -0.00 | OK |  |
| zellers2019hellaswag | 0.015 | — | 0.195 | -0.00 | FLAG | low similarity (embedding=0.195 < 0.2) |
| wang2024mmlu | 0.021 | — | 0.528 | -0.00 | OK |  |
| cobbe2021gsm8k | 0.010 | — | 0.366 | +0.00 | OK |  |
| clark2019boolq | 0.018 | — | 0.278 | +0.00 | CHECK | borderline similarity (embedding=0.278, 0.2–0.3) |
| joshi2017triviaqa | 0.016 | — | 0.295 | +0.01 | CHECK | borderline similarity (embedding=0.295, 0.2–0.3) |
| zadrozny2002transforming | 0.031 | — | 0.477 | +0.00 | OK |  |
| platt1999probabilistic | 0.039 | — | 0.398 | +0.00 | OK |  |
| wang2023selfconsistency | 0.014 | — | 0.413 | -0.00 | OK |  |
| kuhn2023semantic | 0.040 | — | 0.599 | +0.00 | OK |  |
| saito2015precision | 0.023 | — | 0.482 | -0.00 | OK |  |
| naeini2015obtaining | 0.022 | — | 0.522 | +0.00 | OK |  |
| malinin2017predictive | 0.030 | — | 0.494 | -0.00 | OK |  |
| fadeeva2024lmpolygraph | 0.008 | — | 0.227 | +0.01 | CHECK | borderline similarity (embedding=0.227, 0.2–0.3) |
| nixon2019measuring | 0.027 | — | 0.487 | +0.00 | OK |  |
| kadavath2022language | 0.030 | — | 0.444 | -0.00 | OK |  |
| xiong2024llms | 0.060 | — | 0.526 | -0.00 | OK |  |
| lin2024generating | 0.021 | — | 0.620 | +0.00 | OK |  |
| tian2023just | 0.019 | — | 0.341 | +0.03 | OK |  |
| geifman2017selective | 0.031 | — | 0.382 | -0.00 | OK |  |
| mccallum2026hlcc | 0.011 | — | 0.416 | — | OK | title only (no abstract/tldr) |
| ba2016layer | 0.033 | — | 0.573 | +0.70 | CHECK | NLI polarity 0.70 (label=contradiction) |
| liu2020simple | 0.021 | — | 0.355 | +0.00 | OK |  |
| dua2019uci | 0.043 | — | 0.499 | — | OK | title only (no abstract/tldr) |
| lin2017focal | 0.058 | — | 0.607 | +0.00 | OK |  |
| gal2016dropout | 0.057 | — | 0.537 | -0.00 | OK |  |
| lakshminarayanan2017simple | 0.072 | — | 0.478 | +0.00 | OK |  |
| kumar2019verified | 0.025 | — | 0.674 | +0.00 | OK |  |
| corbiere2019addressing | 0.060 | — | 0.576 | -0.00 | OK |  |
| hebbalaguppe2022stitch | 0.038 | — | 0.564 | +0.00 | OK |  |
| mukhoti2020calibrating | 0.031 | — | 0.708 | -0.00 | OK |  |
| zhu2024calibration | 0.023 | — | 0.681 | -0.00 | OK |  |
| corbiere2022confidence | 0.047 | — | 0.619 | -0.00 | OK |  |
| shen2024thermometer | 0.009 | — | 0.381 | -0.00 | OK |  |
| ioffe2015batch | 0.021 | — | 0.529 | -0.00 | OK |  |

## Flagged Citations

### budescu2012optimal
**Title:** Confidence in Aggregation of Opinions from Multiple Sources
**Score:** 0.280
**Best-matching sentence in cited paper:** _In particular I seek to understand and characterize (a) the nature of the aggregation rules used by DMs and (b) the factors that affect the DMs' confidence in the final aggregate._
**NLI:** label=neutral, polarity=+0.000 (>0 means cited paper contradicts the claim)
**Premise (cited paper):** _This chapter summarizes research related to several interrelated questions regarding the process by which single decision makers (DMs) aggregate probabilistic information regarding a certain event from several, possibly asymmetric, advisors who rely _

**Citation context(s):**
> ensures that participating is always dominant over abstention, resolving the omission bias observed in negative-marking schemes where risk-averse test-takers skip questions they partially know budescu2012optimal . Architecture sec:architecture NormShift Confidence Head We propose the NormShift confi
> applied game design principles to computer science education, using confidence-based multiple-choice assessment where students indicate certainty alongside answers with asymmetric reward structures. budescu2012optimal studied optimal scoring rules for confidence-weighted testing, analysing the trade

**Abstract/title:** This chapter summarizes research related to several interrelated questions regarding the process by which single decision makers (DMs) aggregate probabilistic information regarding a certain event fro...

### lin2022truthfulqa
**Title:** TruthfulQA: Measuring How Models Mimic Human Falsehoods
**Score:** 0.249
**Best-matching sentence in cited paper:** _The best model was truthful on 58% of questions, while human performance was 94%._
**NLI:** label=neutral, polarity=-0.001 (>0 means cited paper contradicts the claim)
**Premise (cited paper):** _We propose a benchmark to measure whether a language model is truthful in generating answers to questions. The benchmark comprises 817 questions that span 38 categories, including health, law, finance and politics. We crafted questions that some huma_

**Citation context(s):**
> tering Evaluation datasets. All converted to uniform MCQ format. tab:datasets tabular @ llcl@ Dataset & Source & Options & Domain \\ TruthfulQA & lin2022truthfulqa & 4--5 & Calibration / misconceptions \\ ARC-Easy & clark2018arc & 4 & Science (elementary) \\ ARC-Challenge & clark2018arc & 4 & Scienc

**Abstract/title:** We propose a benchmark to measure whether a language model is truthful in generating answers to questions. The benchmark comprises 817 questions that span 38 categories, including health, law, finance...

### zellers2019hellaswag
**Title:** HellaSwag: Can a Machine Really Finish Your Sentence?
**Score:** 0.195
**Best-matching sentence in cited paper:** _introduced a new task of commonsense natural language inference: given an event description such as "A woman sits at a piano," a machine must select the most likely followup: "She sets her fingers on the keys." With the introduction of BERT (Devlin et al., 2018), near human-level performance was rea_
**NLI:** label=neutral, polarity=-0.000 (>0 means cited paper contradicts the claim)
**Premise (cited paper):** _introduced a new task of commonsense natural language inference: given an event description such as "A woman sits at a piano," a machine must select the most likely followup: "She sets her fingers on the keys." With the introduction of BERT (Devlin e_

**Citation context(s):**
> & 4 & Science (elementary) \\ ARC-Challenge & clark2018arc & 4 & Science (hard) \\ MMLU & hendrycks2021mmlu & 4 & 57 subjects \\ HellaSwag & zellers2019hellaswag & 4 & Commonsense completion \\ MMLU-Pro & wang2024mmlu & 10 & Extended MMLU (A--J) \\ GSM8K & cobbe2021gsm8k & 4 & Math reasoning (conver

**Abstract/title:** introduced a new task of commonsense natural language inference: given an event description such as "A woman sits at a piano," a machine must select the most likely followup: "She sets her fingers on ...

### clark2019boolq
**Title:** BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions
**Score:** 0.278
**Best-matching sentence in cited paper:** _They often query for complex, non-factoid information, and require difficult entailment-like inference to solve._
**NLI:** label=neutral, polarity=+0.001 (>0 means cited paper contradicts the claim)
**Premise (cited paper):** _In this paper we study yes/no questions that are naturally occurring --- meaning that they are generated in unprompted and unconstrained settings. We build a reading comprehension dataset, BoolQ, of such questions, and show that they are unexpectedly_

**Citation context(s):**
> sense completion \\ MMLU-Pro & wang2024mmlu & 10 & Extended MMLU (A--J) \\ GSM8K & cobbe2021gsm8k & 4 & Math reasoning (converted) \\ BoolQ & clark2019boolq & 2 & Yes/no QA \\ TriviaQA & joshi2017triviaqa & 4 & Open-domain QA (converted) \\ tabular table GSM8K is converted to MCQ format by extractin

**Abstract/title:** In this paper we study yes/no questions that are naturally occurring --- meaning that they are generated in unprompted and unconstrained settings. We build a reading comprehension dataset, BoolQ, of s...

### joshi2017triviaqa
**Title:** TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension
**Score:** 0.295
**Best-matching sentence in cited paper:** _We show that, in comparison to other recently introduced large-scale datasets, TriviaQA (1) has relatively complex, compositional questions, (2) has considerable syntactic and lexical variability between questions and corresponding answer-evidence sentences, and (3) requires more cross sentence reas_
**NLI:** label=neutral, polarity=+0.007 (>0 means cited paper contradicts the claim)
**Premise (cited paper):** _We present TriviaQA, a challenging reading comprehension dataset containing over 650K question-answer-evidence triples. TriviaQA includes 95K questionanswer pairs authored by trivia enthusiasts and independently gathered evidence documents, six per q_

**Citation context(s):**
> 10 & Extended MMLU (A--J) \\ GSM8K & cobbe2021gsm8k & 4 & Math reasoning (converted) \\ BoolQ & clark2019boolq & 2 & Yes/no QA \\ TriviaQA & joshi2017triviaqa & 4 & Open-domain QA (converted) \\ tabular table GSM8K is converted to MCQ format by extracting the numeric answer and generating 3 numeric 

**Abstract/title:** We present TriviaQA, a challenging reading comprehension dataset containing over 650K question-answer-evidence triples. TriviaQA includes 95K questionanswer pairs authored by trivia enthusiasts and in...

### fadeeva2024lmpolygraph
**Title:** LM-Polygraph: Uncertainty Estimation for Language Models
**Score:** 0.227
**Best-matching sentence in cited paper:** _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. 2023._
**NLI:** label=neutral, polarity=+0.008 (>0 means cited paper contradicts the claim)
**Premise (cited paper):** _Ekaterina Fadeeva, Roman Vashurin, Akim Tsvigun, Artem Vazhentsev, Sergey Petrakov, Kirill Fedyanin, Daniil Vasilev, Elizaveta Goncharova, Alexander Panchenko, Maxim Panov, Timothy Baldwin, Artem Shelmanov. Proceedings of the 2023 Conference on Empir_

**Citation context(s):**
> ation sec:polygraph To enable direct comparison with the broader uncertainty estimation literature, we implement the NormShift confidence head as a custom estimator within the LM-Polygraph fadeeva2024lmpolygraph benchmarking framework. LM-Polygraph provides over 40 uncertainty estimation methods and
> ective formalised selective prediction as accuracy-coverage tradeoffs. The prediction rejection ratio (PRR) was introduced by malinin2017predictive and adopted by the LM-Polygraph framework fadeeva2024lmpolygraph as a standard metric for uncertainty benchmarking. Confidence-Based Marking. gardnermed
> under the precision-recall curve. Higher is better. PRR (Prediction Rejection Ratio): normalised area between the model's rejection curve and random rejection fadeeva2024lmpolygraph . Higher is better. Acc@ k \% : accuracy on the k \% most confident predictions (selective prediction). itemize Result

**Abstract/title:** Ekaterina Fadeeva, Roman Vashurin, Akim Tsvigun, Artem Vazhentsev, Sergey Petrakov, Kirill Fedyanin, Daniil Vasilev, Elizaveta Goncharova, Alexander Panchenko, Maxim Panov, Timothy Baldwin, Artem Shel...

### ba2016layer
**Title:** Layer Normalization
**Score:** 0.573
**Best-matching sentence in cited paper:** _Empirically, we show that layer normalization can substantially reduce the training time compared with previously published techniques._
**NLI:** label=contradiction, polarity=+0.698 (>0 means cited paper contradicts the claim)
**Premise (cited paper):** _Training state-of-the-art, deep neural networks is computationally expensive. One way to reduce the training time is to normalize the activities of the neurons. A recently introduced technique called batch normalization uses the distribution of the s_

**Citation context(s):**
> h naturally suppresses overconfidence: even at p = 0.8 , the optimal confidence is only = 1.0 mccallum2026hlcc . Normalisation as Uncertainty Signal Layer normalisation ba2016layer rescales activations to zero mean and unit variance. The magnitude of this correction carries information: when pre-nor
> We compare directly against TCP and MDCA in our small-network experiments (Section sec:discussion ). Normalisation. Batch normalisation ioffe2015batch and layer normalisation ba2016layer are standard components of deep networks. Our work is, to our knowledge, the first to explicitly route normalisat

**Abstract/title:** Training state-of-the-art, deep neural networks is computationally expensive. One way to reduce the training time is to normalize the activities of the neurons. A recently introduced technique called ...
