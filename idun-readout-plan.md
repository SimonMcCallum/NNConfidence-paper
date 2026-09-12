# Plan: Kumaran-style readout comparison on large models (Idun, NTNU)

Companion to `docs/kumaran2026_comparison.md`. The local sweep
(`run_readout_sweep.sh`, RTX PRO 4500, 32 GB) covers 1.5B to 32B with
quantisation above 8B. Idun is for the models that decide the argument:
Gemma 3 27B at FP16 (the model Kumaran et al. steered), and 70B-class models
where the local quantisation confound is removed.

## Question each model answers

| Model | Why | GPUs | Precision | Est. time (1000 q + steer) |
|---|---|---|---|---|
| gemma3-27b | Kumaran's Phase 3 model. Direct anchor: their abstention rates, their steering layers (30-40 of 62), our structural statistics on the same forward pass. GeGLU, so gate statistics apply. | 1 x 80 GB | FP16 | 3-4 h |
| gemma2-27b | Same family, earlier generation; separates "Gemma 3" from "Gemma". | 1 x 80 GB | FP16 | 3-4 h |
| qwen2.5-32b | Already run locally at 4-bit; FP16 here isolates the quantisation effect on the structural statistics. | 1 x 80 GB | FP16 | 3 h |
| llama3.1-70b | Scale. Already the NormShift 70B target (`idun_normshift_70b.sh`), so the weights are cached. | 2 x 80 GB | FP16 | 8-10 h |
| qwen2.5-72b | Second 70B family; Kumaran's Qwen result was Qwen3-Next-80B via API with no activation access. | 2 x 80 GB | FP16 | 8-10 h |

Order of submission: gemma3-27b first (smoke, then full), then llama3.1-70b,
then the rest as the queue allows.

## Pipeline per model (`jobs/idun_readout_large.sh`)

1. `readout_comparison.py`: Phases 1, 2, 4 (T = 20, 50, 80) and verbal
   confidence on Kumaran's 1,000 `phase1_questions.csv` with their seed-1042
   arrangement, recording option logits, structural statistics at every layer,
   logit-lens commitment depth, and the residual at every layer (float16 npz,
   about 1 GB for a 70B model).
2. `steer_abstention.py`: difference-of-means steering vectors from the first
   500 questions' Phase 2 residuals, injected at 3 % of residual norm at five
   layers around mid-depth with scales -2 to +2, and the residual-rescale
   intervention at the same layers. Evaluated on 300 held-out questions.
3. `analyse_readout_comparison.py`: the report.

## What to look at first when results come back

1. Section 4 of the report: dAUROC of structural statistics for abstention vs
   for correctness. The prediction is near zero for abstention and positive for
   correctness.
2. Section 5: reverse decodability. R2 from the residual to gate kurtosis and
   near-zero ratio should stay low at every layer if the claim about
   dictionary-mapping holds. Pre-norm std may be partly decodable; report it
   honestly either way.
3. Steering file: does `feats_mean.gate_nearzero_L*` move with steering scale
   while `abstain_rate` moves? If abstention moves and the structural
   statistics do not, they are dissociable. Does rescale move abstention at all?
4. Phase 4 pooled fit: scale and shift against Kumaran's 1.80 / -97.6 for
   GPT-4o and their Gemma value (0.66) for the same model.

## One-off setup on the login node

```bash
cd ~/git/NNConfidence && git pull
bash jobs/setup_env.sh
source jobs/cluster_env.sh
# gated models: accept the licence on huggingface.co first, token in ~/.hf_token
python - <<'EOF'
from huggingface_hub import snapshot_download as s
import os
for rid in ["google/gemma-3-27b-it", "google/gemma-2-27b-it", "Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-72B-Instruct"]:
    s(rid, cache_dir=os.environ["NNCONF_CACHE_DIR"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
EOF
# Kumaran's question set is in the repo under data/kumaran2026/ only on the desktop (data/ is gitignored):
# scp -r D:/git/NNConfidence/data/kumaran2026 idun:~/git/NNConfidence/data/
```

If `snapshot_download` on the login node is too slow, use the samba node as in
`docs/idun-deployment.md`.

## Submitting

```bash
sbatch --time=01:00:00 --export=MODEL=gemma3-27b,MAX=40,SKIP_STEER=1 jobs/idun_readout_large.sh   # smoke
sbatch --export=MODEL=gemma3-27b jobs/idun_readout_large.sh
sbatch --gres=gpu:2 --mem=256G --export=MODEL=llama3.1-70b jobs/idun_readout_large.sh
sbatch --export=MODEL=gemma2-27b jobs/idun_readout_large.sh
sbatch --export=MODEL=qwen2.5-32b jobs/idun_readout_large.sh
sbatch --gres=gpu:2 --mem=256G --export=MODEL=qwen2.5-72b jobs/idun_readout_large.sh
```

Gemma 3 27B-it is multimodal (`Gemma3ForConditionalGeneration`);
`model_loader.py` detects `text_config` + `vision_config` and wraps the
language model, and `LayerHooks` looks for `language_model.model.layers`.
Verify on the smoke run that the hook count equals 62 and gate features are on.

## Bringing results back

```bash
rsync -av idun:/cluster/work/sjmccall/nnconf/results/readout/ D:/git/NNConfidence/data/results/readout/
python analyse_readout_comparison.py --all --summary
```

## Known risks

- Gemma 3 at FP16 on one 80 GB card with `output_hidden_states=True` and
  1000-token prompts is fine; at 70B with two cards `device_map="auto"` splits
  layers and the hooks still fire on each card.
- The verbal-confidence pass uses greedy generation of 12 tokens; reasoning
  models (DeepSeek-R1 distils) emit `<think>` first and the class is not
  parsed. Their rows are mean-filled and flagged in the report.
- `steer_abstention.py` needs the readout run's npz for the same model and
  `MAX`; the job passes `--n-source "$MAX"` for that reason.
