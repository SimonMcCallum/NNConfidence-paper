# Running the 70B NormShift experiment on Idun (NTNU)

Raapoi is oversubscribed, so the 70B run moves to Idun. The job files are in
`jobs/` and follow the conventions already used by the activation-forensics
work on the same cluster (`Mistral-Activation-test/activation-forensics/jobs/`),
including the shared model cache at `/cluster/work/sjmccall/models`.

## What the job does

`jobs/idun_normshift_70b.sh` runs three phases on Llama-3.1-70B-Instruct at FP16
across two 80 GB GPUs:

1. `train_norm_shift_head.py`: combined and norm-shift-only heads, HLCC loss,
   25 epochs on 900 examples (ARC-Easy, TruthfulQA, ARC-Challenge), resumable.
2. `calibration_baselines.py`: entropy, top-k, MSP, temperature scaling and the
   trained head on the full 817 TruthfulQA items.
3. `extract_layer_signals.py`: per-layer Cohen's d across the 80 layers.

Each phase writes its own file under `/cluster/work/sjmccall/nnconf/`, so a job
that is cut short still returns something.

Note that the Raapoi script used the key `mistral-large`, which is
Mistral-Large-2407 at 123B parameters. That does not fit two 80 GB cards at FP16.
`model_loader.py` now carries `llama3.1-70b` and `llama3.3-70b` (70.6B) and the
corrected size for Mistral Large.

## One-off setup (login node)

```bash
cd ~/git
git clone https://github.com/SimonMcCallum/NNConfidence.git
cd NNConfidence
bash jobs/setup_env.sh            # venv at /cluster/work/sjmccall/venv-nnconfidence, datasets cached
```

The 70B weights are gated by Meta. If the activation-forensics download has
already run, the weights are in the shared cache and nothing more is needed:

```bash
ls /cluster/work/sjmccall/models/models--meta-llama--Llama-3.1-70B-Instruct
```

Otherwise, from a login node, with the HF token in `~/.hf_token` (`chmod 600`):

```bash
ssh idun-samba1 'bash ~/git/Mistral-Activation-test/activation-forensics/jobs/samba_download_llama_70b.sh'
```

Compute nodes have no outbound internet; the job sets `HF_HUB_OFFLINE=1` and
fails early with a clear message if the weights are absent.

## Submitting

```bash
# smoke test, about 20 minutes once the model is cached
sbatch --time=01:00:00 --export=EPOCHS=1,TRAIN_EXAMPLES=40,MAX_EXAMPLES=40 jobs/idun_normshift_70b.sh

# full run, two 80 GB GPUs, up to 2 days
sbatch jobs/idun_normshift_70b.sh

# Llama 3.3 instead of 3.1 (same architecture and sizing)
sbatch --export=MODEL=llama3.3-70b jobs/idun_normshift_70b.sh

# one 80 GB card at 4-bit if the queue for pairs is long (reintroduces the quantisation confound)
sbatch --gres=gpu:1 --export=PRECISION=4bit jobs/idun_normshift_70b.sh
```

Account and QoS: the script uses `--account=share-ie-idi` and `--constraint=gpu80g`
on `GPUQ`, as the activation-forensics 70B job does. Confirm with
`sacctmgr show assoc format=Account%15,User,QOS | grep sjmccall` if the
submission is rejected.

## Bringing results back

```bash
rsync -av idun:/cluster/work/sjmccall/nnconf/results/ D:/git/NNConfidence/data/results/
rsync -av idun:/cluster/work/sjmccall/nnconf/checkpoints/llama3.1-70b_norm_shift/ \
          D:/git/NNConfidence/data/checkpoints/llama3.1-70b_norm_shift/
python sync_paper_data.py
python generate_paper_tables.py
```

## Expected duration

At 70B FP16 on two A100-80G or H100-80G cards: about 1 s per forward pass, so
25 epochs over 900 examples is roughly 7 hours, the 817-item baselines about 2
hours (no sampling methods), and the layer extraction under an hour. The 2-day
limit leaves room for a slow node.
