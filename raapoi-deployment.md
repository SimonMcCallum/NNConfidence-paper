# Running NNConfidence on Raapoi (VUW HPC)

## Overview

NNConfidence trains lightweight confidence heads (NormShift) on top of frozen LLMs
and evaluates calibration baselines. On Raapoi we can test at FP16 precision and
on larger models (70B) that don't fit on the desktop GPU.

## Hardware Constraints

| Config | GPU VRAM | 70B FP16 (~140GB) | 70B 4-bit (~35GB) | 24B FP16 (~48GB) |
|--------|----------|-------------------|--------------------|-------------------|
| 2x A100-40GB | 80GB total | CPU offload needed | Fits on 1 GPU | Fits on 1-2 GPUs |
| 2x A100-80GB | 160GB total | Fits across 2 GPUs | Fits on 1 GPU | Fits on 1 GPU |

**Check your actual GPU:** `nvidia-smi` on a compute node will show 40GB or 80GB.

## Setup

### 1. Clone and install

```bash
cd /nfs/scratch/mccallsi
git clone https://github.com/SimonMcCallum/NNConfidence.git
cd NNConfidence

module load Anaconda3
module load CUDA/12.1
conda create -n nnconf python=3.11 -y
conda activate nnconf
pip install -r requirements.txt
```

### 2. Download models

```bash
export HF_HOME=/nfs/scratch/mccallsi/hf_cache
export TRANSFORMERS_CACHE=/nfs/scratch/mccallsi/models

# 7B models (FP16 on any A100)
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

# 70B model (need token for Mistral)
huggingface-cli login
huggingface-cli download mistralai/Mistral-Large-Instruct-2407
```

### 3. Update model cache path

Set in all scripts via environment variable:
```bash
export TRANSFORMERS_CACHE=/nfs/scratch/mccallsi/models
```

Or symlink to match the expected path:
```bash
mkdir -p /nfs/scratch/mccallsi/NNCONFIDENCE/data
ln -s /nfs/scratch/mccallsi/models /nfs/scratch/mccallsi/NNCONFIDENCE/data/models
```

## Slurm Jobs

### Job 1: NormShift training on 7B models (FP16, baseline comparison)

Tests whether FP16 precision changes the NormShift signal compared to
4-bit quantisation on the desktop.

```bash
sbatch jobs/normshift_fp16.sh
```

### Job 2: NormShift on 70B (the big experiment)

Two strategies depending on GPU memory:

**If 2x A100-80GB (160GB total):** Tensor parallelism, no offload needed.
**If 2x A100-40GB (80GB total):** CPU offload for model weights, GPU for compute.

```bash
sbatch jobs/normshift_70b.sh
```

### Job 3: Calibration baselines on 70B

Runs all calibration methods (entropy, MSP, top-k, temperature scaling,
NormShift) on the 70B model.

```bash
sbatch jobs/calibration_70b.sh
```

## Key Differences from Desktop

| | Desktop (RTX PRO 4500) | Raapoi |
|--|------------------------|--------|
| Quantisation | 4-bit / 8-bit | FP16 (eliminates quantisation confound) |
| NormShift signal | May be noisy from quantisation | Clean FP16 magnitudes |
| Model loading | `model_loader.py` auto-detects | Need explicit precision + device_map |
| Training | Single GPU | Can use both GPUs (data parallel or tensor parallel) |
