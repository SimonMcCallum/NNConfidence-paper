#!/bin/bash
# generate_figures.sh — Full pipeline: model → data extraction → gnuplot → LaTeX
#
# Pipeline:
#   1. extract_norm_signals.py  runs llama3.1-8b on 90 MCQ examples,
#      captures per-layer std(h[i]) values and norm-shift signals,
#      saves to data/results/llama8b_norm_shift_signals.json
#
#   2. This script reads the JSON + checkpoint, writes .dat files
#
#   3. Gnuplot scripts read .dat files and produce .tex (epslatex terminal)
#
#   4. pdflatex compiles .tex → .pdf
#
# Data provenance:
#   - Model:      meta-llama/Llama-3.1-8B-Instruct (FP16 precision)
#   - Checkpoint:  data/checkpoints/llama3.1-8b_norm_shift/best_norm_shift_combined.pt
#   - Datasets:    truthfulqa (30 ex), arc-easy (30 ex), arc-challenge (30 ex)
#   - Signals:     data/results/llama8b_norm_shift_signals.json
#
# Usage:
#   cd NNConfidence
#   bash docs/figures/generate_figures.sh          # full pipeline
#   bash docs/figures/generate_figures.sh --plots   # gnuplot + latex only (skip extraction)
#
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"

echo "=== NNConfidence Figure Generation Pipeline ==="
echo "Root: $ROOT"
echo ""

# ── Step 0: Check prerequisites ──────────────────────────────────────────
check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        echo "ERROR: $1 not found. Please install it."
        exit 1
    fi
}

SKIP_EXTRACT=false
if [[ "${1:-}" == "--plots" ]]; then
    SKIP_EXTRACT=true
    echo "Skipping data extraction (--plots mode)"
fi

check_cmd gnuplot
check_cmd pdflatex

# ── Step 1: Extract norm-shift signals from llama3.1-8b ──────────────────
SIGNALS_JSON="$ROOT/data/results/llama8b_norm_shift_signals.json"

if [[ "$SKIP_EXTRACT" == false ]]; then
    echo ""
    echo "── Step 1: Extracting norm-shift signals ──"
    echo "  Script:     extract_norm_signals.py"
    echo "  Model:      meta-llama/Llama-3.1-8B-Instruct (FP16)"
    echo "  Checkpoint: data/checkpoints/llama3.1-8b_norm_shift/best_norm_shift_combined.pt"
    echo "  Datasets:   truthfulqa + arc-easy + arc-challenge (30 each)"
    echo ""
    python "$ROOT/extract_norm_signals.py"
    echo ""
    echo "  Output: $SIGNALS_JSON"
fi

if [[ ! -f "$SIGNALS_JSON" ]]; then
    echo "ERROR: $SIGNALS_JSON not found. Run without --plots first."
    exit 1
fi

# ── Step 2: Generate .dat files from JSON ────────────────────────────────
echo ""
echo "── Step 2: Generating .dat files ──"

python -c "
import json, os, sys
import numpy as np

root = '$ROOT'
with open(f'{root}/data/results/llama8b_norm_shift_signals.json') as f:
    data = json.load(f)

results = data['all_results']
n_layers = data['n_layers']
fig_data = f'{root}/docs/figures/data'
os.makedirs(fig_data, exist_ok=True)

correct = [r['norm_shift_signals'] for r in results if r['is_correct']]
incorrect = [r['norm_shift_signals'] for r in results if not r['is_correct']]
c_avg, c_std = np.mean(correct, axis=0), np.std(correct, axis=0)
i_avg, i_std = np.mean(incorrect, axis=0), np.std(incorrect, axis=0)

with open(f'{fig_data}/norm_shift_avg.dat', 'w') as f:
    f.write('# Layer  Correct_avg  Correct_std  Incorrect_avg  Incorrect_std\n')
    f.write(f'# Source: data/results/llama8b_norm_shift_signals.json\n')
    f.write(f'# n_correct={len(correct)}, n_incorrect={len(incorrect)}\n')
    for k in range(n_layers):
        f.write(f'{k+1}  {c_avg[k]:.4f}  {c_std[k]:.4f}  {i_avg[k]:.4f}  {i_std[k]:.4f}\n')

c_stds = np.mean([r['layer_stds'] for r in results if r['is_correct']], axis=0)
i_stds = np.mean([r['layer_stds'] for r in results if not r['is_correct']], axis=0)
with open(f'{fig_data}/layer_stds.dat', 'w') as f:
    f.write('# Layer  Correct_std  Incorrect_std\n')
    for k in range(n_layers):
        f.write(f'{k+1}  {c_stds[k]:.4f}  {i_stds[k]:.4f}\n')

conf_sorted = sorted([r for r in results if r['confidence'] is not None], key=lambda r: r['confidence'], reverse=True)
high, mid, low = conf_sorted[0], conf_sorted[len(conf_sorted)//2], conf_sorted[-1]
with open(f'{fig_data}/example_signals.dat', 'w') as f:
    f.write(f'# Layer  High_conf({high[\"confidence\"]:.3f})  Mid_conf({mid[\"confidence\"]:.3f})  Low_conf({low[\"confidence\"]:.3f})\n')
    f.write(f'# High: \"{high[\"question\"][:60]}\" correct={high[\"is_correct\"]}\n')
    f.write(f'# Mid:  \"{mid[\"question\"][:60]}\" correct={mid[\"is_correct\"]}\n')
    f.write(f'# Low:  \"{low[\"question\"][:60]}\" correct={low[\"is_correct\"]}\n')
    for k in range(n_layers):
        f.write(f'{k+1}  {high[\"norm_shift_signals\"][k]:.4f}  {mid[\"norm_shift_signals\"][k]:.4f}  {low[\"norm_shift_signals\"][k]:.4f}\n')

import torch
cp = torch.load(f'{root}/data/checkpoints/llama3.1-8b_norm_shift/best_norm_shift_combined.pt', map_location='cpu', weights_only=False)
w = cp['state_dict']['norm_shift_proj.0.weight'].float()
importance = w.abs().sum(dim=0).numpy()
with open(f'{fig_data}/head_weights.dat', 'w') as f:
    f.write('# Layer  Weight_importance\n')
    f.write(f'# Source: data/checkpoints/llama3.1-8b_norm_shift/best_norm_shift_combined.pt\n')
    f.write(f'# norm_shift_proj.0.weight shape=[64,32], sum of abs per input dim\n')
    f.write(f'# epoch={cp[\"epoch\"]}, temperature={cp[\"state_dict\"][\"temperature\"].item():.4f}\n')
    for k in range(n_layers):
        f.write(f'{k+1}  {importance[k]:.4f}\n')

bins = np.linspace(0, 1, 11)
c_hist, _ = np.histogram([r['confidence'] for r in results if r['confidence'] is not None and r['is_correct']], bins)
i_hist, _ = np.histogram([r['confidence'] for r in results if r['confidence'] is not None and not r['is_correct']], bins)
with open(f'{fig_data}/confidence_dist.dat', 'w') as f:
    f.write('# Bin_center  Correct_count  Incorrect_count\n')
    for k in range(len(bins)-1):
        f.write(f'{(bins[k]+bins[k+1])/2:.2f}  {c_hist[k]}  {i_hist[k]}\n')

print('  Generated 5 .dat files in docs/figures/data/')
"
echo "  Done."

# ── Step 3: Run gnuplot scripts ──────────────────────────────────────────
echo ""
echo "── Step 3: Running gnuplot scripts ──"

SCRIPTS_DIR="$ROOT/docs/figures/scripts"
OUTPUT_DIR="$ROOT/docs/figures/output"
mkdir -p "$OUTPUT_DIR"

for gp in "$SCRIPTS_DIR"/*.gp; do
    name=$(basename "$gp" .gp)
    echo "  gnuplot $name.gp → output/$name.tex"
    gnuplot "$gp"
done

echo "  Done."

# ── Step 4: Compile LaTeX → PDF ──────────────────────────────────────────
echo ""
echo "── Step 4: Compiling LaTeX → PDF ──"

cd "$OUTPUT_DIR"
for tex in *.tex; do
    name=$(basename "$tex" .tex)
    echo "  pdflatex $name.tex → $name.pdf"
    pdflatex -interaction=nonstopmode "$tex" > /dev/null 2>&1 || {
        echo "  WARNING: pdflatex failed for $name.tex (may need packages)"
    }
done
cd "$ROOT"

echo ""
echo "  Done."

# ── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "=== Pipeline Complete ==="
echo ""
echo "Data provenance chain:"
echo "  1. Model:      meta-llama/Llama-3.1-8B-Instruct (4-bit NF4)"
echo "  2. Checkpoint:  data/checkpoints/llama3.1-8b_norm_shift/best_norm_shift_combined.pt"
echo "  3. Extraction:  extract_norm_signals.py → data/results/llama8b_norm_shift_signals.json"
echo "  4. Data files:  docs/figures/data/*.dat"
echo "  5. Gnuplot:     docs/figures/scripts/*.gp → docs/figures/output/*.tex"
echo "  6. PDF:         docs/figures/output/*.pdf"
echo ""
echo "Files:"
ls -la "$OUTPUT_DIR"/*.pdf 2>/dev/null || echo "  (no PDFs — gnuplot or pdflatex may not be installed)"
echo ""
echo "To regenerate just the plots (without rerunning the model):"
echo "  bash docs/figures/generate_figures.sh --plots"
