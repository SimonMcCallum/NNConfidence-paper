#!/usr/bin/env python3
"""generate_dat_files.py — Generate all gnuplot .dat files from experiment results.

Reads actual training results and checkpoint data to produce all .dat files
needed by the gnuplot scripts in docs/figures/scripts/. Run this after new
experiments to regenerate figures with updated data.

Data sources:
  - data/results/{model}_layer_signals.json   (per-example SD signal extraction)
  - data/results/llama8b_norm_shift_signals.json  (legacy llama extraction)
  - data/checkpoints/{model}_norm_shift/best_norm_shift_combined.pt  (trained head)

Output:
  - docs/figures/data/norm_shift_avg.dat
  - docs/figures/data/layer_stds.dat
  - docs/figures/data/example_signals.dat
  - docs/figures/data/head_weights.dat
  - docs/figures/data/confidence_dist.dat
  - docs/figures/data/layer27_vs_31_scatter.dat
  - docs/figures/data/layer27_vs_31_boxplot.dat
  - docs/figures/data/layer27_vs_31_density.dat

Usage:
  # Regenerate all .dat files from llama3.1-8b results (default)
  python docs/figures/generate_dat_files.py

  # Regenerate from a specific model's results
  python docs/figures/generate_dat_files.py --model mistral-7b

  # Use a specific signals JSON file
  python docs/figures/generate_dat_files.py --signals-json data/results/mistral-7b_layer_signals.json
"""

import json
import os
import sys
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "docs", "figures", "data")
RESULTS_DIR = os.path.join(ROOT, "data", "results")
CHECKPOINT_DIR = os.path.join(ROOT, "data", "checkpoints")


def load_signals_json(path):
    """Load a layer signals JSON file."""
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded: {path}")
    print(f"  Model: {data.get('model', '?')}")
    print(f"  Examples: {data.get('n_examples', len(data.get('all_results', [])))}")
    print(f"  Layers: {data.get('n_layers', '?')}")
    return data


def find_signals_json(model_key):
    """Find the signals JSON for a given model."""
    # Try new-style name first
    path = os.path.join(RESULTS_DIR, f"{model_key}_layer_signals.json")
    if os.path.exists(path):
        return path
    # Try legacy llama name
    if "llama" in model_key:
        path = os.path.join(RESULTS_DIR, "llama8b_norm_shift_signals.json")
        if os.path.exists(path):
            return path
    return None


def find_checkpoint(model_key):
    """Find the best norm-shift checkpoint for a model."""
    cp_dir = os.path.join(CHECKPOINT_DIR, f"{model_key}_norm_shift")
    for variant in ["best_norm_shift_combined.pt", "best_norm_shift_norm_shift_only.pt"]:
        path = os.path.join(cp_dir, variant)
        if os.path.exists(path):
            return path
    return None


def generate_norm_shift_avg(data, out_path):
    """Generate norm_shift_avg.dat — per-layer correct/incorrect norm-shift averages."""
    results = data.get("all_results", [])
    n_layers = data.get("n_layers", results[0]["n_layers"] if results else 32)
    model = data.get("model", "unknown")

    correct = [r for r in results if r["is_correct"]]
    incorrect = [r for r in results if not r["is_correct"]]

    correct_avg = np.mean([r["norm_shift_signals"] for r in correct], axis=0) if correct else np.zeros(n_layers)
    incorrect_avg = np.mean([r["norm_shift_signals"] for r in incorrect], axis=0) if incorrect else np.zeros(n_layers)
    correct_std = np.std([r["norm_shift_signals"] for r in correct], axis=0) if correct else np.zeros(n_layers)
    incorrect_std = np.std([r["norm_shift_signals"] for r in incorrect], axis=0) if incorrect else np.zeros(n_layers)

    with open(out_path, "w") as f:
        f.write(f"# norm_shift_avg.dat — Average norm-shift by correctness\n")
        f.write(f"# Source: {model} ({len(results)} examples, {len(correct)} correct, {len(incorrect)} incorrect)\n")
        f.write(f"# Layer  Correct_avg  Correct_std  Incorrect_avg  Incorrect_std\n")
        for i in range(n_layers):
            f.write(f"{i+1:4d}  {correct_avg[i]:.6f}  {correct_std[i]:.6f}  "
                    f"{incorrect_avg[i]:.6f}  {incorrect_std[i]:.6f}\n")

    print(f"  Generated: {out_path} ({n_layers} layers)")


def generate_layer_stds(data, out_path):
    """Generate layer_stds.dat — per-layer standard deviation values."""
    results = data.get("all_results", [])
    n_layers = data.get("n_layers", results[0]["n_layers"] if results else 32)
    model = data.get("model", "unknown")

    correct = [r for r in results if r["is_correct"]]
    incorrect = [r for r in results if not r["is_correct"]]

    correct_std_avg = np.mean([r["layer_stds"] for r in correct], axis=0) if correct else np.zeros(n_layers)
    incorrect_std_avg = np.mean([r["layer_stds"] for r in incorrect], axis=0) if incorrect else np.zeros(n_layers)

    with open(out_path, "w") as f:
        f.write(f"# layer_stds.dat — Per-layer activation std(h[i]) averaged by correctness\n")
        f.write(f"# Layer  Correct_std  Incorrect_std\n")
        for i in range(n_layers):
            f.write(f"{i+1:4d}  {correct_std_avg[i]:.6f}  {incorrect_std_avg[i]:.6f}\n")

    print(f"  Generated: {out_path} ({n_layers} layers)")


def generate_example_signals(data, out_path):
    """Generate example_signals.dat — norm-shift signals for high/mid/low confidence examples."""
    results = data.get("all_results", [])
    n_layers = data.get("n_layers", results[0]["n_layers"] if results else 32)

    # Sort by confidence to find high/mid/low examples
    results_with_conf = [r for r in results if r.get("confidence") is not None]
    if not results_with_conf:
        # Fallback: use choice probability of predicted answer as proxy
        for r in results:
            pred = r["predicted_answer"]
            r["_conf"] = r["choice_probs"][pred] if pred < len(r["choice_probs"]) else 0.5
        results_with_conf = results
        conf_key = "_conf"
    else:
        conf_key = "confidence"

    results_with_conf.sort(key=lambda r: r.get(conf_key, r.get("_conf", 0.5)), reverse=True)

    high = results_with_conf[0]
    mid = results_with_conf[len(results_with_conf) // 2]
    low = results_with_conf[-1]

    high_conf = high.get(conf_key, high.get("_conf", 0))
    mid_conf = mid.get(conf_key, mid.get("_conf", 0))
    low_conf = low.get(conf_key, low.get("_conf", 0))

    with open(out_path, "w") as f:
        f.write(f"# example_signals.dat — Norm-shift signals for high/mid/low confidence examples\n")
        f.write(f"# High: conf={high_conf:.3f}, {'correct' if high['is_correct'] else 'wrong'}: {high['question'][:80]}\n")
        f.write(f"# Mid:  conf={mid_conf:.3f}, {'correct' if mid['is_correct'] else 'wrong'}: {mid['question'][:80]}\n")
        f.write(f"# Low:  conf={low_conf:.3f}, {'correct' if low['is_correct'] else 'wrong'}: {low['question'][:80]}\n")
        f.write(f"# Layer  High_conf({high_conf:.3f})  Mid_conf({mid_conf:.3f})  Low_conf({low_conf:.3f})\n")
        for i in range(n_layers):
            f.write(f"{i+1:4d}  {high['norm_shift_signals'][i]:.6f}  "
                    f"{mid['norm_shift_signals'][i]:.6f}  "
                    f"{low['norm_shift_signals'][i]:.6f}\n")

    print(f"  Generated: {out_path} ({n_layers} layers, 3 examples)")


def generate_head_weights(model_key, out_path):
    """Generate head_weights.dat — trained confidence head weight importance per layer."""
    import torch

    cp_path = find_checkpoint(model_key)
    if not cp_path:
        print(f"  WARNING: No checkpoint found for {model_key}, skipping head_weights.dat")
        return

    cp = torch.load(cp_path, map_location="cpu", weights_only=False)
    state_dict = cp.get("state_dict", {})

    # Extract layer importance from norm_shift_proj first layer
    if "norm_shift_proj.0.weight" in state_dict:
        w = state_dict["norm_shift_proj.0.weight"]  # shape [hidden, n_layers]
        layer_importance = w.abs().sum(dim=0).numpy()
    elif "network.0.weight" in state_dict:
        w = state_dict["network.0.weight"]
        layer_importance = w.abs().sum(dim=0).numpy()
    else:
        print(f"  WARNING: Cannot extract layer weights from checkpoint, skipping head_weights.dat")
        return

    n_layers = len(layer_importance)
    epoch = cp.get("epoch", "?")

    # Get temperature if available
    temp = state_dict.get("temperature", None)
    temp_str = f", T={temp.item():.4f}" if temp is not None else ""

    with open(out_path, "w") as f:
        f.write(f"# head_weights.dat — Trained confidence head weight importance per layer\n")
        f.write(f"# Source: {cp_path}\n")
        f.write(f"# Layer weights = sum(abs(norm_shift_proj.0.weight), dim=0)\n")
        f.write(f"# epoch={epoch}{temp_str}\n")
        f.write(f"# Layer  Weight_importance\n")
        for i in range(n_layers):
            f.write(f"{i+1:4d}  {layer_importance[i]:.4f}\n")

    print(f"  Generated: {out_path} ({n_layers} layers, epoch {epoch})")


def generate_confidence_dist(data, out_path):
    """Generate confidence_dist.dat — histogram of confidence by correctness."""
    results = data.get("all_results", [])

    # Use trained head confidence if available, else use max choice probability
    conf_key = "confidence"
    results_with_conf = [r for r in results if r.get(conf_key) is not None]
    if not results_with_conf:
        conf_key = None
        results_with_conf = results

    bins = np.arange(0, 1.05, 0.1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    correct_counts = np.zeros(len(bin_centers), dtype=int)
    incorrect_counts = np.zeros(len(bin_centers), dtype=int)

    for r in results_with_conf:
        if conf_key:
            c = r[conf_key]
        else:
            pred = r["predicted_answer"]
            c = r["choice_probs"][pred] if pred < len(r["choice_probs"]) else 0.5

        bin_idx = min(int(c * 10), 9)
        if r["is_correct"]:
            correct_counts[bin_idx] += 1
        else:
            incorrect_counts[bin_idx] += 1

    with open(out_path, "w") as f:
        f.write(f"# confidence_dist.dat — Confidence distribution histogram\n")
        f.write(f"# Bin_center  Correct_count  Incorrect_count\n")
        for i in range(len(bin_centers)):
            f.write(f"{bin_centers[i]:.2f}  {correct_counts[i]:4d}  {incorrect_counts[i]:4d}\n")

    total = sum(correct_counts) + sum(incorrect_counts)
    print(f"  Generated: {out_path} ({len(bin_centers)} bins, {total} examples)")


def generate_layer27_vs_31_scatter(data, out_path, layer_a=27, layer_b=31):
    """Generate layer_vs_scatter.dat — per-example scatter data for two layers."""
    results = data.get("all_results", [])
    n_layers = data.get("n_layers", results[0]["n_layers"] if results else 32)

    # Adjust for 0-indexed
    idx_a = layer_a - 1
    idx_b = layer_b - 1

    if idx_a >= n_layers or idx_b >= n_layers:
        print(f"  WARNING: Model has {n_layers} layers, cannot use layers {layer_a}/{layer_b}")
        # Fall back to last two interesting layers
        idx_b = n_layers - 1
        idx_a = max(0, int(n_layers * 0.84))  # ~84% through
        layer_a = idx_a + 1
        layer_b = idx_b + 1
        print(f"  Using layers {layer_a} and {layer_b} instead")

    with open(out_path, "w") as f:
        f.write(f"# layer{layer_a}_vs_{layer_b}_scatter.dat — Per-example scatter data\n")
        f.write(f"# Source: {data.get('model', 'unknown')} ({len(results)} examples)\n")
        f.write(f"# Layers: {layer_a} (idx {idx_a}), {layer_b} (idx {idx_b})\n")
        f.write(f"# n_correct={sum(1 for r in results if r['is_correct'])}, "
                f"n_incorrect={sum(1 for r in results if not r['is_correct'])}\n")
        f.write(f"# Example  NormShift_L{layer_a}  NormShift_L{layer_b}  IsCorrect  Confidence\n")
        for i, r in enumerate(results):
            conf = r.get("confidence", 0.0) or 0.0
            f.write(f"{i+1:4d}  {r['norm_shift_signals'][idx_a]:.6f}  "
                    f"{r['norm_shift_signals'][idx_b]:.6f}  "
                    f"{int(r['is_correct'])}  {conf:.6f}\n")

    print(f"  Generated: {out_path} ({len(results)} examples, layers {layer_a}/{layer_b})")


def generate_layer27_vs_31_boxplot(data, out_path, layer_a=27, layer_b=31):
    """Generate layer_vs_boxplot.dat — quartile statistics for two layers."""
    results = data.get("all_results", [])
    n_layers = data.get("n_layers", results[0]["n_layers"] if results else 32)

    idx_a = layer_a - 1
    idx_b = layer_b - 1

    if idx_a >= n_layers or idx_b >= n_layers:
        idx_b = n_layers - 1
        idx_a = max(0, int(n_layers * 0.84))
        layer_a = idx_a + 1
        layer_b = idx_b + 1

    correct = [r for r in results if r["is_correct"]]
    incorrect = [r for r in results if not r["is_correct"]]

    groups = [
        (f"L{layer_a}_correct", [r["norm_shift_signals"][idx_a] for r in correct]),
        (f"L{layer_a}_incorrect", [r["norm_shift_signals"][idx_a] for r in incorrect]),
        (f"L{layer_b}_correct", [r["norm_shift_signals"][idx_b] for r in correct]),
        (f"L{layer_b}_incorrect", [r["norm_shift_signals"][idx_b] for r in incorrect]),
    ]

    with open(out_path, "w") as f:
        f.write(f"# layer{layer_a}_vs_{layer_b}_boxplot.dat — Quartile statistics\n")
        f.write(f"# Source: {data.get('model', 'unknown')}\n")
        f.write(f"# n_correct={len(correct)}, n_incorrect={len(incorrect)}\n")
        f.write(f"# Layers: {layer_a} and {layer_b}\n")
        f.write(f"# Category  Median  Q1  Q3  Min  Max  Mean\n")
        for name, vals in groups:
            if vals:
                arr = np.array(vals)
                f.write(f"{name:<20s}  {np.median(arr):.4f}  {np.percentile(arr, 25):.4f}  "
                        f"{np.percentile(arr, 75):.4f}  {arr.min():.4f}  {arr.max():.4f}  "
                        f"{arr.mean():.4f}\n")
            else:
                f.write(f"{name:<20s}  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000\n")

    print(f"  Generated: {out_path} (4 groups)")


def generate_layer27_vs_31_density(data, out_path, layer_a=27, layer_b=31, n_bins=20):
    """Generate layer_vs_density.dat — histogram density for two layers."""
    results = data.get("all_results", [])
    n_layers = data.get("n_layers", results[0]["n_layers"] if results else 32)

    idx_a = layer_a - 1
    idx_b = layer_b - 1

    if idx_a >= n_layers or idx_b >= n_layers:
        idx_b = n_layers - 1
        idx_a = max(0, int(n_layers * 0.84))
        layer_a = idx_a + 1
        layer_b = idx_b + 1

    correct = [r for r in results if r["is_correct"]]
    incorrect = [r for r in results if not r["is_correct"]]

    # Determine bin range from data
    all_vals = [r["norm_shift_signals"][idx_a] for r in results] + \
               [r["norm_shift_signals"][idx_b] for r in results]
    bin_min = min(all_vals) - 0.02
    bin_max = max(all_vals) + 0.02
    bin_width = (bin_max - bin_min) / n_bins
    bins = np.linspace(bin_min, bin_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    la_correct = np.histogram([r["norm_shift_signals"][idx_a] for r in correct], bins=bins)[0]
    la_incorrect = np.histogram([r["norm_shift_signals"][idx_a] for r in incorrect], bins=bins)[0]
    lb_correct = np.histogram([r["norm_shift_signals"][idx_b] for r in correct], bins=bins)[0]
    lb_incorrect = np.histogram([r["norm_shift_signals"][idx_b] for r in incorrect], bins=bins)[0]

    with open(out_path, "w") as f:
        f.write(f"# layer{layer_a}_vs_{layer_b}_density.dat — Histogram density\n")
        f.write(f"# Source: {data.get('model', 'unknown')}\n")
        f.write(f"# {n_bins} bins over [{bin_min:.2f}, {bin_max:.2f}], bin_width={bin_width:.4f}\n")
        f.write(f"# Bin_center  L{layer_a}_correct  L{layer_a}_incorrect  "
                f"L{layer_b}_correct  L{layer_b}_incorrect\n")
        for i in range(n_bins):
            f.write(f"{bin_centers[i]:.4f}  {la_correct[i]:4d}  {la_incorrect[i]:4d}  "
                    f"{lb_correct[i]:4d}  {lb_incorrect[i]:4d}\n")

    print(f"  Generated: {out_path} ({n_bins} bins)")


def main():
    parser = argparse.ArgumentParser(description="Generate gnuplot .dat files from experiment results")
    parser.add_argument("--model", type=str, default="llama3.1-8b",
                        help="Model key to generate figures for (default: llama3.1-8b)")
    parser.add_argument("--signals-json", type=str, default=None,
                        help="Path to signals JSON file (auto-detected if not specified)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (auto-detected if not specified)")
    parser.add_argument("--output-dir", type=str, default=DATA_DIR,
                        help="Output directory for .dat files")
    parser.add_argument("--layer-a", type=int, default=None,
                        help="First comparison layer (default: auto, ~84%% through model)")
    parser.add_argument("--layer-b", type=int, default=None,
                        help="Second comparison layer (default: auto, last layer)")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated list of .dat files to generate (e.g., norm_shift_avg,head_weights)")
    args = parser.parse_args()

    # Find signals JSON
    if args.signals_json:
        signals_path = args.signals_json
    else:
        signals_path = find_signals_json(args.model)
        if not signals_path:
            print(f"ERROR: No signals JSON found for model '{args.model}'")
            print(f"  Run: python extract_layer_signals.py --model {args.model}")
            sys.exit(1)

    data = load_signals_json(signals_path)
    n_layers = data.get("n_layers", 32)

    # Auto-detect comparison layers
    layer_a = args.layer_a or max(1, int(n_layers * 0.84))
    layer_b = args.layer_b or n_layers

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which files to generate
    all_files = ["norm_shift_avg", "layer_stds", "example_signals", "head_weights",
                 "confidence_dist", "layer27_vs_31_scatter", "layer27_vs_31_boxplot",
                 "layer27_vs_31_density"]
    if args.only:
        to_generate = set(args.only.split(","))
    else:
        to_generate = set(all_files)

    print(f"\nGenerating .dat files for {args.model} (layers: {layer_a}, {layer_b}):")
    print(f"  Output: {args.output_dir}\n")

    if "norm_shift_avg" in to_generate:
        generate_norm_shift_avg(data, os.path.join(args.output_dir, "norm_shift_avg.dat"))

    if "layer_stds" in to_generate:
        generate_layer_stds(data, os.path.join(args.output_dir, "layer_stds.dat"))

    if "example_signals" in to_generate:
        generate_example_signals(data, os.path.join(args.output_dir, "example_signals.dat"))

    if "head_weights" in to_generate:
        generate_head_weights(args.model, os.path.join(args.output_dir, "head_weights.dat"))

    if "confidence_dist" in to_generate:
        generate_confidence_dist(data, os.path.join(args.output_dir, "confidence_dist.dat"))

    if "layer27_vs_31_scatter" in to_generate:
        generate_layer27_vs_31_scatter(data,
            os.path.join(args.output_dir, "layer27_vs_31_scatter.dat"), layer_a, layer_b)

    if "layer27_vs_31_boxplot" in to_generate:
        generate_layer27_vs_31_boxplot(data,
            os.path.join(args.output_dir, "layer27_vs_31_boxplot.dat"), layer_a, layer_b)

    if "layer27_vs_31_density" in to_generate:
        generate_layer27_vs_31_density(data,
            os.path.join(args.output_dir, "layer27_vs_31_density.dat"), layer_a, layer_b)

    print(f"\nDone! Generated {len(to_generate)} .dat files.")
    print(f"\nTo regenerate gnuplot figures:")
    print(f'  cd {ROOT}')
    print(f'  gnuplot docs/figures/scripts/norm_shift_avg.gp')
    print(f'  # ... or run all scripts')


if __name__ == "__main__":
    main()
