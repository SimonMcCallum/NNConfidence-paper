#!/usr/bin/env python3
"""generate_layer27_vs_31.py — Generate Layer 27 vs 31 comparison figures.

Creates publication-quality figures showing the subtle differences in norm-shift
signals between layers 27 and 31 of the Llama 3.1-8B confidence head.

Pipeline:
  1. Reads per-example data from data/results/llama8b_norm_shift_signals.json
  2. Reads trained weights from data/checkpoints/llama3.1-8b_norm_shift/best_norm_shift_combined.pt
  3. Generates 3-panel figure as PDF

Data provenance:
  - Model:      meta-llama/Llama-3.1-8B-Instruct (4-bit NF4)
  - Checkpoint:  data/checkpoints/llama3.1-8b_norm_shift/best_norm_shift_combined.pt (epoch 6)
  - Signals:     data/results/llama8b_norm_shift_signals.json (90 MCQ examples)
  - Datasets:    truthfulqa (30), arc-easy (30), arc-challenge (30)

Usage:
  python docs/figures/generate_layer27_vs_31.py
"""

import json
import os
import sys
import numpy as np

# Try matplotlib for PDF generation
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Install with: pip install matplotlib")
    print("         Only .dat files will be generated (use gnuplot for .tex output)")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIGNALS_JSON = os.path.join(ROOT, "data", "results", "llama8b_norm_shift_signals.json")
CHECKPOINT = os.path.join(ROOT, "data", "checkpoints", "llama3.1-8b_norm_shift", "best_norm_shift_combined.pt")
OUTPUT_DIR = os.path.join(ROOT, "docs", "figures", "output")
DATA_DIR = os.path.join(ROOT, "docs", "figures", "data")

def load_data():
    """Load signals JSON and extract layer 27 vs 31 data."""
    with open(SIGNALS_JSON) as f:
        data = json.load(f)

    results = data["all_results"]
    n_layers = data["n_layers"]

    # Layer indices (0-based): layer 27 = index 26, layer 31 = index 30
    L27_IDX = 26
    L31_IDX = 30

    correct = [r for r in results if r["is_correct"]]
    incorrect = [r for r in results if not r["is_correct"]]

    return {
        "results": results,
        "correct": correct,
        "incorrect": incorrect,
        "n_layers": n_layers,
        "L27_IDX": L27_IDX,
        "L31_IDX": L31_IDX,
    }


def load_checkpoint_weights():
    """Load trained head weights for importance analysis."""
    try:
        import torch
        cp = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        w = cp["state_dict"]["norm_shift_proj.0.weight"].float()
        importance = w.abs().sum(dim=0).numpy()
        temperature = cp["state_dict"]["temperature"].item()
        epoch = cp["epoch"]
        return importance, temperature, epoch
    except Exception as e:
        print(f"  WARNING: Could not load checkpoint: {e}")
        return None, None, None


def generate_matplotlib_figure(d):
    """Generate 3-panel matplotlib figure."""
    if not HAS_MPL:
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    L27 = d["L27_IDX"]
    L31 = d["L31_IDX"]

    # Extract per-example values
    c_l27 = np.array([r["norm_shift_signals"][L27] for r in d["correct"]])
    c_l31 = np.array([r["norm_shift_signals"][L31] for r in d["correct"]])
    i_l27 = np.array([r["norm_shift_signals"][L27] for r in d["incorrect"]])
    i_l31 = np.array([r["norm_shift_signals"][L31] for r in d["incorrect"]])

    # Full layer averages for bottom panel
    c_all = np.array([r["norm_shift_signals"] for r in d["correct"]])
    i_all = np.array([r["norm_shift_signals"] for r in d["incorrect"]])
    c_avg, c_std = c_all.mean(axis=0), c_all.std(axis=0)
    i_avg, i_std = i_all.mean(axis=0), i_all.std(axis=0)

    # Load checkpoint weights
    importance, temperature, epoch = load_checkpoint_weights()

    # Gap calculations
    gap_l27 = c_l27.mean() - i_l27.mean()
    gap_l31 = c_l31.mean() - i_l31.mean()

    # Create figure with explicit white background
    fig = plt.figure(figsize=(8, 11), facecolor='white')
    fig.patch.set_facecolor('white')
    gs = GridSpec(3, 1, height_ratios=[1, 1.2, 1], hspace=0.35)

    # Colors
    GREEN = "#228B22"
    RED = "#B22222"
    BLUE = "#1E90FF"
    CRIMSON = "#DC143C"

    # ── Panel 1: Density histograms ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('white')
    ax1.set_title("Norm-Shift Distribution: Layer 27 vs Layer 31", fontsize=12, fontweight="bold")

    bins_l27 = np.linspace(0.59, 0.68, 20)
    bins_l31 = np.linspace(0.31, 0.47, 20)

    # Layer 27 histograms
    ax1.hist(c_l27, bins=bins_l27, alpha=0.5, color=GREEN, edgecolor=GREEN,
             linewidth=0.8, label=f"L27 correct (mu={c_l27.mean():.3f})")
    ax1.hist(i_l27, bins=bins_l27, alpha=0.5, color=RED, edgecolor=RED,
             linewidth=0.8, label=f"L27 incorrect (mu={i_l27.mean():.3f})")

    # Layer 31 histograms
    ax1.hist(c_l31, bins=bins_l31, alpha=0.5, color=BLUE, edgecolor=BLUE,
             linewidth=0.8, label=f"L31 correct (mu={c_l31.mean():.3f})")
    ax1.hist(i_l31, bins=bins_l31, alpha=0.5, color=CRIMSON, edgecolor=CRIMSON,
             linewidth=0.8, label=f"L31 incorrect (mu={i_l31.mean():.3f})")

    # Gap annotations
    ax1.annotate(f"Delta = {gap_l27:.3f}", xy=(0.641, 0.92), xycoords="axes fraction",
                 fontsize=8, color="#555555")
    ax1.annotate(f"Delta = {gap_l31:.3f} ({gap_l31/gap_l27:.1f}x)", xy=(0.08, 0.92),
                 xycoords="axes fraction", fontsize=8, color="#555555")

    # Mean vertical lines
    ax1.axvline(c_l27.mean(), color=GREEN, linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.axvline(i_l27.mean(), color=RED, linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.axvline(c_l31.mean(), color=BLUE, linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.axvline(i_l31.mean(), color=CRIMSON, linestyle="--", linewidth=1.0, alpha=0.7)

    ax1.set_xlabel("Norm-shift signal s_i = 1 - std(h[i])", fontsize=10)
    ax1.set_ylabel("Count", fontsize=10)
    ax1.legend(fontsize=7, loc="upper center", ncol=2)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Scatter plot L27 vs L31 ─────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('white')
    ax2.set_title("Per-Example Norm-Shift: Layer 27 vs Layer 31", fontsize=12, fontweight="bold")

    ax2.scatter(c_l27, c_l31, c=GREEN, s=40, alpha=0.7, edgecolors="white",
                linewidth=0.5, marker="o", label=f"Correct (n={len(c_l27)})", zorder=5)
    ax2.scatter(i_l27, i_l31, c=RED, s=40, alpha=0.7, edgecolors="white",
                linewidth=0.5, marker="s", label=f"Incorrect (n={len(i_l27)})", zorder=5)

    # Mean crosshairs
    ax2.axvline(c_l27.mean(), color=GREEN, linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.axhline(c_l31.mean(), color=GREEN, linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.axvline(i_l27.mean(), color=RED, linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.axhline(i_l31.mean(), color=RED, linestyle="--", linewidth=0.8, alpha=0.5)

    # Annotate means
    ax2.annotate("Correct\ncluster", xy=(c_l27.mean(), c_l31.mean()),
                 xytext=(0.660, 0.44), fontsize=8, color=GREEN,
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.8))
    ax2.annotate("Incorrect\ncluster", xy=(i_l27.mean(), i_l31.mean()),
                 xytext=(0.610, 0.34), fontsize=8, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))

    # Gap annotations
    ax2.text(0.02, 0.98, f"L27 gap: delta_mu = {gap_l27:.4f}\n"
             f"L31 gap: delta_mu = {gap_l31:.4f} ({gap_l31/gap_l27:.1f}x wider)",
             transform=ax2.transAxes, fontsize=8, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    ax2.set_xlabel("Norm-shift s_27 (Layer 27)", fontsize=10)
    ax2.set_ylabel("Norm-shift s_31 (Layer 31)", fontsize=10)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Zoomed layer progression 25-32 ─────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor('white')
    ax3.set_title("Separation Widening: Layers 25-32", fontsize=12, fontweight="bold")

    layers = np.arange(25, 33)  # layers 25-32
    idx = layers - 1  # 0-based indices

    ax3.fill_between(layers, c_avg[idx] - c_std[idx], c_avg[idx] + c_std[idx],
                     color=GREEN, alpha=0.15)
    ax3.plot(layers, c_avg[idx], "o-", color=GREEN, linewidth=2.0, markersize=6,
             label="Correct (mean +/- 1 sigma)")
    ax3.fill_between(layers, i_avg[idx] - i_std[idx], i_avg[idx] + i_std[idx],
                     color=RED, alpha=0.15)
    ax3.plot(layers, i_avg[idx], "s-", color=RED, linewidth=2.0, markersize=6,
             label="Incorrect (mean +/- 1 sigma)")

    # Highlight layers 27 and 31
    ax3.axvspan(26.7, 27.3, color="#FFFFCC", alpha=0.5, zorder=0)
    ax3.axvspan(30.7, 31.3, color="#FFFFCC", alpha=0.5, zorder=0)

    # Difference annotations at L27 and L31
    diff_27 = c_avg[26] - i_avg[26]
    diff_31 = c_avg[30] - i_avg[30]
    mid_27 = (c_avg[26] + i_avg[26]) / 2
    mid_31 = (c_avg[30] + i_avg[30]) / 2
    ax3.annotate(f"Delta={diff_27:.3f}", xy=(27, mid_27), xytext=(27.5, mid_27 + 0.03),
                 fontsize=8, color="#555555",
                 arrowprops=dict(arrowstyle="->", color="#555555", lw=0.6))
    ax3.annotate(f"Delta={diff_31:.3f}", xy=(31, mid_31), xytext=(31.5, mid_31 + 0.03),
                 fontsize=8, color="#555555",
                 arrowprops=dict(arrowstyle="->", color="#555555", lw=0.6))

    # Weight importance overlay if available
    if importance is not None:
        ax3_twin = ax3.twinx()
        ax3_twin.bar(layers, importance[idx], width=0.3, alpha=0.25, color="#888888",
                     label="Head weight importance")
        ax3_twin.set_ylabel("Weight importance (sum|w|)", fontsize=9, color="#888888")
        ax3_twin.tick_params(axis="y", labelcolor="#888888")
        ax3_twin.legend(fontsize=7, loc="upper left")

    ax3.set_xlabel("Transformer Layer", fontsize=10)
    ax3.set_ylabel("s_i = 1 - std(h[i])", fontsize=10)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(layers)

    # ── Save ─────────────────────────────────────────────────────────────
    output_path = os.path.join(OUTPUT_DIR, "layer27_vs_31.pdf")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
    print(f"  Saved: {output_path}")

    plt.close(fig)
    return output_path


def print_summary(d):
    """Print key statistics for the comparison."""
    L27 = d["L27_IDX"]
    L31 = d["L31_IDX"]

    c_l27 = np.array([r["norm_shift_signals"][L27] for r in d["correct"]])
    c_l31 = np.array([r["norm_shift_signals"][L31] for r in d["correct"]])
    i_l27 = np.array([r["norm_shift_signals"][L27] for r in d["incorrect"]])
    i_l31 = np.array([r["norm_shift_signals"][L31] for r in d["incorrect"]])

    print()
    print("=== Layer 27 vs Layer 31: Key Statistics ===")
    print()
    print(f"  Layer 27 (index {L27}):")
    print(f"    Correct:   mu={c_l27.mean():.4f}, sigma={c_l27.std():.4f}, range=[{c_l27.min():.4f}, {c_l27.max():.4f}]")
    print(f"    Incorrect: mu={i_l27.mean():.4f}, sigma={i_l27.std():.4f}, range=[{i_l27.min():.4f}, {i_l27.max():.4f}]")
    print(f"    Gap:       delta_mu = {c_l27.mean() - i_l27.mean():.4f}")
    print()
    print(f"  Layer 31 (index {L31}):")
    print(f"    Correct:   mu={c_l31.mean():.4f}, sigma={c_l31.std():.4f}, range=[{c_l31.min():.4f}, {c_l31.max():.4f}]")
    print(f"    Incorrect: mu={i_l31.mean():.4f}, sigma={i_l31.std():.4f}, range=[{i_l31.min():.4f}, {i_l31.max():.4f}]")
    print(f"    Gap:       delta_mu = {c_l31.mean() - i_l31.mean():.4f}")
    print()
    ratio = (c_l31.mean() - i_l31.mean()) / (c_l27.mean() - i_l27.mean())
    print(f"  Separation amplification: {ratio:.1f}x from L27 to L31")
    print()

    # Cohen's d effect sizes
    pooled_27 = np.sqrt((c_l27.var() * (len(c_l27)-1) + i_l27.var() * (len(i_l27)-1)) / (len(c_l27) + len(i_l27) - 2))
    pooled_31 = np.sqrt((c_l31.var() * (len(c_l31)-1) + i_l31.var() * (len(i_l31)-1)) / (len(c_l31) + len(i_l31) - 2))
    d27 = (c_l27.mean() - i_l27.mean()) / pooled_27 if pooled_27 > 0 else 0
    d31 = (c_l31.mean() - i_l31.mean()) / pooled_31 if pooled_31 > 0 else 0
    print(f"  Effect sizes (Cohen's d):")
    print(f"    Layer 27: d = {d27:.2f}")
    print(f"    Layer 31: d = {d31:.2f}")
    print()

    importance, temperature, epoch_num = load_checkpoint_weights()
    if importance is not None:
        print(f"  Trained head weights (from checkpoint epoch {epoch_num}, T={temperature:.4f}):")
        print(f"    Layer 27 importance: {importance[L27]:.4f}")
        print(f"    Layer 31 importance: {importance[L31-1]:.4f}")
        print(f"    (for reference: max={importance.max():.4f} at layer {importance.argmax()+1})")
    print()


if __name__ == "__main__":
    print("=== Layer 27 vs 31 Comparison Figure Generation ===")
    print(f"  Signals: {SIGNALS_JSON}")
    print(f"  Checkpoint: {CHECKPOINT}")
    print()

    if not os.path.exists(SIGNALS_JSON):
        print(f"ERROR: {SIGNALS_JSON} not found.")
        print("Run extract_norm_signals.py first.")
        sys.exit(1)

    d = load_data()
    print(f"  Loaded {len(d['results'])} examples ({len(d['correct'])} correct, {len(d['incorrect'])} incorrect)")

    print_summary(d)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if HAS_MPL:
        print("-- Generating matplotlib figure --")
        path = generate_matplotlib_figure(d)
        if path:
            print(f"\n  Output: {path}")
    else:
        print("-- Skipping matplotlib (not installed) --")

    print()
    print("-- To generate LaTeX figures with gnuplot: --")
    print("  gnuplot docs/figures/scripts/layer27_vs_31.gp")
    print("  gnuplot docs/figures/scripts/layer27_vs_31_scatter.gp")
    print()
    print("Done.")
