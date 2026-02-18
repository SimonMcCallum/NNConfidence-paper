# Confidence Head Architecture

Technical documentation for the NormShift confidence heads in the HLCC framework.

## Overview

The confidence head predicts a calibrated confidence score in [0, 1] for each model prediction. Unlike verbalized or entropy-based confidence, the confidence head is a trained neural network that learns to interpret internal model signals — specifically the normalization dynamics across transformer layers.

Three confidence head architectures are available:

| Head | Input Signals | File | Class |
|------|--------------|------|-------|
| `MCQConfidenceHead` | Final hidden state only | `hlcc_mcq_system.py:462` | `MCQConfidenceHead` |
| `ConfidenceHead` | Hidden state + doubt signals (instrumented norms) | `hlcc_loss.py:204` | `ConfidenceHead` |
| `NormShiftConfidenceHead` | Hidden state + norm-shift signals (pre-trained) | `hlcc_loss.py` | `NormShiftConfidenceHead` |
| `NormShiftOnlyConfidenceHead` | Norm-shift signals only (ablation) | `hlcc_loss.py` | `NormShiftOnlyConfidenceHead` |

## The Normalization-Shift Signal

### Intuition

Transformer models apply RMSNorm or LayerNorm between layers. When a layer's output already has well-distributed activations (standard deviation close to 1), the norm applies minimal correction — the model is "settled." When the standard deviation deviates significantly from 1, the norm must make a large correction — the model may be "forcing" a commitment.

### Derivation

Given hidden states `h_i` at layer `i` with shape `(B, T, H)`:

```
signal_i = 1 - std(h_i, dim=-1)    # shape (B, T)
```

- **signal near 1**: hidden state already normalized (low std) — model processing smoothly
- **signal near 0 or negative**: hidden state has high variance — norm is making large corrections

We exclude the embedding layer output (`hidden_states[0]`) and process layers 1 through N, producing a feature vector of dimension `n_layers` per token.

### Implementation

```python
# hlcc_loss.py — extract_norm_shift_signals()
def extract_norm_shift_signals(hidden_states_tuple):
    layer_states = hidden_states_tuple[1:]  # skip embeddings
    signals = []
    for h in layer_states:
        std = h.float().std(dim=-1)
        shift = 1.0 - std
        signals.append(shift)
    return torch.stack(signals, dim=-1)  # (B, T, n_layers)
```

### Relationship to UncertaintyAwareLayerNorm

`UncertaintyAwareLayerNorm` (`hlcc_loss.py:139-197`) captures three doubt signals by instrumenting each norm layer during training:

1. Pre-norm variance
2. Correction scale (inverse std)
3. Pre-norm L2 norm

The norm-shift signal distills these into a single scalar per layer: `1 - std`, which correlates with the correction scale signal. The key advantage is that norm-shift signals can be extracted from **any pre-trained HuggingFace model** using `output_hidden_states=True`, without modifying the model architecture.

## Architecture

### NormShiftConfidenceHead (Combined)

Dual-path architecture that combines norm-shift dynamics with semantic content:

```
Norm-shift signals (n_layers)  ──→  Linear(n_layers, I) → GELU → Dropout
                                                                          ╲
                                                                           → Concat → Linear(2I, I) → GELU → Linear(I, 1) → σ(·/T)
                                                                          ╱
Final hidden state (H)         ──→  Linear(H, I)        → GELU → Dropout
```

Where `I = max(64, n_layers * 2)` and `T` is a learnable temperature parameter.

### NormShiftOnlyConfidenceHead (Ablation)

Tests whether norm-shift signals alone are sufficient:

```
Norm-shift signals (n_layers)  ──→  Linear(n_layers, I) → GELU → Dropout → Linear(I, I) → GELU → Dropout → Linear(I, 1) → σ(·/T)
```

This ablation variant has no access to the final hidden state's semantic content.

## Training Pipeline

### Process

1. **Freeze** the base LLM (all parameters frozen)
2. **Forward pass**: Run MCQ examples through the model with `output_hidden_states=True`
3. **Extract signals**: Pass the full hidden states tuple to the confidence head
4. **HLCC scoring**: Compute the asymmetric score:
   - Correct predictions: `R(c) = 1 + c` (reward confident correct answers)
   - Incorrect predictions: `P(c) = -2c^2` (heavily penalize confident wrong answers)
5. **Loss**: `loss = -mean(HLCC_score) + 0.1 * MSE(confidence, is_correct)`
6. **Update**: Only confidence head parameters are updated

### Usage

```bash
# Train both variants
python train_norm_shift_head.py --model qwen2.5-7b --variant both --epochs 25

# Train combined only
python train_norm_shift_head.py --model phi-2 --variant combined --epochs 10 --train-examples 500

# Load trained head for evaluation
python hlcc_mcq_system.py --model qwen2.5-7b --dataset truthfulqa --max-examples 100
# (pass norm_shift_head via code)
```

### Checkpoints

Checkpoints are saved to `data/checkpoints/` with fields:

```python
{
    "head_type": "combined" | "norm_shift_only",
    "n_layers": int,          # number of model layers
    "hidden_size": int,       # model hidden dimension
    "state_dict": OrderedDict,
    "epoch": int,
    "metrics": {...},
    "timestamp": str,
}
```

The `head_type`, `n_layers`, and `hidden_size` fields enable automatic architecture reconstruction via `load_norm_shift_head()`.

## Component Comparison

| Feature | MCQConfidenceHead | ConfidenceHead | NormShiftConfidenceHead | NormShiftOnlyConfidenceHead |
|---------|------------------|----------------|------------------------|----------------------------|
| Input | Hidden state | Hidden state + doubt signals | Hidden state + norm-shift | Norm-shift only |
| Requires model modification | No | Yes (UncertaintyAwareLayerNorm) | No | No |
| Works with pre-trained models | Yes | No | Yes | Yes |
| Captures norm dynamics | No | Yes (3 signals/layer) | Yes (1 signal/layer) | Yes (1 signal/layer) |
| Semantic content access | Yes | Yes | Yes | No |
| Parameters | ~H^2/16 | ~H^2/8 | ~n_layers*I + H*I | ~n_layers*I |

## Visualization

The confidence chat UI (`confidence_chat.html` served by `confidence_server.py`) provides real-time visualization of per-token confidence during generation. See the chat UI section in `README.md` for usage.
