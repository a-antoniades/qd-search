#!/usr/bin/env python3
"""QD Analysis for nanogpt-faithful-4gpu-pro experiment.

Analyzes the quality and diversity of an LLM-driven evolutionary search over
NanoGPT architecture modifications across 5 epochs (0-4), 86 evaluated entries.

Outputs:
  - 7 PDF figures in figures/
  - 4 CSV tables in data/
  - Summary statistics to stdout
"""

import json
import math
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
AIRS_ROOT = SCRIPT_DIR.parent.parent  # Automated-AI-Researcher/
REPO_ROOT = AIRS_ROOT.parent          # qd-search/
DB_PATH = AIRS_ROOT / "ideas_nanogpt_faithful_4gpu_pro" / "database.json"
FIG_DIR = SCRIPT_DIR / "figures"
DATA_DIR = SCRIPT_DIR / "data"

sys.path.insert(0, str(REPO_ROOT))
from qd.map_elites import GridArchive, Feature
from qd.metrics import coverage, qd_score, best_fitness

# ---------------------------------------------------------------------------
# SECTION 1: Taxonomy definitions
# ---------------------------------------------------------------------------

COMPONENT_NAMES = {
    0: "Normalization",
    1: "Attention",
    2: "MLP / Activation",
    3: "Positional Encoding",
    4: "Embedding",
    5: "Optimizer",
    6: "LR Schedule",
    7: "Init / Architecture",
}
NUM_COMPONENTS = len(COMPONENT_NAMES)

# Keywords for each component category (case-insensitive matching).
# Matched against the full idea text.
COMPONENT_KEYWORDS = {
    0: [  # Normalization
        "rmsnorm", "layernorm", "layer norm", "layer_norm", "sandwich norm",
        "normformer", "normalization layer", "learnable scaling to rmsnorm",
        "learnable rmsnorm", "trainable rmsnorm", "rmsnorm with bias",
        "head-wise rmsnorm", "nn.layernorm",
    ],
    1: [  # Attention
        "qk-norm", "qk norm", "query-key norm", "query key norm",
        "multi-query attention", "mqa", "gqa", "grouped query",
        "attention temp", "attention logit", "soft-capping", "soft capping",
        "hard capping", "hard clamp", "attention dropout",
        "bias in qkv", "c_attn", "attention head", "fewer heads",
        "head dimension", "value head", "expanded value",
        "n_head", "num_head",
    ],
    2: [  # MLP / Activation
        "swiglu", "reglu", "squared relu", "relu activation",
        "silu", "gelu", "activation function", "mlp variant",
        "gated linear", "mlp block", "relu", "swish",
    ],
    3: [  # Positional Encoding
        "rope", "rotary positional", "sinusoidal positional",
        "sinusoidal embed", "alibi", "positional embed",
        "positional encod", "learnable sinusoidal",
    ],
    4: [  # Embedding
        "vocab padding", "vocabulary padding", "untie embedding",
        "untied embedding", "untie weight", "untied weight",
        "embedding scaling", "embedding scale", "embedding norm",
        "embedding dropout", "embedding layer norm",
        "embedding rms", "embedding normalization",
        "ln_emb", "separate embedding learning rate",
        "embedding learning rate", "50304", "50257",
        "wte", "wpe",
    ],
    5: [  # Optimizer
        "adamw beta", "adam beta", "adamw epsilon", "adam epsilon",
        "fused adamw", "fused optimizer", "foreach adamw", "foreach optimizer",
        "weight decay", "optimizer config", "configure_optimizers",
    ],
    6: [  # LR Schedule
        "polynomial", "polynomial decay", "wsd", "warmup-stable-decay",
        "warmup stable decay", "inverse sqrt", "inverse square root",
        "learning rate schedule", "learning rate decay", "lr schedule",
        "lr decay", "cosine decay", "extended warmup", "warmup duration",
        "get_lr",
    ],
    7: [  # Init / Architecture
        "zero-init", "zero init", "zero initialization", "small initialization",
        "layerscale", "layer scale", "layer_scale", "residual scaling",
        "residual projection", "parallel block", "batch size", "micro-batch",
        "grad clip", "gradient clip", "tf32", "tensorfloat",
        "bias=false", "re-introduce bias", "linear bias",
        "logit scaling", "output scaling", "logit soft-capping",
        "output logit", "residual dropout", "c_proj",
        "gpt-2 initialization", "gpt2 initialization", "std=0.02",
    ],
}

# Disambiguation rules: override component assignment in specific contexts
# If keyword match overlaps, these rules resolve ambiguity.

TECHNIQUE_NAMES = {
    0: "Replacement",
    1: "Addition",
    2: "Removal",
    3: "Hyperparameter Tuning",
    4: "Throughput/Systems",
}
NUM_TECHNIQUES = len(TECHNIQUE_NAMES)

TECHNIQUE_KEYWORDS = {
    0: [  # Replacement
        "replace", "swap", "switch to", "instead of", "substitute",
    ],
    1: [  # Addition
        "add ", "introduce", "apply ", "enable", "include",
    ],
    2: [  # Removal
        "remove", "disable", "exclude", "drop ", "eliminate",
        "skip", "without",
    ],
    3: [  # Hyperparameter Tuning
        "tune", "change.*to", "increase.*from", "decrease.*from",
        "reduce.*to", "set.*to", "adjust", "beta1", "beta2",
        "epsilon", "weight_decay", "learning_rate", "warmup",
        "batch size", "0.9", "0.95", "0.98", "0.99", "0.85",
        "0.01", "0.1", "1e-5", "1e-8",
    ],
    4: [  # Throughput/Systems
        "fused", "foreach", "speed", "throughput", "tokens/sec",
        "tokens per second", "kernel", "cuda kernel", "tf32",
        "tensor core", "50304", "alignment",
    ],
}


# ---------------------------------------------------------------------------
# SECTION 2: Data loading
# ---------------------------------------------------------------------------

def load_database(path: Path) -> list[dict]:
    """Load database.json and return list of entry dicts."""
    with open(path) as f:
        entries = json.load(f)
    # Add a sequential index
    for i, entry in enumerate(entries):
        entry["index"] = i
    return entries


def extract_title(idea_text: str) -> str:
    """Extract a human-readable title from the idea text."""
    lines = idea_text.strip().split("\n")
    first_line = lines[0].strip()

    # Remove [Experiment] prefix
    title = re.sub(r"^\[Experiment\]\s*", "", first_line).strip()

    # If title is empty (just "[Experiment]"), try second line
    if not title and len(lines) > 1:
        title = lines[1].strip()
        # Remove numbering prefix like "1. " or "10. "
        title = re.sub(r"^\d+\.\s*", "", title).strip()

    # Remove [Code Changes] if it's appended
    title = re.sub(r"\[Code Changes\].*", "", title).strip()

    # Truncate very long titles
    if len(title) > 80:
        title = title[:77] + "..."

    return title if title else f"Idea (unnamed)"


# ---------------------------------------------------------------------------
# SECTION 3: Feature extraction
# ---------------------------------------------------------------------------

def classify_components(idea_text: str) -> list[int]:
    """Classify idea into component categories (multi-label).

    Returns list of component indices (0-7).
    """
    text_lower = idea_text.lower()
    matched = []

    for comp_id, keywords in COMPONENT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                matched.append(comp_id)
                break

    # Disambiguation: "embedding" near "positional" → Positional, not Embedding
    if 3 in matched and 4 in matched:
        # Check if the primary context is positional encoding
        pos_count = sum(1 for kw in COMPONENT_KEYWORDS[3] if kw in text_lower)
        emb_specific = any(
            kw in text_lower
            for kw in [
                "vocab", "untie", "embedding scaling", "embedding scale",
                "embedding dropout", "embedding norm", "ln_emb",
                "separate embedding learning rate", "50304", "50257",
            ]
        )
        if not emb_specific and pos_count >= 1:
            matched = [c for c in matched if c != 4]

    # If nothing matched, default to Init/Architecture (bin 7)
    if not matched:
        matched = [7]

    return sorted(set(matched))


def primary_component(components: list[int]) -> int:
    """Return the primary component (first in sorted list)."""
    return components[0] if components else 7


def classify_technique(idea_text: str) -> int:
    """Classify the technique type (single-label, 0-4)."""
    text_lower = idea_text.lower()
    scores = {}
    for tech_id, keywords in TECHNIQUE_KEYWORDS.items():
        count = 0
        for kw in keywords:
            if re.search(kw, text_lower):
                count += 1
        scores[tech_id] = count

    best_tech = max(scores, key=scores.get)
    if scores[best_tech] == 0:
        return 0  # Default to Replacement
    return best_tech


def normalize_concept_name(title: str) -> str:
    """Normalize title for concept clustering."""
    s = title.lower().strip()
    # Remove parenthetical details
    s = re.sub(r"\(.*?\)", "", s)
    # Remove numbers and special chars but keep key tokens
    s = re.sub(r"[0-9.]+", "", s)
    # Remove extra whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _concept_key(title: str) -> str:
    """Extract a canonical concept key from a title.

    Uses curated rules to map variant titles to the same key.
    This is more robust than pure fuzzy matching.
    """
    t = title.lower()

    # --- Normalization ---
    if any(k in t for k in ["learnable rmsnorm", "trainable rmsnorm",
                             "learnable scaling to rmsnorm", "learnable rmsnorm scales"]):
        return "learnable_rmsnorm"
    if "sandwich norm" in t:
        return "sandwich_norm"
    if "normformer" in t or "head-wise rmsnorm" in t:
        return "normformer_headwise_rmsnorm"
    if "rmsnorm with bias" in t:
        return "rmsnorm_with_bias"
    if ("layernorm" in t or "layer norm" in t or "standard layernorm" in t) and "embedding" not in t:
        return "standard_layernorm"

    # --- Attention ---
    if "qk-norm" in t or "qk norm" in t or "query-key norm" in t or "query key norm" in t:
        return "qk_norm"
    if "multi-query attention" in t or "mqa" in t:
        return "multi_query_attention"
    if "fewer heads" in t or ("larger dimension" in t and "head" in t):
        return "fewer_heads"
    if "expanded value" in t or ("value head" in t and "dimension" in t):
        return "expanded_value_head"
    if "attention temp" in t or "learnable attention temp" in t:
        return "learnable_attention_temperature"
    if "attention dropout" in t:
        return "attention_dropout"
    if "soft-capping" in t or "soft capping" in t or ("soft" in t and "cap" in t):
        return "attention_logit_softcapping"
    if "hard capping" in t or "hard clamp" in t:
        return "attention_logit_hardcapping"
    if "bias in qkv" in t or "enable bias in qkv" in t:
        return "bias_in_qkv"

    # --- MLP / Activation ---
    if "swiglu" in t:
        return "swiglu_mlp"
    if "reglu" in t:
        return "reglu_mlp"
    if "squared relu" in t:
        return "squared_relu"
    if "silu" in t or "swish" in t:
        return "silu_activation"
    if "relu" in t and "squared" not in t and "reglu" not in t:
        return "relu_activation"

    # --- Positional Encoding ---
    if "rope" in t or "rotary positional" in t or "rotary embed" in t:
        return "rope"
    if "sinusoidal" in t:
        return "sinusoidal_positional"
    if "alibi" in t:
        return "alibi"

    # --- Embedding ---
    if "vocab" in t and ("pad" in t or "50304" in t or "128" in t):
        return "vocab_padding"
    if "untie" in t or "untied" in t:
        return "untied_weights"
    if "embedding scaling" in t or "embedding scale" in t:
        return "embedding_scaling"
    if "embedding dropout" in t:
        return "embedding_dropout"
    if ("embedding norm" in t or "embedding layer norm" in t or
            "embedding rmsnorm" in t or "embedding rms" in t):
        return "embedding_normalization"
    if "separate embedding learning rate" in t or "embedding learning rate" in t:
        return "separate_embedding_lr"

    # --- Optimizer ---
    if "fused adamw" in t or "fused optimizer" in t:
        return "fused_adamw"
    if "foreach adamw" in t or "foreach optimizer" in t:
        return "foreach_adamw"
    if "beta2" in t or "beta 2" in t:
        return "adamw_beta2_tuning"
    if "beta1" in t or "beta 1" in t:
        return "adamw_beta1_tuning"
    if "adamw betas" in t:
        return "adamw_betas_tuning"
    if "epsilon" in t or "eps" in t:
        return "adamw_epsilon_tuning"
    if "weight decay" in t and "exclude" not in t and "smart" not in t:
        return "reduced_weight_decay"
    if "smart weight decay" in t or ("exclude" in t and "weight decay" in t):
        return "exclude_from_weight_decay"
    if "optimizer config" in t or "fix optimizer" in t:
        return "fix_optimizer_config"

    # --- LR Schedule ---
    if "polynomial" in t:
        return "polynomial_lr_decay"
    if "wsd" in t or "warmup-stable-decay" in t or "warmup stable decay" in t:
        return "wsd_schedule"
    if "inverse sq" in t or "inverse square" in t:
        return "inverse_sqrt_schedule"
    if "extended warmup" in t or "warmup duration" in t:
        return "extended_warmup"

    # --- Init / Architecture ---
    if "zero-init" in t or "zero init" in t or "zero initialization" in t:
        return "zero_init_projections"
    if "small initialization" in t or "std=0.01" in t:
        return "small_initialization"
    if "layerscale" in t or "layer scale" in t:
        return "layerscale"
    if "remove manual residual" in t:
        return "remove_residual_scaling"
    if "residual scaling" in t or "residual projection weight" in t:
        return "residual_scaling"
    if "parallel block" in t:
        return "parallel_blocks"
    if "batch size" in t or "micro-batch" in t:
        return "batch_size_tuning"
    if "gradient clip" in t and "disable" in t:
        return "disable_gradient_clipping"
    if "gradient clip" in t and ("value" in t or "clamp" in t):
        return "gradient_clipping_by_value"
    if "gradient clip" in t:
        return "disable_gradient_clipping"
    if "tf32" in t or "tensorfloat" in t:
        return "enable_tf32"
    if "re-introduce" in t and "bias" in t or "linear bias" in t:
        return "reintroduce_linear_biases"
    if "logit scaling" in t or "output scaling" in t or "output logit" in t:
        return "output_logit_scaling"
    if "residual dropout" in t:
        return "residual_dropout"
    if "gpt-2 initialization" in t or "gpt2 initialization" in t or "correct gpt" in t:
        return "gpt2_initialization"

    # Fallback: use normalized title
    return normalize_concept_name(title)


# Canonical display names for concept keys
CONCEPT_DISPLAY_NAMES = {
    "learnable_rmsnorm": "Learnable RMSNorm",
    "sandwich_norm": "Sandwich Norm",
    "normformer_headwise_rmsnorm": "Head-wise RMSNorm (NormFormer)",
    "rmsnorm_with_bias": "RMSNorm with Bias",
    "standard_layernorm": "Standard LayerNorm",
    "qk_norm": "QK-Norm",
    "multi_query_attention": "Multi-Query Attention",
    "fewer_heads": "Fewer Heads, Larger Dim",
    "expanded_value_head": "Expanded Value Head",
    "learnable_attention_temperature": "Learnable Attention Temperature",
    "attention_dropout": "Attention Dropout",
    "attention_logit_softcapping": "Attention Logit Soft-Capping",
    "attention_logit_hardcapping": "Attention Logit Hard-Capping",
    "bias_in_qkv": "Bias in QKV Projections",
    "swiglu_mlp": "SwiGLU MLP",
    "reglu_mlp": "ReGLU MLP",
    "squared_relu": "Squared ReLU",
    "silu_activation": "SiLU Activation",
    "relu_activation": "ReLU Activation",
    "rope": "Rotary Positional Embeddings (RoPE)",
    "sinusoidal_positional": "Sinusoidal Positional Embeddings",
    "vocab_padding": "Vocabulary Padding (50304)",
    "untied_weights": "Untied Embedding Weights",
    "embedding_scaling": "Embedding Scaling",
    "embedding_dropout": "Embedding Dropout",
    "embedding_normalization": "Embedding Normalization",
    "separate_embedding_lr": "Separate Embedding Learning Rate",
    "fused_adamw": "Fused AdamW",
    "foreach_adamw": "Foreach AdamW",
    "adamw_beta2_tuning": "AdamW Beta2 Tuning",
    "adamw_beta1_tuning": "AdamW Beta1 Tuning",
    "adamw_betas_tuning": "AdamW Betas Tuning",
    "adamw_epsilon_tuning": "AdamW Epsilon Tuning",
    "reduced_weight_decay": "Reduced Weight Decay",
    "exclude_from_weight_decay": "Exclude from Weight Decay",
    "fix_optimizer_config": "Fix Optimizer Config",
    "polynomial_lr_decay": "Polynomial LR Decay",
    "wsd_schedule": "WSD Schedule",
    "inverse_sqrt_schedule": "Inverse Sqrt Schedule",
    "extended_warmup": "Extended Warmup",
    "zero_init_projections": "Zero-Init Projections",
    "small_initialization": "Small Initialization",
    "layerscale": "LayerScale",
    "residual_scaling": "Residual Scaling",
    "remove_residual_scaling": "Remove Residual Scaling",
    "parallel_blocks": "Parallel Blocks",
    "batch_size_tuning": "Batch Size Tuning",
    "disable_gradient_clipping": "Disable Gradient Clipping",
    "gradient_clipping_by_value": "Gradient Clipping by Value",
    "enable_tf32": "Enable TF32",
    "reintroduce_linear_biases": "Re-introduce Linear Biases",
    "output_logit_scaling": "Output Logit Scaling",
    "residual_dropout": "Residual Dropout",
    "gpt2_initialization": "GPT-2 Initialization",
}


def cluster_concepts(entries: list[dict]) -> dict[str, list[int]]:
    """Cluster entries into unique concepts using curated keyword rules.

    Returns: {display_name: [list of entry indices]}
    """
    clusters: dict[str, list[int]] = defaultdict(list)

    for e in entries:
        key = _concept_key(e["title"])
        clusters[key].append(e["index"])

    # Convert keys to display names
    result = {}
    for key, members in clusters.items():
        display = CONCEPT_DISPLAY_NAMES.get(key, key.replace("_", " ").title())
        result[display] = members

    return result


# ---------------------------------------------------------------------------
# SECTION 4: Archive construction
# ---------------------------------------------------------------------------

def build_archive(entries: list[dict]) -> GridArchive:
    """Build an 8×5 GridArchive (Component × Technique) with negated val_loss."""
    archive = GridArchive(
        features=[
            Feature("component", min_val=0, max_val=NUM_COMPONENTS - 1, num_bins=NUM_COMPONENTS),
            Feature("technique", min_val=0, max_val=NUM_TECHNIQUES - 1, num_bins=NUM_TECHNIQUES),
        ]
    )

    for entry in entries:
        val_loss = entry["lowest_val_loss"]
        # Skip clearly broken entries
        if val_loss > 5.0:
            continue
        fitness = -val_loss  # Higher is better
        archive.add(
            id=str(entry["index"]),
            fitness=fitness,
            features={
                "component": float(entry["primary_component"]),
                "technique": float(entry["technique"]),
            },
        )
    return archive


# ---------------------------------------------------------------------------
# SECTION 5: Metrics computation
# ---------------------------------------------------------------------------

def compute_qd_metrics(archive: GridArchive) -> dict:
    """Compute standard QD metrics."""
    cov = coverage(archive)
    qd = qd_score(archive)
    bf = best_fitness(archive)

    occupied = archive.occupied_cells()
    val_losses = [-elite.fitness for _, elite in occupied]

    return {
        "coverage": cov,
        "coverage_pct": cov * 100,
        "qd_score": qd,
        "qd_score_val_loss": -qd,  # Converted back to val_loss space
        "best_fitness": bf,
        "best_val_loss": -bf,
        "total_cells": archive.cell_count(),
        "occupied_cells": archive.size,
        "mean_niche_val_loss": np.mean(val_losses) if val_losses else float("nan"),
        "median_niche_val_loss": np.median(val_losses) if val_losses else float("nan"),
    }


def compute_diversity_metrics(entries: list[dict], concepts: dict) -> dict:
    """Compute diversity metrics across all entries."""
    # Component frequency distribution
    comp_counts = Counter()
    for e in entries:
        for c in e["components"]:
            comp_counts[c] += 1

    # Shannon entropy
    total = sum(comp_counts.values())
    probs = [count / total for count in comp_counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(NUM_COMPONENTS)

    # Concept re-use ratio
    single_use = sum(1 for members in concepts.values() if len(members) == 1)
    multi_use = sum(1 for members in concepts.values() if len(members) > 1)
    total_concepts = len(concepts)
    reuse_ratio = multi_use / total_concepts if total_concepts > 0 else 0

    # Pairwise Jaccard distance on label sets
    label_sets = [set(e["components"]) for e in entries]
    n = len(label_sets)
    jaccard_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            intersection = len(label_sets[i] & label_sets[j])
            union = len(label_sets[i] | label_sets[j])
            jaccard_dists.append(1 - intersection / union if union > 0 else 1.0)

    return {
        "unique_concepts": total_concepts,
        "component_entropy": entropy,
        "max_entropy": max_entropy,
        "normalized_entropy": entropy / max_entropy if max_entropy > 0 else 0,
        "concept_reuse_ratio": reuse_ratio,
        "single_use_concepts": single_use,
        "multi_use_concepts": multi_use,
        "mean_jaccard_distance": np.mean(jaccard_dists) if jaccard_dists else 0,
    }


def compute_temporal_metrics(entries: list[dict], concepts: dict) -> pd.DataFrame:
    """Compute per-epoch temporal metrics."""
    epochs = sorted(set(e["epoch"] for e in entries))

    rows = []
    seen_concepts = set()
    cumulative_archive = GridArchive(
        features=[
            Feature("component", min_val=0, max_val=NUM_COMPONENTS - 1, num_bins=NUM_COMPONENTS),
            Feature("technique", min_val=0, max_val=NUM_TECHNIQUES - 1, num_bins=NUM_TECHNIQUES),
        ]
    )
    best_so_far = float("inf")

    # Map entry index to concept name
    idx_to_concept = {}
    for name, members in concepts.items():
        for idx in members:
            idx_to_concept[idx] = name

    for epoch in epochs:
        epoch_entries = [e for e in entries if e["epoch"] == epoch]
        valid = [e for e in epoch_entries if e["lowest_val_loss"] <= 5.0]

        losses = [e["lowest_val_loss"] for e in valid]
        epoch_best = min(losses) if losses else float("nan")
        best_so_far = min(best_so_far, epoch_best) if losses else best_so_far

        # Track new vs. repeated concepts
        epoch_concepts = set()
        for e in epoch_entries:
            concept = idx_to_concept.get(e["index"], e["title"])
            epoch_concepts.add(concept)
        new_concepts = epoch_concepts - seen_concepts
        seen_concepts.update(epoch_concepts)

        # Update cumulative archive
        for e in valid:
            cumulative_archive.add(
                id=str(e["index"]),
                fitness=-e["lowest_val_loss"],
                features={
                    "component": float(e["primary_component"]),
                    "technique": float(e["technique"]),
                },
            )

        cov = coverage(cumulative_archive)

        rows.append({
            "epoch": epoch,
            "total_ideas": len(epoch_entries),
            "valid_ideas": len(valid),
            "failed_ideas": len(epoch_entries) - len(valid),
            "epoch_best_val_loss": epoch_best,
            "epoch_mean_val_loss": np.mean(losses) if losses else float("nan"),
            "epoch_median_val_loss": np.median(losses) if losses else float("nan"),
            "best_so_far": best_so_far,
            "new_concepts": len(new_concepts),
            "repeated_concepts": len(epoch_concepts) - len(new_concepts),
            "cumulative_unique_concepts": len(seen_concepts),
            "cumulative_coverage": cov,
            "cumulative_occupied": cumulative_archive.size,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SECTION 6: Table generation
# ---------------------------------------------------------------------------

def make_concept_table(entries: list[dict], concepts: dict) -> pd.DataFrame:
    """Table 1: Concept Frequency Table."""
    rows = []
    for concept_name, member_indices in concepts.items():
        members = [e for e in entries if e["index"] in member_indices]
        losses = [e["lowest_val_loss"] for e in members if e["lowest_val_loss"] <= 5.0]
        epochs_appeared = sorted(set(e["epoch"] for e in members))
        comp_labels = set()
        for m in members:
            for c in m["components"]:
                comp_labels.add(COMPONENT_NAMES[c])

        rows.append({
            "concept": concept_name,
            "count": len(members),
            "epochs": ", ".join(str(ep) for ep in epochs_appeared),
            "components": ", ".join(sorted(comp_labels)),
            "best_val_loss": min(losses) if losses else float("nan"),
            "mean_val_loss": np.mean(losses) if losses else float("nan"),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("best_val_loss").reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"
    return df


def make_component_summary(entries: list[dict], concepts: dict) -> pd.DataFrame:
    """Table 2: Component Summary."""
    rows = []
    for comp_id in range(NUM_COMPONENTS):
        members = [e for e in entries if comp_id in e["components"]]
        valid = [e for e in members if e["lowest_val_loss"] <= 5.0]
        losses = [e["lowest_val_loss"] for e in valid]

        # Count unique concepts for this component
        comp_concepts = set()
        idx_to_concept = {}
        for name, member_indices in concepts.items():
            for idx in member_indices:
                idx_to_concept[idx] = name
        for m in members:
            comp_concepts.add(idx_to_concept.get(m["index"], m["title"]))

        # Top concept (lowest val_loss)
        if valid:
            best_entry = min(valid, key=lambda e: e["lowest_val_loss"])
            top_concept = idx_to_concept.get(best_entry["index"], best_entry["title"])
        else:
            top_concept = "N/A"

        rows.append({
            "component": COMPONENT_NAMES[comp_id],
            "idea_count": len(members),
            "unique_concepts": len(comp_concepts),
            "best_val_loss": min(losses) if losses else float("nan"),
            "mean_val_loss": np.mean(losses) if losses else float("nan"),
            "top_concept": top_concept,
        })

    return pd.DataFrame(rows)


def make_epoch_summary(temporal_df: pd.DataFrame) -> pd.DataFrame:
    """Table 3: Epoch Summary (already computed in temporal_metrics)."""
    return temporal_df[[
        "epoch", "total_ideas", "valid_ideas", "failed_ideas",
        "epoch_best_val_loss", "epoch_mean_val_loss",
        "new_concepts", "cumulative_unique_concepts", "cumulative_coverage",
    ]].copy()


def make_top20_table(entries: list[dict]) -> pd.DataFrame:
    """Table 4: Top-20 Ideas by val_loss."""
    valid = [e for e in entries if e["lowest_val_loss"] <= 5.0]
    valid_sorted = sorted(valid, key=lambda e: e["lowest_val_loss"])[:20]

    rows = []
    for rank, e in enumerate(valid_sorted, 1):
        rows.append({
            "rank": rank,
            "epoch": e["epoch"],
            "idea_id": e["idea_id"],
            "title": e["title"],
            "val_loss": e["lowest_val_loss"],
            "components": ", ".join(COMPONENT_NAMES[c] for c in e["components"]),
            "technique": TECHNIQUE_NAMES[e["technique"]],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SECTION 7: Figure generation
# ---------------------------------------------------------------------------

# Style configuration
CMAP_HEAT = "YlOrRd_r"  # Lower val_loss = better = darker warm color
CMAP_HEAT_REV = "YlOrRd"
FIG_DPI = 150

sns.set_theme(style="whitegrid", font_scale=1.1)


def fig1_archive_heatmap(entries: list[dict], archive: GridArchive) -> None:
    """Figure 1: 8×5 MAP-Elites archive heatmap."""
    # Build a full grid of best val_loss per cell
    grid = np.full((NUM_COMPONENTS, NUM_TECHNIQUES), np.nan)
    count_grid = np.zeros((NUM_COMPONENTS, NUM_TECHNIQUES), dtype=int)

    valid = [e for e in entries if e["lowest_val_loss"] <= 5.0]
    for e in valid:
        c = e["primary_component"]
        t = e["technique"]
        count_grid[c, t] += 1
        if np.isnan(grid[c, t]) or e["lowest_val_loss"] < grid[c, t]:
            grid[c, t] = e["lowest_val_loss"]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Mask NaN cells
    masked = np.ma.masked_invalid(grid)
    cmap = matplotlib.colormaps[CMAP_HEAT].copy()
    cmap.set_bad(color="#e0e0e0")

    vmin = np.nanmin(grid) if not np.all(np.isnan(grid)) else 3.69
    vmax = np.nanmax(grid) if not np.all(np.isnan(grid)) else 4.0

    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    # Annotate cells
    for i in range(NUM_COMPONENTS):
        for j in range(NUM_TECHNIQUES):
            if not np.isnan(grid[i, j]):
                text = f"{grid[i, j]:.4f}\n(n={count_grid[i, j]})"
                color = "white" if grid[i, j] < (vmin + vmax) / 2 else "black"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=8, fontweight="bold", color=color)
            else:
                ax.text(j, i, "empty", ha="center", va="center",
                        fontsize=8, color="#999999", fontstyle="italic")

    ax.set_xticks(range(NUM_TECHNIQUES))
    ax.set_xticklabels([TECHNIQUE_NAMES[i] for i in range(NUM_TECHNIQUES)],
                       rotation=30, ha="right")
    ax.set_yticks(range(NUM_COMPONENTS))
    ax.set_yticklabels([COMPONENT_NAMES[i] for i in range(NUM_COMPONENTS)])

    ax.set_xlabel("Technique Type")
    ax.set_ylabel("Component Category")
    ax.set_title("MAP-Elites Archive: Best Val Loss per Cell (8×5 Grid)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Best Val Loss (lower = better)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_archive_heatmap.pdf", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [1/7] Archive heatmap saved.")


def fig2_component_frequency(entries: list[dict]) -> None:
    """Figure 2: Component frequency bar chart, colored by mean val_loss."""
    # Component stats
    comp_data = {}
    for comp_id in range(NUM_COMPONENTS):
        members = [e for e in entries if comp_id in e["components"]]
        valid = [e for e in members if e["lowest_val_loss"] <= 5.0]
        losses = [e["lowest_val_loss"] for e in valid]
        comp_data[comp_id] = {
            "name": COMPONENT_NAMES[comp_id],
            "count": len(members),
            "mean_loss": np.mean(losses) if losses else float("nan"),
        }

    # Technique stats
    tech_data = {}
    for tech_id in range(NUM_TECHNIQUES):
        members = [e for e in entries if e["technique"] == tech_id]
        valid = [e for e in members if e["lowest_val_loss"] <= 5.0]
        losses = [e["lowest_val_loss"] for e in valid]
        tech_data[tech_id] = {
            "name": TECHNIQUE_NAMES[tech_id],
            "count": len(members),
            "mean_loss": np.mean(losses) if losses else float("nan"),
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left panel: components
    names = [comp_data[i]["name"] for i in range(NUM_COMPONENTS)]
    counts = [comp_data[i]["count"] for i in range(NUM_COMPONENTS)]
    mean_losses = [comp_data[i]["mean_loss"] for i in range(NUM_COMPONENTS)]

    norm = mcolors.Normalize(
        vmin=min(l for l in mean_losses if not np.isnan(l)),
        vmax=max(l for l in mean_losses if not np.isnan(l)),
    )
    cmap = matplotlib.colormaps[CMAP_HEAT]
    colors = [cmap(norm(l)) if not np.isnan(l) else "#cccccc" for l in mean_losses]

    y_pos = range(NUM_COMPONENTS)
    bars = ax1.barh(y_pos, counts, color=colors, edgecolor="gray", linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names)
    ax1.set_xlabel("Idea Count")
    ax1.set_title("Component Frequency")
    ax1.invert_yaxis()

    for bar, count, ml in zip(bars, counts, mean_losses):
        label = f"{count} (μ={ml:.3f})" if not np.isnan(ml) else f"{count}"
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 label, va="center", fontsize=9)

    # Right panel: techniques
    t_names = [tech_data[i]["name"] for i in range(NUM_TECHNIQUES)]
    t_counts = [tech_data[i]["count"] for i in range(NUM_TECHNIQUES)]
    t_losses = [tech_data[i]["mean_loss"] for i in range(NUM_TECHNIQUES)]

    t_norm = mcolors.Normalize(
        vmin=min(l for l in t_losses if not np.isnan(l)),
        vmax=max(l for l in t_losses if not np.isnan(l)),
    )
    t_colors = [cmap(t_norm(l)) if not np.isnan(l) else "#cccccc" for l in t_losses]

    t_pos = range(NUM_TECHNIQUES)
    bars2 = ax2.barh(t_pos, t_counts, color=t_colors, edgecolor="gray", linewidth=0.5)
    ax2.set_yticks(t_pos)
    ax2.set_yticklabels(t_names)
    ax2.set_xlabel("Idea Count")
    ax2.set_title("Technique Frequency")
    ax2.invert_yaxis()

    for bar, count, ml in zip(bars2, t_counts, t_losses):
        label = f"{count} (μ={ml:.3f})" if not np.isnan(ml) else f"{count}"
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 label, va="center", fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.6, pad=0.02)
    cbar.set_label("Mean Val Loss")

    fig.suptitle("Component & Technique Frequency (colored by mean val_loss)", fontsize=14, y=1.02)
    fig.savefig(FIG_DIR / "02_component_frequency.pdf", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [2/7] Component frequency chart saved.")


def fig3_valloss_distribution(entries: list[dict]) -> None:
    """Figure 3: Val_loss distribution per component (violin + points)."""
    valid = [e for e in entries if e["lowest_val_loss"] <= 5.0]

    # Build a long-form DataFrame
    rows = []
    for e in valid:
        for c in e["components"]:
            rows.append({
                "component": COMPONENT_NAMES[c],
                "component_id": c,
                "val_loss": e["lowest_val_loss"],
            })
    df = pd.DataFrame(rows)

    # Order by component id
    order = [COMPONENT_NAMES[i] for i in range(NUM_COMPONENTS) if COMPONENT_NAMES[i] in df["component"].values]

    fig, ax = plt.subplots(figsize=(14, 7))

    sns.violinplot(
        data=df, x="component", y="val_loss", order=order,
        inner=None, color="lightblue", alpha=0.5, ax=ax,
    )
    sns.stripplot(
        data=df, x="component", y="val_loss", order=order,
        size=5, alpha=0.7, jitter=0.2, ax=ax, color="darkblue",
    )

    # Baseline: median of all valid entries
    baseline = np.median([e["lowest_val_loss"] for e in valid])
    ax.axhline(baseline, color="red", linestyle="--", alpha=0.7, label=f"Global median: {baseline:.4f}")

    ax.set_xlabel("Component Category")
    ax.set_ylabel("Val Loss")
    ax.set_title("Validation Loss Distribution per Component")
    ax.legend()
    plt.xticks(rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_valloss_distribution.pdf", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [3/7] Val loss distribution saved.")


def fig4_coverage_over_epochs(temporal_df: pd.DataFrame) -> None:
    """Figure 4: Cumulative coverage & unique concepts vs. epoch."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "tab:blue"
    ax1.plot(temporal_df["epoch"], temporal_df["cumulative_coverage"] * 100,
             "o-", color=color1, linewidth=2, markersize=8, label="Coverage (%)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cumulative Archive Coverage (%)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.plot(temporal_df["epoch"], temporal_df["cumulative_unique_concepts"],
             "s--", color=color2, linewidth=2, markersize=8, label="Unique Concepts")
    ax2.set_ylabel("Cumulative Unique Concepts", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title("Coverage & Diversity Over Epochs")
    ax1.set_xticks(temporal_df["epoch"])

    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_coverage_over_epochs.pdf", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [4/7] Coverage over epochs saved.")


def fig5_fitness_trajectory(temporal_df: pd.DataFrame, entries: list[dict]) -> None:
    """Figure 5: Best / mean / median val_loss per epoch with success rate."""
    valid = [e for e in entries if e["lowest_val_loss"] <= 5.0]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(temporal_df["epoch"], temporal_df["epoch_best_val_loss"],
             "o-", color="green", linewidth=2, label="Best")
    ax1.plot(temporal_df["epoch"], temporal_df["epoch_mean_val_loss"],
             "s-", color="blue", linewidth=2, label="Mean")
    ax1.plot(temporal_df["epoch"], temporal_df["epoch_median_val_loss"],
             "^-", color="purple", linewidth=2, label="Median")
    ax1.plot(temporal_df["epoch"], temporal_df["best_so_far"],
             "D--", color="darkgreen", linewidth=1.5, alpha=0.7, label="Best-so-far")

    # Min-max shading per epoch
    epoch_min = []
    epoch_max = []
    for ep in temporal_df["epoch"]:
        ep_entries = [e for e in valid if e["epoch"] == ep]
        losses = [e["lowest_val_loss"] for e in ep_entries]
        epoch_min.append(min(losses) if losses else np.nan)
        epoch_max.append(max(losses) if losses else np.nan)

    ax1.fill_between(temporal_df["epoch"], epoch_min, epoch_max,
                     alpha=0.15, color="blue", label="Min-Max range")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val Loss")
    ax1.legend(loc="upper left")
    ax1.set_title("Fitness Trajectory Across Epochs")
    ax1.set_xticks(temporal_df["epoch"].values)

    # Secondary axis: success rate
    ax2 = ax1.twinx()
    success_rate = temporal_df["valid_ideas"] / temporal_df["total_ideas"] * 100
    ax2.bar(temporal_df["epoch"], success_rate, alpha=0.2, color="gray",
            width=0.4, label="Success Rate (%)")
    ax2.set_ylabel("Success Rate (%)", color="gray")
    ax2.set_ylim(0, 110)
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_fitness_trajectory.pdf", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [5/7] Fitness trajectory saved.")


def fig6_concept_freq_performance(entries: list[dict], concepts: dict) -> None:
    """Figure 6: Concept frequency vs. best val_loss (scatter)."""
    rows = []
    for concept_name, member_indices in concepts.items():
        members = [e for e in entries if e["index"] in member_indices]
        valid = [e for e in members if e["lowest_val_loss"] <= 5.0]
        if not valid:
            continue
        best_loss = min(e["lowest_val_loss"] for e in valid)
        # Primary component of first member
        pcomp = members[0]["primary_component"]
        rows.append({
            "concept": concept_name,
            "frequency": len(members),
            "best_val_loss": best_loss,
            "component_id": pcomp,
            "component": COMPONENT_NAMES[pcomp],
        })

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 8))

    palette = sns.color_palette("Set2", NUM_COMPONENTS)
    comp_colors = {COMPONENT_NAMES[i]: palette[i] for i in range(NUM_COMPONENTS)}

    for comp_name, group in df.groupby("component"):
        ax.scatter(group["frequency"], group["best_val_loss"],
                   label=comp_name, color=comp_colors[comp_name],
                   s=100, edgecolors="black", linewidth=0.5, zorder=5)

    # Label top-10 concepts
    top10 = df.nsmallest(10, "best_val_loss")
    for _, row in top10.iterrows():
        short_name = row["concept"][:35] + "..." if len(row["concept"]) > 35 else row["concept"]
        ax.annotate(
            short_name,
            (row["frequency"], row["best_val_loss"]),
            textcoords="offset points", xytext=(8, 4),
            fontsize=7, alpha=0.8,
        )

    ax.set_xlabel("Times Explored (Frequency)")
    ax.set_ylabel("Best Val Loss")
    ax.set_title("Concept Frequency vs. Performance")
    ax.legend(title="Component", bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8, title_fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_concept_freq_performance.pdf", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [6/7] Concept frequency vs. performance saved.")


def fig7_cooccurrence_matrix(entries: list[dict]) -> None:
    """Figure 7: Component co-occurrence matrix (8×8)."""
    cooccur = np.zeros((NUM_COMPONENTS, NUM_COMPONENTS), dtype=int)

    for e in entries:
        comps = e["components"]
        for i, c1 in enumerate(comps):
            for c2 in comps[i:]:
                cooccur[c1, c2] += 1
                if c1 != c2:
                    cooccur[c2, c1] += 1

    labels = [COMPONENT_NAMES[i] for i in range(NUM_COMPONENTS)]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Mask diagonal for off-diagonal focus (but still show it)
    mask = np.zeros_like(cooccur, dtype=bool)

    sns.heatmap(
        cooccur, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        mask=mask, ax=ax, linewidths=0.5,
        cbar_kws={"label": "Co-occurrence Count"},
    )

    ax.set_title("Component Co-occurrence Matrix")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_cooccurrence_matrix.pdf", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [7/7] Co-occurrence matrix saved.")


# ---------------------------------------------------------------------------
# SECTION 8: Main
# ---------------------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("QD Analysis: nanogpt-faithful-4gpu-pro")
    print("=" * 60)

    # --- Load data ---
    print("\n[1] Loading database...")
    entries = load_database(DB_PATH)
    print(f"  Loaded {len(entries)} entries from {DB_PATH.name}")

    # --- Classify ---
    print("\n[2] Classifying ideas...")
    for e in entries:
        e["title"] = extract_title(e["idea"])
        e["components"] = classify_components(e["idea"])
        e["primary_component"] = primary_component(e["components"])
        e["technique"] = classify_technique(e["idea"])

    # Verify classification coverage
    unclassified = [e for e in entries if not e["components"]]
    if unclassified:
        print(f"  WARNING: {len(unclassified)} entries unclassified!")
    else:
        print("  All entries classified (100% coverage).")

    # Print classification summary
    for comp_id in range(NUM_COMPONENTS):
        count = sum(1 for e in entries if comp_id in e["components"])
        print(f"    {COMPONENT_NAMES[comp_id]:25s}: {count:3d} ideas")

    # --- Cluster concepts ---
    print("\n[3] Clustering concepts...")
    concepts = cluster_concepts(entries)
    print(f"  Found {len(concepts)} unique concepts from {len(entries)} ideas.")

    # --- Build archive ---
    print("\n[4] Building MAP-Elites archive (8×5 grid)...")
    archive = build_archive(entries)
    print(f"  Archive: {archive}")

    # --- Compute metrics ---
    print("\n[5] Computing metrics...")
    qd_metrics = compute_qd_metrics(archive)
    diversity_metrics = compute_diversity_metrics(entries, concepts)
    temporal_df = compute_temporal_metrics(entries, concepts)

    print("\n  QD Metrics:")
    print(f"    Coverage:           {qd_metrics['coverage_pct']:.1f}% ({qd_metrics['occupied_cells']}/{qd_metrics['total_cells']} cells)")
    print(f"    QD Score (val_loss): {qd_metrics['qd_score_val_loss']:.4f}")
    print(f"    Best Val Loss:      {qd_metrics['best_val_loss']:.4f}")
    print(f"    Mean Niche Loss:    {qd_metrics['mean_niche_val_loss']:.4f}")
    print(f"    Median Niche Loss:  {qd_metrics['median_niche_val_loss']:.4f}")

    print("\n  Diversity Metrics:")
    print(f"    Unique Concepts:    {diversity_metrics['unique_concepts']}")
    print(f"    Component Entropy:  {diversity_metrics['component_entropy']:.3f} / {diversity_metrics['max_entropy']:.3f} (normalized: {diversity_metrics['normalized_entropy']:.3f})")
    print(f"    Concept Reuse:      {diversity_metrics['concept_reuse_ratio']:.1%} ({diversity_metrics['multi_use_concepts']} multi-use / {diversity_metrics['unique_concepts']} total)")
    print(f"    Mean Jaccard Dist:  {diversity_metrics['mean_jaccard_distance']:.3f}")

    # --- Generate tables ---
    print("\n[6] Generating tables...")
    concept_table = make_concept_table(entries, concepts)
    component_summary = make_component_summary(entries, concepts)
    epoch_summary = make_epoch_summary(temporal_df)
    top20_table = make_top20_table(entries)

    concept_table.to_csv(DATA_DIR / "01_concept_frequency.csv")
    component_summary.to_csv(DATA_DIR / "02_component_summary.csv", index=False)
    epoch_summary.to_csv(DATA_DIR / "03_epoch_summary.csv", index=False)
    top20_table.to_csv(DATA_DIR / "04_top20_ideas.csv", index=False)

    print("  Tables saved to data/")

    # Print tables
    print("\n  --- Table 1: Concept Frequency (top 15) ---")
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.width", 200)
    print(concept_table.head(15).to_string())

    print("\n  --- Table 2: Component Summary ---")
    print(component_summary.to_string(index=False))

    print("\n  --- Table 3: Epoch Summary ---")
    print(epoch_summary.to_string(index=False))

    print("\n  --- Table 4: Top-20 Ideas ---")
    print(top20_table.to_string(index=False))

    # --- Generate figures ---
    print("\n[7] Generating figures...")
    fig1_archive_heatmap(entries, archive)
    fig2_component_frequency(entries)
    fig3_valloss_distribution(entries)
    fig4_coverage_over_epochs(temporal_df)
    fig5_fitness_trajectory(temporal_df, entries)
    fig6_concept_freq_performance(entries, concepts)
    fig7_cooccurrence_matrix(entries)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"  Figures: {FIG_DIR}/")
    print(f"  Data:    {DATA_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
