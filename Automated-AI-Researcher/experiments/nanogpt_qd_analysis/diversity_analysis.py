#!/usr/bin/env python3
"""
Diversity Analysis of NanoGPT Pro (and Flash) Idea Generation
=============================================================
Analyzes concept frequency distribution, temporal novelty decay,
idea text similarity, coverage gaps, and Pro vs Flash comparison.
"""

import json
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from itertools import combinations

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path("/share/edc/home/antonis/qd-search/Automated-AI-Researcher")
PRO_DB = BASE / "runs/nanogpt_faithful_4gpu_pro/ideas/database.json"
FLASH_DB = BASE / "runs/nanogpt_faithful_4gpu/ideas/database.json"
ANALYSIS = BASE / "experiments/nanogpt_qd_analysis/data"
CONCEPT_CSV = ANALYSIS / "01_concept_frequency.csv"
COMPONENT_CSV = ANALYSIS / "02_component_summary.csv"
EPOCH_CSV = ANALYSIS / "03_epoch_summary.csv"
NOVELTY_CSV = ANALYSIS / "05_novelty_classification.csv"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. CONCEPT FREQUENCY DISTRIBUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def analyze_frequency_distribution():
    print("=" * 80)
    print("1. CONCEPT FREQUENCY DISTRIBUTION")
    print("=" * 80)

    concepts = load_csv(CONCEPT_CSV)
    counts = [int(c["count"]) for c in concepts]
    total = sum(counts)
    n = len(counts)

    # Basic stats
    print(f"\nTotal concepts: {n}")
    print(f"Total idea instances (sum of counts): {total}")
    print(f"Max count: {max(counts)} ({concepts[0]['concept']})")
    print(f"Min count: {min(counts)}")
    print(f"Mean count: {total / n:.2f}")
    print(f"Median count: {sorted(counts)[n // 2]}")

    # Count distribution
    count_dist = Counter(counts)
    print(f"\nCount distribution (count -> # concepts):")
    for c in sorted(count_dist.keys(), reverse=True):
        names = [row["concept"] for row in concepts if int(row["count"]) == c]
        print(f"  {c}x: {count_dist[c]} concepts — {', '.join(names[:5])}{'...' if len(names) > 5 else ''}")

    # Singletons vs repeated
    singletons = sum(1 for c in counts if c == 1)
    print(f"\nSingletons (count=1): {singletons}/{n} ({100 * singletons / n:.1f}%)")
    print(f"Repeated (count>=2): {n - singletons}/{n} ({100 * (n - singletons) / n:.1f}%)")

    # Top-10 concepts account for what fraction?
    top10_total = sum(counts[:10])
    print(f"\nTop 10 concepts account for: {top10_total}/{total} instances ({100 * top10_total / total:.1f}%)")
    top5_total = sum(counts[:5])
    print(f"Top 5 concepts account for: {top5_total}/{total} instances ({100 * top5_total / total:.1f}%)")

    # Gini coefficient
    sorted_counts = sorted(counts)
    cum = 0
    gini_sum = 0
    for i, c in enumerate(sorted_counts):
        cum += c
        gini_sum += cum
    gini = 1 - 2 * gini_sum / (n * total) + 1 / n
    # More standard formula
    sorted_c = sorted(counts)
    gini2 = sum(abs(sorted_c[i] - sorted_c[j]) for i in range(n) for j in range(n)) / (2 * n * total)
    print(f"\nGini coefficient: {gini2:.4f}")
    print(f"  (0 = perfectly equal, 1 = maximally concentrated)")

    # Shannon entropy
    probs = [c / total for c in counts]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(n)
    print(f"\nShannon entropy: {entropy:.4f} bits")
    print(f"Max possible entropy (uniform): {max_entropy:.4f} bits")
    print(f"Normalized entropy (evenness): {entropy / max_entropy:.4f}")
    print(f"  (1.0 = perfectly uniform, 0.0 = all mass on one concept)")

    # Effective number of concepts (exponential of entropy)
    effective_n = 2 ** entropy
    print(f"\nEffective number of concepts (2^H): {effective_n:.1f} out of {n}")
    print(f"  (How many 'equally-used' concepts would give same entropy)")

    return concepts, counts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. TEMPORAL ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def analyze_temporal():
    print("\n" + "=" * 80)
    print("2. TEMPORAL ANALYSIS — NEW CONCEPTS PER EPOCH & MODE COLLAPSE")
    print("=" * 80)

    epoch_data = load_csv(EPOCH_CSV)
    pro_ideas = load_json(PRO_DB)

    print(f"\n{'Epoch':>5} | {'Ideas':>5} | {'New':>4} | {'Cumul':>5} | {'Cov%':>5} | {'New%':>5} | {'NewRate':>8}")
    print("-" * 65)

    prev_cum = 0
    for row in epoch_data:
        epoch = int(row["epoch"])
        total = int(row["total_ideas"])
        new = int(row["new_concepts"])
        cumul = int(row["cumulative_unique_concepts"])
        cov = float(row["cumulative_coverage"])
        new_pct = 100 * new / total if total > 0 else 0
        # New concepts per idea
        new_rate = new / total if total > 0 else 0
        print(f"  {epoch:>3}  | {total:>5} | {new:>4} | {cumul:>5} | {cov * 100:>5.1f} | {new_pct:>5.1f} | {new_rate:>8.3f}")
        prev_cum = cumul

    # Concept novelty decay rate
    new_concepts = [int(row["new_concepts"]) for row in epoch_data]
    total_ideas_per_epoch = [int(row["total_ideas"]) for row in epoch_data]
    novelty_rates = [n / t if t > 0 else 0 for n, t in zip(new_concepts, total_ideas_per_epoch)]

    print(f"\nNovelty rate (new concepts / ideas generated):")
    print(f"  Epoch 0: {novelty_rates[0]:.3f} (all new by definition)")
    print(f"  Epochs 1-3: {sum(novelty_rates[1:4]) / 3:.3f} (avg)")
    print(f"  Epochs 4-7: {sum(novelty_rates[4:]) / len(novelty_rates[4:]):.3f} (avg)")

    # Concept repetition analysis
    print(f"\n--- Concept Repetition Across Epochs ---")
    concepts_csv = load_csv(CONCEPT_CSV)
    multi_epoch_concepts = []
    for c in concepts_csv:
        epochs_str = c["epochs"]
        epochs_list = [int(e.strip()) for e in epochs_str.split(",")]
        if len(epochs_list) > 1:
            multi_epoch_concepts.append((c["concept"], int(c["count"]), epochs_list))

    multi_epoch_concepts.sort(key=lambda x: -x[1])
    print(f"\nConcepts appearing in multiple epochs ({len(multi_epoch_concepts)} total):")
    for name, count, epochs in multi_epoch_concepts[:15]:
        span = max(epochs) - min(epochs)
        print(f"  {name:<45} count={count}  epochs={epochs}  span={span}")

    # Is there convergence? Measure epoch-to-epoch overlap
    print(f"\n--- Epoch-to-Epoch Concept Overlap ---")

    # Build per-epoch concept sets from the database
    epoch_concepts = defaultdict(set)
    for idea in pro_ideas:
        epoch = idea["epoch"]
        text = idea["idea"].lower()
        # Match concept names from the CSV
        for c in concepts_csv:
            cname = c["concept"].lower()
            # Check if concept name (or key words) appear in the idea
            if cname in text or any(w in text for w in cname.split()[:2] if len(w) > 4):
                epoch_concepts[epoch].add(c["concept"])

    # Use the epoch data from CSV directly (more reliable)
    epoch_concept_sets = defaultdict(set)
    for c in concepts_csv:
        epochs_list = [int(e.strip()) for e in c["epochs"].split(",")]
        for e in epochs_list:
            epoch_concept_sets[e].add(c["concept"])

    for e in sorted(epoch_concept_sets.keys()):
        if e == 0:
            continue
        overlap = epoch_concept_sets[e] & epoch_concept_sets[e - 1]
        new_in_epoch = epoch_concept_sets[e] - set().union(*[epoch_concept_sets[i] for i in range(e)])
        pct_overlap = 100 * len(overlap) / len(epoch_concept_sets[e]) if epoch_concept_sets[e] else 0
        print(f"  Epoch {e}: {len(epoch_concept_sets[e])} concepts, "
              f"{len(overlap)} overlap with prev ({pct_overlap:.0f}%), "
              f"{len(new_in_epoch)} genuinely new")

    # Mode collapse detection
    print(f"\n--- Mode Collapse Indicators ---")
    # Check: do later epochs converge to a smaller set of concepts?
    for e in sorted(epoch_concept_sets.keys()):
        n_concepts = len(epoch_concept_sets[e])
        n_ideas = int(epoch_data[e]["total_ideas"]) if e < len(epoch_data) else 0
        ratio = n_concepts / n_ideas if n_ideas > 0 else 0
        print(f"  Epoch {e}: {n_concepts} unique concepts in {n_ideas} ideas (diversity ratio: {ratio:.2f})")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. IDEA TEXT SIMILARITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def jaccard_similarity(text1, text2):
    """Word-level Jaccard similarity between two texts."""
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


def trigram_similarity(text1, text2):
    """Character trigram Jaccard similarity."""
    def trigrams(s):
        s = s.lower()
        return set(s[i:i + 3] for i in range(len(s) - 2))
    t1 = trigrams(text1)
    t2 = trigrams(text2)
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def analyze_text_similarity():
    print("\n" + "=" * 80)
    print("3. IDEA TEXT SIMILARITY ANALYSIS")
    print("=" * 80)

    pro_ideas = load_json(PRO_DB)

    # Group by epoch
    by_epoch = defaultdict(list)
    for idea in pro_ideas:
        by_epoch[idea["epoch"]].append(idea)

    # Sample ideas from epoch 0, 4, 7
    sample_epochs = [0, 4, 7]
    print(f"\n--- Sample Ideas from Epochs {sample_epochs} ---")
    for ep in sample_epochs:
        ideas = by_epoch[ep]
        print(f"\nEpoch {ep} ({len(ideas)} ideas total):")
        for idea in ideas[:3]:
            title = idea["idea"].split("\n")[0][:100]
            loss = idea.get("lowest_val_loss", "N/A")
            print(f"  [{idea['idea_id']}] {title}  (val_loss={loss})")

    # Intra-epoch similarity
    print(f"\n--- Intra-Epoch Text Similarity (word-level Jaccard) ---")
    print(f"{'Epoch':>5} | {'N':>3} | {'Mean':>6} | {'Max':>6} | {'Min':>6}")
    print("-" * 40)
    for ep in sorted(by_epoch.keys()):
        ideas = by_epoch[ep]
        texts = [i["idea"] for i in ideas]
        if len(texts) < 2:
            continue
        sims = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sims.append(jaccard_similarity(texts[i], texts[j]))
        print(f"  {ep:>3}  | {len(texts):>3} | {sum(sims) / len(sims):.4f} | {max(sims):.4f} | {min(sims):.4f}")

    # Cross-epoch similarity: how similar are later epoch ideas to earlier ones?
    print(f"\n--- Cross-Epoch Similarity (mean Jaccard of each idea vs ALL prior ideas) ---")
    all_prior = []
    for ep in sorted(by_epoch.keys()):
        if ep == 0:
            all_prior.extend([i["idea"] for i in by_epoch[ep]])
            print(f"  Epoch 0: (baseline, no prior ideas)")
            continue
        current_ideas = [i["idea"] for i in by_epoch[ep]]
        sims = []
        for curr in current_ideas:
            max_sim = max(jaccard_similarity(curr, prior) for prior in all_prior)
            sims.append(max_sim)
        mean_max = sum(sims) / len(sims)
        print(f"  Epoch {ep}: mean max-sim to prior = {mean_max:.4f} (N={len(current_ideas)})")
        all_prior.extend(current_ideas)

    # Find most similar idea pairs across epochs (duplicates)
    print(f"\n--- Most Similar Cross-Epoch Idea Pairs (Jaccard > 0.5) ---")
    all_ideas = [(i["epoch"], i["idea_id"], i["idea"]) for i in pro_ideas]
    high_sim_pairs = []
    for i in range(len(all_ideas)):
        for j in range(i + 1, len(all_ideas)):
            if all_ideas[i][0] == all_ideas[j][0]:
                continue  # skip same-epoch
            sim = jaccard_similarity(all_ideas[i][2], all_ideas[j][2])
            if sim > 0.5:
                high_sim_pairs.append((sim, all_ideas[i], all_ideas[j]))
    high_sim_pairs.sort(reverse=True)
    for sim, (e1, id1, t1), (e2, id2, t2) in high_sim_pairs[:10]:
        title1 = t1.split("\n")[0][:60]
        title2 = t2.split("\n")[0][:60]
        print(f"  J={sim:.3f}  Ep{e1}#{id1} vs Ep{e2}#{id2}")
        print(f"    {title1}")
        print(f"    {title2}")

    # Idea length analysis by epoch
    print(f"\n--- Idea Length by Epoch (characters) ---")
    for ep in sorted(by_epoch.keys()):
        lengths = [len(i["idea"]) for i in by_epoch[ep]]
        print(f"  Epoch {ep}: mean={sum(lengths) / len(lengths):.0f}, "
              f"min={min(lengths)}, max={max(lengths)}")

    # Do later ideas reference earlier concepts?
    print(f"\n--- Do Later Ideas Reference/Build On Earlier Ones? ---")
    reference_words = ["improve", "combine", "extend", "build on", "augment", "modify",
                       "previous", "earlier", "from experiment", "variant", "alternative"]
    for ep in sorted(by_epoch.keys()):
        ideas = by_epoch[ep]
        ref_count = 0
        for i in ideas:
            text = i["idea"].lower()
            if any(w in text for w in reference_words):
                ref_count += 1
        print(f"  Epoch {ep}: {ref_count}/{len(ideas)} ideas contain reference/build-on language")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. COVERAGE GAPS (8×5 grid analysis)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def analyze_coverage_gaps():
    print("\n" + "=" * 80)
    print("4. COVERAGE GAPS — 8-COMPONENT ANALYSIS")
    print("=" * 80)

    components = load_csv(COMPONENT_CSV)
    concepts = load_csv(CONCEPT_CSV)
    novelty = load_csv(NOVELTY_CSV)

    # Component summary
    print(f"\n--- Component Coverage ---")
    print(f"{'Component':<25} | {'Ideas':>5} | {'Concepts':>8} | {'Best Loss':>9} | {'Top Concept'}")
    print("-" * 90)
    for c in components:
        print(f"  {c['component']:<23} | {c['idea_count']:>5} | {c['unique_concepts']:>8} | "
              f"{c['best_val_loss']:>9} | {c['top_concept']}")

    # Total ideas per component
    total_component_ideas = sum(int(c["idea_count"]) for c in components)
    print(f"\nTotal component-idea assignments: {total_component_ideas}")
    print(f"(Ideas can map to multiple components, so this exceeds 128)")

    # Identify under-explored components
    print(f"\n--- Under-Explored Components ---")
    comp_counts = {c["component"]: int(c["idea_count"]) for c in components}
    mean_count = sum(comp_counts.values()) / len(comp_counts)
    for comp, count in sorted(comp_counts.items(), key=lambda x: x[1]):
        ratio = count / mean_count
        status = "UNDER" if ratio < 0.5 else "LOW" if ratio < 0.8 else "OK"
        print(f"  {comp:<25} {count:>3} ideas ({ratio:.2f}x mean) [{status}]")

    # What's NEVER explored?
    print(f"\n--- Techniques NEVER Explored ---")
    never_explored = [
        ("Custom CUDA Kernels", "Init / Architecture", "Custom fused ops, efficient implementations"),
        ("New/Custom Loss Functions", "Init / Architecture", "Beyond cross-entropy: contrastive, distillation, focal"),
        ("Data Augmentation", "Embedding", "Token-level augmentation, cutout, mixup for text"),
        ("Curriculum Learning", "LR Schedule", "Easy-to-hard ordering of training data"),
        ("Mixture of Experts (MoE)", "MLP / Activation", "Sparse gating over multiple MLPs"),
        ("Sparse Attention", "Attention", "Local attention, sliding window, BigBird patterns"),
        ("Knowledge Distillation", "Init / Architecture", "Teacher-student training"),
        ("Gradient Accumulation Tricks", "Optimizer", "Gradient compression, delayed updates"),
        ("Architecture Search/Scaling", "Init / Architecture", "Depth vs width tradeoffs, layer dropping"),
        ("Memory/KV Cache Optimization", "Attention", "Paged attention, compressed KV"),
        ("Tokenizer Changes", "Embedding", "BPE modifications, subword regularization"),
        ("Multi-Task/Auxiliary Losses", "Init / Architecture", "Intermediate layer prediction, denoising"),
        ("Activation Checkpointing", "Init / Architecture", "Memory-compute tradeoff for larger batches"),
        ("Dynamic Learning Rate", "LR Schedule", "Cyclical LR, SGDR, loss-adaptive LR"),
        ("Normalization-Free Architectures", "Normalization", "NFNets-style signal propagation"),
    ]

    # Check which of these appear in the actual ideas using precise multi-word phrases
    pro_ideas = load_json(PRO_DB)
    all_text = " ".join(i["idea"].lower() for i in pro_ideas)

    # Precise phrase checks — avoid false positives from generic words
    precise_checks = {
        "Custom CUDA Kernels": ["custom cuda", "triton kernel", "custom kernel", "write a kernel"],
        "New/Custom Loss Functions": ["contrastive loss", "focal loss", "distillation loss", "custom loss", "new loss function"],
        "Data Augmentation": ["data augment", "token augment", "mixup", "cutout", "back-translation"],
        "Curriculum Learning": ["curriculum"],
        "Mixture of Experts (MoE)": ["mixture of expert", "moe ", "sparse gating", "expert routing"],
        "Sparse Attention": ["sparse attention", "local attention", "sliding window", "bigbird", "longformer"],
        "Knowledge Distillation": ["distillation", "teacher-student", "teacher model"],
        "Gradient Accumulation Tricks": ["gradient compression", "gradient quantiz", "delayed update"],
        "Architecture Search/Scaling": ["architecture search", "nas ", "neural architecture search", "depth vs width"],
        "Memory/KV Cache Optimization": ["kv cache", "paged attention", "compressed kv"],
        "Tokenizer Changes": ["tokenizer change", "bpe modif", "subword regular", "sentencepiece"],
        "Multi-Task/Auxiliary Losses": ["multi-task", "auxiliary loss", "intermediate prediction"],
        "Activation Checkpointing": ["activation checkpoint", "gradient checkpoint"],
        "Dynamic Learning Rate": ["cyclical lr", "sgdr", "loss-adaptive lr", "reduce on plateau"],
        "Normalization-Free Architectures": ["normalization-free", "nfnet", "signal propagation"],
    }

    print(f"{'Technique':<35} | {'Category':<20} | {'Found?':>6} | Description")
    print("-" * 105)
    for tech, cat, desc in never_explored:
        phrases = precise_checks.get(tech, [])
        found = any(phrase in all_text for phrase in phrases)
        status = "YES" if found else "NO"
        print(f"  {tech:<33} | {cat:<20} | {status:>6} | {desc}")

    # Novelty analysis
    print(f"\n--- Novelty Classification Summary ---")
    novelty_counts = Counter(c["novelty_label"] for c in novelty)
    for label, count in novelty_counts.most_common():
        pct = 100 * count / len(novelty)
        print(f"  {label}: {count} ({pct:.1f}%)")

    # Average novelty score
    scores = [int(c["novelty_score"]) for c in novelty]
    print(f"\n  Mean novelty score: {sum(scores) / len(scores):.2f}")
    print(f"  Max novelty score: {max(scores)}")
    print(f"  Score distribution: {dict(Counter(scores))}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. PRO vs FLASH COMPARISON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def normalize_concept_name(text):
    """Extract a rough concept label from idea text."""
    text = text.lower()
    # Key concept keywords to match
    concept_map = {
        "rope": "RoPE",
        "rotary": "RoPE",
        "swiglu": "SwiGLU",
        "silu": "SiLU",
        "gelu": "GELU variant",
        "relu": "ReLU variant",
        "mish": "Mish",
        "rmsnorm": "RMSNorm variant",
        "layernorm": "LayerNorm",
        "fused adamw": "Fused AdamW",
        "fused=true": "Fused AdamW",
        "foreach": "Foreach AdamW",
        "weight decay": "Weight Decay tuning",
        "beta2": "AdamW Beta tuning",
        "beta1": "AdamW Beta tuning",
        "betas": "AdamW Beta tuning",
        "vocabulary padding": "Vocab Padding",
        "50304": "Vocab Padding",
        "zero-init": "Zero-Init Projections",
        "zeros_(p)": "Zero-Init Projections",
        "qk-norm": "QK-Norm",
        "qk norm": "QK-Norm",
        "query-key norm": "QK-Norm",
        "query and key": "QK-Norm",
        "logit soft-cap": "Logit Soft-Capping",
        "soft-cap": "Logit Soft-Capping",
        "embedding norm": "Embedding Normalization",
        "sinusoidal": "Sinusoidal Positional",
        "cosine schedule": "LR Schedule variant",
        "warmup": "LR Schedule variant",
        "wsd": "WSD Schedule",
        "polynomial": "Polynomial LR",
        "inverse sqrt": "Inverse Sqrt LR",
        "dropout": "Dropout variant",
        "label smooth": "Label Smoothing",
        "parallel block": "Parallel Blocks",
        "reglu": "ReGLU",
        "grouped query": "GQA",
        "multi-query": "MQA",
        "output logit scal": "Output Logit Scaling",
        "learnable scale": "Learnable Norm Scale",
        "learnable attention temp": "Learnable Attn Temp",
        "batch size": "Batch Size Tuning",
        "embedding scal": "Embedding Scaling",
        "gradient clip": "Gradient Clipping variant",
        "xavier": "Xavier Init",
        "orthogonal": "Orthogonal Init",
        "sandwich norm": "Sandwich Norm",
        "layerscale": "LayerScale",
        "input jitter": "Input Jitter",
        "token shift": "Token Shifting",
        "z-loss": "Z-Loss",
        "amsgrad": "AMSGrad",
        "radam": "RAdam",
        "elu": "ELU",
        "squared relu": "Squared ReLU",
        "logit temperature": "Logit Temperature",
        "logit_scale": "Logit Temperature",
        "gradient-norm": "Gradient-Norm WD",
        "gradient norm": "Gradient-Norm WD",
    }

    for keyword, concept in concept_map.items():
        if keyword in text:
            return concept
    return "Other"


def analyze_pro_vs_flash():
    print("\n" + "=" * 80)
    print("5. PRO vs FLASH COMPARISON")
    print("=" * 80)

    pro_ideas = load_json(PRO_DB)
    flash_ideas = load_json(FLASH_DB)

    print(f"\n--- Basic Stats ---")
    print(f"{'Metric':<35} | {'Pro':>8} | {'Flash':>8}")
    print("-" * 60)
    print(f"  {'Total ideas':<33} | {len(pro_ideas):>8} | {len(flash_ideas):>8}")
    print(f"  {'Epochs':<33} | {len(set(i['epoch'] for i in pro_ideas)):>8} | {len(set(i['epoch'] for i in flash_ideas)):>8}")

    # Validate losses
    pro_valid = [i for i in pro_ideas if i.get("lowest_val_loss") and i["lowest_val_loss"] < 10]
    flash_valid = [i for i in flash_ideas if i.get("lowest_val_loss") and i["lowest_val_loss"] < 10]
    print(f"  {'Valid ideas (loss < 10)':<33} | {len(pro_valid):>8} | {len(flash_valid):>8}")

    pro_losses = [i["lowest_val_loss"] for i in pro_valid]
    flash_losses = [i["lowest_val_loss"] for i in flash_valid]
    print(f"  {'Best val loss':<33} | {min(pro_losses):>8.4f} | {min(flash_losses):>8.4f}")
    print(f"  {'Mean val loss':<33} | {sum(pro_losses) / len(pro_losses):>8.4f} | {sum(flash_losses) / len(flash_losses):>8.4f}")
    print(f"  {'Median val loss':<33} | {sorted(pro_losses)[len(pro_losses) // 2]:>8.4f} | {sorted(flash_losses)[len(flash_losses) // 2]:>8.4f}")

    # Concept diversity comparison
    pro_concepts = Counter(normalize_concept_name(i["idea"]) for i in pro_ideas)
    flash_concepts = Counter(normalize_concept_name(i["idea"]) for i in flash_ideas)

    print(f"\n--- Concept Diversity ---")
    print(f"  {'Unique concepts (rough match)':<33} | {len(pro_concepts):>8} | {len(flash_concepts):>8}")

    # Entropy
    def compute_entropy(counter):
        total = sum(counter.values())
        probs = [c / total for c in counter.values()]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    pro_H = compute_entropy(pro_concepts)
    flash_H = compute_entropy(flash_concepts)
    print(f"  {'Shannon entropy (bits)':<33} | {pro_H:>8.3f} | {flash_H:>8.3f}")
    print(f"  {'Max entropy':<33} | {math.log2(len(pro_concepts)):>8.3f} | {math.log2(len(flash_concepts)):>8.3f}")
    print(f"  {'Normalized entropy':<33} | {pro_H / math.log2(len(pro_concepts)):>8.3f} | {flash_H / math.log2(len(flash_concepts)):>8.3f}")
    print(f"  {'Effective # concepts':<33} | {2 ** pro_H:>8.1f} | {2 ** flash_H:>8.1f}")

    # Top concepts comparison
    print(f"\n--- Top 15 Concepts Comparison ---")
    print(f"  {'Pro':<30} {'#':>3} | {'Flash':<30} {'#':>3}")
    print("  " + "-" * 70)
    pro_top = pro_concepts.most_common(15)
    flash_top = flash_concepts.most_common(15)
    for i in range(15):
        p_name, p_cnt = pro_top[i] if i < len(pro_top) else ("—", 0)
        f_name, f_cnt = flash_top[i] if i < len(flash_top) else ("—", 0)
        print(f"  {p_name:<30} {p_cnt:>3} | {f_name:<30} {f_cnt:>3}")

    # Concepts unique to each
    pro_set = set(pro_concepts.keys())
    flash_set = set(flash_concepts.keys())
    only_pro = pro_set - flash_set
    only_flash = flash_set - pro_set
    both = pro_set & flash_set

    print(f"\n--- Concept Overlap ---")
    print(f"  Shared concepts: {len(both)}")
    print(f"  Only in Pro: {len(only_pro)} — {sorted(only_pro)}")
    print(f"  Only in Flash: {len(only_flash)} — {sorted(only_flash)}")

    # Per-epoch diversity comparison
    print(f"\n--- Per-Epoch Diversity (unique concepts per epoch) ---")
    print(f"  {'Epoch':>5} | {'Pro concepts':>12} | {'Flash concepts':>14} | {'Pro ideas':>9} | {'Flash ideas':>11}")
    print("  " + "-" * 65)
    for ep in range(8):
        pro_ep = set(normalize_concept_name(i["idea"]) for i in pro_ideas if i["epoch"] == ep)
        flash_ep = set(normalize_concept_name(i["idea"]) for i in flash_ideas if i["epoch"] == ep)
        pro_n = len([i for i in pro_ideas if i["epoch"] == ep])
        flash_n = len([i for i in flash_ideas if i["epoch"] == ep])
        print(f"  {ep:>5} | {len(pro_ep):>12} | {len(flash_ep):>14} | {pro_n:>9} | {flash_n:>11}")

    # Quality distribution: Pro ideas below certain threshold
    print(f"\n--- Quality Distribution (ideas with loss < threshold) ---")
    for thresh in [3.70, 3.72, 3.75, 3.80]:
        pro_below = sum(1 for l in pro_losses if l < thresh)
        flash_below = sum(1 for l in flash_losses if l < thresh)
        print(f"  Loss < {thresh}: Pro={pro_below}/{len(pro_valid)} ({100 * pro_below / len(pro_valid):.0f}%), "
              f"Flash={flash_below}/{len(flash_valid)} ({100 * flash_below / len(flash_valid):.0f}%)")

    # Intra-epoch text similarity comparison
    print(f"\n--- Intra-Epoch Text Similarity (word Jaccard) ---")
    print(f"  {'Epoch':>5} | {'Pro mean':>9} | {'Flash mean':>11}")
    print("  " + "-" * 35)
    for ep in range(8):
        pro_texts = [i["idea"] for i in pro_ideas if i["epoch"] == ep]
        flash_texts = [i["idea"] for i in flash_ideas if i["epoch"] == ep]

        def mean_pairwise(texts):
            if len(texts) < 2:
                return 0.0
            sims = [jaccard_similarity(texts[i], texts[j])
                    for i in range(len(texts)) for j in range(i + 1, len(texts))]
            return sum(sims) / len(sims) if sims else 0.0

        print(f"  {ep:>5} | {mean_pairwise(pro_texts):>9.4f} | {mean_pairwise(flash_texts):>11.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. SYNTHESIS: WHAT LIMITS DIVERSITY?
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def synthesize():
    print("\n" + "=" * 80)
    print("6. SYNTHESIS: WHAT LIMITS DIVERSITY?")
    print("=" * 80)

    concepts = load_csv(CONCEPT_CSV)
    pro_ideas = load_json(PRO_DB)
    epoch_data = load_csv(EPOCH_CSV)

    counts = [int(c["count"]) for c in concepts]
    total = sum(counts)
    n = len(counts)

    # Key metrics
    singletons = sum(1 for c in counts if c == 1)
    top7_total = sum(counts[:7])  # concepts with count >= 3

    probs = [c / total for c in counts]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(n)

    new_rates = []
    for row in epoch_data:
        ep = int(row["epoch"])
        if ep == 0:
            continue
        new_rates.append(int(row["new_concepts"]) / int(row["total_ideas"]))

    print(f"""
DIVERSITY BOTTLENECK ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. MODERATE CONCENTRATION: The top 7 concepts (count >= 3) account for
   {top7_total}/{total} idea instances ({100 * top7_total / total:.0f}%), but
   {singletons}/{n} concepts ({100 * singletons / n:.0f}%) appear only once.
   The distribution is fat-tailed but not extremely concentrated.

2. NORMALIZED ENTROPY: {entropy / max_entropy:.3f} (where 1.0 = perfect uniformity).
   Effective number of concepts: {2 ** entropy:.0f} out of {n}.
   This means ~{100 * (1 - 2 ** entropy / n):.0f}% of concepts are under-utilized.

3. NOVELTY DECAY: The rate of genuinely new concepts per idea:
   Epoch 0: 100% (by definition)
   Epochs 1-3 avg: {100 * sum(new_rates[:3]) / 3:.0f}% of ideas introduce new concepts
   Epochs 4-7 avg: {100 * sum(new_rates[3:]) / len(new_rates[3:]):.0f}% of ideas introduce new concepts
   --> Novelty rate is decaying but not collapsing.

4. CONCEPT RECYCLING: The same ~7 core ideas (SwiGLU, Fused AdamW, Learnable
   RMSNorm, QK-Norm, Zero-Init, RoPE, Embedding tweaks) appear in almost every
   epoch. The LLM treats these as "safe bets" that keep appearing.

5. COMPONENT IMBALANCE: "Init / Architecture" has {[c for c in load_csv(COMPONENT_CSV) if c["component"] == "Init / Architecture"][0]["idea_count"]}
   idea-assignments while "LR Schedule" has only {[c for c in load_csv(COMPONENT_CSV) if c["component"] == "LR Schedule"][0]["idea_count"]}.
   This 10:1 ratio shows the LLM gravitates toward architecture tweaks over
   training dynamics.

6. NOVELTY SCORE: ALL {n} concepts are classified as "Known" or "Known-variant".
   Zero genuinely novel techniques were proposed. The LLM's "creativity" is
   limited to recombining well-known components from its training data.

7. MISSING SEARCH DIRECTIONS: The following major technique families are never
   explored: custom CUDA kernels, mixture of experts, sparse attention, knowledge
   distillation, curriculum learning, data augmentation, tokenizer changes,
   activation checkpointing, normalization-free architectures.
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    analyze_frequency_distribution()
    analyze_temporal()
    analyze_text_similarity()
    analyze_coverage_gaps()
    analyze_pro_vs_flash()
    synthesize()
