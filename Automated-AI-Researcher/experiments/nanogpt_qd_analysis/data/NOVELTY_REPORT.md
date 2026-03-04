# Novelty Evaluation Report: NanoGPT Evolutionary Search

## Overview

- **Run:** nanogpt_faithful_4gpu_pro (Gemini 3 Pro)
- **Total ideas evaluated:** 128
- **Unique concepts:** 75
- **Epochs:** 8 (0–7)
- **Best val_loss:** 3.6900

## Level 1: Reference Literature Classification

Each of the 75 unique concepts was matched against a curated registry of
known Transformer/NanoGPT training techniques with paper citations.

| Novelty Label | Count | Percentage |
|---------------|-------|------------|
| Known | 64 | 85.3% |
| Known-variant | 11 | 14.7% |
| Novel-combination | 0 | 0.0% |
| Novel | 0 | 0.0% |

**Result:** 100% of concepts map to known or known-variant techniques.

### Concept Classification Detail

| Concept | Component | Count | Novelty | Prior Art | Best val_loss |
|---------|-----------|-------|---------|-----------|---------------|
| SwiGLU MLP | MLP / Activation | 7 | Known | Shazeer 2020, GLU Variants Improve Transformer | 3.7007 |
| Fused AdamW | Optimizer | 6 | Known | NVIDIA Apex / PyTorch fused optimizer implementation | 3.7 |
| Learnable RMSNorm | Normalization | 5 | Known | Zhang & Sennrich 2019, Root Mean Square Layer Normalization | 3.6976 |
| QK-Norm | Normalization | 5 | Known | Dehghani et al. 2023, Scaling ViT to 22B; Henry et al. 2020 | 3.7139 |
| Zero-Init Projections | Init / Architecture | 5 | Known | Radford et al. 2019, GPT-2 (scales output projections to 0) | 3.7172 |
| Rotary Positional Embeddings (RoPE) | Attention | 4 | Known | Su et al. 2021, RoFormer: Enhanced Transformer with Rotary P... | 3.69 |
| Exclude from Weight Decay | Embedding | 4 | Known | Loshchilov & Hutter 2019; standard practice for norm/bias pa... | 3.7008 |
| Embedding Normalization | Normalization | 4 | Known | Various Transformer variants apply normalization to embeddin... | 3.7392 |
| AdamW Beta2 Tuning | Optimizer | 3 | Known | Loshchilov & Hutter 2019, Decoupled Weight Decay (hyperparam... | 3.6971 |
| Vocabulary Padding (50304) | Embedding | 3 | Known | Standard GPU optimization (align vocab size to power of 2 or... | 3.6974 |
| Output Logit Scaling | Init / Architecture | 3 | Known | Related to muP (Yang et al. 2022); logit scaling for stable ... | 3.7007 |
| Attention Logit Soft-Capping | Attention | 3 | Known | Gemma 2 Team 2024, Gemma 2 | 3.7053 |
| AdamW Beta1 Tuning | Optimizer | 3 | Known | Loshchilov & Hutter 2019, Decoupled Weight Decay (hyperparam... | 3.7121 |
| Sinusoidal Positional Embeddings | Positional Encoding | 3 | Known | Vaswani et al. 2017, Attention Is All You Need | 4.2364 |
| Reduced Weight Decay | Optimizer | 2 | Known | Loshchilov & Hutter 2019, Decoupled Weight Decay (hyperparam... | 3.704 |
| Batch Size Tuning | Init / Architecture | 2 | Known | Standard hyperparameter; Smith et al. 2018, Don't Decay the ... | 3.7113 |
| WSD Schedule | LR Schedule | 2 | Known | Warmup-Stable-Decay; used in Chinchilla (Hoffmann et al. 202... | 3.7122 |
| Residual Scaling | MLP / Activation | 2 | Known | Various; T5 (Raffel et al. 2020), DeepNet (Wang et al. 2022) | 3.7149 |
| ReLU Activation | MLP / Activation | 2 | Known | Nair & Hinton 2010, Rectified Linear Units | 3.7297 |
| Residual Dropout | Normalization | 2 | Known | Vaswani et al. 2017, Attention Is All You Need | 3.7424 |
| Embedding Scaling | Positional Encoding | 2 | Known | Vaswani et al. 2017 scales embeddings by sqrt(d_model) | 3.7621 |
| Parallel Blocks | Normalization | 2 | Known | Wang & Komatsuzaki 2021, GPT-J-6B | 3.763 |
| SiLU Activation | MLP / Activation | 2 | Known | Elfwing et al. 2018; Ramachandran et al. 2017 (Swish) | 3.7749 |
| Polynomial LR Decay | LR Schedule | 1 | Known | Standard LR schedule; used in BERT (Devlin et al. 2019) | 3.6924 |
| Add Learnable Scale To Rmsnorm | Normalization | 1 | Known | Zhang & Sennrich 2019, RMSNorm (learnable gain is standard) | 3.6932 |
| Foreach AdamW | Optimizer | 1 | Known | PyTorch foreach optimizer implementation | 3.7019 |
| Inverse Sqrt Schedule | LR Schedule | 1 | Known | Vaswani et al. 2017, Attention Is All You Need | 3.7051 |
| Grouped Query Attention | Attention | 1 | Known | Ainslie et al. 2023, GQA: Training Generalized Multi-Query T... | 3.7052 |
| Disable Gradient Clipping | Init / Architecture | 1 | Known-variant | Architectural choice; Pascanu et al. 2013 introduced gradien... | 3.7065 |
| Gradient Clipping by Value | Init / Architecture | 1 | Known | Pascanu et al. 2013, On the Difficulty of Training Recurrent... | 3.7071 |
| Xavier Uniform Initialization | Init / Architecture | 1 | Known | Glorot & Bengio 2010, Understanding Difficulty of Training D... | 3.7085 |
| Input Jitter | Init / Architecture | 1 | Known-variant | Known regularization; related to noise injection techniques | 3.7087 |
| GPT-2 Initialization | Init / Architecture | 1 | Known | Radford et al. 2019, Language Models are Unsupervised Multit... | 3.7103 |
| Embedding Dropout | Positional Encoding | 1 | Known | Standard regularization technique | 3.7107 |
| Fewer Heads, Larger Dim | Attention | 1 | Known | Michel et al. 2019, Are Sixteen Heads Really Better than One... | 3.7125 |
| Squared ReLU | MLP / Activation | 1 | Known | So et al. 2021, Primer: Searching for Efficient Transformers | 3.7133 |
| Learnable Attention Temperature | Attention | 1 | Known | Related to Martins & Astudillo 2016; used in various Transfo... | 3.7133 |
| Token Shifting | Init / Architecture | 1 | Known-variant | Lin et al. 2019, TSM: Temporal Shift Module (adapted for seq... | 3.7156 |
| Orthogonal Initialization | MLP / Activation | 1 | Known | Saxe et al. 2013, Exact Solutions to the Nonlinear Dynamics ... | 3.7185 |
| Attention Dropout | Attention | 1 | Known | Vaswani et al. 2017, Attention Is All You Need | 3.7207 |
| Untied Embedding Weights | Positional Encoding | 1 | Known | Press & Wolf 2017 studied weight tying; untying is architect... | 3.7214 |
| Bias in QKV Projections | Attention | 1 | Known | Radford et al. 2019, GPT-2 (uses bias in attention projectio... | 3.7219 |
| Tanh Approximation For Gelu | MLP / Activation | 1 | Known | Hendrycks & Gimpel 2016, GELU (tanh approximation) | 3.7229 |
| AdamW Betas Tuning | Optimizer | 1 | Known | Loshchilov & Hutter 2019, Decoupled Weight Decay (hyperparam... | 3.7236 |
| Fast Gelu | MLP / Activation | 1 | Known-variant | Hendrycks & Gimpel 2016, GELU (approximate variant) | 3.7238 |
| Standard LayerNorm | Normalization | 1 | Known | Ba et al. 2016, Layer Normalization | 3.7242 |
| Enable TF32 | Init / Architecture | 1 | Known | NVIDIA Ampere TF32 format (hardware feature) | 3.7245 |
| ReGLU MLP | MLP / Activation | 1 | Known | Shazeer 2020, GLU Variants Improve Transformer | 3.7266 |
| Re-introduce Linear Biases | Attention | 1 | Known-variant | GPT-2 used biases; GPT-3+ removed them. Re-introducing is a ... | 3.7323 |
| Z-Loss Regularization | Init / Architecture | 1 | Known | Chowdhery et al. 2022, PaLM: Scaling Language Modeling with ... | 3.7337 |
| Extended Warmup | LR Schedule | 1 | Known | Standard hyperparameter tuning (longer warmup period) | 3.7352 |
| Sandwich Norm | Normalization | 1 | Known | Ding et al. 2021, CogView | 3.7373 |
| Enable Bias In Lm Head | Init / Architecture | 1 | Known-variant | Standard architectural choice | 3.7404 |
| RMSNorm with Bias | Normalization | 1 | Known-variant | Zhang & Sennrich 2019, RMSNorm (variant with bias) | 3.7406 |
| Increase Learning Rate To E- | Init / Architecture | 1 | Known | Standard hyperparameter tuning | 3.751 |
| Scale Input Embeddings | Positional Encoding | 1 | Known | Vaswani et al. 2017, Attention Is All You Need | 3.7554 |
| Head-wise RMSNorm (NormFormer) | Normalization | 1 | Known | Shleifer et al. 2021, NormFormer | 3.7563 |
| LayerScale | Normalization | 1 | Known | Touvron et al. 2021, Going Deeper with Image Transformers (C... | 3.76 |
| Mish Activation | MLP / Activation | 1 | Known | Misra 2019, Mish: A Self Regularized Non-Monotonic Activatio... | 3.7614 |
| Scale Embeddings By $\Sqrt{D {Model}}$ | Embedding | 1 | Known | Vaswani et al. 2017, Attention Is All You Need | 3.7677 |
| Small Initialization | Init / Architecture | 1 | Known | Various; small init for stability is well-known | 3.7702 |
| Droppath | Normalization | 1 | Known | Huang et al. 2016, Deep Networks with Stochastic Depth | 3.7707 |
| Post-Ln Architecture | Normalization | 1 | Known | Xiong et al. 2020, On Layer Normalization in the Transformer... | 3.7732 |
| Amsgrad Optimizer | Optimizer | 1 | Known | Reddi et al. 2018, On the Convergence of Adam and Beyond | 3.7736 |
| Remove Residual Scaling | Positional Encoding | 1 | Known-variant | Architectural choice (removing pre-existing scaling) | 3.7755 |
| Expanded Value Head | Attention | 1 | Known-variant | Common architectural choice in various Transformer variants | 3.7802 |
| Attention Logit Hard-Capping | Attention | 1 | Known-variant | Variant of soft-capping; clamp-based | 3.8012 |
| AdamW Epsilon Tuning | Optimizer | 1 | Known | Loshchilov & Hutter 2019, Decoupled Weight Decay (hyperparam... | 3.8045 |
| Label Smoothing | Init / Architecture | 1 | Known | Szegedy et al. 2016, Rethinking the Inception Architecture | 3.8198 |
| Separate Embedding Learning Rate | Embedding | 1 | Known-variant | Known hyperparameter tuning practice | 3.832 |
| Softmax Attention | Attention | 1 | Known | Vaswani et al. 2017, Attention Is All You Need | 3.8528 |
| Multi-Query Attention | Attention | 1 | Known | Shazeer 2019, Fast Transformer Decoding | 3.952 |
| Elu Activation | MLP / Activation | 1 | Known | Clevert et al. 2015, Fast and Accurate Deep Network Learning... | 3.9821 |
| Scalenorm | Normalization | 1 | Known | Nguyen & Salazar 2019, Transformers without Tears | nan |
| Radam Optimizer | Optimizer | 1 | Known | Liu et al. 2020, On the Variance of the Adaptive Learning Ra... | nan |

## Level 2: LLM Novelty Judge (Gemini 2.0 Flash)

Each unique concept was evaluated by Gemini Flash with a structured
novelty assessment prompt. 73/75 concepts were successfully evaluated
(2 skipped due to missing valid val_loss).

| Novelty Label | Count | Percentage |
|---------------|-------|------------|
| KNOWN | 65 | 89.0% |
| KNOWN-VARIANT | 8 | 11.0% |
| NOVEL-COMBINATION | 0 | 0.0% |
| NOVEL | 0 | 0.0% |

**Level 1 ↔ Level 2 agreement:** 62/73 (84.9%)

### Disagreements Between Levels

| Concept | Registry | LLM Judge | LLM Reasoning |
|---------|----------|-----------|---------------|
| Disable Gradient Clipping | Known-variant | KNOWN | Disabling gradient clipping is a known technique, often used when training is st |
| Enable Bias In Lm Head | Known-variant | KNOWN | Using a bias term in the final linear layer (lm_head) of a language model is sta |
| Fast Gelu | Known-variant | KNOWN | The fast GELU approximation using tanh is a well-known alternative to the exact  |
| Input Jitter | Known-variant | KNOWN | Adding Gaussian noise to the input embeddings is a form of input perturbation, w |
| Output Logit Scaling | Known | KNOWN-VARIANT | Temperature scaling is a well-known technique for calibrating neural network out |
| Re-introduce Linear Biases | Known-variant | KNOWN | Using biases in linear layers is a standard practice in neural networks, includi |
| Remove Residual Scaling | Known-variant | KNOWN | The manual residual scaling is an attempt to stabilize training, but the standar |
| Sandwich Norm | Known | KNOWN-VARIANT | Adding a normalization layer after the MLP block is a variant of post-normalizat |
| Separate Embedding Learning Rate | Known-variant | KNOWN | Using different learning rates for different parameter groups is a standard prac |
| Softmax Attention | Known | KNOWN-VARIANT | Adding a zero column to the attention logits before softmax is a variant of tech |
| WSD Schedule | Known | KNOWN-VARIANT | The proposed WSD schedule is a variant of learning rate schedules with warmups a |

### Most Interesting Ideas (LLM score >= 2)

| Concept | Label | val_loss | Prior Art |
|---------|-------|----------|-----------|
| Attention Logit Hard-Capping | KNOWN-VARIANT | 3.8012 | none |
| Expanded Value Head | KNOWN-VARIANT | 3.7802 | none |
| Output Logit Scaling | KNOWN-VARIANT | 3.7007 | Temperature scaling in classification (e.g., "On Calibration of Modern Neural Ne |
| RMSNorm with Bias | KNOWN-VARIANT | 3.7406 | none |
| Sandwich Norm | KNOWN-VARIANT | 3.7373 | LayerNorm (Ba, Kiros, Hinton, 2016) |
| Softmax Attention | KNOWN-VARIANT | 3.8528 | Similar to techniques used in sparse attention and routing transformer variants, |
| Token Shifting | KNOWN-VARIANT | 3.7156 | Similar to techniques used in recurrent neural networks and convolutional neural |
| WSD Schedule | KNOWN-VARIANT | 3.7122 | Cosine annealing with warm restarts (SGDR) |

### Novelty–Performance Correlation

- **Spearman ρ** = 0.130 (p = 0.2727)
- **No significant correlation** between novelty and performance

## Level 3: Code Diff Structural Analysis

Analyzed code diffs across all epochs to measure implementation
consistency within each concept (do reimplementations of the same
concept produce structurally similar code?).

- **Concepts with diffs:** 75/75
- **Concepts with 2+ instances:** 23
- **Mean intra-concept Jaccard similarity:** 0.402

### Implementation Consistency (concepts with 2+ instances)

| Consistency | Count | Description |
|-------------|-------|-------------|
| High (≥0.7) | 3 | Nearly identical reimplementations |
| Moderate (0.4–0.7) | 9 | Same approach, different details |
| Low (<0.4) | 11 | Structurally diverse implementations |

### Low-Consistency Concepts

These concepts were reimplemented with structurally different code each time,
suggesting the LLM explores different implementation strategies even for the
same well-known technique.

| Concept | Instances | Jaccard Similarity |
|---------|-----------|-------------------|
| ReLU Activation | 2 | 0.0 |
| SiLU Activation | 2 | 0.0 |
| Reduced Weight Decay | 2 | 0.013 |
| Embedding Scaling | 2 | 0.067 |
| Residual Scaling | 2 | 0.143 |
| Batch Size Tuning | 2 | 0.188 |
| Attention Logit Soft-Capping | 3 | 0.189 |
| Rotary Positional Embeddings (RoPE) | 4 | 0.267 |
| Zero-Init Projections | 5 | 0.269 |
| QK-Norm | 5 | 0.353 |
| Output Logit Scaling | 3 | 0.367 |

## Conclusions

### 1. No genuinely novel techniques discovered

All 75 unique concepts generated by Gemini 3 Pro across 8 epochs map to
well-known techniques in the Transformer/NanoGPT literature. Both the curated
registry (Level 1) and the LLM judge (Level 2) agree: **0 concepts are Novel
or Novel-combination.** The search explored known techniques in new configurations,
not genuinely new ideas.

### 2. Known-variants are minor parameter tweaks

The 11 known-variants (Level 1) and 8 (Level 2) are cases like:
- RMSNorm with bias (standard RMSNorm omits bias)
- Attention logit hard-capping (soft-capping is more common)
- Token shifting (adapted from CNN/video to sequence models)
- Input jitter (less common for Transformers specifically)

None represent meaningful innovation — they are straightforward parameter or
implementation variations of established methods.

### 3. Implementation diversity despite conceptual repetition

Among the 23 concepts with multiple implementations, 11/23 (48%)
show low structural consistency (Jaccard < 0.4). This means the LLM
generates meaningfully different code for the same conceptual idea across
epochs. For example, RoPE was implemented 4 times with only 0.267 Jaccard
similarity — different rotation schedules, frequency bases, and integration
points in the code.

### 4. No novelty–performance tradeoff

There is no significant correlation between novelty score and validation loss
(Spearman ρ ≈ 0.13, p ≈ 0.27). Known techniques and known-variants perform
comparably. The best-performing ideas (lowest val_loss) are all well-established
techniques: RoPE, Polynomial LR Decay, Learnable RMSNorm, SwiGLU, etc.

### 5. Implications for the paper

This evaluation rigorously confirms the paper's own acknowledgment that the
evolutionary search "combines known techniques in new configurations." The
value of the approach lies not in discovering novel techniques but in:
- **Automated exploration** of a large combinatorial space of known methods
- **Systematic evaluation** across components (attention, normalization, optimizer, etc.)
- **Implementation diversity** — even for the same concept, the LLM finds
  different implementation strategies, some of which perform better than others

## Output Files

| File | Description |
|------|-------------|
| `data/05_novelty_classification.csv` | Level 1 registry classification (75 concepts) |
| `data/06_llm_novelty_scores.csv` | Level 2 LLM judge results (73 concepts) |
| `data/07_diff_novelty.csv` | Level 3 diff structural analysis (75 concepts) |
| `figures/08_novelty_breakdown.pdf` | Stacked bar: novelty by component |
| `figures/09_novelty_vs_performance.pdf` | Scatter: novelty vs val_loss |
| `figures/10_novelty_over_epochs.pdf` | Temporal novelty trends |
