# ML Agent Benchmarks Research Notes

## Benchmark Landscape

| Benchmark | SOTA | Status |
|-----------|------|--------|
| HumanEval | 99.4% | Saturated |
| SWE-bench Verified | 76% | Near-saturated |
| MLE-bench HIGH | 40% | **Open** |
| Web-Bench | 25.1% | **Open** |
| MLGym novel research | 0% | **Complete gap** |

## MLE-bench Leaderboard

| Agent | Org | LOW (22) | MED (38) | HIGH (15) | Overall | Source |
|-------|-----|----------|----------|-----------|---------|--------|
| PiEvolve | Fractal AI | 80.3% | 58.8% | 40.0% | **61.3%** | Jan 2026 |
| Famou-Agent | Baidu | — | — | — | 59.6% | — |
| ML-Master | — | — | — | — | 56.4% | — |
| AIRA | Meta FAIR | 55.0% | 22.0% | 21.7% | 31.6% | arXiv:2507.02554 |
| AIDE | Weco AI | — | — | — | 17.1% | — |

---

## AIRA vs PiEvolve Architecture

| Category | Aspect | AIRA | PiEvolve |
|----------|--------|------|----------|
| Structure | Graph type | Tree (linear chains) | DAG (preserves lineage) |
| Structure | Failed nodes | Evicted from islands | Kept in DAG, deprioritized |
| Memory | Shows failures to LLM? | ✗ Filtered by `good_nodes` | ✓ Preserved |
| Policy | Exploration control | Fixed complexity schedule | Adaptive priority scoring |
| Policy | Crossover prob | 50% after gen 2 | Score-weighted |
| Execution | Step timeout | 30 min | 24hr total budget |
| Operators | Actions | Draft, Improve, Debug, Crossover, Analyze | Draft, Improve, Debug, Merge |

**Key differentiator**: AIRA filters failed nodes from LLM context; PiEvolve preserves them.

**Evidence**: `memory.py:111` (`nodes = journal.good_nodes`), `evo.yaml:18` (crossover_prob), `draft.yaml:49-55` (complexity schedule)

---

## Master Task Analysis

| Task | Tier | Domain | Winner Approach | Train Time | AIRA | PiEvolve | Failure Root Cause |
|------|------|--------|-----------------|------------|:----:|:--------:|-------------------|
| tabular-playground-dec-2021 | LOW | Tabular | LightGBM + XGBoost + CatBoost, 5-fold | 30-60 min | ✓ | ✓ | — |
| spooky-author | MED | NLP | TF-IDF + MLP (simple wins) | 10 min - 1 hr | ✗ | ✓ Silver | AIRA over-engineers with BERT |
| stanford-covid-vaccine | MED | Scientific | GCN + Bi-GRU + Attention, pseudo-labels | 2-8 hr | ✓ | ✓ | — |
| dog-breed-identification | LOW | CV | Inception-ResNet-V2 + NASNet, TTA | 30 min - 2 hr | ✗ | ✗ | Wrong pretrained models |
| tabular-playground-may-2022 | LOW | Tabular | 5-layer NN, feature interactions | 30 min - 2 hr | ✗ | ✗ | **Bad task**: top-100 within 0.0001 |
| new-york-city-taxi-fare | LOW | Large Tabular | Haversine + landmarks, XGBoost 55M rows | 10-30 min | ✗ | ✗ | Data scale (55M rows) |
| vinbigdata-chest-xray | HIGH | Medical CV | YOLO/EfficientDet detection + clf | 10-15 hr | ✗ | ✓ Gold | Timeout; needs multi-stage |
| jigsaw-toxic | MED | NLP | BiLSTM + CNN + GloVe, multi-label | 15-24 hr | ✗ | ✓ Gold | Timeout |
| essay-scoring-2 | MED | NLP | 4x DeBERTa V3 Large, 2-stage training | 10-25 hr | ✗ | ✓ Gold | Timeout; can't detect distribution shift |
| nfl-player-contact | MED | Multimodal | 3D CNN + video features | 30-50 hr | ✓ 20% | ✗ | PiEvolve: single-modality focus |
| ranzcr-clip-catheter | LOW | Medical CV | EfficientNet ensemble, 1024px | 35-50 hr | ✗ | ✗ | 0.8% gap; needs high-res |
| iwildcam-2019/2020 | HIGH | Wildlife CV | Detection + classification, 150K imgs | 50-90 hr | ✗ | ✓ Gold | Timeout (150K images) |
| billion-word-imputation | MED | Large NLP | KN 5-gram + RNN hybrid | 3 hr - 2 wk | ✓ 30% | ✗ | PiEvolve: wrong problem framing |
| google-contrails | HIGH | Satellite CV | U-Net 1024px + ViT, temporal LSTM | 150-300 hr | ✗ | — | Compute (35GB dataset) |
| siim-isic-melanoma | LOW | Medical CV | 18-model ensemble, EfficientNet | 200-240 hr | ✗ | ✓ Bronze | Compute gap |
| rsna-breast | HIGH | Medical CV | YOLOX-nano ROI + ConvNeXt | 192-240 hr | ✗ | — | Compute (A100 required) |

**Legend**: ✓ = medal, ✗ = no medal, — = not evaluated

---

## Domain Failure Rates

| Domain | AIRA Fail Rate | PiEvolve Fail Rate | Primary Gap Driver |
|--------|----------------|--------------------|--------------------|
| Medical Imaging | 100% (9/9) | 11% (1/9) | Domain preprocessing + timeout |
| NLP | 80% (8/10) | 0% (0/10) | Timeout (30min vs 24hr) |
| Fine-grained CV | 100% (5/5) | 20% (1/5) | Pretrained model selection |
| Large-scale Tabular | 100% (2/2) | 100% (2/2) | Scale fundamentally hard |
| Scientific | 86% (6/7) | 14% (1/7) | Domain knowledge |

---

## Final Assessment: AIRA Improvement Potential

### Current Constraints

| Constraint | Current Value | Impact | Fix Difficulty |
|------------|---------------|--------|----------------|
| Step timeout | 30 min | 75% of winners impossible | Easy (config) |
| Checkpoint resumption | Not supported | Can't train across steps | Medium (sandbox redesign) |
| Memory filtering | Hides failures | Limits learning from mistakes | Easy (config) |
| Model cache | Downloads each step | Wastes 5-10 min/step | Easy (bind mount) |
| Total budget | ~8hr effective | vs PiEvolve's 24hr | Config change |

### Task Feasibility by Constraint

| Category | Tasks | % | Examples |
|----------|-------|---|----------|
| Fits 30min timeout | 4 | 25% | tabular-dec-2021, dog-breed, spooky-author, tabular-may-2022 |
| Fits 2hr timeout | 2 | 12% | stanford-covid-vaccine, vinbigdata |
| Needs checkpoint resumption | 4 | 25% | essay-scoring, jigsaw, nfl-contact, ranzcr |
| Needs 24hr+ / multi-GPU | 4 | 25% | iwildcam, contrails, melanoma, rsna-breast |
| Data scale issues | 2 | 12% | nyc-taxi, billion-word |

### Projected Improvement Path

| Change | Effort | Medal Rate | Gap to PiEvolve |
|--------|--------|------------|-----------------|
| Current AIRA | — | 31.6% | 29.4 pts |
| + Timeout → 2hr | Easy | ~37% | 24 pts |
| + Show failures + model cache | Easy | ~40% | 21 pts |
| + Checkpoint resumption | Medium | **~50%** | 11 pts |
| Theoretical max (same compute) | — | ~55% | 6 pts |

### Verdict

**Yes, significant room for improvement — ceiling at ~50-55%.**

- **Easy wins (+8 pts)**: 2hr timeout, show failures to LLM, model cache
- **Biggest lever (+10 pts)**: Checkpoint resumption unlocks 25% more tasks
- **Hard ceiling**: 38% of tasks require 24hr+ training — compute gap, not fixable by scaffold changes

**Recommendation**: Implement checkpoint resumption. It's the difference between 40% and 50%. The remaining gap to PiEvolve (61%) is primarily compute budget, not architecture.

---

## Local QD Study (Feb 2026)

**Setup**: AIRA_GREEDY vs AIRA_EVO, Gemini-3-flash, 5 tasks × 3 seeds

| Task | Greedy Best | Evo Best | Medal |
|------|-------------|----------|-------|
| tabular-playground-dec-2021 | 0.963 | 0.963 | Gold |
| stanford-covid-vaccine | 0.222 | 0.221 | Gold |
| dog-breed | 0.468 | TBD | No |
| spooky-author | 0.387 | TBD | No |
| essay-scoring | 0.830 | 0.808 | No |

**Error resolution rates**: TIMEOUT 92%, CODE_BUG 92%, CUDA_OOM 100%, SOLVER_BUG 20%

### EVO Diversity Deep-Dive (Feb 11, 2026)

| Task | Valid Solutions | Diversity | Evidence |
|------|-----------------|-----------|----------|
| tabular-playground | 87 | ❌ LOW | 100% XGBoost in top solutions; ~6 meaningful patterns; universal feat. eng. (log-scale, temporal); no NN/linear/SVM attempts |
| spooky-author | 55 | ❌ LOW | 84% DeBERTa-v3-base; only 5 unique archs (1 BiGRU, 7 CNN variants); hyperparameter tuning dominates |
| dog-breed | 67 | ⚠️ MOD-LOW | 8 distinct archs (EfficientNet, ResNet, ConvNeXt, DINOv2, etc.) but top 3 identical: ConvNeXtV2+SoftTargetCE+Mixup |
| stanford-covid | 74 | ✅ HIGH | 7 arch types (GRU, Transformer, BiGRU, CNN, LSTM, GNN, hybrid); 19 unique combos; no dominant approach |
| learning-agency-lab | 8 | ⚠️ TBD | Only 8 solutions; too early; TF-IDF+GBDT vs DeBERTa split |

**Key findings:**
- **3/5 runs show LOW diversity** — algorithm explores many solutions but they're variations on same theme
- **spooky-author**: 84% of attempts use DeBERTa-v3-base with minor tuning (lr, epochs, augmentation); only 1 truly sequential model (Bi-GRU)
- **dog-breed**: Despite 8 architectures, best performers converged identically; exploration happened but didn't influence final output
- **tabular-playground**: All 87 solutions use GBDT family; feature engineering pattern universal across all attempts
- **stanford-covid**: Best QD behavior — genuine architectural variety persisted throughout search

**Implication**: EVO generates breadth during exploration but selection pressure collapses to narrow peaks. May need explicit diversity preservation (MAP-Elites style niching) to maintain variety in final population.

---

## Open Questions

- [ ] Does showing failed nodes to LLM improve diversity? (Test: `memory.py:111` → `include_buggy_nodes=True`)
- [ ] What's the human expert ceiling on MLE-bench?
- [ ] Can checkpoint resumption be added without full sandbox redesign?
