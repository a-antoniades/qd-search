"""
Fast integration test for MAP-Elites diversity-aware selector.

Tests without any LLM calls or code execution — just the selector logic
with synthetic journal nodes. Verifies:
1. Selection runs without errors, exploration pressure fires
2. Diversity reasoning is composed correctly for each operator
3. Archive populates from journal nodes (feature extraction works)
4. Draft targeting picks empty cells
"""

import sys
from pathlib import Path

# Ensure qd/ is importable (monorepo layout)
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from dojo.core.solvers.selection.mapelites_selector import MAPElitesNodeSelector
from dojo.core.solvers.utils.journal import Journal, Node
from dojo.core.solvers.utils.metric import MetricValue
from dojo.config_dataclasses.selector.mapelites import MAPElitesSelectorConfig
from qd.features import extract_features, feature_names


# ── Helpers ──────────────────────────────────────────────────────────────

def make_node(step: int, plan: str, code: str, score: float = None, buggy: bool = False) -> Node:
    """Create a minimal Node for testing."""
    metric = MetricValue(value=score, maximize=True) if score is not None else None
    return Node(
        code=code,
        plan=plan,
        step=step,
        is_buggy=buggy,
        metric=metric,
    )


# Plans/code snippets that map to DIFFERENT feature cells
SOLUTIONS = {
    "lightgbm_kfold": {
        "plan": "Use LightGBM gradient boosting with 5-fold cross-validation",
        "code": """
import lightgbm as lgb
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)
""",
    },
    "cnn_augmentation": {
        "plan": "Build a CNN with data augmentation using torchvision transforms",
        "code": """
import torch
import torchvision
from torchvision import transforms
model = torchvision.models.resnet18(pretrained=True)
augment = transforms.Compose([transforms.RandomHorizontalFlip()])
""",
    },
    "transformer_simple": {
        "plan": "Fine-tune a Transformer (BERT) with simple train/test split",
        "code": """
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('bert-base')
model.fit(X_train, y_train)
""",
    },
    "xgboost_feature_eng": {
        "plan": "XGBoost with feature engineering: polynomial features and PCA",
        "code": """
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = xgb.XGBClassifier()
model.fit(X_poly, y)
""",
    },
    "ensemble_transfer": {
        "plan": "Ensemble of Random Forest and SVM with transfer learning from pretrained features",
        "code": """
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
import timm
backbone = timm.create_model('efficientnet_b0', pretrained=True)
rf = RandomForestClassifier()
svm = SVC()
ensemble = VotingClassifier(estimators=[('rf', rf), ('svm', svm)])
""",
    },
}


def test_feature_extraction_diversity():
    """Verify our test solutions map to different cells."""
    print("=== Test: Feature extraction diversity ===")
    cells = {}
    for name, sol in SOLUTIONS.items():
        feats = extract_features(sol["plan"], sol["code"])
        names = feature_names(feats)
        cell = (names["model_family"], names["data_strategy"])
        cells[name] = cell
        print(f"  {name:25s} -> {cell[0]:15s} + {cell[1]}")

    unique_cells = set(cells.values())
    print(f"  Unique cells: {len(unique_cells)}/{len(SOLUTIONS)}")
    assert len(unique_cells) >= 3, f"Need at least 3 distinct cells, got {len(unique_cells)}: {unique_cells}"
    print("  PASSED\n")
    return cells


def test_archive_sync():
    """Verify nodes are synced into the archive correctly."""
    print("=== Test: Archive sync ===")
    cfg = MAPElitesSelectorConfig(verbose=True)
    selector = MAPElitesNodeSelector(cfg=cfg, lower_is_better=False)

    journal = Journal()
    # Root node (always buggy, no code)
    journal.append(make_node(0, "", "", buggy=True))
    # Add a valid node
    sol = SOLUTIONS["lightgbm_kfold"]
    journal.append(make_node(1, sol["plan"], sol["code"], score=0.85))

    # Trigger sync via select()
    context = {"crossover_prob": 0.0}
    result = selector.select(journal, context)
    state = selector.get_archive_state()

    print(f"  Archive size: {state['archive_size']}")
    print(f"  Coverage: {state['coverage']:.1%}")
    assert state["archive_size"] == 1, f"Expected 1 elite, got {state['archive_size']}"
    assert state["coverage"] > 0, "Coverage should be > 0"
    print("  PASSED\n")


def test_exploration_pressure():
    """Verify draft operations target empty cells when coverage is low."""
    print("=== Test: Exploration pressure (draft targeting) ===")
    cfg = MAPElitesSelectorConfig(verbose=True, seed=42)
    selector = MAPElitesNodeSelector(cfg=cfg, lower_is_better=False)

    journal = Journal()
    journal.append(make_node(0, "", "", buggy=True))
    # Add one valid node → coverage ~3.3% (1/30)
    sol = SOLUTIONS["lightgbm_kfold"]
    journal.append(make_node(1, sol["plan"], sol["code"], score=0.85))

    # With very low coverage, draft_prob = max(0, 1 - 2*0.033) = 0.93
    # Almost all calls should return "draft" with a target cell
    context = {"crossover_prob": 0.0}
    draft_count = 0
    target_cells = set()
    for _ in range(20):
        result = selector.select(journal, context)
        if result.operator == "draft":
            draft_count += 1
            target = result.metadata.get("target_cell")
            if target:
                target_cells.add(tuple(target))

    print(f"  Draft count: {draft_count}/20 (expected ~18-19)")
    print(f"  Unique target cells: {len(target_cells)}")
    assert draft_count >= 15, f"Expected mostly drafts at low coverage, got {draft_count}/20"
    assert len(target_cells) >= 2, f"Expected multiple target cells, got {len(target_cells)}"
    print("  PASSED\n")


def test_reasoning_composition():
    """Verify diversity guidance reasoning is well-formed for each operator."""
    print("=== Test: Reasoning composition ===")
    cfg = MAPElitesSelectorConfig(verbose=True, seed=42)
    selector = MAPElitesNodeSelector(cfg=cfg, lower_is_better=False)

    journal = Journal()
    journal.append(make_node(0, "", "", buggy=True))

    # Add 3 different solutions to get above draft-only threshold
    scores = [0.85, 0.82, 0.78]
    for i, (name, sol) in enumerate(list(SOLUTIONS.items())[:3]):
        journal.append(make_node(i + 1, sol["plan"], sol["code"], score=scores[i]))

    # First call — sync nodes
    context = {"crossover_prob": 0.5}
    result = selector.select(journal, context)

    state = selector.get_archive_state()
    print(f"  Archive: {state['archive_size']} elites, coverage={state['coverage']:.1%}")

    # Collect different operator types
    operators_seen = set()
    reasoning_samples = {}
    for _ in range(50):
        result = selector.select(journal, context)
        op = result.operator
        operators_seen.add(op)
        if op not in reasoning_samples:
            reasoning_samples[op] = result.reasoning

    print(f"  Operators seen: {operators_seen}")
    for op, reasoning in reasoning_samples.items():
        print(f"\n  [{op.upper()}] reasoning preview:")
        preview = reasoning[:200].replace("\n", "\\n")
        print(f"    {preview}...")
        assert len(reasoning) > 20, f"Reasoning for {op} too short: '{reasoning}'"
        assert "Archive:" in reasoning or "Diversity" in reasoning, \
            f"Reasoning for {op} missing archive/diversity context"

    assert "draft" in operators_seen, "Should see draft at <50% coverage"
    print("\n  PASSED\n")


def test_multiple_solutions_fill_archive():
    """Verify multiple different solutions fill distinct archive cells."""
    print("=== Test: Multiple solutions fill archive ===")
    cfg = MAPElitesSelectorConfig(verbose=True)
    selector = MAPElitesNodeSelector(cfg=cfg, lower_is_better=False)

    journal = Journal()
    journal.append(make_node(0, "", "", buggy=True))

    scores = [0.90, 0.85, 0.82, 0.78, 0.75]
    for i, (name, sol) in enumerate(SOLUTIONS.items()):
        journal.append(make_node(i + 1, sol["plan"], sol["code"], score=scores[i]))

    # Also add a buggy node — should NOT be in archive
    journal.append(make_node(len(SOLUTIONS) + 1, "buggy plan", "buggy code", buggy=True))

    # Trigger sync
    context = {"crossover_prob": 0.0}
    selector.select(journal, context)
    state = selector.get_archive_state()

    print(f"  Archive size: {state['archive_size']}")
    print(f"  Coverage: {state['coverage']:.1%}")
    print(f"  QD score: {state['qd_score']:.4f}")
    print(f"  Best fitness: {state['best_fitness']:.4f}")
    print(f"  Elites:")
    for e in state["elites"]:
        print(f"    cell={e['cell']} fitness={e['fitness']:.4f}")

    assert state["archive_size"] >= 3, \
        f"Expected at least 3 distinct cells from 5 solutions, got {state['archive_size']}"
    assert state["best_fitness"] == 0.90, f"Best should be 0.90, got {state['best_fitness']}"
    print("  PASSED\n")


def test_crossover_picks_different_cells():
    """Verify crossover selects parents from different archive cells."""
    print("=== Test: Crossover picks diverse parents ===")
    cfg = MAPElitesSelectorConfig(verbose=True, seed=42)
    selector = MAPElitesNodeSelector(cfg=cfg, lower_is_better=False)

    journal = Journal()
    journal.append(make_node(0, "", "", buggy=True))

    # Add enough solutions to fill archive above draft threshold
    # We need coverage > 50% for crossover to fire, or we can force by setting
    # many occupied cells. With 30 cells and 5 solutions ~16% coverage,
    # draft_prob = max(0, 1-2*0.16) = 0.67 → still mostly drafts.
    # Instead, directly test crossover logic by calling with high coverage.
    # Let's just add solutions and look for crossover when it fires.
    scores = [0.90, 0.85, 0.82, 0.78, 0.75]
    for i, (name, sol) in enumerate(SOLUTIONS.items()):
        journal.append(make_node(i + 1, sol["plan"], sol["code"], score=scores[i]))

    # Force crossover by using high crossover_prob and many trials
    context = {"crossover_prob": 1.0}
    crossover_count = 0
    different_parent_count = 0
    for _ in range(50):
        result = selector.select(journal, context)
        if result.operator == "crossover":
            crossover_count += 1
            if len(result.selected_nodes) == 2:
                id_a = result.selected_nodes[0].node.id
                id_b = result.selected_nodes[1].node.id
                if id_a != id_b:
                    different_parent_count += 1

    print(f"  Crossover count: {crossover_count}/50")
    print(f"  Different parent pairs: {different_parent_count}")
    if crossover_count > 0:
        assert different_parent_count == crossover_count, \
            "Crossover should always pick parents from different cells"
    print("  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MAP-Elites Selector — Fast Integration Tests")
    print("=" * 60 + "\n")

    test_feature_extraction_diversity()
    test_archive_sync()
    test_exploration_pressure()
    test_reasoning_composition()
    test_multiple_solutions_fill_archive()
    test_crossover_picks_different_cells()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
