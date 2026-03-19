"""Feature extraction for MAP-Elites: plan+code text -> feature descriptors.

Classifies ML solutions along two dimensions:
  - model_family (10 bins): Classical ML, GBDT, MLP/Dense, CNN, RNN, Transformer,
    GNN/Graph, State-Space, Generative/EBM, Ensemble
  - data_strategy (7 bins): Simple, K-Fold CV, Augmentation, Transfer Learning,
    Feature Eng., Semi/Self-Supervised, Synthetic Data

Extraction is purely keyword-based (no LLM calls), deterministic, and fast.
"""

from __future__ import annotations

from qd.map_elites import Feature

# ---------------------------------------------------------------------------
# Keyword definitions per bin
# ---------------------------------------------------------------------------

MODEL_FAMILY_KEYWORDS: dict[int, dict] = {
    0: {
        "name": "Classical ML",
        "keywords": [
            "logistic",
            "svm",
            "knn",
            "ridge",
            "lasso",
            "random forest",
            "randomforest",
            "decision tree",
            "naive bayes",
            "naivebayes",
            "extratrees",
            "extra trees",
            "logisticregression",
            "svc",
            "svr",
            "kneighbors",
            "elasticnet",
            "adaboost",
            "sgdclassifier",
            "sgdregressor",
            "gaussianprocess",
            "linearregression",
            "bayesianridge",
            "perceptron",
        ],
    },
    1: {
        "name": "GBDT",
        "keywords": [
            "xgboost",
            "lightgbm",
            "catboost",
            "gradient boost",
            "gradientboosting",
            "gbdt",
            "xgb",
            "lgbm",
            "histgradientboosting",
            "hist_gradient_boosting",
        ],
    },
    2: {
        "name": "MLP/Dense",
        "keywords": [
            "mlp",
            "nn.linear",
            "nn.sequential",
            "mlpclassifier",
            "mlpregressor",
            "dense layer",
            "feedforward",
            "fully connected",
            "tabnet",
            "tab_net",
        ],
    },
    3: {
        "name": "CNN",
        "keywords": [
            "convnext",
            "resnet",
            "resnext",
            "efficientnet",
            "efficientnetv2",
            "vgg",
            "inception",
            "mobilenet",
            "conv2d",
            "cnn",
            "unet",
            "densenet",
            "deeplabv3",
            "segformer",
            "nn.conv",
            "swin",
            "senet",
            "se_net",
            "regnet",
            "nfnet",
            "maxvit",
            "coatnet",
            "squeezenet",
            "shufflenet",
            "yolo",
            "detectron",
            "faster_rcnn",
            "mask_rcnn",
            "retinanet",
        ],
    },
    4: {
        "name": "RNN",
        "keywords": [
            "lstm",
            "gru",
            "bigru",
            "bilstm",
            "rnn",
            "recurrent",
            "nn.lstm",
            "nn.gru",
        ],
    },
    5: {
        "name": "Transformer",
        "keywords": [
            "bert",
            "deberta",
            "roberta",
            "transformer",
            "vit",
            "gpt",
            "attention",
            "automodel",
            "autotokenizer",
            "distilbert",
            "t5",
            "electra",
            "xlnet",
            "albert",
            "longformer",
            "xlm",
            "clip",
            "dino",
        ],
    },
    6: {
        "name": "GNN/Graph",
        "keywords": [
            "gcn",
            "gat",
            "graph_conv",
            "graphconv",
            "message_passing",
            "messagepassing",
            "node2vec",
            "graphsage",
            "graph neural",
            "graph network",
            "torch_geometric",
            "dgl",
            "edge_conv",
            "gin_conv",
        ],
    },
    7: {
        "name": "State-Space",
        "keywords": [
            "mamba",
            "ssm",
            "s4",
            "state_space",
            "state space",
            "hyena",
            "rwkv",
            "linear_attention",
            "linear attention",
            "retnet",
            "mamba_ssm",
        ],
    },
    8: {
        "name": "Generative/EBM",
        "keywords": [
            "vae",
            "variational autoencoder",
            "gan",
            "generative adversarial",
            "diffusion",
            "energy_based",
            "energy based",
            "ebm",
            "normalizing_flow",
            "normalizing flow",
            "score_matching",
            "score matching",
            "denoising_score",
            "flow_matching",
            "ddpm",
            "ddim",
            "stable_diffusion",
        ],
    },
    9: {
        "name": "Ensemble",
        "keywords": [
            "stacking",
            "voting",
            "ensemble",
            "blending",
            "votingclassifier",
            "votingregressor",
            "stackingclassifier",
            "stackingregressor",
        ],
    },
}

DATA_STRATEGY_KEYWORDS: dict[int, dict] = {
    0: {
        "name": "Simple",
        "keywords": [
            "train_test_split",
            "simple split",
            "holdout",
        ],
    },
    1: {
        "name": "K-Fold CV",
        "keywords": [
            "kfold",
            "k-fold",
            "cross_val",
            "stratifiedkfold",
            "groupkfold",
            "cross-val",
            "cross_validation",
            "crossval",
            "repeatedkfold",
            "repeatedstratifiedkfold",
        ],
    },
    2: {
        "name": "Augmentation",
        "keywords": [
            "augment",
            "mixup",
            "cutmix",
            "random crop",
            "randomcrop",
            "flip",
            "rotation",
            "cutout",
            "randomflip",
            "colorjitter",
            "randaugment",
            "albumentations",
            "imgaug",
            "autoaugment",
            "trivialaugment",
            "randomresizedcrop",
            "gaussianblur",
            "randomgrayscale",
            "randomrotation",
            "horizontalflip",
            "verticalflip",
            "randomaffine",
            "randomerasing",
        ],
    },
    3: {
        "name": "Transfer Learning",
        "keywords": [
            "pretrained",
            "fine-tune",
            "fine_tune",
            "finetune",
            "transfer",
            "imagenet",
            "from_pretrained",
            "timm.create_model",
            "freeze",
            "unfreeze",
            "backbone",
            "feature_extract",
            "load_state_dict",
            "torch.hub",
        ],
    },
    4: {
        "name": "Feature Eng.",
        "keywords": [
            "tfidf",
            "tf-idf",
            "pca",
            "svd",
            "feature engineer",
            "count_vectorizer",
            "countvectorizer",
            "tfidfvectorizer",
            "polynomialfeatures",
            "word2vec",
            "standardscaler",
            "minmaxscaler",
            "robustscaler",
            "labelencoder",
            "onehotencoder",
            "ordinalencoder",
            "target_encod",
            "frequency_encod",
            "get_dummies",
        ],
    },
    5: {
        "name": "Semi/Self-Supervised",
        "keywords": [
            "pseudo_label",
            "pseudo label",
            "pseudolabel",
            "self_supervised",
            "self-supervised",
            "self supervised",
            "contrastive",
            "simclr",
            "byol",
            "moco",
            "masked_autoencoder",
            "mae",
            "dino_loss",
            "swav",
            "barlow_twins",
            "barlow twins",
        ],
    },
    6: {
        "name": "Synthetic Data",
        "keywords": [
            "smote",
            "synthetic",
            "oversample",
            "oversampling",
            "adasyn",
            "llm_generate",
            "paraphrase",
            "data_generation",
            "generate_samples",
        ],
    },
}

# ---------------------------------------------------------------------------
# Feature definitions for GridArchive (70-cell grid: 10 x 7)
# ---------------------------------------------------------------------------

MODEL_FAMILY_FEATURE = Feature("model_family", 0, 9, num_bins=10)
DATA_STRATEGY_FEATURE = Feature("data_strategy", 0, 6, num_bins=7)
DEFAULT_FEATURES = [MODEL_FAMILY_FEATURE, DATA_STRATEGY_FEATURE]

# Convenience lookups: bin index -> name
MODEL_FAMILY_NAMES: dict[int, str] = {k: v["name"] for k, v in MODEL_FAMILY_KEYWORDS.items()}
DATA_STRATEGY_NAMES: dict[int, str] = {k: v["name"] for k, v in DATA_STRATEGY_KEYWORDS.items()}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _count_keyword_hits(text: str, keywords: list[str]) -> int:
    """Count total keyword occurrences in *text* (case-insensitive)."""
    text_lower = text.lower()
    return sum(text_lower.count(kw.lower()) for kw in keywords)


def extract_features(plan: str, code: str) -> dict[str, float]:
    """Extract MAP-Elites features from a solution's plan and code.

    Scans the concatenated text for keyword hits in each category.
    The bin with the most hits wins; ties broken by lowest bin index.
    Zero hits default to bin 0 (Classical ML / Simple).

    Returns:
        Dict with ``"model_family"`` (0-9) and ``"data_strategy"`` (0-6),
        suitable for passing to ``GridArchive.add()``.
    """
    text = (plan or "") + "\n" + (code or "")

    # Model family: pick bin with most keyword hits
    best_model_bin = 0
    best_model_hits = 0
    for bin_idx, info in MODEL_FAMILY_KEYWORDS.items():
        hits = _count_keyword_hits(text, info["keywords"])
        if hits > best_model_hits:
            best_model_hits = hits
            best_model_bin = bin_idx

    # Data strategy: pick bin with most keyword hits
    best_data_bin = 0
    best_data_hits = 0
    for bin_idx, info in DATA_STRATEGY_KEYWORDS.items():
        hits = _count_keyword_hits(text, info["keywords"])
        if hits > best_data_hits:
            best_data_hits = hits
            best_data_bin = bin_idx

    return {
        "model_family": float(best_model_bin),
        "data_strategy": float(best_data_bin),
    }


def feature_names(features: dict[str, float]) -> dict[str, str]:
    """Convert numeric feature values to human-readable bin names."""
    model_bin = int(features.get("model_family", 0))
    data_bin = int(features.get("data_strategy", 0))
    return {
        "model_family": MODEL_FAMILY_NAMES.get(model_bin, "Unknown"),
        "data_strategy": DATA_STRATEGY_NAMES.get(data_bin, "Unknown"),
    }
