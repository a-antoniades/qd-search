"""Tests for qd.features — keyword-based feature extraction."""

from qd.features import (
    DATA_STRATEGY_KEYWORDS,
    DEFAULT_FEATURES,
    MODEL_FAMILY_KEYWORDS,
    _count_keyword_hits,
    extract_features,
    feature_names,
)
from qd.map_elites import GridArchive


# ---------------------------------------------------------------------------
# _count_keyword_hits
# ---------------------------------------------------------------------------


class TestCountKeywordHits:
    def test_basic_match(self):
        assert _count_keyword_hits("I use XGBoost", ["xgboost"]) == 1

    def test_case_insensitive(self):
        assert _count_keyword_hits("XGBoost and xgboost", ["xgboost"]) == 2

    def test_multiple_keywords(self):
        text = "Use LightGBM and CatBoost ensemble"
        assert _count_keyword_hits(text, ["lightgbm", "catboost"]) == 2

    def test_no_match(self):
        assert _count_keyword_hits("random text", ["xgboost"]) == 0

    def test_empty_text(self):
        assert _count_keyword_hits("", ["xgboost"]) == 0

    def test_substring_match(self):
        # "xgb" matches inside "XGBClassifier"
        assert _count_keyword_hits("from xgboost import XGBClassifier", ["xgb"]) >= 1


# ---------------------------------------------------------------------------
# extract_features — model family
# ---------------------------------------------------------------------------


class TestModelFamily:
    def test_xgboost_is_gbdt(self):
        features = extract_features("Use XGBoost", "import xgboost")
        assert features["model_family"] == 1.0  # GBDT

    def test_lightgbm_is_gbdt(self):
        features = extract_features("", "import lightgbm as lgb\nlgb.LGBMClassifier()")
        assert features["model_family"] == 1.0

    def test_resnet_is_cnn(self):
        features = extract_features(
            "Use ResNet50 pretrained", "import torchvision.models.resnet"
        )
        assert features["model_family"] == 2.0  # CNN

    def test_efficientnet_is_cnn(self):
        features = extract_features(
            "EfficientNet-B4", "model = timm.create_model('efficientnet_b4')"
        )
        assert features["model_family"] == 2.0

    def test_lstm_is_rnn(self):
        features = extract_features(
            "BiLSTM model", "nn.LSTM(hidden_size=256, bidirectional=True)"
        )
        assert features["model_family"] == 3.0  # RNN

    def test_gru_is_rnn(self):
        features = extract_features("GRU-based approach", "nn.GRU(input_size=128)")
        assert features["model_family"] == 3.0

    def test_bert_is_transformer(self):
        features = extract_features(
            "Fine-tune BERT",
            "from transformers import AutoModel\nmodel = AutoModel.from_pretrained('bert-base')",
        )
        assert features["model_family"] == 4.0  # Transformer

    def test_deberta_is_transformer(self):
        features = extract_features(
            "DeBERTa-v3", "from transformers import DebertaV2Model"
        )
        assert features["model_family"] == 4.0

    def test_ensemble_stacking(self):
        features = extract_features(
            "Stacking ensemble of multiple models",
            "from sklearn.ensemble import StackingClassifier\nVotingClassifier",
        )
        assert features["model_family"] == 5.0  # Ensemble

    def test_random_forest_is_classical(self):
        features = extract_features(
            "Random Forest", "RandomForestClassifier(n_estimators=100)"
        )
        assert features["model_family"] == 0.0  # Classical ML

    def test_logistic_is_classical(self):
        features = extract_features(
            "LogisticRegression", "from sklearn.linear_model import LogisticRegression"
        )
        assert features["model_family"] == 0.0

    def test_default_on_empty(self):
        features = extract_features("", "")
        assert features["model_family"] == 0.0  # Default: Classical ML

    def test_most_hits_wins(self):
        # Code has many CNN references but one xgboost mention
        plan = "CNN-based approach using ResNet"
        code = "resnet\nconv2d\nconv2d\nconv2d\nxgboost"
        features = extract_features(plan, code)
        assert features["model_family"] == 2.0  # CNN wins by count


# ---------------------------------------------------------------------------
# extract_features — data strategy
# ---------------------------------------------------------------------------


class TestDataStrategy:
    def test_train_test_split_is_simple(self):
        features = extract_features(
            "", "from sklearn.model_selection import train_test_split"
        )
        assert features["data_strategy"] == 0.0  # Simple

    def test_kfold_is_cv(self):
        features = extract_features(
            "5-fold cross-validation",
            "StratifiedKFold(n_splits=5)",
        )
        assert features["data_strategy"] == 1.0  # K-Fold CV

    def test_augmentation(self):
        features = extract_features(
            "Data augmentation with random crops",
            "transforms.RandomCrop(224)\ntransforms.RandomFlip()\nCutMix()",
        )
        assert features["data_strategy"] == 2.0  # Augmentation

    def test_transfer_learning(self):
        features = extract_features(
            "Fine-tune pretrained model on ImageNet",
            "model = timm.create_model('resnet50', pretrained=True)",
        )
        assert features["data_strategy"] == 3.0  # Transfer Learning

    def test_feature_engineering(self):
        features = extract_features(
            "TF-IDF features with PCA",
            "TfidfVectorizer(max_features=5000)\nPCA(n_components=100)",
        )
        assert features["data_strategy"] == 4.0  # Feature Eng.

    def test_default_on_empty(self):
        features = extract_features("", "")
        assert features["data_strategy"] == 0.0  # Default: Simple


# ---------------------------------------------------------------------------
# extract_features — combined
# ---------------------------------------------------------------------------


class TestCombined:
    def test_xgboost_with_kfold(self):
        plan = "XGBoost with 5-fold CV"
        code = "import xgboost\nStratifiedKFold(n_splits=5)\nxgb.XGBClassifier()"
        features = extract_features(plan, code)
        assert features["model_family"] == 1.0  # GBDT
        assert features["data_strategy"] == 1.0  # K-Fold CV

    def test_cnn_with_augmentation_and_transfer(self):
        plan = "ResNet50 pretrained on ImageNet with augmentation"
        code = (
            "model = timm.create_model('resnet50', pretrained=True)\n"
            "transforms.RandomCrop(224)\n"
            "transforms.RandomFlip()\n"
            "RandomCrop\n"
            "augment"
        )
        features = extract_features(plan, code)
        assert features["model_family"] == 2.0  # CNN
        # Augmentation has more hits than transfer learning here
        assert features["data_strategy"] == 2.0  # Augmentation

    def test_none_inputs(self):
        features = extract_features(None, None)
        assert features["model_family"] == 0.0
        assert features["data_strategy"] == 0.0


# ---------------------------------------------------------------------------
# feature_names
# ---------------------------------------------------------------------------


class TestFeatureNames:
    def test_known_bins(self):
        names = feature_names({"model_family": 1.0, "data_strategy": 3.0})
        assert names["model_family"] == "GBDT"
        assert names["data_strategy"] == "Transfer Learning"

    def test_defaults(self):
        names = feature_names({})
        assert names["model_family"] == "Classical ML"
        assert names["data_strategy"] == "Simple"


# ---------------------------------------------------------------------------
# Integration with GridArchive
# ---------------------------------------------------------------------------


class TestGridArchiveIntegration:
    def test_features_compatible_with_archive(self):
        archive = GridArchive(DEFAULT_FEATURES)
        features = extract_features("XGBoost model", "import xgboost\nxgb.train()")
        added = archive.add("s1", fitness=0.95, features=features)
        assert added is True
        assert archive.size == 1

    def test_different_approaches_map_to_different_cells(self):
        archive = GridArchive(DEFAULT_FEATURES)

        f1 = extract_features("XGBoost", "import xgboost")
        f2 = extract_features("ResNet CNN", "import torchvision\nresnet\nconv2d")
        f3 = extract_features(
            "BERT transformer", "from transformers import AutoModel\nbert\nbert"
        )

        archive.add("s1", fitness=0.8, features=f1)
        archive.add("s2", fitness=0.7, features=f2)
        archive.add("s3", fitness=0.6, features=f3)

        assert archive.size == 3  # Three different cells

    def test_same_cell_keeps_best(self):
        archive = GridArchive(DEFAULT_FEATURES)

        f1 = extract_features("XGBoost v1", "import xgboost\nxgb.train()")
        f2 = extract_features("XGBoost v2", "import xgboost\nxgb.train()")

        archive.add("s1", fitness=0.8, features=f1)
        archive.add("s2", fitness=0.9, features=f2)

        assert archive.size == 1
        assert archive.elites()[0].fitness == 0.9
        assert archive.elites()[0].id == "s2"


# ---------------------------------------------------------------------------
# Keyword coverage sanity checks
# ---------------------------------------------------------------------------


class TestKeywordCoverage:
    def test_all_model_bins_have_keywords(self):
        for bin_idx in range(6):
            assert bin_idx in MODEL_FAMILY_KEYWORDS
            assert len(MODEL_FAMILY_KEYWORDS[bin_idx]["keywords"]) > 0

    def test_all_data_bins_have_keywords(self):
        for bin_idx in range(5):
            assert bin_idx in DATA_STRATEGY_KEYWORDS
            assert len(DATA_STRATEGY_KEYWORDS[bin_idx]["keywords"]) > 0

    def test_default_features_dimensions(self):
        assert len(DEFAULT_FEATURES) == 2
        assert DEFAULT_FEATURES[0].num_bins == 6
        assert DEFAULT_FEATURES[1].num_bins == 5
        # Total cells: 6 * 5 = 30
        archive = GridArchive(DEFAULT_FEATURES)
        assert archive.cell_count() == 30
