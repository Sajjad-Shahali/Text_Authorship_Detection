"""
features.py
-----------
TF-IDF feature engineering pipeline.

Combines word-level and character-level TF-IDF using FeatureUnion.
This produces a sparse feature matrix suitable for linear classifiers.

Key design decisions:
- Word ngrams (1,2): capture vocabulary and phrase-level style
- Char ngrams (3,5) with char_wb: capture punctuation, spacing, and
  morphological patterns that strongly distinguish LLM families
- sublinear_tf=True: dampens frequency dominance
- min_df / max_df: reduce noise and near-ubiquitous terms
- Fitted INSIDE CV folds to prevent data leakage
"""

from typing import Dict

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

from src.utils import get_logger

logger = get_logger(__name__)


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """
    Pass-through transformer.
    Used to feed text through a Pipeline that expects a transformer
    before the vectorizer in a FeatureUnion.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


def build_word_tfidf(cfg: Dict) -> TfidfVectorizer:
    """Construct word-level TF-IDF vectorizer from config."""
    ngram = tuple(cfg.get("ngram_range", [1, 2]))
    return TfidfVectorizer(
        analyzer=cfg.get("analyzer", "word"),
        ngram_range=ngram,
        max_features=cfg.get("max_features", 100000),
        min_df=cfg.get("min_df", 2),
        max_df=cfg.get("max_df", 0.95),
        sublinear_tf=cfg.get("sublinear_tf", True),
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b",
    )


def build_char_tfidf(cfg: Dict) -> TfidfVectorizer:
    """Construct character-level TF-IDF vectorizer from config."""
    ngram = tuple(cfg.get("ngram_range", [3, 5]))
    return TfidfVectorizer(
        analyzer=cfg.get("analyzer", "char_wb"),
        ngram_range=ngram,
        max_features=cfg.get("max_features", 100000),
        min_df=cfg.get("min_df", 3),
        max_df=cfg.get("max_df", 0.95),
        sublinear_tf=cfg.get("sublinear_tf", True),
    )


def build_feature_union(config: Dict) -> FeatureUnion:
    """
    Build a FeatureUnion combining word and char TF-IDF.

    Returns an unfitted FeatureUnion that expects raw text strings as input.
    """
    feat_cfg = config.get("features", {})
    word_cfg = feat_cfg.get("word_tfidf", {})
    char_cfg = feat_cfg.get("char_tfidf", {})

    word_tfidf = build_word_tfidf(word_cfg)
    char_tfidf = build_char_tfidf(char_cfg)

    feature_union = FeatureUnion(
        transformer_list=[
            ("word_tfidf", word_tfidf),
            ("char_tfidf", char_tfidf),
        ]
    )

    logger.debug(
        f"FeatureUnion: word ngrams={tuple(word_cfg.get('ngram_range', [1,2]))}, "
        f"word max_features={word_cfg.get('max_features', 100000)}, "
        f"char ngrams={tuple(char_cfg.get('ngram_range', [3,5]))}, "
        f"char max_features={char_cfg.get('max_features', 100000)}"
    )
    return feature_union
