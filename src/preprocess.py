"""
preprocess.py
-------------
Minimal, configurable text preprocessing.

Philosophy: preserve stylistic signals.
LLMs have distinctive whitespace, punctuation, and capitalization patterns.
Over-cleaning destroys discriminative features.

The Preprocessor is sklearn-compatible (implements fit/transform)
so it can be embedded in a Pipeline and fitted inside CV folds.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Union

from sklearn.base import BaseEstimator, TransformerMixin

from src.utils import get_logger

logger = get_logger(__name__)


class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Configurable text preprocessor.

    Parameters
    ----------
    normalize_unicode : bool
        Apply NFC unicode normalization (default True).
    strip_whitespace : bool
        Strip leading/trailing whitespace (default True).
    remove_repeated_spaces : bool
        Collapse multiple consecutive spaces into one (default True).
    lowercase : bool
        Lowercase all text (default False — preserves capitalization signals).
    remove_punctuation : bool
        Remove all punctuation (default False — punctuation is discriminative).
    remove_numbers : bool
        Replace digit sequences with a placeholder (default False).
    """

    def __init__(
        self,
        normalize_unicode: bool = True,
        strip_whitespace: bool = True,
        remove_repeated_spaces: bool = True,
        lowercase: bool = False,
        remove_punctuation: bool = False,
        remove_numbers: bool = False,
    ):
        self.normalize_unicode = normalize_unicode
        self.strip_whitespace = strip_whitespace
        self.remove_repeated_spaces = remove_repeated_spaces
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers

    def fit(self, X, y=None):
        """No-op fit: preprocessor has no learned parameters."""
        return self

    def transform(self, X: Union[List[str], List], y=None) -> List[str]:
        """Apply preprocessing to a list of text strings."""
        return [self._clean(text) for text in X]

    def _clean(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        if self.normalize_unicode:
            text = unicodedata.normalize("NFC", text)

        if self.strip_whitespace:
            text = text.strip()

        if self.remove_repeated_spaces:
            text = re.sub(r" {2,}", " ", text)
            # Normalize line endings (CRLF/CR → LF) and convert tabs to spaces,
            # but PRESERVE newline characters — they carry structural signals
            # (paragraph breaks, list items, markdown headers) that are key
            # discriminators between LLMs (especially Gemini's formatted output).
            text = re.sub(r"\r\n?", "\n", text)   # CRLF / CR  → LF
            text = re.sub(r"\t+", " ", text)        # tabs → space
            text = re.sub(r" {2,}", " ", text)      # collapse multiple spaces

        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", "", text)

        if self.remove_numbers:
            text = re.sub(r"\d+", "<NUM>", text)

        return text

    @classmethod
    def from_config(cls, config: Dict) -> "Preprocessor":
        """Construct from the `preprocessing` section of config.yaml."""
        prep_cfg = config.get("preprocessing", {})
        return cls(
            normalize_unicode=prep_cfg.get("normalize_unicode", True),
            strip_whitespace=prep_cfg.get("strip_whitespace", True),
            remove_repeated_spaces=prep_cfg.get("remove_repeated_spaces", True),
            lowercase=prep_cfg.get("lowercase", False),
            remove_punctuation=prep_cfg.get("remove_punctuation", False),
            remove_numbers=prep_cfg.get("remove_numbers", False),
        )
