"""
Text preprocessing module for Twitter sentiment analysis.
Handles cleaning, normalization, and tokenization of raw tweet text.
"""

import logging
import re
import string
import sys
from pathlib import Path
from typing import List, Optional, Set, Union

import contractions
import emoji
import joblib
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)


# All NLTK resources required by this project
NLTK_RESOURCES = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
]


def _find_nltk_resource(resource: str) -> None:
    """Locate an NLTK resource, raising LookupError when it is missing."""
    if "punkt" in resource:
        nltk.data.find(f"tokenizers/{resource}")
    elif resource in ("stopwords", "wordnet", "omw-1.4"):
        nltk.data.find(f"corpora/{resource}")
    else:
        nltk.data.find(f"taggers/{resource}")


def ensure_nltk_resources(resources: Optional[List[str]] = None, force: bool = False) -> bool:
    """
    Download required NLTK resources if missing. Safe to call multiple times.
    
    Args:
        resources: List of NLTK resource names to download. If None, uses default list.
        force: If True, re-download even if resources appear to exist.
        
    Returns:
        True if all resources are available, False if any failed to download.
    """
    if resources is None:
        resources = NLTK_RESOURCES.copy()
    
    all_success = True
    
    for resource in resources:
        try:
            if not force:
                _find_nltk_resource(resource)
                logger.debug("NLTK resource '%s' already available", resource)
                continue
        except LookupError:
            pass  # Resource not found, will download
        
        # Download the resource and verify it is actually usable afterwards.
        # nltk.download() can return False (or silently skip) on failure, so the
        # return value alone is not enough to trust that the resource is ready.
        logger.info("Downloading NLTK resource: %s", resource)
        try:
            if not nltk.download(resource, quiet=True):
                raise RuntimeError("nltk.download() reported failure")
            _find_nltk_resource(resource)
            logger.info("Successfully downloaded NLTK resource: %s", resource)
        except Exception as e:
            logger.error("Failed to download NLTK resource '%s': %s", resource, e)
            all_success = False
    
    return all_success


def download_all_nltk_resources() -> bool:
    """
    Download all NLTK resources required by this project.
    Can be called from command line or programmatically.
    
    Returns:
        True if all resources downloaded successfully, False otherwise.
    """
    logger.info("Downloading all required NLTK resources...")
    success = ensure_nltk_resources(NLTK_RESOURCES, force=False)
    if success:
        logger.info("All NLTK resources are available!")
    else:
        logger.warning("Some NLTK resources failed to download. Check logs above.")
    return success


# Fallback tokenizer in case NLTK punkt fails
def simple_tokenize(text: str) -> List[str]:
    """Simple whitespace-based tokenization as fallback."""
    import re
    # Split on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def main():
    """Command-line entry point to download all NLTK resources."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download NLTK resources required for Twitter sentiment analysis"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of all resources even if they exist"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("Starting NLTK resource download...")
    success = ensure_nltk_resources(NLTK_RESOURCES, force=args.force)
    
    if success:
        logger.info("✓ All NLTK resources downloaded successfully!")
        sys.exit(0)
    else:
        logger.error("✗ Some NLTK resources failed to download")
        sys.exit(1)


if __name__ == "__main__":
    main()


class TextPreprocessor:
    """
    Handles cleaning and normalization of raw text for sentiment models.
    
    The preprocessing pipeline matches the training-time transformations exactly,
    ensuring consistent feature extraction during inference.
    """

    def __init__(
        self,
        language: str = "english",
        min_token_length: int = 3,
        remove_urls: bool = True,
        remove_html: bool = True,
        remove_punctuation: bool = True,
        remove_digits: bool = True,
        expand_contractions: bool = True,
        demojize: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
    ):
        """
        Initialize the text preprocessor with configurable options.
        
        Args:
            language: Language for stopwords (default: "english")
            min_token_length: Minimum token length to keep (default: 3)
            remove_urls: Remove URLs from text
            remove_html: Remove HTML tags
            remove_punctuation: Remove punctuation
            remove_digits: Remove numeric digits
            expand_contractions: Expand contractions (e.g., "don't" -> "do not")
            demojize: Convert emojis to text descriptions
            remove_stopwords: Remove common stopwords
            lemmatize: Apply lemmatization with POS tagging
        """
        ensure_nltk_resources()
        
        self.language = language
        self.min_token_length = min_token_length
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.remove_punctuation = remove_punctuation
        self.remove_digits = remove_digits
        self.expand_contractions = expand_contractions
        self.demojize = demojize
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words: Set[str] = set(stopwords.words(language))
        self._punctuation_table = str.maketrans("", "", string.punctuation)
        
        # Compile regex patterns for efficiency
        self._url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self._html_pattern = re.compile(r"<.*?>")
        self._digit_pattern = re.compile(r"\d+")

    def _get_wordnet_pos(self, word: str) -> str:
        """Map POS tag to the character accepted by WordNetLemmatizer."""
        try:
            tag = nltk.pos_tag([word])[0][1][0].upper()
        except LookupError:
            # Fallback if POS tagger is not available
            logger.warning("POS tagger not available, defaulting to NOUN for word: %s", word)
            return wordnet.NOUN
        
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV,
        }
        return tag_dict.get(tag, wordnet.NOUN)

    def clean(self, text: str) -> str:
        """
        Apply the full cleaning pipeline used during training.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned and normalized text string
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # Expand contractions first (e.g., "don't" -> "do not")
        if self.expand_contractions:
            text = contractions.fix(text)
        
        # Lowercase
        text = text.lower()

        # Remove URLs
        if self.remove_urls:
            text = self._url_pattern.sub("", text)

        # Remove HTML tags
        if self.remove_html:
            text = self._html_pattern.sub("", text)

        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(self._punctuation_table)

        # Remove digits
        if self.remove_digits:
            text = self._digit_pattern.sub("", text)

        # Convert emojis to text (e.g., 😍 -> :smiling_face_with_heart_eyes:)
        if self.demojize:
            text = emoji.demojize(text)

        # Tokenize with fallback
        try:
            tokens = word_tokenize(text)
        except LookupError as e:
            if "punkt_tab" in str(e):
                logger.warning("NLTK punkt_tab not available, using fallback tokenizer")
                tokens = simple_tokenize(text)
            else:
                raise

        # Remove stopwords
        if self.remove_stopwords:
            tokens = [w for w in tokens if w not in self.stop_words]

        # Lemmatize with POS tagging
        if self.lemmatize:
            tokens = [
                self.lemmatizer.lemmatize(w, self._get_wordnet_pos(w)) for w in tokens
            ]

        # Filter by minimum token length
        tokens = [t for t in tokens if len(t) >= self.min_token_length]

        return " ".join(tokens)

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a batch of texts."""
        return [self.clean(text) for text in texts]

    def save(self, path: Union[str, Path]) -> None:
        """Save preprocessor to disk using joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Preprocessor saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TextPreprocessor":
        """Load preprocessor from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {path}")
        # Unpickling bypasses __init__, so NLTK resources must be ensured here.
        # This is what makes `python app.py` work without running training first.
        ensure_nltk_resources()
        preprocessor = joblib.load(path)
        logger.info("Preprocessor loaded from %s", path)
        return preprocessor


# For backward compatibility
def clean_text(text: str, lemmatizer: WordNetLemmatizer, stop_words: Set[str]) -> str:
    """Legacy function for backward compatibility with original code."""
    preprocessor = TextPreprocessor()
    preprocessor.lemmatizer = lemmatizer
    preprocessor.stop_words = stop_words
    return preprocessor.clean(text)