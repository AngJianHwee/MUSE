"""
Inference module for Twitter sentiment analysis.
Provides a clean interface for loading models and making predictions.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import yaml

from text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class SentimentInferenceEngine:
    """
    Loads a trained sentiment model and performs inference with consistent preprocessing.
    
    This class encapsulates the model loading, preprocessing, and prediction logic,
    providing a simple interface for both single and batch predictions.
    """

    AVAILABLE_MODELS = {
        "logistic": "best_logistic_regression.pkl",
        "naive_bayes": "best_naive_bayes.pkl",
        "ensemble": "best_ensemble.pkl",
    }
    
    PREPROCESSOR_FILE = "preprocessor.pkl"

    def __init__(
        self,
        model_name: str = "logistic",
        model_dir: Union[str, Path] = "saved_models",
        preprocessor: Optional[TextPreprocessor] = None,
        model: Optional[Any] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_name: One of the keys in AVAILABLE_MODELS ("logistic", "naive_bayes", "ensemble")
            model_dir: Directory containing the pickled models
            preprocessor: Optional custom preprocessor (defaults to loading from saved preprocessor.pkl)
            model: Optional already-loaded model (skips file loading - useful for testing)
            config_path: Optional path to config.yaml (used only if preprocessor file not found)
        """
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        
        # Load preprocessor - prefer saved preprocessor, fallback to config, then defaults
        if preprocessor is not None:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = self._load_preprocessor(config_path)

        # Load model
        if model is not None:
            self.model = model
            logger.info("Using provided model instance")
        else:
            self.model = self._load_model(model_name)

        # Store class labels for probability mapping
        # For Pipeline, classes_ is on the classifier step
        self.classes_ = self._get_model_classes()

    def _load_preprocessor(self, config_path: Optional[str] = None) -> TextPreprocessor:
        """Load preprocessor from saved file, or create from config, or use defaults."""
        preprocessor_path = self.model_dir / self.PREPROCESSOR_FILE
        
        if preprocessor_path.exists():
            logger.info("Loading preprocessor from %s", preprocessor_path)
            return TextPreprocessor.load(preprocessor_path)
        
        if config_path:
            logger.info("Preprocessor file not found, creating from config: %s", config_path)
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return TextPreprocessor(**config["preprocessing"])
        
        logger.warning("No preprocessor file or config found, using defaults")
        return TextPreprocessor()

    def _load_model(self, model_name: str) -> Any:
        """Load model from disk."""
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(self.AVAILABLE_MODELS.keys())}"
            )

        model_path = self.model_dir / self.AVAILABLE_MODELS[model_name]
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Train and save the model first using train.py"
            )

        logger.info("Loading model '%s' from %s", model_name, model_path)
        return joblib.load(model_path)

    def _get_model_classes(self) -> Optional[List[str]]:
        """Extract class labels from model (handles Pipeline and direct classifiers)."""
        # Direct classifier
        if hasattr(self.model, "classes_"):
            return list(self.model.classes_)
        
        # Pipeline - check classifier step
        if hasattr(self.model, "named_steps"):
            for step_name, step in self.model.named_steps.items():
                if hasattr(step, "classes_"):
                    return list(step.classes_)
        
        return None

    def predict(
        self, texts: Union[str, List[str]], return_probabilities: bool = True
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Predict sentiment for one or more texts.
        
        Args:
            texts: Single text string or list of text strings
            return_probabilities: Whether to include class probabilities in output
            
        Returns:
            A single result dict when a string is passed, otherwise a list of result dicts.
            Each result contains:
              - original_text: The input text
              - cleaned_text: The preprocessed text
              - predicted_sentiment: The predicted class label
              - probabilities: Dict of class -> probability (if return_probabilities=True and model supports it)
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Preprocess all texts
        cleaned = [self.preprocessor.clean(t) for t in texts]
        
        # Make predictions
        predictions = self.model.predict(cleaned)
        
        # Get probabilities if available and requested
        probabilities = None
        if return_probabilities and hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(cleaned)
            if self.classes_ is not None:
                probabilities = [
                    {cls: float(p) for cls, p in zip(self.classes_, row)} for row in proba
                ]
            else:
                probabilities = [None] * len(texts)
        else:
            probabilities = [None] * len(texts)

        # Build results
        results = [
            {
                "original_text": orig,
                "cleaned_text": clean,
                "predicted_sentiment": pred,
                "probabilities": prob,
            }
            for orig, clean, pred, prob in zip(texts, cleaned, predictions, probabilities)
        ]

        return results[0] if single_input else results

    def predict_batch(self, texts: List[str], return_probabilities: bool = True) -> List[Dict[str, Any]]:
        """
        Predict sentiment for a batch of texts (always returns list).
        
        Args:
            texts: List of text strings
            return_probabilities: Whether to include class probabilities
            
        Returns:
            List of result dicts
        """
        results = self.predict(texts, return_probabilities=return_probabilities)
        # Ensure we always return a list
        if isinstance(results, dict):
            return [results]
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            "model_name": self.model_name,
            "model_type": type(self.model).__name__,
            "classes": self.classes_,
        }
        
        # Add pipeline steps if it's a pipeline
        if hasattr(self.model, "named_steps"):
            info["pipeline_steps"] = list(self.model.named_steps.keys())
            if "tfidf" in self.model.named_steps:
                tfidf = self.model.named_steps["tfidf"]
                info["tfidf_vocab_size"] = len(tfidf.vocabulary_) if hasattr(tfidf, "vocabulary_") else None
        
        return info


def load_engine(
    model_name: str = "logistic",
    model_dir: Union[str, Path] = "saved_models",
    config_path: str = "config.yaml"
) -> SentimentInferenceEngine:
    """
    Convenience function to create an inference engine with config.
    
    Args:
        model_name: Model to load ("logistic", "naive_bayes", "ensemble")
        model_dir: Directory containing saved models
        config_path: Path to config.yaml (fallback if preprocessor.pkl not found)
        
    Returns:
        Configured SentimentInferenceEngine instance
    """
    return SentimentInferenceEngine(
        model_name=model_name,
        model_dir=model_dir,
        config_path=config_path,
    )


# ---------------------------------------------------------------------------
# Example usage and testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    test_tweets = [
        "I absolutely love this new update! It works perfectly. 😍",
        "This is the worst experience I've ever had. Terrible customer service.",
        "Just sitting here, waiting for the bus. Nothing much happening today.",
        "@Microsoft Why do I pay for WORD when it functions so poorly? 🙄",
        "So excited for the new game release! Can't wait to play it!",
    ]

    for name in ("logistic", "naive_bayes", "ensemble"):
        print("=" * 60)
        print(f"TESTING {name.upper()}")
        print("=" * 60)

        try:
            engine = SentimentInferenceEngine(model_name=name)
            print(f"Model info: {engine.get_model_info()}")
            
            results: List[Dict[str, Any]] = engine.predict(test_tweets)  # type: ignore

            for res in results:
                print(f"\nOriginal : {res['original_text']}")
                print(f"Cleaned  : {res['cleaned_text']}")
                print(f"Prediction: {str(res['predicted_sentiment']).upper()}")

                if res["probabilities"]:
                    top = sorted(
                        res["probabilities"].items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:2]
                    conf = ", ".join(f"{label} ({p:.1%})" for label, p in top)
                    print(f"Confidence: {conf}")
        except (FileNotFoundError, ValueError) as e:
            logger.error("%s", e)