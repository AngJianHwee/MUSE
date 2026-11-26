"""
Training module for Twitter sentiment analysis.
Handles data loading, preprocessing, model training, and evaluation.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Load and combine training and validation datasets."""
    train_path = config["data"]["train_path"]
    val_path = config["data"]["val_path"]
    text_col = config["data"]["text_column"]
    label_col = config["data"]["label_column"]
    
    logger.info("Loading training data from %s", train_path)
    df_train = pd.read_csv(train_path, header=None)
    df_train.columns = ['tweet_id', 'entity', 'sentiment', 'text']
    
    logger.info("Loading validation data from %s", val_path)
    df_val = pd.read_csv(val_path, header=None)
    df_val.columns = ['tweet_id', 'entity', 'sentiment', 'text']
    
    df = pd.concat([df_train, df_val], ignore_index=True)
    df.dropna(subset=[text_col], inplace=True)
    
    logger.info("Combined data shape: %s", df.shape)
    logger.info("Sentiment distribution:\n%s", df[label_col].value_counts())
    
    return df


def preprocess_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Apply preprocessing filters to the dataframe."""
    label_col = config["data"]["label_column"]
    text_col = config["data"]["text_column"]
    
    # Remove 'Irrelevant' class if configured
    if config["data"].get("drop_irrelevant", True):
        initial_count = len(df)
        df = df[df[label_col] != "Irrelevant"]
        logger.info("Removed %d 'Irrelevant' tweets", initial_count - len(df))
    
    # Remove duplicates if configured
    if config["data"].get("remove_duplicates", True):
        initial_count = len(df)
        df = df[~df.duplicated(subset=[text_col], keep="first")]
        logger.info("Removed %d duplicate tweets", initial_count - len(df))
    
    logger.info("Final data shape after preprocessing: %s", df.shape)
    logger.info("Sentiment distribution:\n%s", df[label_col].value_counts())
    
    return df


def create_preprocessor(config: Dict[str, Any]) -> TextPreprocessor:
    """Create TextPreprocessor from config."""
    prep_config = config["preprocessing"]
    return TextPreprocessor(
        language=prep_config.get("language", "english"),
        min_token_length=prep_config.get("min_token_length", 3),
        remove_urls=prep_config.get("remove_urls", True),
        remove_html=prep_config.get("remove_html", True),
        remove_punctuation=prep_config.get("remove_punctuation", True),
        remove_digits=prep_config.get("remove_digits", True),
        expand_contractions=prep_config.get("expand_contractions", True),
        demojize=prep_config.get("demojize", True),
        remove_stopwords=prep_config.get("remove_stopwords", True),
        lemmatize=prep_config.get("lemmatize", True),
    )


def prepare_features(
    df: pd.DataFrame, 
    preprocessor: TextPreprocessor, 
    config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare features and labels for training."""
    text_col = config["data"]["text_column"]
    label_col = config["data"]["label_column"]
    
    logger.info("Preprocessing text data...")
    cleaned_texts = df[text_col].apply(preprocessor.clean)
    X = cleaned_texts.values
    y = df[label_col].values
    
    logger.info("Feature preparation complete. Samples: %d", len(X))
    return X, y


def split_data(
    X: np.ndarray, 
    y: np.ndarray, 
    config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and test sets."""
    test_size = config["data"].get("test_size", 0.2)
    random_state = config["data"].get("random_state", 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info("Train size: %d, Test size: %d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def create_tfidf_vectorizer(config: Dict[str, Any]) -> TfidfVectorizer:
    """Create TF-IDF vectorizer from config."""
    tfidf_config = config["tfidf"]
    return TfidfVectorizer(
        max_features=tfidf_config.get("max_features", 7000),
        ngram_range=tuple(tfidf_config.get("ngram_range", [1, 2])),
        min_df=tfidf_config.get("min_df", 2),
        max_df=tfidf_config.get("max_df", 0.95),
        sublinear_tf=tfidf_config.get("sublinear_tf", True),
    )


def create_logistic_regression(config: Dict[str, Any]) -> LogisticRegression:
    """Create Logistic Regression model from config."""
    lr_config = config["models"]["logistic_regression"]
    return LogisticRegression(
        C=lr_config.get("C", 10.0),
        solver=lr_config.get("solver", "lbfgs"),
        penalty=lr_config.get("penalty", "l2"),
        multi_class=lr_config.get("multi_class", "multinomial"),
        max_iter=lr_config.get("max_iter", 1000),
        random_state=lr_config.get("random_state", 42),
    )


def create_naive_bayes(config: Dict[str, Any]) -> MultinomialNB:
    """Create Multinomial Naive Bayes model from config."""
    nb_config = config["models"]["naive_bayes"]
    return MultinomialNB(alpha=nb_config.get("alpha", 0.5))


def train_model(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, Any],
    config: Dict[str, Any],
    model_name: str
) -> GridSearchCV:
    """Train a model with GridSearchCV."""
    training_config = config["training"]
    
    logger.info("Starting %s GridSearchCV...", model_name)
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=training_config.get("cv_folds", 3),
        scoring=training_config.get("scoring", "accuracy"),
        verbose=training_config.get("verbose", 1),
        n_jobs=training_config.get("n_jobs", -1),
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info("Best CV score for %s: %.4f", model_name, grid_search.best_score_)
    logger.info("Best params for %s: %s", model_name, grid_search.best_params_)
    
    return grid_search


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str
) -> Dict[str, Any]:
    """Evaluate a model on test data."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    
    logger.info("%s Test Accuracy: %.4f", model_name, accuracy)
    logger.info("%s Classification Report:\n%s", model_name, classification_report(y_test, y_pred, digits=4))
    
    return {
        "accuracy": accuracy,
        "classification_report": report,
        "predictions": y_pred
    }


def create_ensemble(
    lr_grid: GridSearchCV,
    nb_grid: GridSearchCV,
    config: Dict[str, Any],
    n_models: int = 3
) -> VotingClassifier:
    """Create ensemble from top models of each type."""
    ensemble_config = config["models"]["ensemble"]
    
    # Get top N configs from each model type
    lr_results = pd.DataFrame(lr_grid.cv_results_).sort_values("mean_test_score", ascending=False)
    nb_results = pd.DataFrame(nb_grid.cv_results_).sort_values("mean_test_score", ascending=False)
    
    top_lr = lr_results.head(n_models)
    top_nb = nb_results.head(n_models)
    
    estimators = []
    
    # Add top Logistic Regression models
    for i, (_, row) in enumerate(top_lr.iterrows()):
        params = row["params"]
        pipeline = Pipeline([
            ("tfidf", create_tfidf_vectorizer(config)),
            ("clf", create_logistic_regression(config)),
        ])
        pipeline.set_params(**params)
        estimators.append((f"lr_{i}", pipeline))
    
    # Add top Naive Bayes models
    for i, (_, row) in enumerate(top_nb.iterrows()):
        params = row["params"]
        pipeline = Pipeline([
            ("tfidf", create_tfidf_vectorizer(config)),
            ("clf", create_naive_bayes(config)),
        ])
        pipeline.set_params(**params)
        estimators.append((f"nb_{i}", pipeline))
    
    ensemble = VotingClassifier(
        estimators=estimators,
        voting=ensemble_config.get("voting", "soft"),
        n_jobs=ensemble_config.get("n_jobs", -1),
    )
    
    logger.info("Created ensemble with %d models: %s", len(estimators), [name for name, _ in estimators])
    return ensemble


def save_model(model: Any, path: Path) -> None:
    """Save model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)


def main(config_path: str = "config.yaml") -> None:
    """Main training pipeline."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Load config
    config = load_config(config_path)
    
    # Load and preprocess data
    df = load_data(config)
    df = preprocess_data(df, config)
    
    # Create and fit preprocessor
    preprocessor = create_preprocessor(config)
    
    # Prepare features
    X, y = prepare_features(df, preprocessor, config)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, config)
    
    # Create TF-IDF vectorizer
    tfidf = create_tfidf_vectorizer(config)
    
    # Train Logistic Regression
    lr_pipeline = Pipeline([
        ("tfidf", tfidf),
        ("clf", create_logistic_regression(config)),
    ])
    
    lr_param_grid = {
        "tfidf__max_features": [5000, 7000],
        "tfidf__ngram_range": [(1, 2), (1, 3)],
        "clf__C": [1.0, 10.0],
        "clf__solver": ["lbfgs"],
        "clf__penalty": ["l2"],
    }
    
    lr_grid = train_model(lr_pipeline, X_train, y_train, lr_param_grid, config, "Logistic Regression")
    
    # Train Naive Bayes
    nb_pipeline = Pipeline([
        ("tfidf", tfidf),
        ("clf", create_naive_bayes(config)),
    ])
    
    nb_param_grid = {
        "tfidf__max_features": [5000, 7000],
        "tfidf__ngram_range": [(1, 2), (1, 3)],
        "clf__alpha": [0.1, 0.5, 1.0],
    }
    
    nb_grid = train_model(nb_pipeline, X_train, y_train, nb_param_grid, config, "Naive Bayes")
    
    # Evaluate best individual models
    logger.info("\n" + "="*60)
    logger.info("EVALUATING BEST INDIVIDUAL MODELS")
    logger.info("="*60)
    
    lr_results = evaluate_model(lr_grid.best_estimator_, X_test, y_test, "Best Logistic Regression")
    nb_results = evaluate_model(nb_grid.best_estimator_, X_test, y_test, "Best Naive Bayes")
    
    # Create and evaluate ensemble
    logger.info("\n" + "="*60)
    logger.info("TRAINING AND EVALUATING ENSEMBLE")
    logger.info("="*60)
    
    ensemble = create_ensemble(lr_grid, nb_grid, config, n_models=3)
    ensemble.fit(X_train, y_train)
    ensemble_results = evaluate_model(ensemble, X_test, y_test, "Ensemble")
    
    # Save models
    paths_config = config["paths"]
    model_dir = Path(paths_config["model_dir"])
    
    save_model(lr_grid.best_estimator_, model_dir / paths_config["best_lr_model"])
    save_model(nb_grid.best_estimator_, model_dir / paths_config["best_nb_model"])
    save_model(ensemble, model_dir / paths_config["best_ensemble_model"])
    preprocessor.save(model_dir / paths_config["preprocessor_path"])
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE - ALL MODELS SAVED")
    logger.info("="*60)
    logger.info("Best Logistic Regression: %.4f", lr_results["accuracy"])
    logger.info("Best Naive Bayes: %.4f", nb_results["accuracy"])
    logger.info("Ensemble: %.4f", ensemble_results["accuracy"])


if __name__ == "__main__":
    main()