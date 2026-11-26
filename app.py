"""
Gradio web interface for Twitter sentiment analysis.
Provides an interactive UI for real-time sentiment prediction.
Supports MCP (Model Context Protocol) server for tool integration.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import yaml

from inference import SentimentInferenceEngine, load_engine

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class GradioApp:
    """Gradio application for sentiment analysis."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.gradio_config = self.config.get("gradio", {})
        self.engines: Dict[str, SentimentInferenceEngine] = {}
        self._load_models()
    
    def _load_models(self) -> None:
        """Load all available models."""
        model_dir = self.config["paths"]["model_dir"]
        for model_name in ("logistic", "naive_bayes", "ensemble"):
            try:
                self.engines[model_name] = load_engine(
                    model_name=model_name,
                    model_dir=model_dir,
                    config_path="config.yaml"
                )
                logger.info("Loaded model: %s", model_name)
            except FileNotFoundError:
                logger.warning("Model not found: %s", model_name)
            except Exception as e:
                logger.error("Failed to load model %s: %s", model_name, e)
    
    def predict(
        self,
        text: str,
        model_choice: str = "ensemble"
    ) -> Dict[str, Any]:
        """
        Predict sentiment for input text.
        
        Args:
            text: The tweet text to analyze
            model_choice: Which model to use (logistic, naive_bayes, ensemble)
            
        Returns:
            Dictionary with prediction results
        """
        if not text or not text.strip():
            return {"error": "Please enter some text"}
        
        if model_choice not in self.engines:
            return {"error": f"Model '{model_choice}' not available"}
        
        engine = self.engines[model_choice]
        result = engine.predict(text)
        
        predicted = str(result["predicted_sentiment"]).upper()
        cleaned = result["cleaned_text"]
        probabilities = result["probabilities"] or {}
        
        # Format probabilities for display
        prob_dict = {k.upper(): v for k, v in probabilities.items()}
        
        return {
            "predicted_sentiment": predicted,
            "cleaned_text": cleaned,
            "probabilities": prob_dict,
            "model_used": model_choice
        }
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        title = self.gradio_config.get("title", "Twitter Sentiment Analysis")
        description = self.gradio_config.get(
            "description",
            "Enter a tweet to analyze its sentiment (Positive, Negative, Neutral)"
        )
        examples = self.gradio_config.get("examples", [
            "I absolutely love this new update! It works perfectly. 😍",
            "This is the worst experience I've ever had. Terrible customer service.",
            "Just sitting here, waiting for the bus. Nothing much happening today.",
        ])
        
        available_models = list(self.engines.keys())
        if not available_models:
            available_models = ["logistic", "naive_bayes", "ensemble"]
        
        with gr.Blocks(title=title, theme=gr.themes.Soft()) as demo:
            gr.Markdown(f"# {title}")
            gr.Markdown(description)
            
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="Enter Tweet",
                        placeholder="Type or paste a tweet here...",
                        lines=3,
                        max_lines=5,
                    )
                    model_dropdown = gr.Dropdown(
                        choices=available_models,
                        value=available_models[0] if available_models else "logistic",
                        label="Select Model",
                        info="Choose which trained model to use for prediction"
                    )
                    predict_btn = gr.Button("Analyze Sentiment", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    predicted_label = gr.Label(
                        label="Predicted Sentiment",
                        num_top_classes=3,
                    )
            
            with gr.Row():
                with gr.Column():
                    cleaned_text = gr.Textbox(
                        label="Preprocessed Text",
                        lines=3,
                        interactive=False,
                    )
            
            with gr.Row():
                with gr.Column():
                    prob_output = gr.Label(
                        label="Class Probabilities",
                        num_top_classes=3,
                    )
            
            # Examples
            gr.Examples(
                examples=examples,
                inputs=text_input,
                label="Try these examples:"
            )
            
            # Event handlers
            def predict_wrapper(text, model_choice):
                result = self.predict(text, model_choice)
                if "error" in result:
                    return result["error"], "", {}
                return result["predicted_sentiment"], result["cleaned_text"], result["probabilities"]
            
            predict_btn.click(
                fn=predict_wrapper,
                inputs=[text_input, model_dropdown],
                outputs=[predicted_label, cleaned_text, prob_output],
            )
            
            # Also trigger on Enter key
            text_input.submit(
                fn=predict_wrapper,
                inputs=[text_input, model_dropdown],
                outputs=[predicted_label, cleaned_text, prob_output],
            )
        
        return demo
    
    def launch(self, mcp_server: bool = False, **kwargs) -> None:
        """Launch the Gradio app."""
        demo = self.create_interface()
        
        # Default launch parameters from config
        launch_kwargs = {
            "server_port": self.gradio_config.get("server_port", 7860),
            "share": self.gradio_config.get("share", False),
            "server_name": "0.0.0.0",
            "mcp_server": mcp_server,
        }
        launch_kwargs.update(kwargs)
        
        logger.info("Launching Gradio app on port %d (MCP: %s)",
                    launch_kwargs["server_port"], launch_kwargs["mcp_server"])
        demo.launch(**launch_kwargs)


def main():
    """Main entry point for the Gradio app."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    app = GradioApp()
    app.launch(mcp_server=True)


if __name__ == "__main__":
    main()