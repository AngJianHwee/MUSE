"""
Gradio web interface for Twitter sentiment analysis.
Provides an interactive UI for real-time sentiment prediction.
Supports MCP (Model Context Protocol) server for tool integration.
"""

import argparse
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


# Color mapping for sentiment labels
SENTIMENT_COLORS = {
    "POSITIVE": "#22c55e",  # green-500
    "NEGATIVE": "#ef4444",  # red-500
    "NEUTRAL": "#3b82f6",   # blue-500
    "IRRELEVANT": "#6b7280", # gray-500
}


def get_sentiment_color(sentiment: str) -> str:
    """Get color for sentiment label."""
    return SENTIMENT_COLORS.get(sentiment.upper(), "#6b7280")


def format_colored_sentiment(sentiment: str, confidence: Optional[float] = None) -> str:
    """Format sentiment with colored HTML for Gradio display."""
    color = get_sentiment_color(sentiment)
    label = sentiment.upper()
    if confidence is not None:
        return f'<span style="color: {color}; font-weight: bold; font-size: 1.2em;">{label}</span> <span style="color: {color}; opacity: 0.7;">({confidence:.1%})</span>'
    return f'<span style="color: {color}; font-weight: bold; font-size: 1.2em;">{label}</span>'


def format_probabilities_html(probabilities: Dict[str, float]) -> str:
    """Format probabilities as colored HTML bars."""
    if not probabilities:
        return ""
    
    html_parts = []
    for label, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        color = get_sentiment_color(label)
        bar_width = prob * 100
        html_parts.append(f'''
            <div style="margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: {color}; font-weight: 500;">{label.upper()}</span>
                    <span style="color: {color}; font-weight: bold;">{prob:.1%}</span>
                </div>
                <div style="background: #e5e7eb; border-radius: 4px; height: 20px; overflow: hidden;">
                    <div style="background: {color}; width: {bar_width}%; height: 100%; border-radius: 4px; transition: width 0.3s ease;"></div>
                </div>
            </div>
        ''')
    return "".join(html_parts)


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
        
        # Get confidence for predicted class
        confidence = prob_dict.get(predicted, 0.0)
        
        return {
            "predicted_sentiment": predicted,
            "cleaned_text": cleaned,
            "probabilities": prob_dict,
            "confidence": confidence,
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
                    # Use HTML for colored sentiment display
                    predicted_html = gr.HTML(
                        label="Predicted Sentiment",
                        value="<div style='text-align: center; padding: 20px; color: #9ca3af;'>Enter text and click Analyze</div>"
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
                    # Use HTML for colored probability bars
                    prob_html = gr.HTML(
                        label="Class Probabilities",
                        value="<div style='padding: 20px; color: #9ca3af; text-align: center;'>Probabilities will appear here</div>"
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
                    error_html = f'<div style="text-align: center; padding: 20px; color: #ef4444;">{result["error"]}</div>'
                    return error_html, "", '<div style="padding: 20px; color: #9ca3af; text-align: center;">Error</div>'
                
                # Format colored sentiment
                sentiment_html = format_colored_sentiment(
                    result["predicted_sentiment"],
                    result.get("confidence", 0)
                )
                sentiment_display = f'<div style="text-align: center; padding: 20px;">{sentiment_html}</div>'
                
                # Format colored probabilities
                prob_html = format_probabilities_html(result["probabilities"])
                prob_display = f'<div style="padding: 10px;">{prob_html}</div>' if prob_html else '<div style="padding: 20px; color: #9ca3af; text-align: center;">No probabilities available</div>'
                
                return sentiment_display, result["cleaned_text"], prob_display
            
            predict_btn.click(
                fn=predict_wrapper,
                inputs=[text_input, model_dropdown],
                outputs=[predicted_html, cleaned_text, prob_html],
            )
            
            # Also trigger on Enter key
            text_input.submit(
                fn=predict_wrapper,
                inputs=[text_input, model_dropdown],
                outputs=[predicted_html, cleaned_text, prob_html],
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
            "mcp_server": mcp_server or self.gradio_config.get("mcp_server", False),
            "debug": self.gradio_config.get("debug", False),
        }
        launch_kwargs.update(kwargs)
        
        logger.info("Launching Gradio app on port with kwargs: %s", launch_kwargs)
        demo.launch(**launch_kwargs)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Twitter Sentiment Analysis Gradio App")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (overrides config)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link"
    )
    parser.add_argument(
        "--mcp-server",
        action="store_true",
        help="Enable MCP server"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    return parser.parse_args()


def main():
    """Main entry point for the Gradio app."""
    args = parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    app = GradioApp(config_path=args.config)
    
    # Override config with CLI args if provided
    launch_kwargs = {}
    if args.port is not None:
        launch_kwargs["server_port"] = args.port
    if args.share:
        launch_kwargs["share"] = True
    if args.mcp_server:
        launch_kwargs["mcp_server"] = True
    if args.debug:
        launch_kwargs["debug"] = True
    
    app.launch(**launch_kwargs)


if __name__ == "__main__":
    main()