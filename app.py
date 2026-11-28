"""
Gradio web interface for Twitter sentiment analysis.
Provides an interactive UI for real-time sentiment prediction.
Supports MCP (Model Context Protocol) server for tool integration.
"""

import argparse
import html
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import spaces
import yaml

from inference import SentimentInferenceEngine, load_engine
from text_preprocessor import ensure_nltk_resources

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

# Icons used by the result card
SENTIMENT_ICONS = {
    "POSITIVE": "😊",
    "NEGATIVE": "😞",
    "NEUTRAL": "😐",
    "IRRELEVANT": "⚪",
    "ERROR": "⚠️",
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


# Scoped styles for the result card. Kept as a plain string so CSS braces do not
# need escaping inside an f-string.
RESULT_CARD_CSS = """
<style>
@keyframes sc-rise {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: none; }
}
.sc-card {
  position: relative;
  overflow: hidden;
  box-sizing: border-box;
  border-radius: 18px;
  padding: 22px 22px 18px;
  background: linear-gradient(150deg, #ffffff 0%, #f8fafc 100%);
  border: 1px solid rgba(15, 23, 42, 0.07);
  box-shadow: 0 12px 32px rgba(15, 23, 42, 0.09), 0 2px 8px rgba(15, 23, 42, 0.04);
  color: #0f172a;
  font-family: inherit;
  animation: sc-rise 0.35s ease both;
}
.sc-card * { box-sizing: border-box; }
.sc-card__glow {
  position: absolute;
  top: -70px;
  right: -60px;
  width: 190px;
  height: 190px;
  border-radius: 50%;
  background: var(--sc-soft);
  pointer-events: none;
}
.sc-card__header {
  position: relative;
  display: flex;
  align-items: center;
  gap: 14px;
}
.sc-card__icon {
  flex: 0 0 auto;
  width: 54px;
  height: 54px;
  display: grid;
  place-items: center;
  font-size: 26px;
  border-radius: 15px;
  background: var(--sc-soft);
  box-shadow: inset 0 0 0 1px var(--sc-soft);
}
.sc-card__heading { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.sc-card__eyebrow {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #94a3b8;
}
.sc-card__label {
  font-size: 25px;
  font-weight: 800;
  line-height: 1.1;
  letter-spacing: -0.02em;
  color: var(--sc-accent);
  word-break: break-word;
}
.sc-card__confidence {
  margin-left: auto;
  flex: 0 0 auto;
  text-align: right;
  padding: 8px 12px;
  border-radius: 12px;
  background: var(--sc-soft);
}
.sc-card__confidence-value {
  display: block;
  font-size: 20px;
  font-weight: 800;
  line-height: 1;
  color: var(--sc-accent);
  font-variant-numeric: tabular-nums;
}
.sc-card__confidence-label {
  display: block;
  margin-top: 3px;
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #64748b;
}
.sc-card__divider {
  height: 1px;
  margin: 18px 0 16px;
  background: linear-gradient(90deg, var(--sc-soft), rgba(148, 163, 184, 0.12));
}
.sc-card__bars { position: relative; display: flex; flex-direction: column; gap: 11px; }
.sc-bar__top {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 5px;
  font-size: 11px;
}
.sc-bar__label {
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: #475569;
}
.sc-bar__value { font-weight: 800; color: #0f172a; font-variant-numeric: tabular-nums; }
.sc-bar__track {
  height: 7px;
  border-radius: 99px;
  background: #eef2f7;
  overflow: hidden;
}
.sc-bar__fill {
  height: 100%;
  border-radius: 99px;
  transition: width 0.45s ease;
}
.sc-card__empty {
  padding: 10px 0;
  font-size: 12px;
  font-style: italic;
  color: #94a3b8;
}
.sc-card__footer {
  position: relative;
  margin-top: 18px;
  padding-top: 13px;
  border-top: 1px dashed #e2e8f0;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 7px;
}
.sc-card__model {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 99px;
  background: #f1f5f9;
  color: #475569;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.04em;
}
.sc-card__preview {
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 11px;
  color: #64748b;
}
</style>
"""


def format_result_card(
    sentiment: Optional[str] = None,
    confidence: Optional[float] = None,
    probabilities: Optional[Dict[str, float]] = None,
    model_used: Optional[str] = None,
    cleaned_text: str = "",
) -> str:
    """Render a polished sentiment result card as HTML for the Gradio UI.

    Args:
        sentiment: Predicted label; omit to render the empty state.
        confidence: Confidence for the predicted label.
        probabilities: Class-to-probability mapping used for the bars.
        model_used: Name of the model that produced the result.
        cleaned_text: Preprocessed text shown in the card footer.

    Returns:
        HTML string containing scoped styles and the card markup.
    """
    is_empty = not sentiment
    label = (sentiment or "AWAITING ANALYSIS").upper()

    if label == "ERROR":
        accent = "#ef4444"
    elif is_empty:
        accent = "#94a3b8"
    else:
        accent = get_sentiment_color(label)

    soft = f"{accent}1a"
    icon = "💬" if is_empty else SENTIMENT_ICONS.get(label, "🔍")
    confidence_text = "—" if is_empty or confidence is None else f"{confidence:.0%}"

    if is_empty:
        bars_html = '<div class="sc-card__empty">Run an analysis to see class probabilities.</div>'
    else:
        rows = []
        for cls, prob in sorted(
            (probabilities or {}).items(), key=lambda item: float(item[1]), reverse=True
        ):
            value = float(prob)
            color = get_sentiment_color(str(cls))
            rows.append(f"""
              <div class="sc-bar">
                <div class="sc-bar__top">
                  <span class="sc-bar__label">{html.escape(str(cls).upper())}</span>
                  <span class="sc-bar__value">{value:.1%}</span>
                </div>
                <div class="sc-bar__track">
                  <div class="sc-bar__fill" style="width: {max(0.0, min(1.0, value)) * 100:.1f}%; background: linear-gradient(90deg, {color}, {color}cc);"></div>
                </div>
              </div>
            """)
        bars_html = "".join(rows) or '<div class="sc-card__empty">No probabilities available.</div>'

    model_text = f"Model · {model_used}" if model_used else "Model · —"
    preview = (
        f"&ldquo;{html.escape(cleaned_text)}&rdquo;"
        if cleaned_text
        else "Preprocessed text will appear here"
    )

    return RESULT_CARD_CSS + f"""
<div class="sc-card" style="--sc-accent: {accent}; --sc-soft: {soft};">
  <div class="sc-card__glow"></div>
  <div class="sc-card__header">
    <div class="sc-card__icon">{icon}</div>
    <div class="sc-card__heading">
      <span class="sc-card__eyebrow">Sentiment result</span>
      <span class="sc-card__label">{html.escape(label)}</span>
    </div>
    <div class="sc-card__confidence">
      <span class="sc-card__confidence-value">{confidence_text}</span>
      <span class="sc-card__confidence-label">confidence</span>
    </div>
  </div>
  <div class="sc-card__divider"></div>
  <div class="sc-card__bars">{bars_html}</div>
  <div class="sc-card__footer">
    <span class="sc-card__model">{html.escape(model_text)}</span>
    <span class="sc-card__preview">{preview}</span>
  </div>
</div>
"""


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
        
        @spaces.GPU
        def _dummy_gpu():
            """Internal ZeroGPU probe; not exposed as an MCP tool."""
            return None
        
        with gr.Blocks(title=title, theme=gr.themes.Soft()) as demo:
            gr.Markdown(f"# {title}")
            gr.Markdown(description)
            
            # Hidden button bound to the dummy GPU function so ZeroGPU's scanner detects it
            _gpu_probe = gr.Button(visible=False)
            # api_name=False keeps this internal helper out of the MCP tool list.
            _gpu_probe.click(fn=_dummy_gpu, api_name=False)
            
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
                        value=format_result_card()
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
            def predict_wrapper(text: str, model_choice: str = "logistic") -> Tuple[str, str, str]:
                """Classify a tweet's sentiment and return formatted results.

                Args:
                    text: The raw tweet text to analyze and classify.
                    model_choice: The classifier model to run ('logistic', 'naive_bayes', or 'ensemble').

                Returns:
                    Tuple of (sentiment result card HTML, preprocessed text, probability bars HTML).
                """
                result = self.predict(text, model_choice)
                if "error" in result:
                    error_html = format_result_card(
                        sentiment="ERROR",
                        model_used=model_choice,
                        cleaned_text=result["error"],
                    )
                    return error_html, "", '<div style="padding: 20px; color: #9ca3af; text-align: center;">Error</div>'
                
                # Render the main result card on the right-hand side
                sentiment_display = format_result_card(
                    sentiment=result["predicted_sentiment"],
                    confidence=result.get("confidence"),
                    probabilities=result["probabilities"],
                    model_used=result.get("model_used"),
                    cleaned_text=result["cleaned_text"],
                )
                
                # Format colored probabilities
                prob_html = format_probabilities_html(result["probabilities"])
                prob_display = f'<div style="padding: 10px;">{prob_html}</div>' if prob_html else '<div style="padding: 20px; color: #9ca3af; text-align: center;">No probabilities available</div>'
                
                return sentiment_display, result["cleaned_text"], prob_display
            
            # One dependency = one MCP tool. Registering the same function on two
            # events would create duplicate tools (`predict_wrapper` and
            # `predict_wrapper_1`) with identical, empty descriptions.
            gr.on(
                triggers=[predict_btn.click, text_input.submit],
                fn=predict_wrapper,
                inputs=[text_input, model_dropdown],
                outputs=[predicted_html, cleaned_text, prob_html],
                api_name="predict_sentiment",
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
    
    # Download NLTK data before loading models. This is essential when the app
    # is launched directly (e.g. `python app.py`) without running training first,
    # because loading a pickled preprocessor bypasses TextPreprocessor.__init__.
    if not ensure_nltk_resources():
        logger.warning(
            "Some NLTK resources could not be downloaded. "
            "Predictions that need them may fail."
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