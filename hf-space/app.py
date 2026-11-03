"""Inference API for the transformer. Deployed as a Gradio Space on ZeroGPU."""
import gradio as gr
import spaces
from transformers import pipeline

MODEL_ID = "anshy047/fake-news-detector-transformer"
MAX_TEXT_CHARS = 10000

# Runs on CPU on purpose: this small model answers quickly there, while ZeroGPU calls
# wait in a queue, use up a daily quota and were failing intermittently.
classifier = pipeline("text-classification", model=MODEL_ID, max_length=512, truncation=True,
                      top_k=None, device=-1)


def predict(text):
    return classifier(text[:MAX_TEXT_CHARS])[0]


@spaces.GPU(duration=1)
def _unused_gpu_function():
    """ZeroGPU hardware won't start a Space without one decorated function. Never called."""


gr.Interface(fn=predict, inputs=gr.Textbox(label="Text"), outputs=gr.JSON(label="Scores"),
             title="Fake News Detector", api_name="predict").launch()
