"""Configuration for the LangGraph + LangChain GenAI summary service."""
from __future__ import annotations

import os

# --- Primary provider: Google Gemini (free tier) -----------------------------
# Create a free Gemini API key at https://aistudio.google.com/apikey and set
# GOOGLE_API_KEY. The default model below is served on Google's free tier.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL_ID = os.environ.get("SUMMARY_GEMINI_MODEL_ID", "gemini-3.6-flash")

# --- Fallback provider: Hugging Face inference (Qwen) -------------------------
# Used only if the Gemini call fails/rate-limits. Requires a Hugging Face
# access token with Inference API permissions.
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
QWEN_MODEL_ID = os.environ.get("SUMMARY_QWEN_MODEL_ID", "Qwen/Qwen3.8-27B")

MAX_NEW_TOKENS = int(os.environ.get("SUMMARY_MAX_NEW_TOKENS", "2000"))
TEMPERATURE = float(os.environ.get("SUMMARY_TEMPERATURE", "0.2"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("SUMMARY_REQUEST_TIMEOUT_SECONDS", "30"))
MAX_BATCH_PROJECTS = int(os.environ.get("SUMMARY_MAX_BATCH_PROJECTS", "25"))
MODEL_VERSION = "summary-langgraph-gemini-qwen-v3"
