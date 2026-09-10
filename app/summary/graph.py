"""LangGraph orchestration for PAIMANA GenAI summaries.

The graph tries the free Google Gemini model first and falls through to the
Hugging Face-hosted Qwen model when the primary model/provider is
unavailable, rate-limited, times out, or returns another retryable upstream
failure. A missing/invalid API key for one candidate does not abort the
whole graph -- Gemini and the Hugging Face Qwen fallback use independent
credentials, so the graph simply moves on to the next candidate.
"""
from __future__ import annotations

from typing import TypedDict

from fastapi import HTTPException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import END, START, StateGraph

from app.common.http import logger

from . import config as cfg


class SummaryGraphState(TypedDict, total=False):
    messages: list[BaseMessage]
    text: str
    model: str
    errors: list[str]
    candidate_index: int


# Each candidate is a (provider, model_id) pair. Gemini (free tier) is tried
# first; the Hugging Face-hosted Qwen model is the fallback.
_CLIENTS: dict[tuple[str, str], BaseChatModel] = {}
_GRAPH = None


def _candidate_models() -> list[tuple[str, str]]:
    candidates = [
        ("google", cfg.GEMINI_MODEL_ID),
        ("huggingface", cfg.QWEN_MODEL_ID),
    ]
    ordered = [(provider, model) for provider, model in candidates if model]
    return list(dict.fromkeys(ordered))


def _coerce_messages(messages: list[dict[str, str] | BaseMessage]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, BaseMessage):
            out.append(message)
            continue
        role = message.get("role", "user")
        content = str(message.get("content", ""))
        if role == "system":
            out.append(SystemMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out


def _build_client(provider: str, model: str) -> BaseChatModel:
    if provider == "google":
        if not cfg.GOOGLE_API_KEY:
            raise HTTPException(
                status_code=503,
                detail="GenAI summary is not configured. Set GOOGLE_API_KEY in the backend environment.",
            )
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=cfg.GOOGLE_API_KEY,
            max_output_tokens=cfg.MAX_NEW_TOKENS,
            temperature=cfg.TEMPERATURE,
            timeout=cfg.REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
            thinking_level="low"
        )

    if provider == "huggingface":
        if not cfg.HUGGINGFACEHUB_API_TOKEN:
            raise HTTPException(
                status_code=503,
                detail="GenAI summary fallback is not configured. Set HUGGINGFACEHUB_API_TOKEN in the backend environment.",
            )
        endpoint = HuggingFaceEndpoint(
            repo_id=model,
            huggingfacehub_api_token=cfg.HUGGINGFACEHUB_API_TOKEN,
            max_new_tokens=cfg.MAX_NEW_TOKENS,
            temperature=cfg.TEMPERATURE,
            timeout=cfg.REQUEST_TIMEOUT_SECONDS,
        )
        return ChatHuggingFace(llm=endpoint)

    raise HTTPException(status_code=503, detail=f"Unknown GenAI provider '{provider}'.")


def _get_client(provider: str, model: str) -> BaseChatModel:
    key = (provider, model)
    client = _CLIENTS.get(key)
    if client is None:
        client = _build_client(provider, model)
        _CLIENTS[key] = client
    return client


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if text:
                    chunks.append(str(text))
            elif block is not None:
                chunks.append(str(block))
        return "".join(chunks).strip()
    return str(content or "").strip()


def _status_code(exc: Exception) -> int | None:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def _is_retryable(exc: Exception) -> bool:
    status = _status_code(exc)
    if status is not None:
        return status == 408 or status == 409 or status == 425 or status == 429 or 500 <= status < 600 or status == 404
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return any(
        marker in name or marker in text
        for marker in ("timeout", "temporarily", "connection", "connect", "reset", "unavailable")
    )


def _make_node(index: int, provider: str, model: str):
    def call_model(state: SummaryGraphState) -> SummaryGraphState:
        try:
            response = _get_client(provider, model).invoke(state["messages"])
            text = _extract_text(response.content)
            if not text:
                raise RuntimeError("model returned an empty response")
            return {"text": text, "model": model, "candidate_index": index}
        except HTTPException as exc:
            # Missing/invalid API key for THIS candidate: record it and let
            # routing move on to the next provider instead of aborting the
            # whole graph, since Gemini and the HF fallback use independent
            # credentials.
            logger.warning("GenAI model %s (%s) is not configured: %s", model, provider, exc.detail)
            return {
                "errors": [*state.get("errors", []), f"{model}: {exc.detail}"],
                "candidate_index": index,
            }
        except Exception as exc:
            logger.warning("GenAI model %s (%s) failed: %s", model, provider, exc)
            if not _is_retryable(exc):
                raise HTTPException(
                    status_code=502,
                    detail=f"GenAI request failed for model {model}: {exc}",
                ) from exc
            return {
                "errors": [*state.get("errors", []), f"{model}: {exc}"],
                "candidate_index": index,
            }

    call_model.__name__ = f"call_model_{index}"
    return call_model


def _route_after_candidate(state: SummaryGraphState):
    if state.get("text"):
        return END
    next_index = state.get("candidate_index", 0) + 1
    models = _candidate_models()
    return f"call_model_{next_index}" if next_index < len(models) else END


def get_summary_graph():
    global _GRAPH
    if _GRAPH is not None:
        return _GRAPH

    models = _candidate_models()
    if not models:
        raise HTTPException(status_code=503, detail="No GenAI models are configured.")

    graph = StateGraph(SummaryGraphState)
    for index, (provider, model) in enumerate(models):
        graph.add_node(f"call_model_{index}", _make_node(index, provider, model))
    graph.add_edge(START, "call_model_0")
    for index in range(len(models)):
        graph.add_conditional_edges(f"call_model_{index}", _route_after_candidate)
    _GRAPH = graph.compile()
    return _GRAPH


def generate_with_graph(messages: list[dict[str, str] | BaseMessage]) -> tuple[str, str]:
    """Invoke the LangGraph and return ``(text, model_used)``."""
    result = get_summary_graph().invoke({
        "messages": _coerce_messages(messages),
        "candidate_index": 0,
        "errors": [],
    })
    if result.get("text"):
        return result["text"], result.get("model", cfg.GEMINI_MODEL_ID)
    errors = result.get("errors", [])
    raise HTTPException(
        status_code=502,
        detail="All configured GenAI models failed. " + (" | ".join(errors) if errors else "No model response was produced."),
    )
