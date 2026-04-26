"""
Answer generator supporting two provider modes:
  - "ollama"      : self-hosted Ollama via OpenAI-compatible API (default)
  - "huggingface" : HuggingFace Inference API

Controlled by the LLM_PROVIDER environment variable.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

from rag.prompts import PromptTemplates

logger = logging.getLogger(__name__)


@dataclass
class GeneratedAnswer:
    """A generated answer with source attribution and confidence."""
    answer: str
    sources: list[dict]
    confidence: float
    model: str
    prompt_tokens: Optional[int] = None
    eval_tokens: Optional[int] = None


class AnswerGenerator:
    """
    Generates answers from retrieved IMS document chunks.

    Provider is selected via LLM_PROVIDER env var:
        LLM_PROVIDER=ollama       → self-hosted Ollama (OpenAI-compatible)
        LLM_PROVIDER=huggingface  → HuggingFace Inference API
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._provider = os.getenv("LLM_PROVIDER", "ollama").lower()

        if self._provider == "huggingface":
            self._init_huggingface()
        else:
            self._init_ollama()

    def _init_ollama(self):
        from openai import OpenAI
        endpoint = os.getenv("OLLAMA_ENDPOINT", "http://ollama-service:11434")
        self._client = OpenAI(base_url=f"{endpoint}/v1", api_key="ollama")
        logger.info("LLM provider: Ollama | endpoint=%s | model=%s", endpoint, self._model)

    def _init_huggingface(self):
        from huggingface_hub import InferenceClient
        token = os.getenv("HUGGING_FACE_HUB_TOKEN", "")
        if not token:
            raise RuntimeError("HUGGING_FACE_HUB_TOKEN is required for LLM_PROVIDER=huggingface")
        hf_model = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B")
        self._model = hf_model
        self._client = InferenceClient(model=hf_model, token=token)
        logger.info("LLM provider: HuggingFace | model=%s", hf_model)

    def _compute_confidence(self, reranked_results: list) -> float:
        if not reranked_results:
            return 0.0
        scores = [r.reranked_score for r in reranked_results]
        avg_score = sum(scores) / len(scores)
        confidence = min(1.0, max(0.0, avg_score))
        if reranked_results[0].boost_reasons:
            confidence = min(1.0, confidence + 0.05)
        return round(confidence, 3)

    def generate(self, question: str, reranked_results: list) -> GeneratedAnswer:
        chunks = []
        sources = []
        for r in reranked_results:
            chunks.append({
                "text": r.document,
                "source_file": r.metadata.get("source_file", "Unknown"),
                "subfolder": r.metadata.get("subfolder", ""),
            })
            sources.append({
                "file": r.metadata.get("source_file", "Unknown"),
                "subfolder": r.metadata.get("subfolder", ""),
                "score": round(r.reranked_score, 3),
                "chunk_id": r.chunk_id,
            })

        system_prompt, user_prompt = PromptTemplates.build_prompt(question, chunks)
        confidence = self._compute_confidence(reranked_results)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            if self._provider == "huggingface":
                response = self._client.chat_completion(
                    messages=messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
                answer = response.choices[0].message.content
                prompt_tokens = getattr(response.usage, "prompt_tokens", None)
                eval_tokens = getattr(response.usage, "completion_tokens", None)
            else:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
                answer = response.choices[0].message.content
                prompt_tokens = getattr(response.usage, "prompt_tokens", None)
                eval_tokens = getattr(response.usage, "completion_tokens", None)

            return GeneratedAnswer(
                answer=answer,
                sources=sources,
                confidence=confidence,
                model=self._model,
                prompt_tokens=prompt_tokens,
                eval_tokens=eval_tokens,
            )

        except Exception as e:
            logger.error("Generation failed (%s): %s", self._provider, e)
            return GeneratedAnswer(
                answer=f"Generation failed: {e}",
                sources=sources,
                confidence=confidence,
                model=self._model,
            )
