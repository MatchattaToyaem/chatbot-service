"""
Answer generator supporting two provider modes:
  - "ollama"      : self-hosted Ollama via OpenAI-compatible API (default)
  - "huggingface" : HuggingFace Inference API

Takes retrieved and reranked document chunks, builds a domain-specific
prompt, and generates a grounded answer via HuggingFace InferenceClient.
Includes confidence scoring based on retrieval quality.

CHANGES (v2.2 — 05 May 2026):
  - Added document_path (full file path) to sources for frontend document display
  - Added seed=42 to LLM calls for deterministic answers
CHANGES (v2.1 — 24 Apr 2026):
  - Default temperature changed from 0.1 to 0.0 for consistent answers
  - Sources output now includes clean document names (not raw paths)
  - Removed chunk_id from client-facing sources (internal detail)
"""

import logging
import os
import re
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
        elif self._provider == "azure-foundry":
            self._init_azure_foundry()
        else:
            self._init_ollama()

    def _init_azure_foundry(self):
        from openai import OpenAI
        key = os.getenv("AZURE_FOUNDRY_KEY", "")
        if not key:
            raise RuntimeError("AZURE_FOUNDRY_KEY is required for LLM_PROVIDER=azure-foundry")
        endpoint = os.getenv("AZURE_FOUNDRY_ENDPOINT", "")
        if not endpoint:
            raise RuntimeError("AZURE_FOUNDRY_ENDPOINT is required for LLM_PROVIDER=azure-foundry")
        self._model = os.getenv("AZURE_FOUNDRY_MODEL", self._model)
        self._client = OpenAI(base_url=endpoint, api_key=key)
        logger.info("LLM provider: Azure Foundry | endpoint=%s | model=%s", endpoint, self._model)

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
        hf_model = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        hf_provider = os.getenv("HF_PROVIDER", "featherless-ai")
        self._model = hf_model
        self._client = InferenceClient(token=token, provider=hf_provider)
        logger.info("LLM provider: HuggingFace | model=%s | provider=%s", hf_model, hf_provider)

    def _compute_confidence(self, reranked_results: list) -> float:
        if not reranked_results:
            return 0.0
        scores = [r.reranked_score for r in reranked_results]
        avg_score = sum(scores) / len(scores)
        confidence = min(1.0, max(0.0, avg_score))
        if reranked_results[0].boost_reasons:
            confidence = min(1.0, confidence + 0.05)
        return round(confidence, 3)

    @staticmethod
    def _clean_source_name(raw_path: str, subfolder: str = "") -> str:
        """Convert raw file path to clean document name for API response."""
        name = os.path.basename(raw_path)
        name = os.path.splitext(name)[0]
        name = re.sub(r'\s*ID\s+\d{2}[\s.]\d{2}[\s.]\d{2,4}', '', name)
        name = re.sub(r'\s*_?\s*v\.?\d+\s+\d{2}\.\d{2}\.\d{2,4}', '', name)
        name = re.sub(r'\s*_?\s*Issue\s+\d+\s+\d{2}\s+\d{2}\s+\d{4}', '', name)
        name = re.sub(r'\s*_?\s*REV\s+\w+\s*', ' ', name)
        name = name.replace(' _ ', ' — ').replace('_', ' ')
        name = re.sub(r'\s+', ' ', name).strip()
        if subfolder and "MSDS" in subfolder and "MSDS" not in name:
            name = f"{name} MSDS"
        return name

    def generate(
        self,
        question: str,
        reranked_results: list,
    ) -> GeneratedAnswer:
        """
        Generate an answer from reranked retrieval results.

        Args:
            question: The user's question.
            reranked_results: List of RankedResult objects from the reranker.

        Returns:
            GeneratedAnswer with the response, sources, and confidence.
        """
        chunks = []
        sources = []
        seen_files = set()

        for r in reranked_results:
            raw_file = r.metadata.get("source_file", "Unknown")
            subfolder = r.metadata.get("subfolder", "")

            chunks.append({
                "text": r.document,
                "source_file": raw_file,
                "subfolder": subfolder,
            })

            # Deduplicate sources — only list each document once
            if raw_file not in seen_files:
                seen_files.add(raw_file)
                sources.append({
                    "file": self._clean_source_name(raw_file, subfolder),
                    "document_path": raw_file,
                    "subfolder": subfolder,
                    "score": round(r.reranked_score, 3),
                    "chunk_id": r.chunk_id,
                })

        # Build prompt
        system_prompt, user_prompt = PromptTemplates.build_prompt(question, chunks)
        confidence = self._compute_confidence(reranked_results)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            if self._provider == "huggingface":
                response = self._client.chat_completion(
                    model=self._model,
                    messages=messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
            else:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    seed=42,
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
                answer=(
                    "An error occurred while processing your question. "
                    "Please try again or contact your supervisor."
                ),
                sources=sources,
                confidence=confidence,
                model=self._model,
            )
