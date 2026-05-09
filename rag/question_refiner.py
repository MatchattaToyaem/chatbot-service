"""
Question refinement and relevance checking via a single LLM call.

Given a new question and recent conversation history the LLM:
  1. Decides whether the new question is a follow-up / related to the
     conversation context (is_context_relevant).
  2. If RELATED  → rewrites the question as a fully self-contained,
     standalone query optimised for vector database retrieval
     (resolves pronouns, expands implicit references).
  3. If UNRELATED → returns the original question unchanged.

The refined question is what gets sent to the vector DB; the original
question is still what the answer generator answers.
"""

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a query reformulation assistant. Your task is to analyse a new "
    "question in the context of a recent conversation and produce the optimal "
    "standalone search query for a vector database retrieval system.\n\n"
    "RULES:\n"
    "1. Determine whether the new question is a follow-up or related to the "
    "conversation history.\n"
    "2. If RELATED: Rewrite the question as a fully self-contained query that:\n"
    "   - Resolves all pronouns (it, they, them, this, that) to their explicit "
    "referents from the conversation.\n"
    "   - Expands implicit references ('the procedure above', 'the chemical "
    "mentioned', 'that standard').\n"
    "   - Incorporates enough context so the query stands alone without the history.\n"
    "   - Is concise and optimised for semantic vector search.\n"
    "3. If NOT RELATED (completely new topic): Keep the question exactly as given.\n\n"
    "RESPOND with a JSON object only — no preamble, no explanation:\n"
    "{\n"
    "  \"is_context_relevant\": true | false,\n"
    "  \"refined_question\": \"<the refined or original question>\"\n"
    "}"
)


class QuestionRefiner:
    """
    Uses the configured LLM to refine a question given prior chat history.

    Shares the same provider/client as the AnswerGenerator so no extra
    LLM connection is created.
    """

    def __init__(self, client, model: str, provider: str, temperature: float = 0.0):
        self._client = client
        self._model = model
        self._provider = provider
        self._temperature = temperature

    def refine(self, question: str, history: list[dict]) -> tuple[str, bool]:
        """
        Refine the question using conversation history.

        Args:
            question: The new user question.
            history:  Last N chat history dicts with 'question' and 'answer'.

        Returns:
            (refined_question, is_context_relevant)
            On any error returns (original_question, False) — graceful fallback.
        """
        if not history:
            return question, False

        history_text = self._format_history(history)
        user_content = (
            f"Conversation history:\n{history_text}\n\n"
            f"New question: {question}"
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            if self._provider == "huggingface":
                response = self._client.chat_completion(
                    model=self._model,
                    messages=messages,
                    max_tokens=256,
                    temperature=self._temperature,
                )
            else:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=256,
                    temperature=self._temperature,
                )

            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw).strip()

            parsed = json.loads(raw)
            refined = parsed.get("refined_question", "").strip() or question
            is_relevant = bool(parsed.get("is_context_relevant", False))

            logger.info(
                "Question refiner [relevant=%s]: '%s' → '%s'",
                is_relevant,
                question[:80],
                refined[:80],
            )
            return refined, is_relevant

        except Exception as exc:
            logger.warning("Question refinement failed (%s): %s", type(exc).__name__, exc)
            return question, False

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        parts = []
        for i, entry in enumerate(history, 1):
            q = entry.get("question", "")
            a = entry.get("answer", "")
            # Truncate long answers so the refinement prompt stays lean
            if len(a) > 400:
                a = a[:400] + "..."
            parts.append(f"Q{i}: {q}\nA{i}: {a}")
        return "\n\n".join(parts)
