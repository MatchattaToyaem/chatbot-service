"""
Prompt templates for O'Connors IMS chatbot answer generation.

These prompts are tuned for high faithfulness — the LLM must ground
its answers strictly in the retrieved document chunks and cite sources.
Any information not present in the context must be flagged as such.

The system prompt establishes O'Connors' domain (HVAC, mechanical
engineering, Australian workplace safety) and instructs the model
to behave as a knowledgeable IMS assistant.
"""


class PromptTemplates:
    """
    Prompt templates for the RAG answer generation pipeline.

    All templates use {context} and {question} placeholders that are
    filled at runtime with the retrieved chunks and user query.
    """

    SYSTEM_PROMPT = (
        "You are the O'Connors Services IMS Assistant, an expert on the company's "
        "Integrated Management System covering HVAC, mechanical engineering, "
        "workplace health and safety, and Australian Standards compliance.\n\n"
        "RULES:\n"
        "1. Answer ONLY based on the provided context documents. Do not use "
        "outside knowledge.\n"
        "2. If the context does not contain enough information to answer the "
        "question, say: \"The available IMS documents do not contain sufficient "
        "information to answer this question.\"\n"
        "3. Always cite the source document name when referencing specific "
        "information (e.g., \"According to SOP-064...\").\n"
        "4. Be specific and precise. Quote relevant sections where appropriate.\n"
        "5. If the question asks about a procedure or process, list the steps "
        "in the order they appear in the source document.\n"
        "6. Use professional language appropriate for a workplace safety context.\n"
        "7. If multiple documents contain relevant information, synthesise them "
        "and cite each source."
    )

    USER_PROMPT_TEMPLATE = (
        "Context documents:\n"
        "---\n"
        "{context}\n"
        "---\n\n"
        "Question: {question}\n\n"
        "Provide a detailed answer based strictly on the context documents above. "
        "Cite the source document for each key point."
    )

    @classmethod
    def format_context(cls, chunks: list[dict]) -> str:
        """
        Format retrieved chunks into a context string for the prompt.

        Each chunk is labelled with its source file and subfolder for
        clear attribution in the generated answer.
        """
        parts = []
        for i, chunk in enumerate(chunks):
            source = chunk.get("source_file", "Unknown")
            subfolder = chunk.get("subfolder", "")
            text = chunk.get("text", "")

            header = f"[Document {i+1}: {source}]"
            if subfolder:
                header = f"[Document {i+1}: {source} ({subfolder})]"

            parts.append(f"{header}\n{text}")

        return "\n\n".join(parts)

    @classmethod
    def build_prompt(cls, question: str, chunks: list[dict]) -> tuple[str, str]:
        """
        Build the system and user prompts for answer generation.

        Args:
            question: The user's question.
            chunks: List of chunk dicts with 'text', 'source_file', 'subfolder'.

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        context = cls.format_context(chunks)
        user_prompt = cls.USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )
        return cls.SYSTEM_PROMPT, user_prompt
