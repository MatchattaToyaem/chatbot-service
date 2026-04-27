"""
Prompt templates for O'Connors IMS chatbot answer generation.

These prompts are tuned for high faithfulness — the LLM must ground
its answers strictly in the retrieved document chunks and cite sources.
Any information not present in the context must be flagged as such.

The system prompt establishes O'Connors' domain (HVAC, mechanical
engineering, Australian workplace safety) and instructs the model
to behave as a knowledgeable IMS assistant.

CHANGES (v2.1 — 24 Apr 2026):
  - Prompt now instructs LLM to cite sources naturally (not "Document 1:")
  - format_context uses clean document names (not raw file paths)
  - Added rules 8-12 for technical precision (numbers, standards, GHS, PPE)
  - Added _clean_source_name helper to convert paths to readable names
"""

import os
import re


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
        "3. Cite sources naturally within your answer using the document name "
        "shown in square brackets. For example: \"According to the R134A "
        "Material Safety Data Sheet, Section 7...\" or \"The After Hours "
        "On-Call Coverage Policy states...\"\n"
        "4. NEVER show raw file paths, document numbers (Document 1, Document 2), "
        "chunk IDs, confidence scores, or any internal system identifiers in "
        "your answer. The user is a field technician, not a developer.\n"
        "5. Be specific and precise. Include exact numbers, thresholds, "
        "distances, temperatures, and concentrations from the source documents.\n"
        "6. If the question asks about a procedure or process, list the steps "
        "in the order they appear in the source document.\n"
        "7. Use professional language appropriate for a workplace safety context.\n"
        "8. If multiple documents contain relevant information, synthesise them "
        "and cite each source naturally.\n"
        "9. When referencing Australian Standards, always include the full "
        "standard number (e.g., AS/NZS 3666.1-2011).\n"
        "10. For MSDS/SDS questions, always specify the GHS section number and "
        "title (e.g., Section 7: Handling and Storage).\n"
        "11. For PPE requirements, list each item specifically (e.g., "
        "\"chemical-resistant gloves, safety goggles, P2 respirator\").\n"
        "12. At the end of your answer, add a brief source line starting with "
        "\"Source:\" listing the document names used, separated by commas."
    )

    USER_PROMPT_TEMPLATE = (
        "Context documents:\n"
        "---\n"
        "{context}\n"
        "---\n\n"
        "Question: {question}\n\n"
        "Provide a clear, professional answer based strictly on the context "
        "documents above. Cite sources naturally within the answer. End with "
        "a brief \"Source:\" line listing the documents referenced."
    )

    @staticmethod
    def _clean_source_name(source_file: str, subfolder: str = "") -> str:
        """
        Convert a raw file path into a clean, human-readable document name.

        Examples:
            "MSDS's/R134A ID 18 07 2022.pdf" → "R134A Material Safety Data Sheet"
            "Policies/After Hours On-Call Coverage Policy ID  11.07.23.pdf"
                → "After Hours On-Call Coverage Policy"
            "SOP's/SOP-001 Personal Protective Equipment _ v.2 01.09.25.pdf"
                → "SOP-001 Personal Protective Equipment"
            "Australian Standards/AS NZS 3666.1-2011 Air-handling..."
                → "AS NZS 3666.1-2011 Air-handling and water systems..."
        """
        # Get just the filename without folder path
        name = os.path.basename(source_file)

        # Remove file extension
        name = os.path.splitext(name)[0]

        # Remove date patterns: "ID 18 07 2022", "ID  11.07.23", "ID 29 05 2022"
        name = re.sub(r'\s*ID\s+\d{2}[\s.]\d{2}[\s.]\d{2,4}', '', name)

        # Remove version patterns: "_ v.2 01.09.25", "_ v.3 04.11.24", "_v.3 04.11.24"
        name = re.sub(r'\s*_?\s*v\.?\d+\s+\d{2}\.\d{2}\.\d{2,4}', '', name)

        # Remove revision patterns: "_ Issue 8 18 09 2023", "_ REV A _v.4 19.11.24"
        name = re.sub(r'\s*_?\s*Issue\s+\d+\s+\d{2}\s+\d{2}\s+\d{4}', '', name)
        name = re.sub(r'\s*_?\s*REV\s+\w+\s*', ' ', name)

        # Clean up underscores used as separators
        name = name.replace(' _ ', ' — ')
        name = name.replace('_', ' ')

        # Clean up multiple spaces
        name = re.sub(r'\s+', ' ', name).strip()

        # Add "Material Safety Data Sheet" suffix for MSDS files
        if subfolder and "MSDS" in subfolder and "MSDS" not in name and "SDS" not in name:
            name = f"{name} Material Safety Data Sheet"

        # Add "Policy" suffix if in Policies folder but not in name
        if subfolder == "Policies" and "Policy" not in name:
            name = f"{name} Policy"

        return name

    @classmethod
    def format_context(cls, chunks: list[dict]) -> str:
        """
        Format retrieved chunks into a context string for the prompt.

        Each chunk is labelled with a clean, human-readable document name
        (not raw file paths) for natural citation in the generated answer.
        """
        parts = []
        for i, chunk in enumerate(chunks):
            source = chunk.get("source_file", "Unknown")
            subfolder = chunk.get("subfolder", "")
            text = chunk.get("text", "")

            clean_name = cls._clean_source_name(source, subfolder)
            parts.append(f"[{clean_name}]\n{text}")

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
