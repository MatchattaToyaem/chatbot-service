"""
Interactive Q&A for O'Connors IMS Chatbot.

Usage:
    python ask.py

Type a question and press Enter. The RAG system retrieves relevant
IMS documents and generates an answer grounded in the source material.
Type 'quit' to exit.
"""

from rag.service import RAGService

print("Loading RAG service...")
rag = RAGService()
print("Ready!\n")
print("=" * 50)
print("  O'Connors IMS Chatbot")
print("  Type a question and press Enter")
print("  Type 'quit' to exit")
print("=" * 50)

while True:
    print()
    q = input("You: ").strip()
    if not q or q.lower() in ("quit", "exit", "q"):
        print("Session ended.")
        break

    r = rag.ask(q)
    print(f"\nAssistant: {r.answer}")

    # Show clean source names (deduplicated)
    seen = set()
    print(f"\n{'─' * 40}")
    for s in r.sources:
        doc = s.get("document", s.get("file", "Unknown"))
        if doc not in seen:
            seen.add(doc)
            cat = s.get("category", s.get("subfolder", ""))
            print(f"  Source: {doc} ({cat})")
    print(f"{'─' * 40}")
