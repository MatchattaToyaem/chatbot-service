from rag.service import RAGService

print("Loading RAG service...")
rag = RAGService()
print("Ready!\n")

while True:
    q = input("Ask: ")
    if q.lower() in ("quit", "exit"):
        break
    r = rag.ask(q)
    print(f"\n{r.answer}")
    print(f"\nConfidence: {r.confidence}")
    for s in r.sources:
        print(f"  Source: {s['file']}")
    print()