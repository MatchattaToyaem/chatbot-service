import requests

resp = requests.get(
    "https://huggingface.co/api/models/meta-llama/Llama-3.2-3B-Instruct",
    params={"expand": "inferenceProviderMapping"}
)
mapping = resp.json().get("inferenceProviderMapping", {})

if not mapping:
    print("No providers available at all.")
else:
    for provider, info in mapping.items():
        print(f"{provider}: {info.get('status')}")