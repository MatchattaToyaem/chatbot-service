import grpc
import logging
import os
import sys
from concurrent import futures

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chatbot_service_pb2
import chatbot_service_pb2_grpc
from config import AppConfig
from rag.service import RAGService

log_file = os.environ.get("LOG_FILE", "/var/log/app/chatbot-service.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
)
logger = logging.getLogger(__name__)


def sanity_check():
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "huggingface":
        token = os.getenv("HUGGING_FACE_HUB_TOKEN", "")
        if not token:
            raise RuntimeError("Sanity check failed: HUGGING_FACE_HUB_TOKEN is not set")
        logger.info("Sanity check passed: LLM_PROVIDER=huggingface, token present.")

    elif provider == "azure-foundry":
        key = os.getenv("AZURE_FOUNDRY_KEY", "")
        endpoint = os.getenv("AZURE_FOUNDRY_ENDPOINT", "")
        if not key:
            raise RuntimeError("Sanity check failed: AZURE_FOUNDRY_KEY is not set")
        if not endpoint:
            raise RuntimeError("Sanity check failed: AZURE_FOUNDRY_ENDPOINT is not set")
        logger.info("Sanity check passed: LLM_PROVIDER=azure-foundry, key and endpoint present.")

    else:
        endpoint = os.getenv("OLLAMA_ENDPOINT", "http://ollama-service:11434")
        model = os.getenv("LLM_MODEL", "llama3.2:3b")
        logger.info("Sanity check: Ollama endpoint=%s model=%s", endpoint, model)
        try:
            r = requests.get(f"{endpoint}/v1/models", timeout=10)
            r.raise_for_status()
            available = [m["id"] for m in r.json().get("data", [])]
            if model not in available:
                raise RuntimeError(f"Model '{model}' not found in Ollama. Available: {available}")
            logger.info("Sanity check passed: Ollama reachable, model '%s' is loaded.", model)
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Sanity check failed: cannot reach Ollama at {endpoint}")
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Sanity check failed: Ollama at {endpoint} timed out")


class AIServiceServicer(chatbot_service_pb2_grpc.HuggingFaceServiceServicer):
    def __init__(self):
        sanity_check()
        logger.info("Initializing RAG service...")
        self.rag = RAGService(AppConfig())
        logger.info("Pre-warming BM25 index...")
        self.rag.retrieve("warmup", retrieval_k=1, final_k=1)
        logger.info("RAG service initialized successfully.")

    def GenerateResponse(self, request, context):
        session_id = request.session_id or ""
        logger.info(f"Received prompt: {request.prompt} [session_id={session_id or 'none'}]")
        try:
            response = self.rag.ask(question=request.prompt, session_id=session_id)
            sources = [
                chatbot_service_pb2.Source(
                    file=s.get("file", ""),
                    subfolder=s.get("subfolder", ""),
                    score=s.get("score", 0.0),
                    chunk_id=s.get("chunk_id", ""),
                )
                for s in response.sources
            ]
            return chatbot_service_pb2.InferenceReply(
                result=response.answer,
                confidence=response.confidence,
                sources=sources,
                model=response.model,
            )
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return chatbot_service_pb2.InferenceReply()


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    chatbot_service_pb2_grpc.add_HuggingFaceServiceServicer_to_server(AIServiceServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    logger.info("AI gRPC Server is running on port 50051...")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()