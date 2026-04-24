import grpc
import logging
import os
import sys
from concurrent import futures

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


class AIServiceServicer(chatbot_service_pb2_grpc.HuggingFaceServiceServicer):
    def __init__(self):
        logger.info("Initializing RAG service...")
        self.rag = RAGService(AppConfig())
        logger.info("RAG service initialized successfully.")

    def GenerateResponse(self, request, context):
        logger.info(f"Received prompt: {request.prompt}")
        try:
            response = self.rag.ask(question=request.prompt)
            return chatbot_service_pb2.InferenceReply(result=response.answer)
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