import grpc
import logging
import os
from concurrent import futures

import chatbot_service_pb2
import chatbot_service_pb2_grpc
from huggingface_hub import InferenceClient

HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
if not HUGGINGFACEHUB_API_TOKEN:
    raise RuntimeError("HUGGING_FACE_HUB_TOKEN environment variable is not set")

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
        logger.info("Initializing Hugging Face InferenceClient...")
        self.client = InferenceClient(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            provider="featherless-ai",
            token=HUGGINGFACEHUB_API_TOKEN,
        )
        logger.info("Model initialized successfully.")

    def GenerateResponse(self, request, context):
        logger.info(f"Received prompt: {request.prompt}")
        try:
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": request.prompt}],
                max_tokens=256,
                temperature=0.7,
            )
            response_text = response.choices[0].message.content
            return chatbot_service_pb2.InferenceReply(result=response_text)
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