import grpc
from concurrent import futures
import os

# Import the generated gRPC code
import chatbot_service_pb2
import chatbot_service_pb2_grpc

# Import Hugging Face InferenceClient
from huggingface_hub import InferenceClient

# Read Hugging Face API token from environment variable
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")

class AIServiceServicer(chatbot_service_pb2_grpc.HuggingFaceServiceServicer):
    def __init__(self):
        print("Initializing Hugging Face InferenceClient...")
        self.client = InferenceClient(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            provider="featherless-ai",
            token=HUGGINGFACEHUB_API_TOKEN,
        )
        print("Model initialized successfully.")

    def GenerateResponse(self, request, context):
        print(f"Received prompt: {request.prompt}")
        
        try:
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": request.prompt}],
                max_tokens=256,
                temperature=0.7,
            )
            response_text = response.choices[0].message.content
            
            # Return the result via gRPC
            return chatbot_service_pb2.InferenceReply(result=response_text)
            
        except Exception as e:
            # Handle errors and return gRPC error status
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return chatbot_service_pb2.InferenceReply()

def serve():
    # Set up the gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    chatbot_service_pb2_grpc.add_HuggingFaceServiceServicer_to_server(AIServiceServicer(), server)
    
    # Listen on port 50051
    server.add_insecure_port('[::]:50051')
    server.start()
    print("AI gRPC Server is running on port 50051...")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()