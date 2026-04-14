import grpc
import chatbot_service_pb2
import chatbot_service_pb2_grpc

def run():
    # Connect to the gRPC server
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = chatbot_service_pb2_grpc.HuggingFaceServiceStub(channel)
        
        # Send the request
        print("Sending prompt to AI Service...")
        response = stub.GenerateResponse(chatbot_service_pb2.PromptRequest(prompt="What is gRPC?"))
        
        print(f"AI Response: {response.result}")

if __name__ == '__main__':
    run()