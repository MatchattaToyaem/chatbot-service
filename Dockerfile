FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for gRPC
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY server.py .
COPY config.py .
COPY chatbot_service.proto .
COPY chatbot_service_pb2.py .
COPY chatbot_service_pb2_grpc.py .
COPY rag/ ./rag/
COPY ingest/ ./ingest/

# Expose gRPC port
EXPOSE 50051

# Run the gRPC server
CMD ["python", "server.py"]
