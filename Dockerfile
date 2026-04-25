FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for gRPC
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model so cold starts don't fetch 2.3 GB at runtime
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

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
