FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for gRPC
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model files so cold starts don't fetch 2.3 GB at runtime.
# Uses snapshot_download (no model load into memory) to avoid OOM during build.
ARG HUGGING_FACE_HUB_TOKEN
RUN HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN python -c \
    "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BAAI/bge-m3')"

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
