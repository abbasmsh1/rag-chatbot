FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
COPY scripts/ scripts/
# registry, stored docs, embedded qdrant, and fastembed model cache live here
ENV RAG_DATA_DIR=/data HF_HOME=/data/models
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
