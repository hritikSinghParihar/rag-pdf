# RAG PDF - Production-Grade Retrieval-Augmented Generation

A scalable, containerized RAG system for processing and querying over documents using FastAPI, Qdrant, Cloudflare R2, and Celery.

## 🚀 Architecture

- **Backend**: FastAPI (Python 3.10+)
- **Vector Database**: [Qdrant](https://qdrant.tech/) for high-performance similarity search
- **Document Store**: Cloudflare R2 (S3-compatible) for persistent storage
- **Metadata/Relational**: PostgreSQL 15
- **Task Queue**: Celery with Redis for asynchronous background ingestion
- **LLM Support**: OpenAI (GPT-4o) and Google Gemini (2.0 Flash)

## ✨ Features

- **Scalable Ingestion**: Upload documents via API and process them asynchronously.
- **Advanced Chunking**: Structure-aware chunking to preserve context.
- **Fast Retrieval**: Hybrid search capabilities via Qdrant.
- **Production-Ready**: Dockerized stack with dedicated workers and monitoring.
- **Auto-Documentation**: Integrated Swagger UI and ReDoc for all API endpoints.

## 🛠 Setup & Installation

### Local Development

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd rag-pdf
   ```

2. **Setup virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Environment Configuration**:
   Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

4. **Run with Docker (Recommended)**:
   ```bash
   docker-compose up --build
   ```

### API Documentation

Once the server is running, you can access the interactive API documentation at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

> [!TIP]
> Navigating to the root URL [http://localhost:8000/](http://localhost:8000/) will automatically redirect you to the Swagger documentation.

### Postman Collection

For manual API testing, a Postman collection is included in the root directory:
- [rag_pdf_postman_collection.json](file:///home/carl/Desktop/wisipay/rag-pdf/rag_pdf_postman_collection.json)

**How to use**:
1. Open Postman.
2. Click **Import**.
3. Drag and drop the `rag_pdf_postman_collection.json` file.
4. Set the `base_url` variable (default is `http://localhost:8000`).

## 📡 API Endpoints

### Ingestion
- `POST /api/v1/ingest/upload`: Upload a document for processing.

### Querying
- `POST /api/v1/query/`: Ask questions based on your indexed documents.

### Health
- `GET /health`: Monitor system status.

## 🏗 Project Structure

```text
├── app/
│   ├── api/          # API routes (v1)
│   ├── core/         # Config, logging, security
│   ├── integrations/ # Third-party clients (Qdrant, R2, LLMs)
│   ├── models/       # Database schemas
│   ├── pipeline/     # RAG logic (chunking, embedding, orchestration)
│   ├── services/     # Business logic
│   └── workers/      # Celery task definitions
├── docker/           # Docker configuration
├── tests/            # Test suite
└── docker-compose.yml
```

## 🤝 Contributing

Contributions are welcome! Please follow the standard fork-and-pull-request workflow.

## 📄 License

[Apache 2.0](LICENSE)
