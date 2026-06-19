# Cloud Engineering Handoff - RAG Pipeline And Document Ingestion

This repository contains a containerized RAG stack with three Docker Compose services:

| Service | Container | Purpose | External Port |
| --- | --- | --- | --- |
| `retrieval-api` | `retrieval-api` | RAG question answering API | `8093` |
| `rag-ingestion-admin` | `rag-ingestion-admin` | Document upload, ingestion status, listing, and delete API | `8094` |
| `rag-ingestion` | `rag-ingestion` | CLI/batch ingestion job driven by `REQUEST_JSON` or `INGEST_SOURCE` | none |

The active root deployment file is:

```text
docker-compose.yml
```

There is also a standalone compose file under:

```text
Doc_ingetion/docker-compose.yml
```

Use the root `docker-compose.yml` for deploying the full stack.

## 1. Runtime Dependencies

Provision these external services before deployment:

| Dependency | Notes |
| --- | --- |
| OpenAI API | Used for embeddings and answer generation. |
| Qdrant | Vector database used for `main_memory` retrieval and `cache_memory` semantic cache. |
| Persistent storage | Needed for logs, uploaded files, output files, and the document file registry. |

## 2. Required Environment Variables

Create environment configuration from the values currently stored in local `.env` files. Do not commit real secrets.

Root `.env` is used by:

```text
rag-ingestion
retrieval-api
```

Doc ingestion `.env` is used by:

```text
rag-ingestion-admin
```

Required or important variables:

| Variable | Required | Used By | Notes |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | yes | all services | OpenAI key for embeddings/chat. Store as a cloud secret. |
| `QDRANT_URL` | yes | all services | URL reachable from containers. |
| `QDRANT_API_KEY` | if Qdrant auth enabled | all services | Store as a cloud secret. |
| `OPENAI_BASE_URL` | no | retrieval/ingestion | Defaults to `https://api.openai.com/v1`. |
| `OPENAI_EMBEDDING_MODEL` | no | all services | Defaults to `text-embedding-3-large`. |
| `OPENAI_EMBEDDING_DIMENSIONS` | no | all services | Defaults to `1024`; Qdrant collection vector size must match. |
| `OPENAI_CHAT_MODEL` | no | retrieval API | Defaults to `gpt-4.1-mini` in `retrieval_new.py`. |
| `OPENAI_EMBEDDING_TIMEOUT` | no | retrieval API | Defaults to `120`. |
| `OPENAI_EMBEDDING_MAX_RETRIES` | no | retrieval API | Defaults to `3`. |
| `VECTOR_STORE_ID` | no | all services | Compose defaults to `2`. |
| `KNOWLEDGE_BASE_ID` | no | ingestion/admin | Defaults vary by code path; set explicitly for production. |
| `REQUEST_JSON` | for CLI ingestion job | `rag-ingestion` | Batch ingestion payload or shorthand source. |
| `INGEST_SOURCE` | alternative to `REQUEST_JSON` | `rag-ingestion` | Batch ingestion source if `REQUEST_JSON` is not set. |
| `ADMIN_API_PORT` | no | `rag-ingestion-admin` | Defaults to `8094`. |
| `CHUNK_SIZE` | no | `rag-ingestion-admin` | Defaults to `1024`. |
| `CHUNK_OVERLAP` | no | `rag-ingestion-admin` | Defaults to `50`. |

Recommended production values:

```text
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_EMBEDDING_DIMENSIONS=1024
OPENAI_CHAT_MODEL=gpt-4.1-mini
VECTOR_STORE_ID=2
KNOWLEDGE_BASE_ID=<tenant_or_kb_id>
```

## 3. Persistent Volumes

The Docker Compose file mounts these host paths:

| Service | Host Path | Container Path | Purpose |
| --- | --- | --- | --- |
| `retrieval-api` | `./logs` | `/app/logs` | API logs |
| `retrieval-api` | `./output_files` | `/app/output_files` | Generated/output files |
| `retrieval-api` | `./testfiles` | `/app/testfiles` | Local test or mounted ingestion files |
| `rag-ingestion` | `./logs` | `/app/logs` | Job logs |
| `rag-ingestion` | `./output_files` | `/app/output_files` | Job output files |
| `rag-ingestion` | `./testfiles` | `/app/testfiles` | Local files for batch ingestion |
| `rag-ingestion-admin` | `./Doc_ingetion/logs` | `/app/logs` | Admin API logs |
| `rag-ingestion-admin` | `./Doc_ingetion/uploads` | `/app/uploads` | Uploaded source documents |
| `rag-ingestion-admin` | `./Doc_ingetion/file_registry` | `/app/file_registry` | File status registry JSON |
| `rag-ingestion-admin` | `./Doc_ingetion/output_files` | `/app/output_files` | Ingestion output files |
| `rag-ingestion-admin` | `./Doc_ingetion/testfiles` | `/app/testfiles` | Local test files |

For cloud deployment, back these paths with persistent storage if uploaded file history and registry state must survive container restarts.

Important registry path:

```text
/app/file_registry/files.json
```

## 4. Build Commands

Run from repository root:

```powershell
docker compose build
```

Build a single service:

```powershell
docker compose build retrieval-api
docker compose build rag-ingestion-admin
docker compose build rag-ingestion
```

## 5. Run Commands

Start retrieval API:

```powershell
docker compose up -d retrieval-api
```

Start document ingestion admin API:

```powershell
docker compose up -d rag-ingestion-admin
```

Start both public APIs:

```powershell
docker compose up -d retrieval-api rag-ingestion-admin
```

Run the CLI/batch RAG ingestion pipeline:

```powershell
docker compose run --rm rag-ingestion
```

Run everything:

```powershell
docker compose up -d
```

Check status:

```powershell
docker compose ps
```

View logs:

```powershell
docker logs -f retrieval-api
docker logs -f rag-ingestion-admin
docker logs -f rag-ingestion
```

Stop services:

```powershell
docker compose down
```

## 6. API Endpoints

### Retrieval API

Base URL:

```text
http://<host>:8093
```

Swagger:

```text
http://<host>:8093/docs
```

Health:

```text
GET /retrieve/health
```

Question answering:

```text
POST /retrieve
```

Example:

```powershell
curl.exe -X POST "http://localhost:8093/retrieve" `
  -H "Content-Type: application/json" `
  -d '{"query":"What is this document about?","collection":"main_memory","limit":5,"use_cache":true}'
```

Important request fields:

| Field | Notes |
| --- | --- |
| `query` | User question. |
| `collection` | Usually `main_memory`. |
| `limit` | Number of chunks to retrieve. |
| `score_threshold` | Optional relevance threshold. |
| `filters` | Optional Qdrant payload filter. |
| `session_id` | Optional conversation/session id. |
| `conversation_history` | Optional previous messages. |
| `use_cache` | Enables semantic cache lookup. |

### Document Ingestion Admin API

Base URL:

```text
http://<host>:8094
```

Swagger:

```text
http://<host>:8094/docs
```

Health:

```text
GET /health
```

Upload and ingest file:

```text
POST /admin/files/upload
```

Example:

```powershell
curl.exe -X POST "http://localhost:8094/admin/files/upload" `
  -F "file=@C:\path\to\file.pdf" `
  -F "knowledge_base_id=1"
```

List files:

```text
GET /admin/files
```

Get file status:

```text
GET /admin/files/{file_id}
```

Delete file vectors:

```text
DELETE /admin/files/{file_id}
```

File statuses:

| Status | Meaning |
| --- | --- |
| `queued` | Upload accepted and background ingestion queued. |
| `processing` | Extraction, chunking, embedding, or vector write is running. |
| `active` | File is ingested and searchable. |
| `failed` | Ingestion failed after upload. |
| `deleted` | File vectors were deleted. |

## 7. Batch Ingestion Job

The `rag-ingestion` service runs:

```text
python main.py
```

It requires one of:

```text
REQUEST_JSON
INGEST_SOURCE
```

Example shorthand values:

```text
REQUEST_JSON=https://www.example.com/
REQUEST_JSON=./testfiles/sample.pdf
REQUEST_JSON=["https://www.example.com/","./testfiles/sample.pdf"]
```

Example full JSON:

```json
{
  "knowledge_base_id": 1,
  "name": "Production Ingestion",
  "source_type_name": "File Upload",
  "chunking_details": {
    "chunking_type": "SEMANTIC",
    "chunkSize": 1024,
    "chunkOverlap": 50
  },
  "embedding_details": {
    "embedding_model_name": "text-embedding-3-large",
    "dimensions": 1024
  },
  "vector_store_details": {
    "vector_store_id": 2,
    "collection_name": "main_memory"
  },
  "source_details": {
    "folder_path": "./testfiles"
  }
}
```

## 8. Qdrant Collections

Expected collections:

| Collection | Purpose |
| --- | --- |
| `main_memory` | Primary document chunks used for RAG retrieval. |
| `cache_memory` | Semantic cache for retrieval answers. |

Embedding dimensions must match the Qdrant collection vector size. Current default is:

```text
1024
```

## 9. Supported Source Types

The ingestion code supports:

| Source | Notes |
| --- | --- |
| File upload/local folder | Files can be uploaded through admin API or loaded from mounted paths. |
| Website URL | Supported in the unified retrieval API and CLI ingestion paths. |
| API source | Supported in the CLI ingestion path via JSON request. |

Supported file extensions include:

```text
.pdf, .docx, .doc, .txt, .md, .markdown, .json, .html, .htm, .xml,
.csv, .tsv, .xlsx, .xls, .xlsm, .zip,
.png, .jpg, .jpeg, .gif, .bmp, .webp, .tiff, .tif
```

## 10. Cloud Deployment Notes

Expose only the required APIs:

```text
8093 - retrieval-api
8094 - rag-ingestion-admin
```

Recommended controls:

| Area | Recommendation |
| --- | --- |
| Secrets | Use cloud secret manager for OpenAI and Qdrant credentials. |
| Networking | Allow app containers to reach Qdrant and OpenAI. |
| TLS | Terminate HTTPS at load balancer, ingress, or reverse proxy. |
| Auth | Add authentication in front of both APIs before public exposure, especially `8094`. |
| Storage | Persist upload and registry folders for admin ingestion. |
| Logging | Ship container logs and mounted `/app/logs` to centralized logging. |
| Scaling | Retrieval API can scale horizontally. Admin ingestion can scale only if registry and upload paths are shared safely. |
| Timeouts | File ingestion can be slow for large PDFs, archives, OCR, or website crawls. |
| Resource sizing | Ingestion needs more CPU/RAM than retrieval because it installs OCR/document-processing dependencies. |

## 11. Smoke Tests

After deployment:

```powershell
curl.exe http://localhost:8093/retrieve/health
curl.exe http://localhost:8094/health
```

Upload test file:

```powershell
curl.exe -X POST "http://localhost:8094/admin/files/upload" `
  -F "file=@C:\path\to\file.pdf" `
  -F "knowledge_base_id=1"
```

Check ingestion status:

```powershell
curl.exe http://localhost:8094/admin/files
```

Ask a retrieval question:

```powershell
curl.exe -X POST "http://localhost:8093/retrieve" `
  -H "Content-Type: application/json" `
  -d '{"query":"Summarize the uploaded document","collection":"main_memory","limit":5}'
```

## 12. Known Naming Notes

The directory name is currently spelled:

```text
Doc_ingetion
```

Keep this exact path in deployment scripts unless the repository is renamed and all references are updated.

The route name `/retrieve/file-uploads` in `retrieval_new.py` is historical and may include website crawls as well as uploaded files.

