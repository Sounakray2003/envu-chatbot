# RAG API Handoff

This project currently exposes two API services:

| Service | Base URL | Purpose |
| --- | --- | --- |
| Retrieval API | `http://localhost:8093` | RAG question answering |
| Admin Ingestion API | `http://localhost:8094` | Upload, view, status check, and delete ingested files |

## 1. RAG Question Answering API

Base URL:

```text
http://localhost:8093
```

Swagger docs:

```text
http://localhost:8093/docs
```

### Ask A Question

Endpoint:

```text
POST /retrieve
```

Example:

```powershell
curl.exe -X POST "http://localhost:8093/retrieve" `
  -H "Content-Type: application/json" `
  -d '{"query":"What is this document about?","collection":"main_memory","limit":5}'
```

### Ask A Question For One Uploaded File

Use the uploaded file's `file_id` as a Qdrant payload filter.

Example:

```powershell
curl.exe -X POST "http://localhost:8093/retrieve" `
  -H "Content-Type: application/json" `
  -d '{"query":"What is this document about?","collection":"main_memory","limit":5,"filters":{"must":[{"key":"file_id","match":{"value":"770cd6da-4ec7-4f6c-9413-4e1a79213b18"}}]}}'
```

Common request fields:

| Field | Type | Notes |
| --- | --- | --- |
| `query` | string | User question |
| `collection` | string | Usually `main_memory` |
| `limit` | number | Number of chunks to retrieve |
| `filters` | object | Optional Qdrant filter |
| `session_id` | string | Optional conversation/session id |
| `conversation_history` | array | Optional previous chat messages |

## 2. Admin Ingestion API

Base URL:

```text
http://localhost:8094
```

Swagger docs:

```text
http://localhost:8094/docs
```

Health check:

```powershell
curl.exe http://localhost:8094/health
```

### Upload And Ingest File

Endpoint:

```text
POST /admin/files/upload
```

Example:

```powershell
curl.exe -X POST "http://localhost:8094/admin/files/upload" `
  -F "file=@C:\path\to\your-file.pdf" `
  -F "knowledge_base_id=1"
```

Successful response returns:

```json
{
  "file_id": "770cd6da-4ec7-4f6c-9413-4e1a79213b18",
  "filename": "1671122754.pdf",
  "status": "queued",
  "collection_name": "main_memory"
}
```

`queued` means the file was accepted and ingestion is running in the background.

### List All Files

Endpoint:

```text
GET /admin/files
```

Example:

```powershell
curl.exe http://localhost:8094/admin/files
```

Response includes:

```json
{
  "files": [],
  "total": 0
}
```

### Check One File Status

Endpoint:

```text
GET /admin/files/{file_id}
```

Example:

```powershell
curl.exe http://localhost:8094/admin/files/770cd6da-4ec7-4f6c-9413-4e1a79213b18
```

Important statuses:

| Status | Meaning |
| --- | --- |
| `queued` | Upload accepted, waiting for background ingestion |
| `processing` | Ingestion is running |
| `active` | File is successfully ingested and searchable |
| `failed` | Upload saved, but extraction/embedding/vector storage failed |
| `deleted` | File vectors were deleted |

Successful ingestion includes fields like:

```json
{
  "status": "active",
  "chunk_count": 16,
  "vector_count": 16,
  "activated_points": 16
}
```

### Delete One File

Endpoint:

```text
DELETE /admin/files/{file_id}
```

Example:

```powershell
curl.exe -X DELETE http://localhost:8094/admin/files/770cd6da-4ec7-4f6c-9413-4e1a79213b18
```

This marks the file inactive and deletes its vectors from Qdrant.

## Docker Commands

Start retrieval API:

```powershell
docker compose up -d retrieval-api
```

Start admin ingestion API from root compose:

```powershell
docker compose up -d rag-ingestion-admin
```

Check running services:

```powershell
docker compose ps
docker ps
```

View logs:

```powershell
docker logs -f retrieval-api
docker logs -f rag-ingestion-admin
```

## Summary

```text
8093 = RAG question answering
8094 = file ingestion admin API
main_memory = active vector collection
file_id = key used to query/delete one uploaded file
```
