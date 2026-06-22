# Data Ingestion API Handoff

Base URL:

```text
http://34.69.4.12:8094
```

Swagger docs:

```text
http://34.69.4.12:8094/docs
```

## 1. Health Check

Endpoint:

```http
GET /health
```

Example:

```bash
curl "http://34.69.4.12:8094/health"
```

Expected output:

```json
{
  "status": "ok",
  "collection": "main_memory"
}
```

## 2. Upload File For Ingestion

Endpoint:

```http
POST /admin/files/upload
```

Content type:

```text
multipart/form-data
```

Payload:

| Field | Required | Type | Notes |
| --- | --- | --- | --- |
| `file` | Yes | file | File to ingest |
| `file_id` | No | string | Optional custom file id |
| `knowledge_base_id` | No | integer | Defaults to `1` |
| `member_id` | No | string | Optional user/member id |
| `org_id` | No | string | Optional organization id |
| `source_mapping_id` | No | string | Optional source mapping id |

Example:

```bash
curl -X POST "http://34.69.4.12:8094/admin/files/upload" \
  -F "file=@/path/to/file.pdf" \
  -F "knowledge_base_id=1" \
  -F "org_id=org_123" \
  -F "member_id=user_123"
```

Expected output:

```json
{
  "file_id": "770cd6da-4ec7-4f6c-9413-4e1a79213b18",
  "filename": "file.pdf",
  "status": "queued",
  "collection_name": "main_memory"
}
```

`queued` means the upload was accepted and ingestion is running in the background.

## 3. List All Uploaded Files

Endpoint:

```http
GET /admin/files
```

Example:

```bash
curl "http://34.69.4.12:8094/admin/files"
```

Expected output:

```json
{
  "files": [
    {
      "file_id": "770cd6da-4ec7-4f6c-9413-4e1a79213b18",
      "filename": "file.pdf",
      "status": "active",
      "collection_name": "main_memory",
      "knowledge_base_id": 1,
      "member_id": "user_123",
      "org_id": "org_123",
      "chunk_count": 16,
      "vector_count": 16,
      "uploaded_at": "2026-06-09T10:08:27.203760Z",
      "completed_at": "2026-06-09T10:09:48.021520Z"
    }
  ],
  "total": 1
}
```

## 4. Get One File Status

Endpoint:

```http
GET /admin/files/{file_id}
```

Example:

```bash
curl "http://34.69.4.12:8094/admin/files/770cd6da-4ec7-4f6c-9413-4e1a79213b18"
```

Expected output for successful ingestion:

```json
{
  "file_id": "770cd6da-4ec7-4f6c-9413-4e1a79213b18",
  "filename": "file.pdf",
  "status": "active",
  "collection_name": "main_memory",
  "knowledge_base_id": 1,
  "member_id": "user_123",
  "org_id": "org_123",
  "source_mapping_id": null,
  "storage_path": "/app/uploads/770cd6da-4ec7-4f6c-9413-4e1a79213b18_file.pdf",
  "chunk_count": 16,
  "vector_count": 16,
  "activated_points": 16,
  "results": {
    "status": "success",
    "total_files_processed": 1,
    "total_chunks_created": 16,
    "total_vectors_stored": 16,
    "errors": [],
    "warnings": []
  },
  "uploaded_at": "2026-06-09T10:08:27.203760Z",
  "started_at": "2026-06-09T10:08:27.222072Z",
  "completed_at": "2026-06-09T10:09:48.021520Z"
}
```

Expected output while ingestion is still running:

```json
{
  "file_id": "770cd6da-4ec7-4f6c-9413-4e1a79213b18",
  "filename": "file.pdf",
  "status": "processing",
  "collection_name": "main_memory"
}
```

Expected output if ingestion failed:

```json
{
  "file_id": "181f1032-9c77-4299-a1af-b2c95cf5d5af",
  "filename": "file.pdf",
  "status": "failed",
  "collection_name": "main_memory",
  "error": "Ingestion failed or stored zero vectors",
  "results": {
    "status": "FAILED",
    "errors": [
      "file.pdf: No content extracted"
    ],
    "total_chunks_created": 0,
    "total_vectors_stored": 0
  }
}
```

## 5. Delete File

Endpoint:

```http
DELETE /admin/files/{file_id}
```

Example:

```bash
curl -X DELETE "http://34.69.4.12:8094/admin/files/770cd6da-4ec7-4f6c-9413-4e1a79213b18"
```

Expected output:

```json
{
  "file_id": "770cd6da-4ec7-4f6c-9413-4e1a79213b18",
  "status": "deleted",
  "deleted_points": 16
}
```

## Status Meanings

| Status | Meaning |
| --- | --- |
| `queued` | Upload accepted, ingestion waiting |
| `processing` | Ingestion running |
| `active` | Successfully ingested and searchable |
| `failed` | Ingestion failed |
| `deleted` | Vectors deleted |
