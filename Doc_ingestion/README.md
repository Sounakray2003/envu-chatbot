Doc Ingestion Admin API
=======================

This repository contains a FastAPI service for ingesting files and website URLs
into Qdrant, then managing the ingested sources through admin endpoints.

The current production-style entrypoint is:

```text
admin_api.py
```

It runs on port `8094` and stores vectors in the Qdrant collection:

```text
main_memory
```

Use this README as the implementation handoff/spec if rebuilding similar code
with Codex later.


Core Requirements
-----------------

The service must provide:

- File upload ingestion.
- Website URL ingestion through the same upload endpoint.
- Background ingestion jobs with status tracking.
- Qdrant-backed listing of completed ingestions.
- File/source deletion by `file_id`.
- Semantic chunking only.
- OpenAI embeddings only.
- Docker support on port `8094`.


Main Files
----------

```text
admin_api.py
```

Admin FastAPI app. Exposes health, upload, list, status, and delete endpoints.
This is the default Docker entrypoint.

```text
services/ingestion_service.py
```

Main ingestion orchestration:

- discover source files/pages
- extract text/content
- chunk documents
- generate embeddings
- store vectors in Qdrant

```text
services/sources/website_source.py
```

Website crawler. Handles URL normalization, same-site crawl, sitemap discovery,
robots.txt support, HTML-only filtering, and page document creation.

```text
services/chunking_service.py
services/chunking/semantic_chunking.py
services/chunking/base_chunking.py
```

Semantic chunking implementation.

```text
services/payload_builder.py
```

Builds standardized Qdrant payloads. Important fields include:

- `file_id`
- `filename`
- `source_url`
- `source_type_name`
- `source_mapping_id`
- `knowledge_base_id`
- `member_id`
- `org_id`
- `isActive`
- `source_details`
- `ingestion_metadata`

```text
services/vectore_store/qudrant_vector.py
```

Direct Qdrant REST backend. Supports upsert, count by `file_id`, set active,
delete by `file_id`, scroll collection, and collection metadata.


Runtime Model
-------------

The API keeps two kinds of state:

1. Qdrant vectors in `main_memory`
2. A local registry file at `/app/file_registry/files.json`

Important behavior:

- Completed/active records in `GET /admin/files` must come from Qdrant.
- The local registry is only for transient jobs such as `queued`, `processing`,
  `failed`, `delete_failed`, and local upload metadata.
- Restarting or recreating the API container must not make completed Qdrant
  records disappear from `GET /admin/files`.
- Deleted records must not appear in `GET /admin/files`.
- Direct lookup by `GET /admin/files/{file_id}` should first show local
  queued/processing/failed state, then fall back to Qdrant for completed
  records.
- Delete should work even if the local registry disappeared, as long as Qdrant
  still has points for the given `file_id`.


Admin API Handoff
-----------------

Base URL:

```text
http://localhost:8094
```

Swagger docs:

```text
http://localhost:8094/docs
```


1. Health Check
---------------

Endpoint:

```text
GET /health
```

Example:

```powershell
curl.exe "http://localhost:8094/health"
```

Expected output:

```json
{
  "status": "ok",
  "collection": "main_memory"
}
```


2. Upload File For Ingestion
----------------------------

Endpoint:

```text
POST /admin/files/upload
```

Compatibility alias:

```text
POST /ingest/file-upload
```

Content type:

```text
multipart/form-data
```

Payload fields:

| Field | Required | Type | Notes |
| --- | --- | --- | --- |
| file | Yes, unless `url` is provided | file | File to ingest |
| url | Yes, unless `file` is provided | string | Website URL to crawl |
| file_id | No | string | Optional custom file id |
| knowledge_base_id | No | integer | Defaults to `1` |
| member_id | No | string | Optional user/member id |
| org_id | No | string | Optional organization id |
| source_mapping_id | No | string | Optional source mapping id |

Rules:

- Provide exactly one of `file` or `url`.
- `file` triggers file upload ingestion.
- `url` triggers website crawl ingestion.
- The response returns `queued` because ingestion runs in the background.

File upload example:

```powershell
curl.exe -X POST "http://localhost:8094/admin/files/upload" `
  -F "file=@C:\path\to\file.pdf" `
  -F "knowledge_base_id=1" `
  -F "org_id=org_123" `
  -F "member_id=user_123"
```

Website ingestion example:

```powershell
curl.exe -X POST "http://localhost:8094/admin/files/upload" `
  -F "url=https://www.in.envu.com/" `
  -F "knowledge_base_id=1"
```

Expected queued output:

```json
{
  "file_id": "f1060042-8e2b-4ebb-908b-e160374650db",
  "filename": null,
  "source_url": "https://www.in.envu.com/",
  "status": "queued",
  "collection_name": "main_memory"
}
```


3. List All Active Sources
--------------------------

Endpoint:

```text
GET /admin/files
```

Compatibility alias:

```text
GET /ingest/file-uploads
```

Example:

```powershell
curl.exe "http://localhost:8094/admin/files"
```

Expected output:

```json
{
  "files": [
    {
      "file_id": "f1060042-8e2b-4ebb-908b-e160374650db",
      "filename": "https://www.in.envu.com/",
      "source_url": "https://www.in.envu.com/",
      "status": "active",
      "collection_name": "main_memory",
      "knowledge_base_id": 1,
      "knowledge_base_name": "Admin Website - www.in.envu.com",
      "member_id": null,
      "org_id": null,
      "source_mapping_id": null,
      "source_type_name": "Website",
      "uploaded_at": "2026-06-18T11:08:31.528073Z",
      "completed_at": "2026-06-18T11:08:31.528073Z",
      "chunk_count": 1995,
      "vector_count": 1995,
      "results": {
        "status": "success",
        "total_files_processed": 1,
        "total_chunks_created": 1995,
        "total_vectors_stored": 1995,
        "errors": [],
        "warnings": []
      }
    }
  ],
  "total": 1
}
```

Implementation requirement:

- This endpoint must scroll Qdrant collection `main_memory` and group points by
  `file_id` or source identity.
- It must count grouped points as `chunk_count` and `vector_count`.
- It must not depend only on `/app/file_registry/files.json`.
- It must hide deleted records.
- It may merge local queued/processing/failed records from the registry because
  those states may not exist in Qdrant yet.


4. Get One File/Source Status
-----------------------------

Endpoint:

```text
GET /admin/files/{file_id}
```

Example:

```powershell
curl.exe "http://localhost:8094/admin/files/f1060042-8e2b-4ebb-908b-e160374650db"
```

Expected successful output:

```json
{
  "file_id": "f1060042-8e2b-4ebb-908b-e160374650db",
  "filename": "https://www.in.envu.com/",
  "source_url": "https://www.in.envu.com/",
  "status": "active",
  "collection_name": "main_memory",
  "knowledge_base_id": 1,
  "source_type_name": "Website",
  "chunk_count": 1995,
  "vector_count": 1995,
  "results": {
    "status": "success",
    "total_files_processed": 1,
    "total_chunks_created": 1995,
    "total_vectors_stored": 1995,
    "errors": [],
    "warnings": []
  }
}
```

Expected while ingestion is still running:

```json
{
  "file_id": "770cd6da-4ec7-4f6c-9413-4e1a79213b18",
  "filename": "file.pdf",
  "status": "processing",
  "collection_name": "main_memory"
}
```

Expected if not found:

```json
{
  "detail": "file_id not found"
}
```

Implementation requirement:

- If local registry has `queued`, `processing`, `failed`, or `delete_failed`,
  return that local record.
- Otherwise query Qdrant by payload field `file_id`.
- Return 404 only if neither local registry nor Qdrant has the file/source.


5. Delete File/Source
---------------------

Endpoint:

```text
DELETE /admin/files/{file_id}
```

Compatibility alias:

```text
DELETE /ingest/file-uploads/{file_id}
```

Example:

```powershell
curl.exe -X DELETE "http://localhost:8094/admin/files/f1060042-8e2b-4ebb-908b-e160374650db"
```

Expected output:

```json
{
  "file_id": "f1060042-8e2b-4ebb-908b-e160374650db",
  "status": "deleted",
  "deleted_points": 1995
}
```

Implementation requirement:

- Delete all Qdrant points whose payload `file_id` matches.
- Delete should work even if `/app/file_registry/files.json` is missing.
- Deleted files/sources must not appear in `GET /admin/files`.
- It is acceptable to keep a local registry tombstone with `status=deleted`,
  but the list endpoint must filter it out.


Status Meanings
---------------

| Status | Meaning |
| --- | --- |
| queued | Upload accepted, waiting for background ingestion |
| processing | Ingestion is running |
| active | Successfully ingested and searchable |
| inactive | Points exist but `isActive=false` |
| failed | Ingestion failed |
| delete_failed | Delete operation failed |
| deleted | Vectors deleted |


Website Crawl Behavior
----------------------

When `url` is posted to `/admin/files/upload`, the service builds a Website
ingestion request.

Defaults:

```text
WEBSITE_MAX_PAGES=500
WEBSITE_MAX_DEPTH=5
WEBSITE_RESPECT_ROBOTS_TXT=true
WEBSITE_DISCOVER_SITEMAPS=true
WEBSITE_SCOPE_TO_START_PATH=false
```

Crawler behavior:

- Normalize the start URL.
- Crawl only `http` and `https`.
- Stay on the same host by default.
- Fetch and respect `robots.txt` by default.
- Discover URLs from `robots.txt` sitemap entries.
- Also try `/sitemap.xml` and `/sitemap_index.xml`.
- Breadth-first crawl discovered HTML links.
- Skip non-HTML responses and common static assets.
- Parse HTML with BeautifulSoup when available.
- Remove `script`, `style`, and `noscript`.
- Extract title, headings, main/body text, links, and images.
- Convert each page into markdown before chunking.

For website crawls, `file_id` represents the whole crawl/source, not one page.


Chunking Strategy
-----------------

Only semantic chunking is supported.

Default chunking details:

```json
{
  "chunking_type": "SEMANTIC",
  "chunkSize": 1024,
  "chunkOverlap": 50
}
```

Environment overrides:

```text
CHUNK_SIZE
CHUNK_OVERLAP
```

Implementation details:

- Split markdown into structured units: headings, code blocks, lists, and
  paragraphs.
- Group adjacent units up to the target token size.
- Use Jaccard similarity over 4+ character words to decide whether adjacent
  units are semantically similar enough to keep together.
- Add token overlap between chunks.
- Split oversized units by sentence or token windows.
- Merge very small chunks when possible.
- Use OpenAI-compatible token counting with `cl100k_base`.

This is local semantic/structure-aware chunking, not embedding-based semantic
chunking.


Embedding And Storage
---------------------

Default embedding model:

```text
text-embedding-3-large
```

Default embedding dimensions:

```text
1024
```

The service normalizes ingestion to:

```text
collection_name=main_memory
vector_store_id=2
```

Qdrant payloads must include a top-level `file_id` so list, status, and delete
can operate directly from Qdrant.


Environment Variables
---------------------

Required:

```text
OPENAI_API_KEY
QDRANT_URL
```

Optional:

```text
QDRANT_API_KEY
KNOWLEDGE_BASE_ID
VECTOR_STORE_ID
OPENAI_BASE_URL
OPENAI_EMBEDDING_MODEL
OPENAI_EMBEDDING_DIMENSIONS
CHUNK_SIZE
CHUNK_OVERLAP
WEBSITE_MAX_PAGES
WEBSITE_MAX_DEPTH
WEBSITE_RESPECT_ROBOTS_TXT
WEBSITE_DISCOVER_SITEMAPS
WEBSITE_SCOPE_TO_START_PATH
MAX_UPLOAD_BYTES
MAX_BATCH_FILES
MAX_ZIP_FILES
MAX_ZIP_UNCOMPRESSED_BYTES
MAX_PDF_PAGES
MAX_EXTRACTED_CHARS_PER_FILE
MAX_CHUNKS_PER_JOB
INGESTION_CONCURRENCY
INGESTION_QUEUE_TIMEOUT_SECONDS
MAX_INGESTION_SECONDS_PER_ITEM
ADMIN_OPERATION_TIMEOUT_SECONDS
UPLOAD_DIR
FILE_REGISTRY_PATH
```


Supported File Types
--------------------

```text
.pdf
.docx
.doc
.txt
.md
.markdown
.json
.csv
.tsv
.xlsx
.xls
.xlsm
.html
.htm
.xml
.zip
.png
.jpg
.jpeg
.gif
.bmp
.webp
.tiff
.tif
```


Docker
------

The Dockerfile must expose `8094` and run:

```text
uvicorn admin_api:app --host 0.0.0.0 --port 8094
```

Build and start:

```powershell
docker compose up -d --build
```

Check status:

```powershell
docker compose ps
curl.exe "http://localhost:8094/health"
```

View logs:

```powershell
docker compose logs -f rag-ingestion-api
```

Recommended persistent mounts:

```yaml
services:
  rag-ingestion-api:
    volumes:
      - ./file_registry:/app/file_registry
      - ./uploads:/app/uploads
```

Even with these mounts, completed source listing should still come from Qdrant.
The mounts help preserve queued/processing/failed local state and uploaded file
copies.


PowerShell Curl Notes
---------------------

In PowerShell, `curl` can resolve to `Invoke-WebRequest`. Use `curl.exe` for
multipart commands.

Correct:

```powershell
curl.exe -X POST "http://localhost:8094/admin/files/upload" `
  -F "url=https://www.in.envu.com/" `
  -F "knowledge_base_id=1"
```

Line continuation in PowerShell is a backtick:

```text
`
```

not a Unix backslash.


Troubleshooting
---------------

`GET /admin/files` returns `{"files":[],"total":0}`

- Check that `QDRANT_URL` and `QDRANT_API_KEY` point to the Qdrant instance that
  contains `main_memory`.
- Check that ingested Qdrant payloads have a top-level `file_id`.
- Check container logs for Qdrant scroll errors.
- Confirm the running container is the compose container:

```powershell
docker compose ps
docker ps -a
```

Port `8094` is served by the wrong container

- Stop old anonymous containers that map `8094`.
- Start the compose service again:

```powershell
docker compose up -d --build
```

PowerShell says `A parameter cannot be found that matches parameter name 'X'`

- Use `curl.exe`, not `curl`.

Deleted records still show in list

- `GET /admin/files` must filter out local registry records with
  `status=deleted`.
- If records are rebuilt from Qdrant after delete, then Qdrant deletion did not
  remove all points for that `file_id`.

Status route returns 404 after container recreate

- The route should query Qdrant by `file_id` when the local registry is missing.
- If it still returns 404, verify the Qdrant payload has top-level `file_id` and
  the collection is `main_memory`.


Future Codex Implementation Checklist
-------------------------------------

When rebuilding similar code, implement these exact behaviors:

1. `admin_api.py` is the Docker entrypoint.
2. Port is `8094`.
3. Health returns `{"status":"ok","collection":"main_memory"}`.
4. Upload endpoint accepts exactly one of `file` or `url`.
5. Upload returns immediately with `status="queued"`.
6. Background ingestion updates local registry for queued/processing/failed
   state.
7. Successful chunks are stored in Qdrant with top-level `file_id`.
8. `GET /admin/files` scrolls Qdrant `main_memory`, groups by `file_id`, and
   returns active records.
9. `GET /admin/files` merges local queued/processing/failed records.
10. `GET /admin/files` excludes deleted records.
11. `GET /admin/files/{file_id}` falls back to Qdrant if local registry is
    missing.
12. `DELETE /admin/files/{file_id}` deletes Qdrant points by `file_id` even if
    the local registry is missing.
13. Chunking is semantic only, with default `chunkSize=1024` and
    `chunkOverlap=50`.
14. Embeddings use OpenAI, default `text-embedding-3-large`, dimensions `1024`.
15. All ingestion stores into Qdrant collection `main_memory`.
