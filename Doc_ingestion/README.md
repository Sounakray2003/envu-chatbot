Q0 Knowledge Base Pipeline
This repository contains two active entrypoints:

main.py for CLI-driven ingestion.
retrieval_new.py for the unified FastAPI service that handles retrieval and HTTP-based ingestion.
The current stack is centered on:

semantic chunking
OpenAI embeddings
OpenAI chat generation for answers
Qdrant as the vector store
Main Files
main.py: parses REQUEST_JSON or INGEST_SOURCE and runs ingestion from the command line
retrieval_new.py: FastAPI app for retrieval, file upload ingestion, website URL ingestion, listing stored sources, and deleting stored sources by file_id
services/ingestion_service.py: orchestration for discovery, extraction, chunking, embedding, and storage
services/sources/: source adapters for file/folder, website, and API ingestion
services/payload_builder.py: normalizes stored payloads written to Qdrant
services/vectore_store/qudrant_vector.py: writes chunk vectors into Qdrant
What The Code Does
Ingestion
The ingestion pipeline:

normalizes the request
discovers files, pages, or API records
extracts text or structured content
chunks content
creates OpenAI embeddings
stores vectors in Qdrant
The main storage target is main_memory.

Retrieval
The retrieval API:

embeds the query with OpenAI
checks semantic cache in cache_memory
falls back to main_memory on cache miss
generates an answer from retrieved context
stores cacheable misses back into cache_memory
Current Support Model
Chunking: SEMANTIC only
Retrieval mode: dense search only
Embeddings: OpenAI embeddings only
Default embedding model: text-embedding-3-large
Default embedding dimensions: 1024
Retrieval collection: main_memory
Retrieval backend: user-defined Qdrant from QDRANT_URL and optional QDRANT_API_KEY
Supported Sources
The current code supports these ingestion source families:

File upload or local folder
Website crawl
API source
Notes:

The FastAPI ingestion route only accepts file or url.
API-source ingestion is still supported through explicit JSON requests handled by main.py and services/ingestion_service.py.
Multi-source ingestion is supported in the CLI path through JSON lists or bracketed shorthand lists.
Supported File Types
The extractor stack currently handles:

Documents: .pdf, .docx, .doc, .txt, .md, .markdown, .json, .html, .htm, .xml
Spreadsheet and row-wise files: .csv, .tsv, .xlsx, .xls, .xlsm
Archives: .zip
Images: .png, .jpg, .jpeg, .gif, .bmp, .webp, .tiff, .tif
Image ingestion uses the configured vision-capable OpenAI path when image description is enabled in the extractor flow.

File IDs And Stored Source Rows
Stored source rows in GET /retrieve/file-uploads now include both:

real file uploads
website crawls
Current behavior:

file uploads get a deterministic file_id
website crawls also get a deterministic crawl-level file_id
the same stored file_id is used by DELETE /retrieve/file-uploads/{file_id}
For website crawls, the file_id represents the crawl source, not a single page.

Environment Variables
Common Required Settings
OPENAI_API_KEY
QDRANT_URL
QDRANT_API_KEY if your Qdrant instance requires authentication
CLI Ingestion Inputs
Set one of:

REQUEST_JSON
INGEST_SOURCE
Useful Optional Settings
OPENAI_BASE_URL
OPENAI_EMBEDDING_MODEL
OPENAI_EMBEDDING_DIMENSIONS
OPENAI_EMBEDDING_TIMEOUT
OPENAI_EMBEDDING_MAX_RETRIES
OPENAI_CHAT_MODEL
KNOWLEDGE_BASE_ID
VECTOR_STORE_ID
WEBSITE_MAX_PAGES
WEBSITE_MAX_DEPTH
WEBSITE_RESPECT_ROBOTS_TXT
WEBSITE_DISCOVER_SITEMAPS
WEBSITE_SCOPE_TO_START_PATH
Notes:

main.py uses KNOWLEDGE_BASE_ID and VECTOR_STORE_ID defaults when it builds shorthand ingestion requests.
retrieval_new.py uses the unified API defaults for HTTP ingestion and always stores into main_memory.
In docker-compose.yml, the CLI ingestion service and the API service both default VECTOR_STORE_ID to 2.
FastAPI Endpoints
The unified API lives in retrieval_new.py.

GET /retrieve/health
Simple health response:

{
  "status": "ok",
  "service": "retrieval-api"
}
POST /retrieve
Dense retrieval with semantic cache support.

Important request fields:

query
limit
score_threshold
is_active
filters
use_cache
session_id
reset_history
resolve_references
previous_query
previous_answer
conversation_history
POST /ingest/file-upload
Multipart ingestion endpoint.

Accepted form fields:

file
url
is_active
Rules:

provide exactly one of file or url
file triggers file-upload ingestion
url triggers website crawl ingestion
Response fields include:

status
file_id
filename
source_url
knowledge_base_id
collection_name
results
GET /retrieve/file-uploads
Lists distinct stored file uploads and website ingestions from main_memory.

Returned fields include:

file_id
filename
source_url
knowledge_base_id
knowledge_base_name
source_type_name
total_chunks
last_ingested_at
DELETE /retrieve/file-uploads/{file_id}
Deletes all stored chunks that belong to the given stored source file_id.

This applies to:

uploaded files
website crawls
Hidden Compatibility Route
POST /retrieve/semantic-cache still exists as a backward-compatible alias for /retrieve, but it is excluded from the generated schema.

Quick Start
1. Install Dependencies
pip install -r requirements.txt
2. Start The Unified API
uvicorn retrieval_new:app --host 0.0.0.0 --port 8093
The API will be available at http://localhost:8093.

3. Ingest A File Through HTTP
curl -X POST "http://localhost:8093/ingest/file-upload" \
  -F "file=@testfiles/Einstein.html" \
  -F "is_active=true"
4. Ingest A Website Through HTTP
curl -X POST "http://localhost:8093/ingest/file-upload" \
  -F "url=https://www.in.envu.com/" \
  -F "is_active=true"
5. List Stored Sources
curl "http://localhost:8093/retrieve/file-uploads"
6. Delete A Stored Source By file_id
curl -X DELETE "http://localhost:8093/retrieve/file-uploads/<file_id>"
7. Run Retrieval
curl -X POST "http://localhost:8093/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Filing Buddy?",
    "limit": 5,
    "score_threshold": 0.35,
    "use_cache": true
  }'
CLI Ingestion Examples
Full JSON Request
export OPENAI_API_KEY="your-key"
export QDRANT_URL="http://localhost:6333"
export QDRANT_API_KEY="your-qdrant-api-key"
export VECTOR_STORE_ID="2"
export REQUEST_JSON='{
  "knowledge_base_id": 13,
  "name": "Local Folder Ingestion",
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
}'

python main.py
Website Shorthand
export OPENAI_API_KEY="your-key"
export QDRANT_URL="http://localhost:6333"
export VECTOR_STORE_ID="2"
export REQUEST_JSON='https://www.in.envu.com/'

python main.py
Local File Shorthand
export REQUEST_JSON='sample_zip.zip'
python main.py
Multi-Source Shorthand
export REQUEST_JSON='["https://www.in.envu.com/", "./testfiles/Einstein.html"]'
python main.py
Docker Usage
Build The Image
docker build -t q0-knowledge-base .
Run The Unified API Container
docker run --rm \
  --env-file .env \
  -p 8093:8093 \
  -v "$(pwd)/logs:/app/logs" \
  -v "$(pwd)/output_files:/app/output_files" \
  -v "$(pwd)/testfiles:/app/testfiles" \
  q0-knowledge-base
Run CLI Ingestion In Docker
docker run --rm \
  --env-file .env \
  -e REQUEST_JSON='https://www.in.envu.com/' \
  -v "$(pwd)/logs:/app/logs" \
  -v "$(pwd)/output_files:/app/output_files" \
  -v "$(pwd)/testfiles:/app/testfiles" \
  q0-knowledge-base \
  python main.py
Docker Compose
Build services:

docker compose build
Run CLI ingestion:

docker compose run --rm rag-ingestion
Start the unified API:

docker compose up retrieval-api
Notes
docker-compose.yml still uses two services: one for CLI ingestion and one for the unified API.
The route name /retrieve/file-uploads is historical; it now lists both uploaded files and website crawls.
uploaded_files/ is temporary workspace data for HTTP ingestion and should not be committed.
Retrieval uses cache_memory for semantic caching and main_memory for source retrieval.
Troubleshooting
REQUEST_JSON Or INGEST_SOURCE Is Missing
Set one of them before running python main.py.

OpenAI Calls Fail
Check:

OPENAI_API_KEY
OPENAI_BASE_URL
OPENAI_EMBEDDING_MODEL
OPENAI_CHAT_MODEL
Qdrant Connection Fails
Check:

QDRANT_URL
QDRANT_API_KEY if required
network reachability from your host or container
No Files Or Pages Are Discovered
Check:

source_details.file_path for file ingestion
source_details.folder_path for folder ingestion
source_details.start_url or HTTP url for website ingestion
GET /retrieve/file-uploads Shows More Chunks Than One Ingestion Run
The endpoint groups stored rows by source identity. Re-ingesting the same file or website can increase the displayed total_chunks for that stored source.