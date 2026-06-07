# XRayVision API Documentation

This document describes the REST API for the XRayVision application, a DICOM processor with OpenAI integration.

## Base URL

The API is accessible at `http://localhost:8000` by default. The port can be configured in the `xrayvision.cfg` file under `[dashboard]` section with the `DASHBOARD_PORT` parameter.

## Authentication

All API endpoints (except static file serving) require HTTP Basic Authentication. Credentials are configured in the `xrayvision.cfg` file under the `[users]` section.

```
Authorization: Basic base64(username:password)
```

Two roles exist: `admin` (full access, unredacted names) and `user` (radiologist names anonymized to initials).

## Rate Limiting

Heavy AI endpoints (`/api/check`, `/api/analyse`, `/api/translate`) are limited to 10 requests per IP per minute. All other `/api/*` endpoints are limited to 60 requests per IP per minute. Exceeded limits return HTTP 429.

## Common Response Fields

### Exam object

```json
{
  "uid": "string",
  "patient": {
    "name": "string",
    "cnp": "string",
    "id": "string",
    "age": "integer",
    "birthdate": "string (YYYY-MM-DD)",
    "sex": "string (M|F|O)",
    "county": "integer"
  },
  "exam": {
    "created": "string (datetime)",
    "date": "string (YYYYMMDD)",
    "time": "string (HHMMSS)",
    "protocol": "string",
    "region": "string",
    "status": "string (none|queued|processing|done|error|ignore|requeue)",
    "type": "string",
    "study": "string",
    "series": "string",
    "id": "string"
  },
  "report": {
    "ai": {
      "text": "string",
      "summary": "string",
      "created": "string (datetime)",
      "updated": "string (datetime)",
      "positive": "integer (-1|0|1)",
      "severity": "integer (-1..10)",
      "confidence": "integer (-1..100)",
      "model": "string",
      "latency": "integer"
    },
    "rad": {
      "text": "string",
      "text_en": "string or null",
      "positive": "integer (-1|0|1)",
      "severity": "integer (-1..10)",
      "summary": "string",
      "created": "string (datetime)",
      "updated": "string (datetime)",
      "id": "string",
      "type": "string",
      "radiologist": "string (initials for non-admin)",
      "justification": "string",
      "model": "string",
      "latency": "integer"
    },
    "correct": "integer (-1|0|1)",
    "reviewed": "integer (0|1)"
  }
}
```

---

## Dashboard Pages

#### GET /
Serve the main dashboard HTML page.

#### GET /stats
Serve the statistics HTML page.

#### GET /stats/radiologists
Serve the radiologist statistics HTML page.

#### GET /stats/diagnostics
Serve the diagnostics statistics HTML page.

#### GET /stats/insights
Serve the insights HTML page.

#### GET /about
Serve the about HTML page.

#### GET /check
Serve the report check HTML page.

#### GET /favicon.ico
Serve the favicon.ico file.

---

## WebSocket

#### GET /ws
Handle WebSocket connections for real-time dashboard updates. Sends JSON events for queue status, processing state, AI health, and radiologist reviews.

---

## REST API Endpoints

### Exams

#### GET /api/exams
Provide paginated exam data with optional filters.

**Query Parameters:**
- `page` (integer, default: 1) — Page number
- `reviewed` (string: any|yes|no) — Filter by review status (yes = severity > -1)
- `positive` (string: any|yes|no) — Filter by AI prediction
- `correct` (string: any|yes|no) — Filter by AI correctness
- `region` (string) — Filter by anatomic region (partial match)
- `status` (string) — Filter by processing status
- `search` (string) — Filter by patient name, CNP, patient ID, or UID
- `diagnostic` (string) — Filter by radiologist diagnostic summary
- `radiologist` (string) — Filter by radiologist name
- `severity` (string) — Filter by severity with interval notation (e.g. `3-6`, `-8`, `2-`, `5`)

**Response:**
```json
{
  "exams": ["<Exam object>"],
  "total": "integer",
  "pages": "integer",
  "filters": "object"
}
```

#### GET /api/exams/{uid}
Provide a single exam's data by UID.

**Path Parameters:**
- `uid` (string) — Exam UID

**Response:** Exam object

**Error (404):**
```json
{"error": "Exam not found"}
```

---

### Patients

#### GET /api/patients
Provide paginated patient data with optional filters.

**Query Parameters:**
- `page` (integer, default: 1) — Page number
- `search` (string) — Filter by patient name or CNP

**Response:**
```json
{
  "patients": [
    {
      "cnp": "string",
      "id": "string",
      "name": "string",
      "age": "integer",
      "birthdate": "string (YYYY-MM-DD)",
      "sex": "string (M|F|O)"
    }
  ],
  "total": "integer",
  "pages": "integer",
  "filters": "object"
}
```

#### GET /api/patients/{cnp}
Provide a single patient's data by CNP.

**Path Parameters:**
- `cnp` (string) — Patient CNP

**Response:**
```json
{
  "cnp": "string",
  "id": "string",
  "name": "string",
  "age": "integer",
  "birthdate": "string (YYYY-MM-DD)",
  "sex": "string (M|F|O)",
  "exams": ["string (uid)"]
}
```

**Error (404):**
```json
{"error": "Patient not found"}
```

---

### Statistics

#### GET /api/stats
Provide overall statistical data for the dashboard.

**Response:**
```json
{
  "total": "integer",
  "reviewed": "integer",
  "positive": "integer",
  "correct": "integer",
  "wrong": "integer",
  "region": "object (per-region breakdown)",
  "trends": "object",
  "monthly_trends": "object",
  "avg_processing_time": "number",
  "throughput": "number",
  "error_stats": "object"
}
```

#### GET /api/stats/radiologists
Provide per-radiologist performance statistics. Non-admin users receive anonymized names (initials only).

**Response:** Object keyed by radiologist name/initials, each value containing report counts, accuracy, and timing metrics.

#### GET /api/stats/radiologists/monthly_trends
Provide monthly report counts per radiologist over the last 12 months. Non-admin users receive anonymized names.

**Response:** Object keyed by month (`YYYY-MM`), each value an object of radiologist → count.

#### GET /api/stats/diagnostics
Provide per-diagnostic performance statistics including accuracy, sensitivity, and PPV.

**Response:** Object keyed by diagnostic summary, each value containing counts and statistical metrics.

#### GET /api/stats/insights
Provide analytical insights including age distribution, AI processing times, and radiologist workload metrics.

**Response:**
```json
{
  "age_distribution": "object",
  "processing_times": "object",
  "radiologist_metrics": "object"
}
```

---

### Reference Data

#### GET /api/regions
Provide the list of supported anatomic regions.

**Response:**
```json
["chest", "abdomen", "..."]
```

#### GET /api/diagnostics
Provide distinct diagnostic summaries and their report counts.

**Response:** Object mapping diagnostic summary string to report count integer.

#### GET /api/diagnostics/monthly_trends
Provide monthly counts for the top 10 diagnostics over the last 12 months.

**Response:** Object keyed by month (`YYYY-MM`), each value an object of diagnostic → count.

#### GET /api/radiologists
Provide distinct radiologist names. Non-admin users receive initials only.

**Response:**
```json
["string"]
```

#### GET /api/severity
Provide severity levels and their report counts.

**Response:** Object mapping severity level (string key `"0"`–`"10"`) to report count integer.

#### GET /api/config
Provide global configuration parameters to the frontend.

**Response:**
```json
{
  "OPENAI_URL_PRIMARY": "string",
  "OPENAI_URL_SECONDARY": "string",
  "NTFY_URL": "string",
  "AE_TITLE": "string",
  "AE_PORT": "integer",
  "REMOTE_AE_TITLE": "string",
  "REMOTE_AE_IP": "string",
  "REMOTE_AE_PORT": "integer"
}
```

---

### Actions

#### POST /api/radreview
Record a radiologist's review of an exam as normal or abnormal.

**Request Body:**
```json
{
  "uid": "string",
  "normal": "boolean"
}
```

**Response:**
```json
{"status": "success"}
```

#### POST /api/requeue
Re-queue an exam for AI processing (sets status to `requeue`).

**Request Body:**
```json
{"uid": "string"}
```

**Response:**
```json
{"status": "string", "message": "string"}
```

#### POST /api/dicomquery
Trigger a manual DICOM C-FIND/C-MOVE query for recent studies.

**Request Body:**
```json
{"hours": "integer (default: 3)"}
```

**Response:**
```json
{"status": "string", "message": "string"}
```

#### POST /api/getrad
Retrieve radiologist report for an exam from FHIR and queue it for LLM processing.

**Request Body:**
```json
{"uid": "string"}
```

**Response:**
```json
{"status": "string", "message": "string"}
```

**Error (400):** Missing or invalid UID.

---

### AI Endpoints

These endpoints are rate-limited to 10 requests per IP per minute and return HTTP 429 when exceeded.

#### POST /api/check
Analyze a free-text radiology report for pathological findings.

**Request Body:**
```json
{"report": "string"}
```

**Response:**
```json
{
  "pathologic": "string (yes|no)",
  "severity": "integer (0-10)",
  "summary": "string (1-3 words)"
}
```

**Error (400):** Empty report text.

#### POST /api/analyse
Perform detailed three-pass analysis of a radiology report.

**Request Body:**
```json
{"report": "string"}
```

**Response:**
```json
{
  "first_pass": {"topic": "string", "purpose": "string", "primary_findings": "string"},
  "second_pass": {"main_points": ["string"], "supporting_evidence": ["string"], "conclusions_valid": "boolean"},
  "third_pass": {"assumptions": ["string"], "missing_info": ["string"], "critique": "string"},
  "overall_assessment": "string"
}
```

**Error (400):** Empty report text.

#### POST /api/translate
Translate a Romanian radiology report to English.

**Request Body:**
```json
{"report": "string"}
```

**Response:**
```json
{"translation": "string"}
```

---

### Specification

#### GET /api/spec
Serve the OpenAPI specification file (`static/spec.json`).

---

## Static File Serving

#### GET /images/{filename}
Serve image files (PNG, DCM) from the `images/` directory.

#### GET /static/{filename}
Serve static files (CSS, JS, HTML, WAV, PNG, ICO) from the `static/` directory.
