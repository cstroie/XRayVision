# XRayVision API Documentation

This document describes the REST API for the XRayVision application, a DICOM processor with OpenAI integration.

## Base URL

The API is accessible at `http://localhost:8000` by default. The port can be configured in the `xrayvision.cfg` file under `[dashboard]` section with the `DASHBOARD_PORT` parameter.

## Authentication

All API endpoints (except static file serving) require HTTP Basic Authentication. Credentials are configured in the `xrayvision.cfg` file under the `[users]` section.

Example:
```
Authorization: Basic base64(username:password)
```

## API Endpoints

### Dashboard Pages

#### GET /
Serve the main dashboard HTML page.

#### GET /stats
Serve the statistics HTML page.

#### GET /about
Serve the about HTML page.

#### GET /check
Serve the report check HTML page.

#### GET /favicon.ico
Serve the favicon.ico file.

### WebSocket

#### GET /ws
Handle WebSocket connections for real-time dashboard updates.

### API Endpoints

#### GET /api/exams
Provide paginated exam data with optional filters.

**Query Parameters:**
- `page` (integer, default: 1) - Page number
- `reviewed` (string, enum: any, yes, no) - Filter by review status
- `positive` (string, enum: any, yes, no) - Filter by AI prediction
- `valid` (string, enum: any, yes, no) - Filter by validation status
- `region` (string) - Filter by anatomic region
- `status` (string) - Filter by processing status
- `search` (string) - Filter by patient name

**Response:**
```json
{
  "exams": [
    {
      "uid": "string",
      "patient": {
        "name": "string",
        "cnp": "string",
        "id": "string",
        "age": "integer",
        "sex": "string"
      },
      "exam": {
        "created": "string",
        "date": "string",
        "time": "string",
        "protocol": "string",
        "region": "string",
        "status": "string",
        "type": "string",
        "study": "string",
        "series": "string",
        "id": "string"
      },
      "report": {
        "ai": {
          "text": "string",
          "short": "string",
          "created": "string",
          "updated": "string",
          "positive": "boolean",
          "confidence": "integer",
          "model": "string",
          "latency": "integer"
        },
        "rad": {
          "text": "string",
          "positive": "boolean",
          "severity": "integer",
          "summary": "string",
          "created": "string",
          "updated": "string",
          "id": "string",
          "type": "string",
          "radiologist": "string",
          "justification": "string",
          "model": "string",
          "latency": "integer"
        },
        "correct": "integer",
        "reviewed": "integer"
      }
    }
  ],
  "total": "integer",
  "pages": "integer",
  "filters": "object"
}
```

#### GET /api/stats
Provide statistical data for the dashboard.

**Response:**
```json
{
  "total": "integer",
  "reviewed": "integer",
  "positive": "integer",
  "correct": "integer",
  "wrong": "integer",
  "region": "object",
  "trends": "object",
  "monthly_trends": "object",
  "avg_processing_time": "number",
  "throughput": "number",
  "error_stats": "object"
}
```

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

#### GET /api/regions
Provide supported regions for the frontend dropdown.

**Response:**
```json
["string"]
```

#### GET /api/patients
Provide paginated patient data with optional filters.

**Query Parameters:**
- `page` (integer, default: 1) - Page number
- `search` (string) - Filter by patient name or CNP

**Response:**
```json
{
  "patients": [
    {
      "cnp": "string",
      "id": "string",
      "name": "string",
      "age": "integer",
      "sex": "string"
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
- `cnp` (string) - Patient CNP

**Response:**
```json
{
  "cnp": "string",
  "id": "string",
  "name": "string",
  "age": "integer",
  "sex": "string",
  "exams": ["string"]
}
```

**Error Response:**
```json
{
  "error": "Patient not found"
}
```

#### GET /api/exams/{uid}
Provide a single exam's data by UID.

**Path Parameters:**
- `uid` (string) - Exam UID

**Response:**
```json
{
  "uid": "string",
  "patient": {
    "name": "string",
    "cnp": "string",
    "id": "string",
    "age": "integer",
    "sex": "string"
  },
  "exam": {
    "created": "string",
    "date": "string",
    "time": "string",
    "protocol": "string",
    "region": "string",
    "status": "string",
    "type": "string",
    "study": "string",
    "series": "string",
    "id": "string"
  },
  "report": {
    "ai": {
      "text": "string",
      "short": "string",
      "created": "string",
      "updated": "string",
      "positive": "boolean",
      "confidence": "integer",
      "model": "string",
      "latency": "integer"
    },
    "rad": {
      "text": "string",
      "positive": "boolean",
      "severity": "integer",
      "summary": "string",
      "created": "string",
      "updated": "string",
      "id": "string",
      "type": "string",
      "radiologist": "string",
      "justification": "string",
      "model": "string",
      "latency": "integer"
    },
    "correct": "integer",
    "reviewed": "integer"
  }
}
```

**Error Response:**
```json
{
  "error": "Exam not found"
}
```

#### POST /api/radreview
Record radiologist's review of an exam as normal or abnormal.

**Request Body:**
```json
{
  "uid": "string",
  "normal": "boolean"
}
```

**Response:**
```json
{
  "status": "string"
}
```

#### POST /api/requeue
Re-queue an exam for processing.

**Request Body:**
```json
{
  "uid": "string"
}
```

**Response:**
```json
{
  "status": "string",
  "message": "string"
}
```

#### POST /api/dicomquery
Trigger a manual DICOM query/retrieve operation.

**Request Body:**
```json
{
  "hours": "integer"
}
```

**Response:**
```json
{
  "status": "string",
  "message": "string"
}
```

#### POST /api/check
Analyze a free-text radiology report for pathological findings.

**Request Body:**
```json
{
  "report": "string"
}
```

**Response:**
```json
{
  "pathologic": "string",
  "severity": "integer",
  "summary": "string"
}
```

#### GET /api/spec
Serve the OpenAPI specification file.

**Response:**
```json
{
  "openapi": "string",
  "info": "object",
  "servers": "array",
  "components": "object",
  "paths": "object"
}
```

### Static File Serving

#### GET /images/{filename}
Serve static image files (PNG, DCM) from the `images/` directory.

#### GET /static/{filename}
Serve other static files (CSS, JS, HTML, WAV, PNG, ICO) from the `static/` directory.
