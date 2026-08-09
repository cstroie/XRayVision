# XRayVision — Architecture

## Overview

XRayVision is a single-file async Python application (`xrayvision.py`, ~6500 lines) that acts as a DICOM relay and AI-assisted radiology analysis platform. It receives X-ray studies from a PACS, converts them to PNG, sends them to a local OpenAI-compatible vision model (default: MedGemma 4B-IT), stores AI findings alongside radiologist reports fetched from FHIR, and surfaces everything via a live web dashboard.

---

## Major Components

### 1. Configuration & Startup (lines 1–320)

- `DEFAULT_CONFIG` dict merged with `xrayvision.cfg` and `local.cfg` via `configparser`
- Global constants extracted at module level: `SEVERITY_THRESHOLD`, `PAGE_SIZE`, `REGIONS`, `REGION_RULES`, `USERS`, etc.
- `load_prompts()` reads `prompts/*.txt` into the `PROMPTS` dict at startup
- `USERS` dict built from the `[users]` config section (username → password + role)

### 2. Database Layer (lines 335–2100)

Layered helpers, bottom-up:

| Function | Role |
|---|---|
| `db_execute_query()` | Read-only queries; no retry |
| `db_execute_query_retry()` | Write queries with `BEGIN IMMEDIATE` and exponential backoff (up to 5 retries) |
| `db_analyze()` | Introspects table schema via `PRAGMA table_info`; results cached in `_db_analyze_cache` |
| `db_create_insert_query()` / `db_create_select_query()` | Parameterised query string builders |
| `db_select()` / `db_insert()` / `db_update()` / `db_select_one()` | Typed CRUD helpers |
| Domain functions | `db_get_exams()`, `db_get_stats()`, `db_get_ai_report()`, `db_get_rad_report()`, `db_get_requeue_analysis()`, etc. |

**Schema:** `patients` → `exams` → `ai_reports` + `rad_reports`, all linked by `uid` (exam) or `cnp` (patient).

SQLite is configured with WAL journal mode, `isolation_level=None`, and `BEGIN IMMEDIATE` for all writes to handle concurrent async access safely.

### 3. DICOM Server (lines ~2100–2500)

- `AE` instance (pynetdicom) listening on `AE_PORT` for inbound C-STORE
- `dicom_store()` — C-STORE handler: receives dataset, calls `extract_dicom_metadata()`, classifies region via `identify_anatomic_region()`, converts to PNG, writes to DB, signals `QUEUE_EVENT`
- `send_c_move()` / `send_c_get()` — outbound retrieval from remote PACS (generator must be iterated to drive the transfer)
- `query_retrieve_loop()` — periodic async task: C-FIND on PACS → C-MOVE or C-GET for new studies not yet in the database

### 4. Image Processing (lines ~2350–2550)

| Function | Role |
|---|---|
| `extract_dicom_metadata()` | Pulls patient/exam info from DICOM tags (SeriesDate, SeriesTime, PatientID, etc.) |
| `dicom_to_png()` | Converts DICOM pixel array to PNG via OpenCV/NumPy |
| `apply_gamma_correction()` | Adaptive gamma based on mean pixel intensity; falls back to identity (γ=1.0) when mean ≤ 0 |
| `identify_anatomic_region()` | Keyword-matching against `REGION_RULES` loaded from `[regions]` config |

### 5. AI Pipeline (lines ~3800–5650)

| Function | Role |
|---|---|
| `send_to_llm(session, headers, payload, url=...)` | Single POST to the given (or default) LLM URL; 300 s timeout; returns parsed JSON or None |
| `send_exam_to_llm(exam)` | Full exam processing: builds prompt, retries 3× with exponential backoff, parses FINDINGS/IMPRESSION structure, writes `ai_reports`, triggers downstream checks |
| `check_ai_report_and_update(uid)` | Sends AI report text to model with `CHK_PROMPT`; extracts severity/confidence/summary; updates `ai_reports` |
| `check_rad_report_and_update(uid)` | Same flow for radiologist reports |
| `translate_report(text)` | Sends Romanian report to model with `TRN_PROMPT`; stores English result in `rad_reports.text_en` |
| `check_report(text)` | Ad-hoc single-pass check via `/api/check` |
| `detailed_analysis_report(text)` | Multi-pass analysis (overview → detail → critique → assessment) via `ANA_PROMPT` |

All AI calls use a shared `aiohttp.ClientSession` scoped to the operation. Each task (`exam`, `translation`, `check`, `analysis`) is routed to its own resolved backend/model in `TASK_ACTIVE`; failover across the ordered `[llm] backends = ...` list is managed by `llm_health_check()`.

### 6. FHIR / HIS Integration (lines ~4900–5300)

- `get_fhir_patient()` — searches Hipocrate HIS by CNP or patient name
- `get_fhir_service_requests()` — fetches service requests for a patient
- `get_fhir_diagnostic_reports()` — fetches completed `DiagnosticReport` resources
- `extract_report_data()` — extracts report text and radiologist name from `presentedForm`
- `update_patient_info_from_fhir()` — called before AI processing to enrich exam data (patient ID, birthdate, sex)
- `fhir_loop()` — async task that continuously polls for new radiologist reports on `done` exams

All FHIR calls use HTTP Basic Auth (`FHIR_USERNAME` / `FHIR_PASSWORD`) against `FHIR_URL`.

### 7. Web / API Layer (lines ~3500–3800, 5650–6400)

- Static page handlers (`serve_dashboard_page()`, etc.) — serve `static/*.html`
- REST handlers under `/api/*`:
  - `GET /api/exams` — paginated exam list with filters
  - `GET /api/stats/*` — statistics endpoints (global, regional, radiologists, insights)
  - `POST /api/check` / `/api/analyse` — ad-hoc report analysis
  - `POST /api/rad_review` — radiologist review submission
  - `POST /api/requeue` — re-queue exam for reprocessing
  - `GET /api/config` — exposes runtime config (including `USER_ROLE`) to frontend
  - `GET /api/spec` — OpenAPI specification
- `ws_handler()` — WebSocket endpoint at `/ws`; adds client to `websocket_clients` set
- `broadcast_dashboard_update()` — pushes JSON event to all connected WebSocket clients
- `auth_middleware()` — HTTP Basic Auth gate; sets `request.user_role`; logs auth events to audit log
- `rate_limit_middleware()` — per-IP sliding window rate limiter; evicts stale IPs after each request

### 8. Background Loops

All loops are `async def` tasks started in `main()` via `asyncio.gather()`.

| Task | Function | Trigger | Role |
|---|---|---|---|
| Processing | `relay_to_llm_loop()` | `QUEUE_EVENT` (async) | Dequeues `queued`/`requeue` exams, calls `send_exam_to_llm()` |
| Query/Retrieve | `query_retrieve_loop()` | Timer (`QUERY_INTERVAL`) | Periodic C-FIND → C-MOVE/GET from PACS |
| FHIR polling | `fhir_loop()` | Timer | Checks `done` exams for missing radiologist reports |
| AI health check | `llm_health_check()` | Timer (300 s) | Resolves each task's backend/model into `TASK_ACTIVE`; sets `active_llm_url`; signals `QUEUE_EVENT` when a backend recovers |
| Maintenance | `maintenance_loop()` | Timer (daily) | DB backup, dead WebSocket cleanup, cache eviction, purge old errors |

---

## Data Flow

```
PACS
  │  C-STORE (inbound) or C-MOVE/C-GET (outbound retrieval)
  ▼
dicom_store()
  ├─ extract_dicom_metadata()    → patient + exam dict
  ├─ identify_anatomic_region()  → region string (or '' if unknown)
  ├─ dicom_to_png()              → images/<uid>.png
  ├─ db_add_exam()               → exams table (status = queued)
  └─ QUEUE_EVENT.set()
        │
        ▼
relay_to_llm_loop()               ← wakes on QUEUE_EVENT
  └─ send_exam_to_llm(exam)
       ├─ update_patient_info_from_fhir()   → enriches patient dict from HIS
       ├─ prepare_exam_data()               → region, projection, gender text
       ├─ create_exam_prompt()              → assembles REP_PROMPT + USR_PROMPT + base64 image
       ├─ send_to_llm()                     → POST /v1/chat/completions  (retry ×3)
       ├─ db_insert('ai_reports', text=…)   → exams.status = done
       └─ check_ai_report_and_update()
            ├─ send_to_llm()                → CHK_PROMPT → severity / confidence / summary
            ├─ db_update('ai_reports', …)
            ├─ broadcast_dashboard_update() → WebSocket → browser
            └─ send_ntfy_notification()     → ntfy.sh  (if severity ≥ SEVERITY_THRESHOLD)

fhir_loop()  (runs in parallel)
  └─ get_fhir_diagnostic_reports()
       └─ extract_report_data()
            ├─ db_insert / db_update 'rad_reports'
            ├─ check_rad_report_and_update()   → CHK_PROMPT → severity / summary
            └─ translate_report()              → TRN_PROMPT → rad_reports.text_en

Browser
  ├─ WebSocket /ws  ← server-push on new_exam / review / requeue events
  └─ REST /api/*    ← on-demand: exams, stats, config, check, analyse, review
```

---

## Key Shared State

| Variable | Type | Purpose |
|---|---|---|
| `QUEUE_EVENT` | `asyncio.Event` | Signals relay loop when exams are queued or endpoint recovers |
| `active_llm_url` | `str \| None` | Currently healthy LLM endpoint for the 'exam' task (set by health check loop) |
| `TASK_ACTIVE` | `dict[str, dict]` | Per-task resolved `{backend, url, model, api_key}` (exam, translation, check, analysis) |
| `health_status` | `dict[str, bool]` | Reachability of each configured LLM backend (by name) |
| `timings` | `dict[str, int]` | Rolling average latencies (ms) per AI call type |
| `dashboard` | `dict` | Queue sizes and processing state pushed to WebSocket clients |
| `websocket_clients` | `set` | Live WebSocket connections; cleaned up by maintenance loop |
| `_db_analyze_cache` | `dict` | Table schema cache; cleared periodically to prevent unbounded growth |
| `_translation_cache` | `dict` | Translation dedup cache (hash → translated text) |
| `_rate_limit_store` | `dict` | Per-IP request timestamps for rate limiting |

---

## Prompt System

Prompts live in `prompts/` as plain-text files loaded at startup into the `PROMPTS` dict:

| Key | File | Used in |
|---|---|---|
| `REP_PROMPT` | `rep_prompt.txt` | System prompt for X-ray report generation |
| `USR_PROMPT` | `usr_prompt.txt` | User-turn preamble for exam context |
| `REV_PROMPT` | `rev_prompt.txt` | Review prompt appended for re-queued exams |
| `CHK_PROMPT` | `chk_prompt.txt` | Structured extraction: severity, confidence, summary |
| `ANA_PROMPT` | `ana_prompt.txt` | Multi-pass detailed analysis |
| `TRN_PROMPT` | `trn_prompt.txt` | Romanian → English translation |

---

## Status Values for Exams

| Status | Meaning |
|---|---|
| `none` | Received, not yet queued |
| `queued` | Waiting in processing queue |
| `processing` | Currently being sent to AI |
| `done` | AI analysis complete |
| `error` | Processing failed |
| `ignore` | Unsupported region, skipped |
| `requeue` | Manually re-queued for reprocessing |

---

## Severity and Scoring

| Field | Table | Range | Meaning |
|---|---|---|---|
| `positive` | both | -1 / 0 / 1 | -1 = not assessed, 0 = no findings, 1 = findings present |
| `severity` | both | -1 … 10 | -1 = not assessed; 0–10 scale |
| `confidence` | `ai_reports` | -1 … 100 | AI self-confidence percentage |

`SEVERITY_THRESHOLD` (default 5) is the cut-off used for positive/negative classification, correctness calculation (TP/TN/FP/FN), and ntfy.sh notifications.
