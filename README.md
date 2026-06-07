# XRayVision

**AI-assisted radiology analysis for clinical X-ray workflows**

XRayVision bridges your PACS and a local AI vision model. It receives X-ray studies via DICOM, runs them through an OpenAI-compatible vision model (tested with [MedGemma 4B-IT](https://huggingface.co/google/medgemma-4b-it)), and presents findings alongside radiologist reports in a live web dashboard — helping radiologists prioritise worklists and track AI performance over time.

> Designed for Romanian hospital workflows (Hipocrate HIS / FHIR), but adaptable to any DICOM-compliant PACS.

---

## How it works

```
PACS ──C-STORE/C-MOVE──► XRayVision ──► AI Vision Model
                               │              │
                          SQLite DB  ◄─────────┘
                               │
                          FHIR / HIS ──► Radiologist reports
                               │
                          Web Dashboard (live, WebSocket)
```

1. Studies arrive via **C-STORE** (push) or are pulled via periodic **C-FIND/C-MOVE**
2. DICOM files are converted to PNG and sent to the configured vision model
3. AI findings are stored alongside radiologist reports fetched from **FHIR**
4. The **live dashboard** lets radiologists review, flag, and re-queue exams

---

## Features

| | |
|---|---|
| **DICOM** | C-STORE receiver + periodic C-FIND/C-MOVE/C-GET from PACS |
| **AI analysis** | OpenAI-compatible API, primary + secondary endpoint failover |
| **Radiologist reports** | FHIR integration, auto-translation (Romanian → English) |
| **Dashboard** | Live WebSocket updates, lightbox previews, search & filter |
| **Statistics** | Per-region, per-radiologist, per-diagnostic accuracy metrics |
| **Notifications** | ntfy.sh push alerts for high-severity positive findings |
| **Security** | HTTP Basic Auth, role-based access (admin / user), audit log |
| **Storage** | SQLite with WAL, automatic backups |

---

## Requirements

- Python 3.8+
- An OpenAI-compatible vision API endpoint (local or remote)
- A DICOM-compliant PACS (for QueryRetrieve)

```bash
pip install aiohttp pydicom pynetdicom opencv-python numpy
```

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/cstroie/XRayVision.git
cd XRayVision
```

### 2. Configure

Copy or edit `xrayvision.cfg`. The most important sections:

```ini
[openai]
OPENAI_URL_PRIMARY   = http://127.0.0.1:8080/v1/chat/completions
OPENAI_URL_SECONDARY = http://127.0.0.1:11434/v1/chat/completions
OPENAI_API_KEY       = sk-your-api-key
MODEL_NAME           = medgemma-4b-it

[dicom]
AE_TITLE        = XRAYVISION
AE_PORT         = 4010
REMOTE_AE_TITLE = YOUR_PACS
REMOTE_AE_IP    = 192.168.1.1
REMOTE_AE_PORT  = 104

[users]
admin = yourpassword,admin
```

For local overrides (passwords, paths), create a `local.cfg` — it is gitignored and takes precedence.

### 3. Run

```bash
python3 xrayvision.py
```

### 4. Open the dashboard

```
http://localhost:8000
```

Log in with the credentials from `[users]` in your config.

---

## Command-line options

| Option | Description |
|---|---|
| `--keep-dicom` | Keep `.dcm` files after PNG conversion |
| `--load-dicom` | Load any existing `.dcm` files into the queue on startup |
| `--no-query` | Disable automatic DICOM QueryRetrieve |
| `--enable-ntfy` | Enable ntfy.sh push notifications |
| `--model NAME` | Override the AI model name |
| `--retrieval-method` | `C-MOVE` (default) or `C-GET` |
| `--log-level` | `DEBUG`, `INFO` (default), `WARNING`, or `ERROR` |
| `--translate-existing` | Translate any existing radiologist reports that lack an English translation |

---

## Dashboard pages

| Page | URL | Description |
|---|---|---|
| Dashboard | `/` | Live exam queue, thumbnails, review controls |
| Statistics | `/stats` | Overall accuracy, region breakdown, monthly trends |
| Radiologists | `/stats/radiologists` | Per-radiologist workload and accuracy |
| Diagnostics | `/stats/diagnostics` | Per-diagnostic category performance |
| Insights | `/stats/insights` | Age distribution, processing times, workload metrics |
| Check | `/check` | Free-text report analysis (paste any report) |
| About | `/about` | System status, model health, configuration |

---

## Configuration reference

All options live in `xrayvision.cfg`. The full file is commented. Key sections:

| Section | Purpose |
|---|---|
| `[general]` | Database path, backup directory |
| `[users]` | Credentials — `username = password,role` (`admin` or `user`) |
| `[dicom]` | AE title/port, remote PACS address, retrieval method |
| `[openai]` | AI API URLs, key, model name |
| `[fhir]` | FHIR server URL and credentials (Hipocrate HIS) |
| `[dashboard]` | Web server port |
| `[notifications]` | ntfy.sh URL, optional image base URL for attachments |
| `[processing]` | Page size, DICOM file handling, query interval, severity threshold |
| `[regions]` | Keyword rules for anatomic region detection |
| `[questions]` | Region-specific clinical questions sent to the AI |
| `[supported_regions]` | Which regions to process (`true`) or skip (`false`) |

---

## Roles and access

Two roles are supported:

- **admin** — full access; sees real radiologist names everywhere
- **user** — radiologist names are anonymised to initials in the dashboard and API

---

## Logging

Application events are written to `xrayvision.log` and the console.

Security and clinical actions are written separately to `xrayvision_audit.log`:

| Event | Trigger |
|---|---|
| `AUTH_OK` | Successful login |
| `AUTH_FAIL` | Failed login attempt |
| `RAD_REVIEW` | Radiologist marks exam normal / abnormal |
| `REQUEUE` | Exam re-queued for AI reprocessing |
| `DICOM_QUERY` | Manual DICOM QueryRetrieve triggered |

---

## Documentation

- [`API.md`](API.md) — REST API reference
- [`DATABASE.md`](DATABASE.md) — Database schema
- [`static/spec.json`](static/spec.json) — OpenAPI 3.0 specification (also served at `/api/spec`)
- [`CLAUDE.md`](CLAUDE.md) — Developer guide

---

## License

Copyright © 2026 Costin Stroie. See repository for licence details.
