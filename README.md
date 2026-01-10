# XRayVision

A real-time DICOM relay and analysis system with AI integration, persistent history, and a live web dashboard for reviewing and flagging X-ray images.

## Features

* Asynchronous TCP DICOM server (storescp-compatible)
* Real-time queue processing with AI API integration
* Automatic DICOM-to-PNG conversion (OpenCV)
* Persistent processing history with SQLite storage
* WebSocket-powered live dashboard updates
* Manual review of processed items (normal/abnormal)
* Manual DICOM QueryRetrieve trigger with configurable time span
* Automatic QueryRetrieve of CR modality studies at configurable intervals
* Fully responsive PicoCSS dashboard with lightbox image previews
* Comprehensive statistics and performance metrics
* Configuration file support
* Patient age calculation from Romanian ID numbers
* Anatomic region identification and region-specific analysis
* Previous report comparison for longitudinal studies
* Notification system for positive findings via ntfy.sh
* Database backup and maintenance routines
* Logging with timestamps to both console and file
* FHIR integration for patient data and radiologist reports
* AI-powered analysis of radiologist reports for quality metrics
* Report checking functionality for free-text radiology reports

---

## Requirements

* Python 3.8+
* DICOM peer system for QueryRetrieve (pynetdicom-compatible)
* OpenAI API endpoint (can be local or remote)

### Python Packages

```bash
pip install aiohttp pydicom pynetdicom opencv-python numpy
```

---

## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/cstroie/XRayVision.git
cd XRayVision
```

2. Update the configuration in `xrayvision.cfg`:

```ini
[openai]
OPENAI_URL_PRIMARY = http://127.0.0.1:8080/v1/chat/completions
OPENAI_URL_SECONDARY = http://127.0.0.1:11434/v1/chat/completions
OPENAI_API_KEY = sk-your-api-key
MODEL_NAME = medgemma-4b-it

[dicom]
AE_TITLE = XRAYVISION
AE_PORT = 4010
REMOTE_AE_TITLE = DICOM_SERVER
REMOTE_AE_IP = 192.168.1.1
REMOTE_AE_PORT = 104
```

3. Run the server:

```bash
export OPENAI_API_KEY="sk-your-api-key"
python xrayvision.py
```

Optional arguments:
* `--keep-dicom` - Do not delete .dcm files after conversion
* `--load-dicom` - Load existing .dcm files in queue
* `--no-query` - Do not query the DICOM server automatically
* `--enable-ntfy` - Enable ntfy.sh notifications
* `--model` - Model name to use for analysis
* `--retrieval-method` - DICOM retrieval method (C-MOVE or C-GET)
* `--log-level` - Set logging level (DEBUG, INFO, WARNING, ERROR)

4. Open the dashboard:

```
http://localhost:8000
```

---

## Dashboard Features

* Live processing statistics (queue, current file, success, failure)
* Paginated processed files with thumbnails
* Real-time review functionality (normal/abnormal)
* Manual QueryRetrieve trigger (select time span: 1, 3, 6, 12, 24 hours)
* Lightbox image preview with reviewed and positive highlighting
* Comprehensive statistics page with charts and metrics
* Configuration display
* Search and filtering capabilities
* Patient details view
* Radiologist report integration and analysis
* Report checking page for free-text analysis

---

## Configuration

XRayVision uses a configuration file (`xrayvision.cfg`) for all settings. A default configuration is provided in the file, and you can override settings by creating a `local.cfg` file with your custom values.

Key configuration sections include:
* `general` - Database path, backup directory
* `users` - User credentials with roles (admin, user)
* `dicom` - DICOM server settings (AE title, port, remote server details)
* `openai` - OpenAI API endpoints and credentials
* `dashboard` - Dashboard port
* `notifications` - ntfy.sh notification URL
* `processing` - Processing options (page size, DICOM file handling, query settings)
* `regions` - Anatomic region identification rules (keywords for region detection)
* `questions` - Region-specific questions for AI analysis
* `supported_regions` - List of regions to process (enable/disable regions)

---

## Logging

All events are logged to:

* `xrayvision.log` file
* Console output

Timestamps, info, warnings, and errors are all captured.

---

## API Endpoints

* `/` - Main dashboard
* `/stats` - Statistics page
* `/about` - About page
* `/check` - Report check page
* `/ws` - WebSocket for real-time updates
* `/` - Main dashboard
* `/stats` - Statistics page
* `/about` - About page
* `/check` - Report check page
* `/ws` - WebSocket for real-time updates
* `/api/exams` - Get exams with pagination and filtering
* `/api/stats` - Get statistics data
* `/api/config` - Get configuration parameters
* `/api/regions` - Get supported regions
* `/api/patients` - Get patients with pagination and filtering
* `/api/patients/{cnp}` - Get patient by CNP
* `/api/exams/{uid}` - Get exam by UID
* `/api/radreview` - Record radiologist review of an exam
* `/api/requeue` - Re-queue an exam for processing
* `/api/dicomquery` - Manually trigger DICOM QueryRetrieve
* `/api/check` - Analyze a free-text radiology report
* `/api/analyse` - Perform detailed three-pass analysis of a radiology report
* `/api/translate` - Translate a Romanian radiology report to English
* `/api/getrad` - Retrieve radiologist report from FHIR
* `/api/spec` - Get OpenAPI specification

---

## Future Improvements

* Enhanced user management with role-based access control
* More detailed audit logging
* Export functionality for reports
* Integration with more DICOM modalities
* Improved error handling and recovery mechanisms
* Enhanced statistics and reporting capabilities
