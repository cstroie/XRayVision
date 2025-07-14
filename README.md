# XRayVision

A real-time DICOM relay and analysis system with OpenAI integration, persistent history, and a live web dashboard for reviewing and flagging X-ray images.

## Features

* Asynchronous TCP DICOM server (storescp-compatible)
* Real-time queue processing and OpenAI API integration
* Automatic DICOM-to-PNG conversion (OpenCV)
* Persistent processing history with SQlite storage
* WebSocket-powered live dashboard updates
* Manual flagging/unflagging of processed items
* Manual DICOM QueryRetrieve trigger with configurable time span
* Automatic hourly QueryRetrieve of CR modality studies
* Fully responsive PicoCSS dashboard with lightbox image previews
* Logging with timestamps to both console and file

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

2. Update the configuration in `xrayvision.py`:

```python
OPENAI_API_URL = 'http://127.0.0.1:8080/v1/chat/completions'
AE_TITLE = 'XRAYVISION'
AE_PORT  = 4010
REMOTE_AE_TITLE = '3DNETCLOUD'
REMOTE_AE_IP = '192.168.3.50'
REMOTE_AE_PORT = 104
```

3. Run the server:

```bash
export OPENAI_API_KEY="sk-your-api-key"
python xrayvision.py
```

4. Open the dashboard:

```
http://localhost:8000
```

---

## Dashboard Features

* Live processing statistics (queue, current file, success, failure)
* Last 100 processed files with thumbnails
* Real-time flag/unflag functionality
* Manual QueryRetrieve trigger (select time span: 1, 3, 6, 12, 24 hours)
* Lightbox image preview with flagged and positive highlighting

---

## Logging

All events are logged to:

* `xrayvision.log` file
* Console output

Timestamps, info, warnings, and errors are all captured.

---

## Future Improvements

* Pagination and filtering for large histories
* Live logs displayed in the dashboard
* Docker support
