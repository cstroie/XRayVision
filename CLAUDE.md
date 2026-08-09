# XRayVision — Developer Guide for Claude

## Project overview

XRayVision is an async Python application that acts as a DICOM relay and AI-assisted radiology analysis platform. It receives X-ray studies from a PACS via C-STORE or C-MOVE/C-GET, converts them to PNG, sends them to a local/remote OpenAI-compatible vision model (default: MedGemma 4B-IT), stores the AI findings alongside radiologist reports fetched from FHIR, and surfaces everything via a live web dashboard.

The entire backend lives in a **single file**: `xrayvision.py` (~6600 lines). Do not split it into modules unless explicitly asked.

---

## Architecture

```
DICOM server (pynetdicom)
    └─ C-STORE handler → dicom_store()
         └─ QUEUE_EVENT → relay_to_llm_loop()
              └─ send_exam_to_llm()
                   ├─ update_patient_info_from_fhir()  ← FHIR integration
                   ├─ prepare_exam_data()               ← region/projection/gender
                   ├─ create_exam_prompt()              ← prompt assembly
                   └─ send_to_llm()                     ← HTTP to LLM server

query_retrieve_loop()    ← periodic C-FIND + C-MOVE/C-GET from PACS
fhir_loop()             ← polls FHIR for radiologist reports on processed exams
maintenance_loop()      ← DB backup, dead WebSocket cleanup

aiohttp web server
    ├─ Static HTML pages  (static/*.html)
    ├─ REST API           (/api/*)
    └─ WebSocket          (/ws) → broadcast_dashboard_update()
```

---

## Key design decisions

### Single-file layout
All Python logic stays in `xrayvision.py`. Functions are ordered: config → DB → DICOM → AI → web handlers → loops → main. Do not reorganise this order.

### Configuration
- `xrayvision.cfg` — tracked default config (safe to commit, no secrets)
- `local.cfg` — untracked local overrides (in `.gitignore`)
- Config is loaded once at startup via `configparser`. To add a new option, add it to `DEFAULT_CONFIG` and read it in the globals block below `load_prompts()`.

### Logging format
All log lines use a single global logger with the format:
```
%(asctime)s | %(levelname)8s | %(message)s
```
Example: `2026-01-10 10:26:37,940 |    ERROR | Failed to parse AI translation response`

- Use `logging.info/warning/error/debug` — never `print()`.
- Level names are right-aligned to 8 chars via `%(levelname)8s`.
- Third-party loggers (`aiohttp`, `asyncio`, `pynetdicom`, `pydicom`) are suppressed to `WARNING`.
- A separate `xrayvision_audit.log` file exists for audit events (access, reviews, re-queuing).

### Database
- SQLite with WAL journal mode, NORMAL sync, foreign keys ON.
- Schema: `patients` → `exams` → `ai_reports` + `rad_reports`.
- All writes go through `db_execute_query_retry()` (exponential backoff, up to 5 retries).
- Read-only queries use `db_execute_query()`.
- Helper builders: `db_create_insert_query()`, `db_create_select_query()`, `db_select()`, `db_insert()`, `db_update()`.
- Result unpacking: `db_unpack_result(result, keys)` converts a list of tuples into a dict.
- Schema changes require updating both the `CREATE TABLE` in `db_init()` and any relevant helper queries throughout the file.

### AI integration
- LLM backends are configured as an arbitrary, ordered list: `[llm] backends = primary, secondary, ...` names matching `[llm:<name>]` sections, each with its own `url`, optional `api_key`, and a model per task (`exam`, `translation`, `check`, `analysis`). A task left blank in a backend section falls back to that same backend's `exam` model. Loaded into `LLM_BACKEND_NAMES` / `LLM_BACKENDS` at startup.
- `llm_health_check()` runs every 300 s, probes `<backend>/v1/models` for each backend (reachability into `health_status[name]`, parsed model-id set into `available_models[name]`), then calls `resolve_task_backend(task)` for each of the 4 tasks to pick the highest-priority backend that is reachable **and** has that task's model — result stored per-task in `TASK_ACTIVE[task] = {'backend', 'url', 'model', 'api_key'}`. If a backend doesn't return a parseable model list, it's treated permissively (not gated on model presence) to avoid breaking servers without `/v1/models` support.
- `active_llm_url` is kept as a back-compat alias for `TASK_ACTIVE['exam']['url']` — it's what gates `relay_to_llm_loop`, the startup wait loop, and the coarse "AI reachable" check in `check_rad_report_and_update()`.
- Each AI-calling function (`send_exam_to_llm`/`prepare_ai_request_data`, `translate_report`, `check_report`, `detailed_analysis_report`) reads its own task's entry from `TASK_ACTIVE` for the model, backend URL, and API key — never a single global model/URL. `build_ai_headers(api_key)` omits the `Authorization` header entirely when a backend's `api_key` is blank.
- All AI calls go through `send_to_llm(session, headers, payload, url=...)`.
- `send_exam_to_llm()` implements exponential backoff (3 retries, 2 s / 4 s / 8 s delays).
- AI responses are expected as JSON with specific keys; parsing failures are logged at ERROR level (see `TODO` for known edge cases with MedGemma returning plain text).
- **Different backends may use different model-id strings for the same model** (e.g. `qwen3-4b` on llama.cpp vs `qwen/qwen3-4b` on LM Studio) — this is exactly why models are configured per-backend rather than once globally; never assume a model id is portable across backends.

### Prompt system
- Prompts live in `prompts/` as `.txt` files, loaded at startup by `load_prompts()` into the `PROMPTS` dict.
- Keys: `REP_PROMPT` (report), `USR_PROMPT` (user), `REV_PROMPT` (review), `CHK_PROMPT` (check), `ANA_PROMPT` (analysis), `TRN_PROMPT` (translation).
- Never inline prompt text in `xrayvision.py` — all prompt changes go in the `prompts/` files.
- `USR_PROMPT` supports `{anatomy}` and `{subject}` placeholders. The `{question}` placeholder is passed but currently unused (no token in the template file) — `[questions]` config is loaded but superseded by `[templates]` for supported regions.

### Web frontend
- Framework: **PicoCSS v2** (slate theme, dark mode default) loaded from CDN.
- Font: Inter via CSS variable `--pico-font-family-sans-serif`.
- All pages share `static/styles.css` and the same `<nav>` structure.
- Pages: `dashboard.html`, `stats.html`, `radiologists.html`, `diagnostics.html`, `insights.html`, `check.html`, `about.html`.
- Navigation order: Dashboard → Statistics (dropdown: Stats / Radiologists / Diagnostics / Insights) → Check → About.
- Active page link gets class `contrast` to highlight current page.
- Real-time updates delivered via WebSocket (`/ws`), not polling.
- WebSocket `onmessage` handler must JSON.parse inside try/catch — malformed frames must not kill the handler.
- Image previews use a lightbox pattern (JS in the HTML files).
- XSS prevention: use `sanitize()` before injecting into `innerHTML`; use `textContent` (no sanitize needed) for plain text nodes. Do not double-escape by combining both.
- Radiologist name anonymization is server-side only (`extract_radiologist_initials()` for non-admin users). Frontend pages must not re-anonymize names already returned by the API.
- Chart.js with `indexAxis: 'y'`: `x` is the value axis, `y` is the label axis. Place `beginAtZero`, `max`, and axis titles on `x`, not `y`.

### Domain specifics
- **Romanian healthcare context**: patient IDs are CNP (Personal Numeric Code), validated and parsed in `validate_romanian_cnp()` / `compute_age_from_cnp()`.
- Reports may be in Romanian; `translate_report()` translates them to English (stored in `rad_reports.text_en`).
- Anatomic region detection uses keyword rules from `[regions]` in config.
- Only regions listed as `true` in `[supported_regions]` are processed; others get status `ignore`.
- Per-region AI reporting checklists are defined in `[templates]` (delimiter `|`, not `,` — values contain commas). Loaded into `REGION_TEMPLATES` at startup; injected into `create_exam_prompt()` as an `ASSESS IN ORDER` section in the user-turn prompt.
- FHIR server is the Hipocrate HIS (Romanian hospital information system).

### Severity and scoring
- `positive`: -1 = not assessed, 0 = no findings, 1 = findings present (both AI and rad reports).
- `severity`: 0–10 scale, -1 = not assessed (rad reports only in the DB; AI reports also have this column).
- `confidence`: 0–100, -1 = not assessed (AI reports only).
- `correct`: 1 = AI correct, 0 = AI incorrect, -1 = not yet reviewed. **Never use truthiness checks** (`correct and ...`) — `-1` is truthy and indistinguishable from `1`. Always use `== 1`, `== 0`, `== -1`.
- `SEVERITY_THRESHOLD` (default 5) gates which positive findings trigger ntfy.sh notifications.
- `rad_reports.id = -1` is the "stop retrying" sentinel for FHIR lookups. `db_get_exams_without_rad_report()` filters `rr.id IS NULL OR rr.id > 0`, so any exam with `id = -1` is permanently excluded from `fhir_loop`. Insert this stub when a patient cannot be resolved in FHIR (404 or ambiguous name match) to prevent infinite retries.

---

## Status values for exams

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

## Adding new features

- **New API endpoint**: add `async def <name>_handler(request)` then register it in `start_dashboard()` with `app.router.add_*()`.
- **New config option**: add to `DEFAULT_CONFIG`, read in the globals block, document in `xrayvision.cfg` with a comment.
- **New DB column**: add to `CREATE TABLE` in `db_init()`, add `IF NOT EXISTS` / `ALTER TABLE` migration guard, update all relevant `db_select`/`db_insert`/`db_update` call sites.
- **New prompt**: add a file in `prompts/`, add its key to `load_prompts()`, reference it from `PROMPTS['NEW_KEY']`.
- **New region template**: add an entry to `[templates]` in `xrayvision.cfg` using `|` as the item delimiter. No code change needed.
- **New dashboard page**: create `static/<page>.html` following the existing nav structure, add a `serve_<page>_page()` handler and route.

---

## Issue workflow

When fixing issues from `issues.txt`:
1. Fix one issue at a time — no bundling unrelated changes.
2. Verify the fix makes sense against the actual code before committing.
3. Commit with a focused message referencing the issue number.
4. Move to the next issue only after the commit is done.

---

## Known issues / active work

- `translate_report()` expects the model to wrap the translation in ` ```text``` ` code fences. If the model returns plain text (no fences), the response is discarded and the function returns `None`. This is a known model behaviour issue with some versions of MedGemma.
- Issues backlog in `issues.txt`: transaction isolation, WebSocket cleanup, FHIR response validation, path validation.
- `acronyms.txt` / `find_acronyms.py` tools exist for expanding Romanian medical abbreviations (in progress).
- `[questions]` config section is loaded into `REGION_QUESTIONS` and passed to `create_exam_prompt()` but the `{question}` placeholder is not present in `usr_prompt.txt`, so questions are currently unused. Left in place intentionally — superseded by `[templates]`.

## Common pitfalls (learned from bug-fix sessions)

- **SQL in `db_analyze`**: table name is interpolated via `PRAGMA` — validate against an allowlist before interpolating; never use user input directly in SQL f-strings.
- **`GROUP_CONCAT` separator**: default separator `,` breaks `.split()` when values contain commas. Use `'||'` as separator and split on `'||'`.
- **Config delimiter for multi-item values with commas**: `[templates]` uses `|` as separator because items contain commas inside parentheses. Apply the same pattern for any future config list whose items may contain commas.
- **Blocking calls in async functions**: `process_dicom_file()` (PIL conversion) and file reads in `prepare_exam_data()` must run via `asyncio.to_thread()` — calling them directly blocks the event loop and makes the web server unreachable during DICOM ingestion.
- **Translation before AI is ready**: `translate_existing_reports()` polls `active_llm_url` (30 s interval) before querying the DB. Inline translation in `get_rad_report()` returns `False` immediately if the URL is not set, letting `fhir_loop` retry. Never call `translate_report()` when `active_llm_url is None`.
- **`SUM(CASE …)` returns NULL** (not 0) when no rows match — always guard with `or 0` in Python after fetching.
- **`HAVING` with column aliases**: SQLite does not allow `HAVING alias > N`. Use `HAVING COUNT(*) > N` or repeat the expression.
- **`isCorrect === null` vs `=== false`** in JS: `null` means not reviewed, `false` means wrong. Never use `!isCorrect` to mean "incorrect" — it catches both.
- **Empty query params**: a query param present but empty (e.g. `?positive=`) passes `!= 'any'` checks but `value[0]` raises `IndexError`. Use `value.lower().startswith('y')` or check `len(value)` first.
- **`result["choices"][0]` accesses**: the AI API can return an error dict without `choices`. Wrap in try/except or check key existence first — but note all current call sites are already inside broad `try/except Exception` blocks.

---

## File map

| Path | Purpose |
|---|---|
| `xrayvision.py` | Entire application |
| `xrayvision.cfg` | Default configuration (committed) |
| `local.cfg` | Local overrides (gitignored) |
| `prompts/*.txt` | AI prompt templates |
| `static/*.html` | Dashboard pages |
| `static/styles.css` | Shared CSS (PicoCSS overrides) |
| `static/spec.json` | OpenAPI specification |
| `tools/` | Offline utilities (dataset export, fine-tuning) |
| `tests.py` | Test suite |
| `DATABASE.md` | DB schema reference |
| `API.md` | REST API reference |
