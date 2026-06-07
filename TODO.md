# XRayVision — Feature Planning

## In progress

- Expand medical abbreviations: integrate `tools/find_acronyms.py` output into
  `MEDICAL_ACRONYMS` dict; add auto-discovery pass for unknown abbreviations
  found in incoming Romanian reports.

## Planned

- Enhanced user management: role-based access control beyond admin/user;
  per-user activity log.
- Export functionality: CSV/PDF export of exam lists and AI report summaries.
- Integration with additional DICOM modalities beyond CR (e.g. DX, MR).
- Improved statistics: AI accuracy trends over time (longitudinal accuracy drift).
- Report templates: configurable finding templates per anatomic region.
- Multi-language support: extend translation beyond Romanian → English.
- Limit the number of database backups in the backup directory to a safe default.

## Ideas / low priority

- Dark/light theme toggle in dashboard (PicoCSS supports it).
- Keyboard shortcuts for radiologist review workflow.
- Mobile-friendly dashboard layout improvements.
