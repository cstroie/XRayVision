# XRayVision Database Schema

This document describes the database schema used by the XRayVision application.

## Overview

The XRayVision application uses a SQLite database to store patient information, exam metadata, AI-generated reports, and radiologist reports. The schema is designed to be normalized to reduce data redundancy and improve query performance.

## Tables

### patients

Stores patient demographic information.

| Column | Type | Description |
|--------|------|-------------|
| cnp | TEXT (PRIMARY KEY) | Romanian personal identification number |
| id | TEXT | Patient ID from hospital system |
| name | TEXT | Patient full name |
| age | INTEGER | Patient age in years |
| sex | TEXT | Patient sex ('M', 'F', or 'O') |

### exams

Stores exam metadata and processing status.

| Column | Type | Description |
|--------|------|-------------|
| uid | TEXT (PRIMARY KEY) | Unique exam identifier (SOP Instance UID) |
| cnp | TEXT (FOREIGN KEY) | References patients.cnp |
| created | TIMESTAMP | Exam timestamp from DICOM |
| protocol | TEXT | Imaging protocol name from DICOM |
| region | TEXT | Anatomic region identified from protocol |
| type | TEXT | Exam type/modality |
| status | TEXT | Processing status ('none', 'queued', 'processing', 'done', 'error', 'ignore') |
| study | TEXT | Study Instance UID |
| series | TEXT | Series Instance UID |

### ai_reports

Stores AI-generated reports and analysis results.

| Column | Type | Description |
|--------|------|-------------|
| uid | TEXT (PRIMARY KEY, FOREIGN KEY) | References exams.uid |
| created | TIMESTAMP | Report creation timestamp (default: CURRENT_TIMESTAMP) |
| updated | TIMESTAMP | Report last update timestamp (default: CURRENT_TIMESTAMP) |
| text | TEXT | AI-generated report content |
| positive | INTEGER | Binary indicator (-1=not assessed, 0=no findings, 1=findings) |
| confidence | INTEGER | AI self-confidence score (0-100, -1 if not assessed) |
| is_correct | INTEGER | Validation status (-1=not assessed, 0=incorrect, 1=correct) |
| model | TEXT | Name of the model used to analyze the image |
| latency | INTEGER | Time in seconds needed to analyze the image by the AI (-1 if not assessed) |

### rad_reports

Stores radiologist reports and clinical information.

| Column | Type | Description |
|--------|------|-------------|
| uid | TEXT (PRIMARY KEY, FOREIGN KEY) | References exams.uid |
| id | TEXT | HIS report ID |
| created | TIMESTAMP | Report creation timestamp (default: CURRENT_TIMESTAMP) |
| updated | TIMESTAMP | Report last update timestamp (default: CURRENT_TIMESTAMP) |
| text | TEXT | Radiologist report content |
| positive | INTEGER | Binary indicator (-1=not assessed, 0=no findings, 1=findings) |
| severity | INTEGER | Severity score (0-10, -1 if not assessed) |
| summary | TEXT | Brief summary of findings |
| type | TEXT | Exam type |
| radiologist | TEXT | Identifier for the radiologist |
| justification | TEXT | Clinical diagnostic text |
| model | TEXT | Name of the model used to summarize the radiologist report |
| latency | INTEGER | Time in seconds needed by the radiologist to fill in the report (-1 if not assessed) |

## Indexes

To optimize query performance, the following indexes are created:

- `idx_exams_status`: Fast filtering by exam status
- `idx_exams_region`: Quick regional analysis
- `idx_exams_cnp`: Efficient patient lookup
- `idx_exams_created`: Fast sorting by exam creation time
- `idx_exams_study`: Efficient study-based queries
- `idx_ai_reports_created`: Fast sorting by AI report creation time
- `idx_rad_reports_created`: Fast sorting by radiologist report creation time
- `idx_patients_name`: Fast patient name searches

## Relationships

- `exams.cnp` references `patients.cnp`
- `ai_reports.uid` references `exams.uid`
- `rad_reports.uid` references `exams.uid`

## Constraints

- Foreign key constraints are enabled to maintain referential integrity
- The `sex` column in the `patients` table is constrained to values 'M', 'F', or 'O'
- The `positive` column in `ai_reports` is constrained to values -1, 0, or 1
- The `confidence` column in `ai_reports` is constrained to values between -1 and 100
- The `is_correct` column in `ai_reports` is constrained to values -1, 0, or 1
- The `positive` column in `rad_reports` is constrained to values -1, 0, or 1
- The `severity` column in `rad_reports` is constrained to values between -1 and 10
