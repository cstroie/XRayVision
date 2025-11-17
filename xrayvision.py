#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# XRayVision - Async DICOM processor with OpenAI and WebSocket dashboard.
# Copyright (C) 2025 Costin Stroie <costinstroie@eridu.eu.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# Standard library imports
import argparse
import asyncio
import base64
import json
import logging
import math
import os
import re
import sqlite3
import random
from datetime import datetime, timedelta
from typing import Optional

# Third-party imports
import aiohttp
import cv2
import numpy as np
from aiohttp import web
from pydicom import dcmread
from pydicom.dataset import Dataset
from pynetdicom import AE, evt, QueryRetrievePresentationContexts, StoragePresentationContexts
from pynetdicom.sop_class import (
    ComputedRadiographyImageStorage,
    DigitalXRayImageStorageForPresentation,
    PatientRootQueryRetrieveInformationModelFind,
    PatientRootQueryRetrieveInformationModelMove
)

# Logger config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(message)s',
    handlers=[
        logging.FileHandler("xrayvision.log"),
        logging.StreamHandler()
    ]
)
# Filter out noisy module logs
logging.getLogger('aiohttp').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)
# DICOM network operations
logging.getLogger('pynetdicom').setLevel(logging.WARNING)
# DICOM file operations
logging.getLogger('pydicom').setLevel(logging.WARNING)

# Set default log level
logging.getLogger().setLevel(logging.INFO)

# Configuration
import configparser

# Default configuration values
DEFAULT_CONFIG = {
    'general': {
        'XRAYVISION_DB_PATH': 'xrayvision.db',
        'XRAYVISION_BACKUP_DIR': 'backup'
    },
    'dicom': {
        'AE_TITLE': 'XRAYVISION',
        'AE_PORT': '4010',
        'REMOTE_AE_TITLE': 'DICOM_SERVER',
        'REMOTE_AE_IP': '192.168.1.1',
        'REMOTE_AE_PORT': '104'
    },
    'openai': {
        'OPENAI_URL_PRIMARY': 'http://127.0.0.1:8080/v1/chat/completions',
        'OPENAI_URL_SECONDARY': 'http://127.0.0.1:11434/v1/chat/completions',
        'OPENAI_API_KEY': 'sk-your-api-key',
        'MODEL_NAME': 'medgemma-4b-it'
    },
    'dashboard': {
        'DASHBOARD_PORT': '8000'
    },
    'notifications': {
        'NTFY_URL': 'https://ntfy.sh/xrayvision-alerts'
    },
    'processing': {
        'PAGE_SIZE': '10',
        'KEEP_DICOM': 'False',
        'LOAD_DICOM': 'False',
        'NO_QUERY': 'False',
        'ENABLE_NTFY': 'False'
    }
}

# County names mapping for Romanian CNP validation
county_names = {
    1: "Alba", 2: "Arad", 3: "Argeș", 4: "Bacău", 5: "Bihor", 6: "Bistrița-Năsăud",
    7: "Botoșani", 8: "Brașov", 9: "Brăila", 10: "Buzău", 11: "Caraș-Severin",
    12: "Cluj", 13: "Constanța", 14: "Covasna", 15: "Dâmbovița", 16: "Dolj",
    17: "Galați", 18: "Gorj", 19: "Harghita", 20: "Hunedoara", 21: "Ialomița",
    22: "Iași", 23: "Ilfov", 24: "Maramureș", 25: "Mehedinți", 26: "Mureș",
    27: "Neamț", 28: "Olt", 29: "Prahova", 30: "Satu Mare", 31: "Sălaj",
    32: "Sibiu", 33: "Suceava", 34: "Teleorman", 35: "Timiș", 36: "Tulcea",
    37: "Vaslui", 38: "Vâlcea", 39: "Vrancea", 40: "București", 41: "București",
    42: "București", 43: "București", 44: "București", 45: "București", 46: "București",
    51: "Călărași", 52: "Giurgiu",
    70: "Diaspora", 71: "Diaspora", 72: "Diaspora", 73: "Diaspora", 74: "Diaspora",
    75: "Diaspora", 76: "Diaspora", 77: "Diaspora", 78: "Diaspora", 79: "Diaspora",
    90: "Special", 91: "Special", 92: "Special", 93: "Special", 94: "Special",
    95: "Special", 96: "Special", 97: "Special", 98: "Special", 99: "Special"
}

# Load configuration from file if it exists, otherwise use defaults
config = configparser.ConfigParser()
config.read_dict(DEFAULT_CONFIG)
try:
    config.read('xrayvision.cfg')
    logging.info("Configuration loaded from xrayvision.cfg")
    # Check for local configuration file to override settings
    local_config_files = config.read('local.cfg')
    if local_config_files:
        logging.info("Local configuration loaded from local.cfg")
except Exception as e:
    logging.info("Using default configuration values")

# User roles configuration
USERS = {}
if 'users' in config:
    for user in config['users']:
        password, role = config.get('users', user).split(',', 1)
        USERS[user.strip()] = {
            'password': password.strip(),
            'role': role.strip()
        }

# Extract configuration values
OPENAI_URL_PRIMARY = config.get('openai', 'OPENAI_URL_PRIMARY')
OPENAI_URL_SECONDARY = config.get('openai', 'OPENAI_URL_SECONDARY')
OPENAI_API_KEY = config.get('openai', 'OPENAI_API_KEY')
NTFY_URL = config.get('notifications', 'NTFY_URL')
DASHBOARD_PORT = config.getint('dashboard', 'DASHBOARD_PORT')
AE_TITLE = config.get('dicom', 'AE_TITLE')
AE_PORT = config.getint('dicom', 'AE_PORT')
REMOTE_AE_TITLE = config.get('dicom', 'REMOTE_AE_TITLE')
REMOTE_AE_IP = config.get('dicom', 'REMOTE_AE_IP')
REMOTE_AE_PORT = config.getint('dicom', 'REMOTE_AE_PORT')
FHIR_URL = config.get('fhir', 'FHIR_URL')
FHIR_USERNAME = config.get('fhir', 'FHIR_USERNAME')
FHIR_PASSWORD = config.get('fhir', 'FHIR_PASSWORD')
IMAGES_DIR = 'images'
STATIC_DIR = 'static'
DB_FILE = config.get('general', 'XRAYVISION_DB_PATH')
BACKUP_DIR = config.get('general', 'XRAYVISION_BACKUP_DIR')
MODEL_NAME = config.get('openai', 'MODEL_NAME')

SYS_PROMPT = ("""
You are an experienced emergency radiologist analyzing imaging studies.

OUTPUT FORMAT:
You must respond with ONLY valid JSON in this exact format:
{
  "short": "yes" or "no",
  "report": "detailed findings as a string",
  "confidence": integer from 0 to 100
}

CRITICAL RULES:
- Output ONLY the JSON object - no additional text, explanations, or apologies before or after
- The "short" field must be exactly "yes" or "no" (lowercase, in quotes)
- "yes" means pathological findings are present
- "no" means no significant findings detected
- The "confidence" field must be a number between 0-100 (no quotes)
- Use double quotes for all keys and string values
- Properly escape special characters in the report string

EXAMPLES:

Example 1 - Chest X-ray with pneumonia:
Input: Chest X-ray, patient with cough and fever
Output: {"short": "yes", "report": "Consolidation in the right lower lobe consistent with pneumonia. No pleural effusion or pneumothorax. Heart size normal.", "confidence": 92}

Example 2 - Normal chest X-ray:
Input: Chest X-ray, routine screening
Output: {"short": "no", "report": "Clear lung fields bilaterally. No consolidation, pleural effusion, or pneumothorax. Cardiac silhouette within normal limits. No acute bony abnormalities.", "confidence": 95}

Example 3 - Abdominal X-ray with uncertain findings:
Input: Abdominal X-ray, abdominal pain
Output: {"short": "yes", "report": "Dilated small bowel loops measuring up to 3.5 cm with air-fluid levels, concerning for possible small bowel obstruction. No free air under the diaphragm. Limited assessment of solid organs on plain film.", "confidence": 78}

ANALYSIS APPROACH:
- Systematically examine the entire image for all abnormalities
- Report all identified lesions and pathological findings
- Be factual - if uncertain, describe what you observe without assuming
- Use professional radiological terminology
- Review the image multiple times if findings are ambiguous

REPORT CONTENT:
The "report" field should contain a complete radiological description including:
- Primary findings related to the clinical question
- Additional incidental findings or lesions
- Relevant negative findings if clinically important

CONFIDENCE SCORING:
- 90-100: High confidence in findings
- 70-89: Moderate confidence, some uncertainty
- 50-69: Low confidence, significant uncertainty
- 0-49: Very low confidence, speculative findings

Remember: Output ONLY the JSON object with no other text.
""")
USR_PROMPT = ("""
{question} in this {anatomy} X-ray of a {subject}?
""")
REV_PROMPT = ("""
Your previous report was incorrect. Carefully re-examine the image.

Review checklist:
- Verify each finding you previously reported
- Look for any missed abnormalities
- Reassess systematically from top to bottom

Output ONLY the JSON format. No apologies or explanations.
""")

CHK_PROMPT = ("""
You are a medical assistant analyzing radiology reports.

TASK: Read the report and extract the main pathological information in JSON format.

OUTPUT FORMAT (JSON):
{
  "pathologic": "yes/no",
  "severity": 1-10,
  "summary": "1-5 words"
}

RULES:
- "pathologic": "yes" if any anomaly exists, otherwise "no"
- "severity": 1=minimal, 5=moderate, 10=critical/urgent
- "summary": diagnosis in maximum 5 words (e.g., "fracture", "pneumonia", "lung nodule")
- If everything is normal: {"pathologic": "no", "severity": 0, "summary": "normal"}
- Ignore spelling errors
- Respond ONLY with the JSON, without additional text

EXAMPLES:

Report: "Hazy opacity in the left mid lung field, possibly representing consolidation or infiltrate."
Response: {"pathologic": "yes", "severity": 6, "summary": "pulmonary consolidation"}

Report: "No pathological changes. Heart of normal size."
Response: {"pathologic": "no", "severity": 0, "summary": "normal"}
""")

# Images directory
os.makedirs(IMAGES_DIR, exist_ok=True)
# Static directory
os.makedirs(STATIC_DIR, exist_ok=True)

# Global variables
MAIN_LOOP = None  # Main asyncio event loop reference
websocket_clients = set()  # Set of connected WebSocket clients for dashboard updates
QUEUE_EVENT = asyncio.Event()  # Event to signal when items are added to the processing queue
next_query = None  # Timestamp for the next scheduled DICOM query operation

# Global variables to store the servers
dicom_server = None  # DICOM server instance for receiving studies
web_server = None  # Web server instance for dashboard and API

# OpenAI health
active_openai_url = None  # Currently active OpenAI API endpoint
health_status = {
    OPENAI_URL_PRIMARY: False,  # Health status of primary OpenAI endpoint
    OPENAI_URL_SECONDARY: False,  # Health status of secondary OpenAI endpoint
    FHIR_URL: False  # Health status of FHIR endpoint
}
# OpenAI timings
timings = {
    'total': 0,       # Total processing time (milliseconds)
    'average': 0      # Average processing time (milliseconds)
}

# Global parameters
PAGE_SIZE = config.getint('processing', 'PAGE_SIZE')      # Number of exams to display per page in the dashboard
KEEP_DICOM = config.getboolean('processing', 'KEEP_DICOM')  # Whether to keep DICOM files after processing
LOAD_DICOM = config.getboolean('processing', 'LOAD_DICOM')  # Whether to load existing DICOM files at startup
NO_QUERY = config.getboolean('processing', 'NO_QUERY')    # Whether to disable automatic DICOM query/retrieve
ENABLE_NTFY = config.getboolean('processing', 'ENABLE_NTFY') # Whether to enable ntfy.sh notifications for positive findings

# Load region identification rules from config
REGION_RULES = {}
region_config = config['regions']
for key in region_config:
    REGION_RULES[key] = [word.strip() for word in region_config[key].split(',')]

# Load region-specific questions from config
REGION_QUESTIONS = {}
question_config = config['questions']
for key in question_config:
    REGION_QUESTIONS[key] = question_config[key]

# Load supported regions from config
REGIONS = []
region_config = config['supported_regions']
for key in region_config:
    if config.getboolean('supported_regions', key):
        REGIONS.append(key)

# Dashboard state
dashboard = {
    'queue_size': 0,        # Number of exams waiting in the processing queue
    'processing': None,     # Currently processing exam patient name
    'success_count': 0,     # Number of successfully processed exams in the last week
    'error_count': 0,       # Number of exams that failed processing
    'ignore_count': 0       # Number of exams that were ignored (wrong region)
}



# Database operations
def db_init():
    """
    Initialize the SQLite database with normalized tables and indexes.

    Creates tables for patients, exams, AI reports, and radiologist reports.
    Also creates indexes for efficient query operations.

    Database Schema:
    
    patients:
        - cnp (TEXT, PRIMARY KEY): Romanian personal identification number
        - id (TEXT): Patient ID from hospital system
        - name (TEXT): Patient full name
        - age (INTEGER): Patient age in years
        - sex (TEXT): Patient sex ('M', 'F', or 'O')
    
    exams:
        - uid (TEXT, PRIMARY KEY): Unique exam identifier (SOP Instance UID)
        - cnp (TEXT, FOREIGN KEY): References patients.cnp
        - id (TEXT): Imaging study ID from HIS
        - created (TIMESTAMP): Exam timestamp from DICOM
        - protocol (TEXT): Imaging protocol name from DICOM
        - region (TEXT): Anatomic region identified from protocol
        - type (TEXT): Exam type/modality
        - status (TEXT): Processing status ('none', 'queued', 'processing', 'done', 'error', 'ignore')
        - study (TEXT): Study Instance UID
        - series (TEXT): Series Instance UID
    
    ai_reports:
        - uid (TEXT, PRIMARY KEY, FOREIGN KEY): References exams.uid
        - created (TIMESTAMP): Report creation timestamp (default: CURRENT_TIMESTAMP)
        - updated (TIMESTAMP): Report last update timestamp (default: CURRENT_TIMESTAMP)
        - text (TEXT): AI-generated report content
        - positive (INTEGER): Binary indicator (-1=not assessed, 0=no findings, 1=findings)
        - confidence (INTEGER): AI self-confidence score (0-100, -1 if not assessed)
        - model (TEXT): Name of the model used to analyze the image
        - latency (INTEGER): Time in seconds needed to analyze the image by the AI (-1 if not assessed)
    
    rad_reports:
        - uid (TEXT, PRIMARY KEY, FOREIGN KEY): References exams.uid
        - id (TEXT): Diagnostic report ID from HIS
        - created (TIMESTAMP): Report creation timestamp (default: CURRENT_TIMESTAMP)
        - updated (TIMESTAMP): Report last update timestamp (default: CURRENT_TIMESTAMP)
        - text (TEXT): Radiologist report content
        - positive (INTEGER): Binary indicator (-1=not assessed, 0=no findings, 1=findings)
        - severity (INTEGER): Severity score (0-10, -1 if not assessed)
        - summary (TEXT): Brief summary of findings
        - type (TEXT): Exam type
        - radiologist (TEXT): Identifier for the radiologist
        - justification (TEXT): Clinical diagnostic text
        - model (TEXT): Name of the model used to summarize the radiologist report
        - latency (INTEGER): Time in seconds needed by the radiologist to fill in the report (-1 if not assessed)
    
    Indexes:
        - idx_exams_status: Fast filtering by exam status
        - idx_exams_region: Quick regional analysis
        - idx_exams_cnp: Efficient patient lookup
        - idx_patients_name: Fast patient name searches
    """
    with sqlite3.connect(DB_FILE) as conn:
        # Enable foreign key constraints
        conn.execute('PRAGMA foreign_keys = ON')
        
        # Patients table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                cnp TEXT PRIMARY KEY,
                id TEXT,
                name TEXT,
                age INTEGER,
                sex TEXT CHECK(sex IN ('M', 'F', 'O'))
            )
        ''')
        
        # Exams table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS exams (
                uid TEXT PRIMARY KEY,
                cnp TEXT,
                id TEXT,
                created TIMESTAMP,
                protocol TEXT,
                region TEXT,
                type TEXT,
                status TEXT DEFAULT 'none',
                study TEXT,
                series TEXT,
                FOREIGN KEY (cnp) REFERENCES patients(cnp)
            )
        ''')
        
        # AI reports table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS ai_reports (
                uid TEXT PRIMARY KEY,
                created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                text TEXT,
                positive INTEGER DEFAULT -1 CHECK(positive IN (-1, 0, 1)),
                confidence INTEGER DEFAULT -1 CHECK(confidence BETWEEN -1 AND 100),
                model TEXT,
                latency INTEGER DEFAULT -1,
                FOREIGN KEY (uid) REFERENCES exams(uid)
            )
        ''')
        
        # Radiologist reports table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS rad_reports (
                uid TEXT PRIMARY KEY,
                id TEXT,
                created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                text TEXT,
                positive INTEGER DEFAULT -1 CHECK(positive IN (-1, 0, 1)),
                severity INTEGER DEFAULT -1 CHECK(severity BETWEEN -1 AND 10),
                summary TEXT,
                type TEXT,
                radiologist TEXT,
                justification TEXT,
                model TEXT,
                latency INTEGER DEFAULT -1,
                FOREIGN KEY (uid) REFERENCES exams(uid)
            )
        ''')
        
        # Indexes for common query filters
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_exams_status
            ON exams(status)
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_exams_region
            ON exams(region)
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_exams_cnp
            ON exams(cnp)
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_exams_created
            ON exams(created)
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_exams_study
            ON exams(study)
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_ai_reports_created
            ON ai_reports(created)
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_rad_reports_created
            ON rad_reports(created)
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_patients_name
            ON patients(name)
        ''')
        logging.info("Initialized SQLite database with normalized schema.")


def db_execute_query(query: str, params: tuple = (), fetch_mode: str = 'all') -> Optional[list]:
    """Execute a database query and return results.

    Executes a parameterized SQL query and returns results based on the
    specified fetch mode.

    Args:
        query (str): SQL query to execute
        params (tuple): Query parameters
        fetch_mode (str): 'all', 'one', or 'none' for fetchall(), fetchone(), or no fetch

    Returns:
        Query results based on fetch_mode, or None on error
    """
    with sqlite3.connect(DB_FILE) as conn:
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)

            if fetch_mode == 'all':
                return cursor.fetchall()
            elif fetch_mode == 'one':
                return cursor.fetchone()
            elif fetch_mode == 'none':
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            return handle_error(e, "database query execution", None, raise_on_error=True)


def db_execute_query_retry(query: str, params: tuple = (), max_retries: int = 3) -> Optional[int]:
    """Execute a database query with retry logic.

    Executes a database query with exponential backoff retry logic in case
    of failures.

    Args:
        query (str): SQL query to execute
        params (tuple): Query parameters
        max_retries (int): Maximum number of retry attempts

    Returns:
        Number of affected rows or None on error
    """
    with sqlite3.connect(DB_FILE) as conn:
        for attempt in range(max_retries):
            try:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                return cursor.rowcount
            except Exception as e:
                if attempt < max_retries - 1:
                    # Use sync sleep for synchronous function
                    import time
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                    continue
                return handle_error(e, "database query with retry", None, raise_on_error=True)
    return None


def db_create_insert_query(table_name, *columns):
    """
    Convenience function to build INSERT OR REPLACE query strings.

    Args:
        table_name: Name of the table to insert into
        *columns: Variable number of column names

    Returns:
        str: Formatted SQL query string with placeholders
    """
    placeholders = ', '.join(['?'] * len(columns))
    columns_str = ', '.join(columns)
    return f'INSERT OR REPLACE INTO {table_name} ({columns_str}) VALUES ({placeholders})'


def db_add_patient(cnp, id, name, age, sex):
    """
    Add a new patient to the database or update existing patient information.

    Args:
        cnp: Romanian personal identification number (primary key)
        id: Patient ID from hospital system
        name: Patient full name
        age: Patient age in years
        sex: Patient sex ('M', 'F', or 'O')
    """
    query = db_create_insert_query('patients', 'cnp', 'id', 'name', 'age', 'sex')
    params = (cnp, id, name, age, sex)
    return db_execute_query_retry(query, params)


def db_add_ai_report(uid, report_text, positive, confidence, model, latency):
    """
    Add or update an AI report entry in the database.

    Args:
        uid: Exam unique identifier
        report_text: AI-generated report content
        positive: AI prediction result (True/False)
        confidence: AI confidence score (0-100)
        model: Name of the model used
        latency: Processing time in seconds
    """
    values = (
        uid,
        report_text,
        int(positive),
        confidence,
        model,
        latency
    )
    query = db_create_insert_query('ai_reports', 'uid', 'text', 'positive', 'confidence', 'model', 'latency')
    db_execute_query_retry(query, values)


def db_add_rad_report(uid, report_id, report_text, positive, severity, summary, report_type, radiologist, justification, model, latency):
    """
    Add or update a radiologist report entry in the database.

    Args:
        uid: Exam unique identifier
        report_id: HIS report ID
        report_text: Radiologist report content
        positive: Report positivity (-1=not assessed, 0=no findings, 1=findings)
        severity: Severity score (0-10, -1 if not assessed)
        summary: Brief summary of findings
        report_type: Exam type
        radiologist: Identifier for the radiologist
        justification: Clinical diagnostic text
        model: Name of the model used
        latency: Processing time in seconds
    """
    values = (
        uid,
        report_id,
        report_text,
        positive,
        severity,
        summary,
        report_type,
        radiologist,
        justification,
        model,
        latency
    )
    query = db_create_insert_query('rad_reports', 'uid', 'id', 'text', 'positive', 'severity', 'summary', 'type', 'radiologist', 'justification', 'model', 'latency')
    db_execute_query_retry(query, values)

def db_update_rad_report(uid, positive, severity, summary, model, latency):
    """
    Update a radiologist report with LLM analysis results.

    Args:
        uid: Exam unique identifier
        positive: Report positivity (-1=not assessed, 0=no findings, 1=findings)
        severity: Severity score (0-10, -1 if not assessed)
        summary: Brief summary of findings
        model: Name of the model used
        latency: Processing time in seconds
    """
    query = """
        UPDATE rad_reports 
        SET positive = ?, severity = ?, summary = ?, model = ?, latency = ?
        WHERE uid = ?
    """
    params = (positive, severity, summary, model, latency, uid)
    db_execute_query_retry(query, params)

def db_get_exam_without_rad_report():
    """
    Get a random exam that doesn't have a radiologist report yet or has a report with null ID.

    Returns:
        dict: Exam data or None if not found
    """
    query = """
        SELECT 
            e.uid, e.created, e.protocol, e.region, e.status, e.type, e.study, e.series, e.id,
            p.name, p.cnp, p.id, p.age, p.sex
        FROM exams e
        INNER JOIN patients p ON e.cnp = p.cnp
        LEFT JOIN rad_reports rr ON e.uid = rr.uid
        WHERE (rr.id IS NULL OR rr.id = '')
        AND e.status = 'done'
        ORDER BY RANDOM()
        LIMIT 1
    """
    row = db_execute_query(query, fetch_mode='one')
    if row:
        (uid, exam_created, exam_protocol, exam_region, exam_status, exam_type, exam_study, exam_series, exam_id,
         patient_name, patient_cnp, patient_id, patient_age, patient_sex) = row
        
        return {
            'uid': uid,
            'exam': {
                'created': exam_created,
                'protocol': exam_protocol,
                'region': exam_region,
                'status': exam_status,
                'type': exam_type,
                'study': exam_study,
                'series': exam_series,
                'id': exam_id,
            },
            'patient': {
                'name': patient_name,
                'cnp': patient_cnp,
                'id': patient_id,
                'age': patient_age,
                'sex': patient_sex,
            }
        }
    return None

def db_update_patient_id(cnp, patient_id):
    """
    Update patient ID in the database.

    Args:
        cnp: Patient CNP
        patient_id: Patient ID from HIS
    """
    query = "UPDATE patients SET id = ? WHERE cnp = ?"
    params = (patient_id, cnp)
    db_execute_query_retry(query, params)


def db_add_exam(info, report=None, positive=None, confidence=None):
    """
    Add or update an exam entry in the database.

    This function handles queuing new exams for processing. It sets status to 'queued'
    and stores exam metadata. Patient information is stored in the patients table.
    If report data is provided, it also creates an entry in the ai_reports table.

    Args:
        info: Dictionary containing exam metadata (uid, patient info, exam details)
        report: Optional AI report text
        positive: Optional AI positive finding indicator (True/False)
        confidence: Optional AI confidence score (0-100)
    """
    # Add or update patient information
    patient = info["patient"]
    db_add_patient(
        patient["cnp"],
        patient.get("id",""),
        patient["name"],
        patient["age"],
        patient["sex"]
    )
    
    # Set status to queued for new exams
    status = 'queued'
    
    # Insert into database
    exam = info["exam"]
    params = (
        info['uid'],
        patient["cnp"],
        exam.get("id",""),
        exam['created'],
        exam["protocol"],
        exam['region'],
        exam.get("type", ""),
        status,
        exam.get("study"),
        exam.get("series")
    )
    query = db_create_insert_query('exams', 'uid', 'cnp', 'id', 'created', 'protocol', 'region', 'type', 'status', 'study', 'series')
    db_execute_query_retry(query, params)

    # If report is provided, add it to ai_reports table
    if report is not None and positive is not None:
        db_add_ai_report(
            info['uid'],
            report,
            positive,
            confidence if confidence is not None else -1,
            MODEL_NAME,
            -1
        )


def db_get_exams(limit = PAGE_SIZE, offset = 0, **filters):
    """
    Load exams from the database with optional filters and pagination.

    Retrieves exams with associated patient information, AI reports, and radiologist
    reports. Calculates correctness based on agreement between AI and radiologist
    predictions.

    Args:
        limit: Maximum number of exams to return (default: PAGE_SIZE)
        offset: Number of exams to skip for pagination (default: 0)
        **filters: Optional filters for querying exams:
            - reviewed: Filter by review status (0/1)
            - positive: Filter by AI prediction (0/1)
            - correct: Filter by correctness status (0/1)
            - region: Filter by anatomic region (case-insensitive partial
              match)
            - status: Filter by processing status (case-insensitive exact
              match or list of statuses)
            - search: Filter by patient name, CNP, or UID (case-insensitive
              partial match for name/CNP, exact for UID)

    Returns:
        tuple: (exams_list, total_count) where exams_list is a list of
               exam dictionaries containing patient, exam, and report data,
               and total_count is the total number of exams matching the filters
    """
    conditions = []
    params = []

    # Update the conditions with proper parameterization
    if 'reviewed' in filters:
        conditions.append("rr.positive > ?")
        params.append("-1")
    if 'positive' in filters:
        conditions.append("ar.positive = ?")
        params.append(filters['positive'])
    if 'correct' in filters:
        conditions.append("correct = ?")
        params.append(filters['correct'])
    if 'region' in filters:
        conditions.append("LOWER(e.region) LIKE ?")
        params.append(f"%{filters['region'].lower()}%")
    if 'status' in filters:
        status_value = filters['status']
        if isinstance(status_value, list):
            # Handle list of statuses
            placeholders = ','.join(['?'] * len(status_value))
            conditions.append(f"LOWER(e.status) IN ({placeholders})")
            params.extend([s.lower() for s in status_value])
        else:
            # Handle single status
            conditions.append("LOWER(e.status) = ?")
            params.append(status_value.lower())
    if 'search' in filters:
        conditions.append("(LOWER(p.name) LIKE ? OR LOWER(p.cnp) LIKE ? OR e.uid LIKE ?)")
        search_term = f"%{filters['search']}%"
        params.extend([search_term, search_term, filters['search']])

    # Build WHERE clause
    where = ""
    if conditions:
        where = "WHERE " + " AND ".join(conditions)

    # Apply the limits (pagination)
    query = f"""
        SELECT 
            e.uid, e.created, e.protocol, e.region, e.status, e.type, e.study, e.series, e.id,
            p.name, p.cnp, p.id, p.age, p.sex,
            ar.created, ar.text, ar.positive, ar.updated, ar.confidence, ar.model, ar.latency,
            rr.text, rr.positive, rr.severity, rr.summary, rr.created, rr.updated, rr.id, rr.type, rr.radiologist, rr.justification, rr.model, rr.latency,
            CASE 
                WHEN rr.positive = -1 THEN -1
                WHEN ar.positive = rr.positive THEN 1
                ELSE 0
            END AS correct,
            CASE
                WHEN rr.positive > -1 THEN 1
                ELSE 0
            END AS reviewed
        FROM exams e
        INNER JOIN patients p ON e.cnp = p.cnp
        LEFT JOIN ai_reports ar ON e.uid = ar.uid
        LEFT JOIN rad_reports rr ON e.uid = rr.uid
        {where}
        ORDER BY e.created DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])

    # Get the exams
    exams = []
    rows = db_execute_query(query, params, fetch_mode='all')
    if rows:
        for row in rows:
            # Unpack row into named variables for better readability
            (uid, exam_created, exam_protocol, exam_region, exam_status, exam_type, exam_study, exam_series, exam_id,
             patient_name, patient_cnp, patient_id, patient_age, patient_sex,
             ai_created, ai_text, ai_positive, ai_updated, ai_confidence, ai_model, ai_latency,
             rad_text, rad_positive, rad_severity, rad_summary, rad_created, rad_updated, rad_id, rad_type, rad_radiologist, rad_justification, rad_model, rad_latency,
             correct, reviewed) = row
                
            dt = datetime.strptime(exam_created, "%Y-%m-%d %H:%M:%S")
            exams.append({
                'uid': uid,
                'patient': {
                    'name': patient_name,
                    'cnp': patient_cnp,
                    'id': patient_id,
                    'age': patient_age,
                    'sex': patient_sex,
                },
                'exam': {
                    'created': exam_created,
                    'date': dt.strftime('%Y%m%d'),
                    'time': dt.strftime('%H%M%S'),
                    'protocol': exam_protocol,
                    'region': exam_region,
                    'status': exam_status,
                    'type': exam_type,
                    'study': exam_study,
                    'series': exam_series,
                    'id': exam_id,
                },
                'report': {
                    'ai': {
                        'text': ai_text,
                        'short': ai_positive and 'yes' or 'no' if ai_positive is not None and ai_positive > -1 else 'no',
                        'created': ai_created,
                        'updated': ai_updated,
                        'positive': bool(ai_positive) if ai_positive is not None and ai_positive > -1 else False,
                        'confidence': ai_confidence,
                        'model': ai_model,
                        'latency': ai_latency,
                    },
                    'rad': {
                        'text': rad_text,
                        'positive': bool(rad_positive) if (rad_positive is not None and rad_positive > -1) else False,
                        'severity': rad_severity,
                        'summary': rad_summary,
                        'created': rad_created,
                        'updated': rad_updated,
                        'id': rad_id,
                        'type': rad_type,
                        'radiologist': rad_radiologist,
                        'justification': rad_justification,
                        'model': rad_model,
                        'latency': rad_latency,
                    },
                    'correct': correct,
                    'reviewed': reviewed,
                },
            })
    # Get the total for pagination
    count_query = """
        SELECT COUNT(*) 
        FROM exams e
        INNER JOIN patients p ON e.cnp = p.cnp
        LEFT JOIN ai_reports ar ON e.uid = ar.uid
        LEFT JOIN rad_reports rr ON e.uid = rr.uid
    """
    count_params = []
    if conditions:
        count_query += ' WHERE ' + " AND ".join(conditions)
        count_params = params[:-2]  # Exclude limit and offset parameters
    total_row = db_execute_query(count_query, count_params, fetch_mode='one')
    total = total_row[0] if total_row else 0
    return exams, total


def db_get_previous_reports(patient_cnp, region, months=3):
    """
    Get previous reports for the same patient and region from the last few months.

    Args:
        patient_cnp: Patient identifier
        region: Anatomic region to match
        months: Number of months to look back (default: 3)

    Returns:
        list: List of tuples containing (report_text, updated_timestamp)
    """
    cutoff_date = datetime.now() - timedelta(days=months*30)
    cutoff_date_str = cutoff_date.strftime('%Y-%m-%d %H:%M:%S')

    query = """
        SELECT ar.text, ar.updated
        FROM exams e
        INNER JOIN ai_reports ar ON e.uid = ar.uid
        WHERE e.cnp = ?
        AND e.region = ?
        AND ar.updated >= ?
        AND ar.text IS NOT NULL
        AND ar.positive IS NOT NULL
        ORDER BY ar.updated DESC
    """
    params = (patient_cnp, region, cutoff_date_str)
    results = db_execute_query(query, params, fetch_mode='all')
    return results if results else []


def db_check_already_processed(uid):
    """
    Check if an exam has already been processed, is queued, or is being processed.

    Args:
        uid: Unique identifier of the exam (SOP Instance UID)

    Returns:
        bool: True if exam exists with status 'done', 'queued', or 'processing'
    """
    query = "SELECT status FROM exams WHERE uid = ? AND status IN ('done', 'queued', 'processing')"
    params = (uid,)
    result = db_execute_query(query, params, fetch_mode='one')
    return result is not None


async def db_get_stats():
    """
    Retrieve comprehensive statistics from the database for dashboard
    display.

    Calculates various metrics including total exams, reviewed counts,
    positive findings, wrong predictions, regional breakdowns, temporal
    trends, processing performance, and error statistics. Computes
    precision, negative predictive value, sensitivity, and specificity
    for each anatomic region.

    To reduce memory usage, temporal trends are limited to:
    - Daily trends for the last 30 days
    - Monthly trends for the last 12 months

    Returns:
        dict: Dictionary containing all statistical data organized by
              category
    """
    stats = {
        "total": 0,
        "reviewed": 0,
        "positive": 0,
        "correct": 0,
        "wrong": 0,
        "region": {},
        "trends": {},
        "monthly_trends": {},
        "avg_processing_time": 0,
        "throughput": 0,
        "error_stats": {}
    }
    
    # Get count total and reviewed statistics in a single query
    query = """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN rr.positive > -1 THEN 1 ELSE 0 END) AS reviewed
        FROM exams e
        LEFT JOIN rad_reports rr ON e.uid = rr.uid
        WHERE e.status LIKE 'done'
    """
    row = db_execute_query(query, fetch_mode='one')
    if row:
        (total, reviewed) = row
        stats["total"] = total
        stats["reviewed"] = reviewed or 0

    # Calculate correct (TP + TN) and wrong (FP + FN) predictions
    query = """
        SELECT
            SUM(CASE WHEN (ar.positive = 1 AND rr.positive = 1) THEN 1 ELSE 0 END) AS tpos,
            SUM(CASE WHEN (ar.positive = 0 AND rr.positive = 0) THEN 1 ELSE 0 END) AS tneg,
            SUM(CASE WHEN (ar.positive = 1 AND rr.positive = 0) THEN 1 ELSE 0 END) AS fpos,
            SUM(CASE WHEN (ar.positive = 0 AND rr.positive = 1) THEN 1 ELSE 0 END) AS fneg
        FROM exams e
        LEFT JOIN ai_reports ar ON e.uid = ar.uid
        LEFT JOIN rad_reports rr ON e.uid = rr.uid
        WHERE e.status LIKE 'done'
          AND ar.positive > -1
          AND rr.positive > -1;
    """
    metrics_row = db_execute_query(query, fetch_mode='one')
    if metrics_row:
        (tpos, tneg, fpos, fneg) = metrics_row
        tpos = tpos or 0
        tneg = tneg or 0
        fpos = fpos or 0
        fneg = fneg or 0
        stats["correct"] = tpos + tneg
        stats["wrong"] = fpos + fneg

        # Calculate Matthews Correlation Coefficient (MCC) for totals
        denominator = math.sqrt((tpos + fpos) * (tpos + fneg) * (tneg + fpos) * (tneg + fneg))
        if denominator == 0:
            stats["mcc"] = 0.0
        else:
            mcc = (tpos * tneg - fpos * fneg) / denominator
            stats["mcc"] = round(mcc, 2)

    # Get processing time statistics (last day only)
    query = """
        SELECT
            AVG(CAST(strftime('%s', ar.created) - strftime('%s', e.created) AS REAL)) AS avg_processing_time,
            COUNT(*) * 1.0 / (SUM(CAST(strftime('%s', ar.created) - strftime('%s', e.created) AS REAL)) + 1) AS throughput
        FROM exams e
        LEFT JOIN ai_reports ar ON e.uid = ar.uid
        WHERE e.status LIKE 'done'
          AND ar.created IS NOT NULL
          AND e.created IS NOT NULL
          AND e.created >= datetime('now', '-1 days')
    """
    timing_row = db_execute_query(query, fetch_mode='one')
    if timing_row and timing_row[0] is not None:
        (avg_processing_time, throughput) = timing_row
        stats["avg_processing_time"] = round(avg_processing_time, 2)
        stats["throughput"] = round(throughput * 3600, 2)  # exams per hour

    # Get error statistics
    query = """
        SELECT status, COUNT(*) as count
        FROM exams
        WHERE status IN ('error', 'ignore')
        GROUP BY status
    """
    error_data = db_execute_query(query, fetch_mode='all')
    if error_data:
        for row in error_data:
            (status, count) = row
            stats["error_stats"][status] = count

    # Totals per anatomic part
    query = """
        SELECT e.region,
                COUNT(*) AS total,
                SUM(CASE WHEN rr.positive > -1 THEN 1 ELSE 0 END) AS reviewed,
                SUM(CASE WHEN ar.positive = 1 THEN 1 ELSE 0 END) AS positive,
                SUM(CASE WHEN (ar.positive != rr.positive AND ar.positive > -1 AND rr.positive > -1) THEN 1 ELSE 0 END) AS wrong,
                SUM(CASE WHEN (ar.positive = 1 AND rr.positive = 1) THEN 1 ELSE 0 END) AS tpos,
                SUM(CASE WHEN (ar.positive = 0 AND rr.positive = 0) THEN 1 ELSE 0 END) AS tneg,
                SUM(CASE WHEN (ar.positive = 1 AND rr.positive = 0) THEN 1 ELSE 0 END) AS fpos,
                SUM(CASE WHEN (ar.positive = 0 AND rr.positive = 1) THEN 1 ELSE 0 END) AS fneg
        FROM exams e
        LEFT JOIN ai_reports ar ON e.uid = ar.uid
        LEFT JOIN rad_reports rr ON e.uid = rr.uid
        WHERE e.status LIKE 'done' AND ar.positive > -1
        GROUP BY e.region
    """
    region_data = db_execute_query(query, fetch_mode='all')
    if region_data:
        for row in region_data:
            (region, total, reviewed, positive, wrong, tpos, tneg, fpos, fneg) = row
            region = region or 'unknown'
            stats["region"][region] = {
                "total": total,
                "reviewed": reviewed,
                "positive": positive,
                "wrong": wrong,
                "tpos": tpos,
                "tneg": tneg,
                "fpos": fpos,
                "fneg": fneg,
                "ppv": '-',
                "pnv": '-',
                "snsi": '-',
                "spci": '-',
            }
            # Calculate metrics safely
            if (tpos + fpos) != 0:
                stats["region"][region]["ppv"] = int(100.0 * tpos / (tpos + fpos))
            if (tneg + fneg) != 0:
                stats["region"][region]["pnv"] = int(100.0 * tneg / (tneg + fneg))
            if (tpos + fneg) != 0:
                stats["region"][region]["snsi"] = int(100.0 * tpos / (tpos + fneg))
            if (tneg + fpos) != 0:
                stats["region"][region]["spci"] = int(100.0 * tneg / (tneg + fpos))

            # Calculate Matthews Correlation Coefficient (MCC)
            tp = tpos or 0
            tn = tneg or 0
            fp = fpos or 0
            fn = fneg or 0
            denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            if denominator == 0:
                stats["region"][region]["mcc"] = 0.0
            else:
                mcc = (tp * tn - fp * fn) / denominator
                # Round to 2 decimal places
                stats["region"][region]["mcc"] = round(mcc, 2)

    # Get temporal trends (last 30 days only to reduce memory usage)
    query = """
        SELECT DATE(e.created) as date,
               e.region,
               COUNT(*) as total,
               SUM(CASE WHEN ar.positive = 1 THEN 1 ELSE 0 END) as positive
        FROM exams e
        LEFT JOIN ai_reports ar ON e.uid = ar.uid
        WHERE e.status LIKE 'done'
          AND e.created >= date('now', '-30 days')
        GROUP BY DATE(e.created), e.region
        ORDER BY date
    """
    trends_data = db_execute_query(query, fetch_mode='all')
    if trends_data:
        # Process trends data into a structured format
        for row in trends_data:
            (date, region, total, positive) = row
            if region not in stats["trends"]:
                stats["trends"][region] = []
            stats["trends"][region].append({
                "date": date,
                "total": total,
                "positive": positive
            })

    # Get monthly trends (last 12 months only to reduce memory usage)
    query = """
        SELECT strftime('%Y-%m', e.created) as month,
               e.region,
               COUNT(*) as total,
               SUM(CASE WHEN ar.positive = 1 THEN 1 ELSE 0 END) as positive
        FROM exams e
        LEFT JOIN ai_reports ar ON e.uid = ar.uid
        WHERE e.status LIKE 'done'
          AND e.created >= date('now', '-12 months')
        GROUP BY strftime('%Y-%m', e.created), e.region
        ORDER BY month
    """
    monthly_trends_data = db_execute_query(query, fetch_mode='all')
    if monthly_trends_data:
        # Process monthly trends data into a structured format
        for row in monthly_trends_data:
            (month, region, total, positive) = row
            if region not in stats["monthly_trends"]:
                stats["monthly_trends"][region] = []
            stats["monthly_trends"][region].append({
                "month": month,
                "total": total,
                "positive": positive
            })

    # Return stats
    return stats


def db_get_queue_size():
    """
    Get the current number of exams waiting in the processing queue.

    Returns:
        int: Number of exams with status 'queued'
    """
    query = "SELECT COUNT(*) FROM exams WHERE status = 'queued'"
    result = db_execute_query(query, fetch_mode='one')
    return result[0] if result else 0


def db_get_error_stats():
    """
    Get statistics for exams that failed processing or were ignored.

    Returns:
        dict: Dictionary with 'error' and 'ignore' counts
    """
    stats = {'error': 0, 'ignore': 0}
    query = """
        SELECT status, COUNT(*) as count
        FROM exams
        WHERE status IN ('error', 'ignore')
        GROUP BY status
    """
    rows = db_execute_query(query, fetch_mode='all')
    if rows:
        for row in rows:
            (status, count) = row
            stats[status] = count
    return stats


def db_get_weekly_processed_count():
    """
    Get the count of successfully processed exams in the last 7 days.

    Returns:
        int: Number of exams with status 'done' reported in the last week
    """
    query = """
        SELECT COUNT(*)
        FROM exams e
        LEFT JOIN ai_reports ar ON e.uid = ar.uid
        WHERE e.status = 'done'
        AND ar.created >= datetime('now', '-7 days')
    """
    result = db_execute_query(query, fetch_mode='one')
    return result[0] if result else 0


# Convenience functions for database access
def db_get_exam_by_uid(uid):
    """
    Get a single exam by its UID.

    Args:
        uid: Unique identifier of the exam

    Returns:
        dict: Exam data or None if not found
    """
    exams, _ = db_get_exams(limit=1, search=uid)
    return exams[0] if exams else None


def db_get_recent_exams(limit=10):
    """
    Get the most recent exams.

    Args:
        limit: Maximum number of exams to return

    Returns:
        list: List of recent exams
    """
    exams, _ = db_get_exams(limit=limit)
    return exams


def db_get_exams_by_region(region, limit=PAGE_SIZE, offset=0):
    """
    Get exams filtered by region.

    Args:
        region: Anatomic region to filter by
        limit: Maximum number of exams to return
        offset: Number of exams to skip

    Returns:
        tuple: (exams_list, total_count)
    """
    return db_get_exams(limit=limit, offset=offset, region=region)


def db_get_positive_exams(limit=PAGE_SIZE, offset=0):
    """
    Get exams with positive AI findings.

    Args:
        limit: Maximum number of exams to return
        offset: Number of exams to skip

    Returns:
        tuple: (exams_list, total_count)
    """
    return db_get_exams(limit=limit, offset=offset, positive=1)


def db_get_unreviewed_exams(limit=PAGE_SIZE, offset=0):
    """
    Get exams that haven't been reviewed yet.

    Args:
        limit: Maximum number of exams to return
        offset: Number of exams to skip

    Returns:
        tuple: (exams_list, total_count)
    """
    return db_get_exams(limit=limit, offset=offset, reviewed=0)


def db_get_patient_exams(patient_cnp, limit=PAGE_SIZE, offset=0):
    """
    Get all exams for a specific patient.

    Args:
        patient_cnp: Patient identifier
        limit: Maximum number of exams to return
        offset: Number of exams to skip

    Returns:
        tuple: (exams_list, total_count)
    """
    exams, total = db_get_exams(limit=limit, offset=offset, search=patient_cnp)
    # Filter to ensure we only get exams for this specific patient
    patient_exams = [exam for exam in exams if exam['patient']['cnp'] == patient_cnp]
    return patient_exams, total


def db_count_exams_by_status(status):
    """
    Count exams by status.

    Args:
        status: Status to count ('queued', 'processing', 'done', 'error', 'ignore')

    Returns:
        int: Count of exams with the specified status
    """
    query = "SELECT COUNT(*) FROM exams WHERE status = ?"
    params = (status,)
    result = db_execute_query(query, params, fetch_mode='one')
    return result[0] if result else 0


def db_get_exam_ai_report(uid):
    """
    Get AI report for a specific exam.

    Args:
        uid: Unique identifier of the exam

    Returns:
        dict: Report data or None if not found
    """
    query = """
        SELECT text, positive, confidence, model, latency, created, updated
        FROM ai_reports WHERE uid = ?
    """
    params = (uid,)
    result = db_execute_query(query, params, fetch_mode='one')
    if result:
        (text, positive, confidence, model, latency, created, updated) = result
        return {
            'text': text,
            'positive': positive,
            'confidence': confidence,
            'model': model,
            'latency': latency,
            'created': created,
            'updated': updated
        }
    return None


def db_get_exam_rad_report(uid):
    """
    Get radiologist report for a specific exam.

    Args:
        uid: Unique identifier of the exam

    Returns:
        dict: Report data or None if not found
    """
    query = """
        SELECT id, text, positive, severity, summary, type, radiologist, justification, model, latency, created, updated
        FROM rad_reports WHERE uid = ?
    """
    params = (uid,)
    result = db_execute_query(query, params, fetch_mode='one')
    if result:
        (id, text, positive, severity, summary, type, radiologist, justification, model, latency, created, updated) = result
        return {
            'id': id,
            'text': text,
            'positive': positive,
            'severity': severity,
            'summary': summary,
            'type': type,
            'radiologist': radiologist,
            'justification': justification,
            'model': model,
            'latency': latency,
            'created': created,
            'updated': updated
        }
    return None


def db_get_patient_by_cnp(cnp):
    """
    Get patient information by CNP.

    Args:
        cnp: Romanian personal identification number

    Returns:
        dict: Patient data or None if not found
    """
    query = """
        SELECT cnp, id, name, age, sex FROM patients WHERE cnp = ?
    """
    params = (cnp,)
    result = db_execute_query(query, params, fetch_mode='one')
    if result:
        (cnp, id, name, age, sex) = result
        return {
            'cnp': cnp,
            'id': id,
            'name': name,
            'age': age,
            'sex': sex
        }
    return None


def db_get_patient_exam_uids(cnp):
    """
    Get all exam UIDs for a specific patient.

    Args:
        cnp: Romanian personal identification number

    Returns:
        list: List of exam UIDs for this patient
    """
    query = "SELECT uid FROM exams WHERE cnp = ? ORDER BY created DESC"
    params = (cnp,)
    rows = db_execute_query(query, params, fetch_mode='all')
    return [uid for (uid,) in rows] if rows else []


def db_get_regions():
    """
    Get distinct regions from the database for exams with status 'done'.

    Returns:
        list: List of distinct regions that have been processed with status 'done'
    """
    query = "SELECT DISTINCT region FROM exams WHERE region IS NOT NULL AND region != '' AND status = 'done' ORDER BY region"
    rows = db_execute_query(query, fetch_mode='all')
    return [region for (region,) in rows] if rows else []


def db_get_patients(limit=PAGE_SIZE, offset=0, **filters):
    """
    Load patients from the database with optional filters and pagination.

    Args:
        limit: Maximum number of patients to return (default: PAGE_SIZE)
        offset: Number of patients to skip for pagination (default: 0)
        **filters: Optional filters for querying patients:
            - search: Filter by patient name or CNP (case-insensitive partial match)

    Returns:
        tuple: (patients_list, total_count) where patients_list is a list of
               patient dictionaries and total_count is the total number of
               patients matching the filters
    """
    conditions = []
    params = []

    # Update the conditions with proper parameterization
    if 'search' in filters:
        conditions.append("(LOWER(name) LIKE ? OR LOWER(cnp) LIKE ?)")
        search_term = f"%{filters['search']}%"
        params.extend([search_term, search_term])

    # Build WHERE clause
    where = ""
    if conditions:
        where = "WHERE " + " AND ".join(conditions)

    # Apply the limits (pagination)
    query = f"""
        SELECT cnp, id, name, age, sex
        FROM patients
        {where}
        ORDER BY name
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])

    # Get the patients
    patients = []
    rows = db_execute_query(query, params, fetch_mode='all')
    if rows:
        for row in rows:
            (cnp, id, name, age, sex) = row
            patients.append({
                'cnp': cnp,
                'id': id,
                'name': name,
                'age': age,
                'sex': sex,
            })
    # Get the total for pagination
    count_query = "SELECT COUNT(*) FROM patients"
    count_params = []
    if conditions:
        count_query += ' WHERE ' + " AND ".join(conditions)
        count_params = params[:-2]  # Exclude limit and offset parameters
    total_row = db_execute_query(count_query, count_params, fetch_mode='one')
    total = total_row[0] if total_row else 0
    return patients, total


def db_purge_ignored_errors():
    """
    Delete ignored and erroneous records older than 1 week and their associated files.

    Removes database entries with status 'ignore' or 'error' that are older than
    1 week, along with their corresponding DICOM and PNG files from the filesystem
    and associated AI and radiologist reports.

    Returns:
        int: Number of records deleted
    """
    # First delete associated AI reports
    ai_query = '''
        DELETE FROM ai_reports
        WHERE uid IN (
            SELECT uid FROM exams
            WHERE status IN ('ignore', 'error')
            AND created < datetime('now', '-7 days')
        )
    '''
    db_execute_query_retry(ai_query)
    
    # Then delete associated radiologist reports
    rad_query = '''
        DELETE FROM rad_reports
        WHERE uid IN (
            SELECT uid FROM exams
            WHERE status IN ('ignore', 'error')
            AND created < datetime('now', '-7 days')
        )
    '''
    db_execute_query_retry(rad_query)
    
    # Get UIDs for file cleanup before deleting exams
    uid_query = '''
        SELECT uid FROM exams
        WHERE status IN ('ignore', 'error')
        AND created < datetime('now', '-7 days')
    '''
    uid_rows = db_execute_query(uid_query, fetch_mode='all')
    deleted_uids = [row[0] for row in uid_rows] if uid_rows else []
    
    # Finally delete the exams
    exam_query = '''
        DELETE FROM exams
        WHERE status IN ('ignore', 'error')
        AND created < datetime('now', '-7 days')
    '''
    deleted_count = db_execute_query_retry(exam_query)
    
    # Delete associated files
    for uid in deleted_uids:
        for ext in ('dcm', 'png'):
            file_path = os.path.join(IMAGES_DIR, f"{uid}.{ext}")
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass
    logging.info(f"Purged {deleted_count} old records from database and their files.")
    return deleted_count


def db_backup():
    """
    Create a timestamped backup of the database.

    Creates a backup copy of the SQLite database file with a timestamp in the filename
    and stores it in the backup directory. Uses SQLite's built-in backup API for
    consistent backups.

    Returns:
        str: Path to the created backup file
    """
    try:
        # Create backup directory if it doesn't exist
        os.makedirs(BACKUP_DIR, exist_ok=True)
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"xrayvision_{timestamp}.db"
        backup_path = os.path.join(BACKUP_DIR, backup_filename)
        # Create backup using SQLite backup API
        with sqlite3.connect(DB_FILE) as conn:
            backup_conn = sqlite3.connect(backup_path)
            conn.backup(backup_conn)
            backup_conn.close()
        logging.info(f"Database backed up to {backup_path}")
        return backup_path
    except Exception as e:
        logging.error(f"Failed to create database backup: {e}")
        return None


def db_rad_review(uid, normal, radiologist='rad'):
    """
    Update radiologist report with normal/abnormal status.

    When a radiologist reviews a case, they indicate if the finding is normal (negative)
    or abnormal (positive). This function updates the radiologist report with that information.
    If no report entry exists for this UID, a new one is created with default values.

    Args:
        uid: The unique identifier of the exam
        normal: Whether the radiologist marked the case as normal (True) or abnormal (False)
        radiologist: Name/identifier of the radiologist (default: 'rad')

    Returns:
        None
    """
    positive = 0 if normal else 1
    
    # Check if a row already exists for this UID
    check_query = "SELECT 1 FROM rad_reports WHERE uid = ?"
    check_params = (uid,)
    result = db_execute_query(check_query, check_params, fetch_mode='one')
    
    if result:
        # Row exists, update it
        query = "UPDATE rad_reports SET positive = ?, updated = CURRENT_TIMESTAMP, radiologist = ? WHERE uid = ?"
        params = (positive, radiologist, uid)
    else:
        # Row doesn't exist, insert a new one
        query = """INSERT INTO rad_reports 
                  (uid, id, created, updated, text, positive, severity, summary, type, radiologist, justification, model, latency)
                  VALUES (?, '', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, '', ?, -1, '', '', ?, '', '', -1)"""
        params = (uid, positive, radiologist)
    
    db_execute_query_retry(query, params)


def db_set_status(uid, status):
    """
    Set the processing status for a specific exam.

    Args:
        uid: Unique identifier of the exam
        status: New status value (e.g., 'queued', 'processing', 'done', 'error', 'ignore')

    Returns:
        str: The status that was set
    """
    query = "UPDATE exams SET status = ? WHERE uid = ?"
    params = (status, uid)
    db_execute_query_retry(query, params)
    # Return the status
    return status


def db_requeue_exam(uid):
    """
    Re-queue an exam for processing.

    This function sets the exam status to 'queued' so it will be processed again.
    It clears most existing AI report data but preserves the text for reference.

    Args:
        uid: Unique identifier of the exam to re-queue

    Returns:
        bool: True if successfully re-queued, False otherwise
    """
    try:
        # Set the status to queued
        db_set_status(uid, 'queued')
        
        # Clear existing AI report data but preserve the text
        query = """
            UPDATE ai_reports 
            SET positive = -1, confidence = -1, model = NULL, latency = -1, updated = CURRENT_TIMESTAMP
            WHERE uid = ?
        """
        params = (uid,)
        db_execute_query_retry(query, params)
        
        return True
    except Exception as e:
        logging.error(f"Failed to re-queue exam {uid}: {e}")
        return False


# DICOM network operations
async def query_and_retrieve(minutes=15):
    """
    Query and Retrieve new studies from the remote DICOM server.

    This function queries the remote PACS for studies performed within the last 'minutes'
    and requests them to be sent to our DICOM server. It handles the complexity of
    time ranges that cross midnight by splitting into two separate queries.

    Args:
        minutes: Number of minutes to look back for new studies (default: 15)
    """
    ae = AE(ae_title=AE_TITLE)
    ae.requested_contexts = QueryRetrievePresentationContexts
    ae.connection_timeout = 30
    # Create the association
    assoc = ae.associate(REMOTE_AE_IP, REMOTE_AE_PORT, ae_title=REMOTE_AE_TITLE)
    if assoc.is_established:
        logging.info(
            f"QueryRetrieve association established. "
            f"Asking for studies in the last {minutes} minutes."
        )
        # Prepare the timespan
        current_time = datetime.now()
        past_time = current_time - timedelta(minutes=minutes)
        # Check if the time span crosses midnight, split into two queries
        # This is necessary because DICOM time ranges can't wrap around midnight
        if past_time.date() < current_time.date():
            date_yesterday = past_time.strftime('%Y%m%d')
            time_yesterday = f"{past_time.strftime('%H%M%S')}-235959"
            date_today = current_time.strftime('%Y%m%d')
            time_today = f"000000-{current_time.strftime('%H%M%S')}"
            queries = [(date_yesterday, time_yesterday), (date_today, time_today)]
        else:
            time_range = f"{past_time.strftime('%H%M%S')}-{current_time.strftime('%H%M%S')}"
            date_today = current_time.strftime('%Y%m%d')
            queries = [(date_today, time_range)]
        # Perform one or two queries, as needed
        for study_date, time_range in queries:
            # The query dataset
            ds = Dataset()
            ds.QueryRetrieveLevel = "STUDY"
            ds.StudyDate = study_date
            ds.StudyTime = time_range
            ds.Modality = "CR"
            # Get the responses list
            responses = assoc.send_c_find(
                ds,
                PatientRootQueryRetrieveInformationModelFind
            )
            # Ask for each one to be sent
            for (status, identifier) in responses:
                if status and status.Status in (0xFF00, 0xFF01):
                    study_instance_uid = identifier.StudyInstanceUID
                    logging.info(f"Found Study {study_instance_uid}")
                    await send_c_move(ae, study_instance_uid)
        # Release the association
        assoc.release()
    else:
        logging.error("Could not establish QueryRetrieve association.")

async def send_c_move(ae, study_instance_uid):
    """
    Request a study to be sent from the remote PACS to our DICOM server.

    Sends a C-MOVE request to the remote DICOM server to transfer a specific
    study (identified by Study Instance UID) to our AE.

    Args:
        ae: Application Entity instance
        study_instance_uid: Unique identifier of the study to retrieve
    """
    # Create the association
    assoc = ae.associate(REMOTE_AE_IP, REMOTE_AE_PORT, ae_title=REMOTE_AE_TITLE)
    if assoc.is_established:
        # The retrieval dataset
        ds = Dataset()
        ds.QueryRetrieveLevel = "STUDY"
        ds.StudyInstanceUID = study_instance_uid
        # Get the response
        responses = assoc.send_c_move(
            ds,
            AE_TITLE,
            PatientRootQueryRetrieveInformationModelMove
        )
        # Release the association
        assoc.release()
    else:
        logging.error("Could not establish C-MOVE association.")


def dicom_store(event):
    """
    Callback function for handling received DICOM C-STORE requests.

    This function is called whenever a DICOM file is sent to our DICOM server.
    It saves the DICOM file, extracts metadata, converts to PNG, and adds the
    exam to the processing queue.

    Args:
        event: pynetdicom event containing the DICOM dataset

    Returns:
        int: DICOM status code (0x0000 for success)
    """
    # Get the dataset
    ds = event.dataset
    ds.file_meta = event.file_meta
    uid = f"{ds.SOPInstanceUID}"
    # Check if already processed
    if db_check_already_processed(uid):
        logging.info(f"Skipping already processed image {uid}")
    elif ds.Modality == "CR":
        # Check the Modality
        dicom_file = os.path.join(IMAGES_DIR, f"{uid}.dcm")
        # Save the DICOM file
        ds.save_as(dicom_file, enforce_file_format = True)
        logging.info(f"DICOM file saved to {dicom_file}")
        # Process the DICOM file
        process_dicom_file(dicom_file, uid)
        # Notify the queue
        asyncio.run_coroutine_threadsafe(broadcast_dashboard_update(), MAIN_LOOP)
    # Return success
    return 0x0000


# DICOM files operations
async def load_existing_dicom_files():
    """
    Load existing DICOM files from the images directory into the processing queue.

    Scans the images directory for .dcm files that haven't been processed yet,
    converts them to PNG format, extracts metadata, and adds them to the queue
    for AI analysis. Updates the dashboard after processing.
    """
    for dicom_file in os.listdir(IMAGES_DIR):
        uid, ext = os.path.splitext(os.path.basename(dicom_file.lower()))
        if ext == '.dcm':
            if db_check_already_processed(uid):
                logging.info(f"Skipping already processed image {uid}")
            else:
                logging.info(f"Adding {uid} into processing queue...")
                full_path = os.path.join(IMAGES_DIR, dicom_file)
                # Process the DICOM file
                process_dicom_file(full_path, uid)
    # At the end, update the dashboard
    await broadcast_dashboard_update()


def extract_dicom_metadata(ds):
    """
    Extract relevant information from a DICOM dataset.

    Parses patient demographics, exam details, and timestamps from a DICOM dataset.
    Handles missing or malformed data gracefully with fallback values.

    Args:
        ds: pydicom Dataset object

    Returns:
        dict: Dictionary containing structured exam information
    """
    age = -1
    county = None
    if 'PatientAge' in ds:
        age = str(ds.PatientAge).lower().replace("y", "").strip()
        try:
            age = int(age)
        except Exception as e:
            logging.error(f"Cannot convert age to number: {e}")
            # Try to compute age from PatientID if available
            if 'PatientID' in ds:
                age = compute_age_from_cnp(ds.PatientID)
    else:
        # Try to compute age from PatientID if PatientAge is not available
        if 'PatientID' in ds:
            age = compute_age_from_cnp(ds.PatientID)
            # Also try to get county information
            cnp_result = validate_romanian_cnp(ds.PatientID)
            if cnp_result['valid']:
                county = cnp_result['county']
    # Get the reported timestamp (now)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Get the exam timestamp
    if str(ds.SeriesDate) and str(ds.SeriesTime) and \
        len(str(ds.SeriesDate)) == 8 and len(str(ds.SeriesTime)) >= 6:
        try:
            dt = datetime.strptime(f'{str(ds.SeriesDate)} {str(ds.SeriesTime)[:6]}', "%Y%m%d %H%M%S")
            created = dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            created = now
    else:
        created = now

    info = {
        'uid': str(ds.SOPInstanceUID),
        'patient': {
            'name':  str(ds.PatientName),
            'cnp':    str(ds.PatientID),
            'age':   age,
            'sex':   str(ds.PatientSex),
            'bdate': str(ds.PatientBirthDate),
        },
        'exam': {
            'protocol': str(ds.ProtocolName),
            'created': created,
            'region': str(ds.ProtocolName),
            'study': str(ds.StudyInstanceUID) if 'StudyInstanceUID' in ds else None,
            'series': str(ds.SeriesInstanceUID) if 'SeriesInstanceUID' in ds else None,
            'id': str(ds.StudyInstanceUID) if 'StudyInstanceUID' in ds else None,  # HIS study ID
        }
    }
    # Add county if available
    if county is not None:
        info['patient']['county'] = county
    # Check gender
    if not info['patient']['sex'] in ['M', 'F', 'O']:
        # Try to determine from ID only if it's a valid Romanian ID
        result = validate_romanian_cnp(info['patient']['cnp'])
        if result['valid']:
            info['patient']['sex'] = result['sex']
            # Also add county if not already added
            if 'county' not in info['patient'] and 'county' in result:
                info['patient']['county'] = result['county']
        else:
            info['patient']['sex'] = 'O'
    # Return the dicom info
    return info


def extract_patient_initials(name):
    """
    Extract initials from a patient name.

    Args:
        name: Patient name string

    Returns:
        str: Patient initials with dots
    """
    if not name or not isinstance(name, str):
        return "NoName"
    # Split by spaces, hyphens, and carets and take first letter of each part
    parts = re.split(r'[-^ ]', name)
    initials = ''.join([part[0] + '.' for part in parts if part])
    return initials.upper() if initials else "NoName"


# Image processing operations
def apply_gamma_correction(image, gamma = 1.2):
    """
    Apply gamma correction to an image to adjust brightness and contrast.

    Gamma correction is a nonlinear operation used to encode and decode luminance
    values in video or still image systems. It helps to optimize our images for
    better visualization by the AI model.

    When gamma is None, it's automatically calculated based on the image's median value.
    A gamma < 1 makes the image brighter, while gamma > 1 makes it darker.

    Args:
        image: Input image as numpy array
        gamma: Gamma value for correction. If None, will be automatically calculated

    Returns:
        numpy array: Gamma-corrected image
    """
    # If gamma is None, compute it based on image statistics
    if gamma is None:
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mid = 0.5
        mean = np.median(image)
        gamma = math.log(mid * 255) / math.log(mean)
        logging.debug(f"Calculated gamma is {gamma:.2f}")
    # Build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def convert_dicom_to_png(dicom_file, max_size = 800):
    """
    Convert DICOM to PNG with preprocessing for optimal AI analysis.

    This function performs several important preprocessing steps:
    1. Reads the DICOM pixel data
    2. Resizes the image while maintaining aspect ratio
    3. Applies percentile clipping to remove outliers
    4. Normalizes pixel values to 0-255 range
    5. Applies automatic gamma correction for better visualization
    6. Saves as PNG for efficient processing by the AI model

    Args:
        dicom_file: Path to the DICOM file
        max_size: Maximum dimension for the output image (default: 800)

    Returns:
        str: Path to the saved PNG file
    """
    try:
        # Get the dataset
        ds = dcmread(dicom_file)
        # Check for PixelData
        if 'PixelData' not in ds:
            raise ValueError(f"DICOM file {dicom_file} has no pixel data!")
        # Convert to float
        image = ds.pixel_array.astype(np.float32)
        # Resize while maintaining aspect ratio
        height, width = image.shape[:2]
        if max(height, width) > max_size:
            if height > width:
                new_height = max_size
                new_width = int(width * (max_size / height))
            else:
                new_width = max_size
                new_height = int(height * (max_size / width))
            image = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_AREA)
        # Clip to 1..99 percentiles to remove outliers and improve contrast
        minval = np.percentile(image, 1)
        maxval = np.percentile(image, 99)
        image = np.clip(image, minval, maxval)
        # Normalize image to 0-255
        image -= image.min()
        if image.max() != 0:
            image /= image.max()
        image *= 255.0
        # Save as 8 bit
        image = image.astype(np.uint8)
        # Auto adjust gamma
        image = apply_gamma_correction(image, None)
        # Save the PNG file
        base_name = os.path.splitext(os.path.basename(dicom_file))[0]
        png_file = os.path.join(IMAGES_DIR, f"{base_name}.png")
        cv2.imwrite(png_file, image)
        logging.info(f"Converted PNG saved to {png_file}")
        # Return the PNG file name
        return png_file
    except Exception as e:
        logging.error(f"Error converting DICOM to PNG: {e}")
        raise


# WebSocket and WebServer operations
async def serve_dashboard_page(request):
    """Serve the main dashboard HTML page.

    Args:
        request: aiohttp request object

    Returns:
        web.FileResponse: Dashboard HTML file response
    """
    return web.FileResponse(path=os.path.join(STATIC_DIR, "dashboard.html"))

async def serve_stats_page(request):
    """Serve the statistics HTML page.

    Args:
        request: aiohttp request object

    Returns:
        web.FileResponse: Statistics HTML file response
    """
    return web.FileResponse(path=os.path.join(STATIC_DIR, "stats.html"))

async def serve_about_page(request):
    """Serve the about HTML page.

    Args:
        request: aiohttp request object

    Returns:
        web.FileResponse: About HTML file response
    """
    return web.FileResponse(path=os.path.join(STATIC_DIR, "about.html"))


async def serve_check_page(request):
    """Serve the report check HTML page.

    Args:
        request: aiohttp request object

    Returns:
        web.FileResponse: Check HTML file response
    """
    return web.FileResponse(path=os.path.join(STATIC_DIR, "check.html"))

async def serve_favicon(request):
    """Serve the favicon.ico file.

    Args:
        request: aiohttp request object

    Returns:
        web.FileResponse: Favicon file response
    """
    return web.FileResponse(path=os.path.join(STATIC_DIR, "favicon.ico"))


async def serve_api_spec(request):
    """Serve the OpenAPI specification file from static/spec.json.

    Args:
        request: aiohttp request object

    Returns:
        web.Response: OpenAPI spec as JSON with updated server URL
    """
    import json
    
    # Read the static spec file
    spec_path = os.path.join(STATIC_DIR, "spec.json")
    with open(spec_path, 'r') as f:
        spec = json.load(f)
    
    # Update the server URL based on the request
    server_url = f"{request.scheme}://{request.host}"
    spec['servers'][0]['url'] = server_url
    
    # Return as JSON response
    return web.json_response(spec)


async def websocket_handler(request):
    """Handle WebSocket connections for real-time dashboard updates.

    Manages WebSocket connections from dashboard clients, adds them to the
    client set, sends connection notifications, and handles disconnections.

    Args:
        request: aiohttp request object containing connection info

    Returns:
        web.WebSocketResponse: WebSocket response object
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    websocket_clients.add(ws)
    await broadcast_dashboard_update(event = "connected", payload = {'address': request.remote}, client = ws)
    logging.info(f"Dashboard connected via WebSocket from {request.remote}")
    try:
        async for msg in ws:
            pass
    finally:
        websocket_clients.remove(ws)
        logging.info("Dashboard WebSocket disconnected.")
    return ws


async def exams_handler(request):
    """Provide paginated exam data with optional filters.

    Retrieves exams from database with pagination and filtering options.
    Supports filtering by review status, positivity, validity, region,
    processing status, and patient name search.
    Anonymizes patient names and IDs for non-admin users.

    Args:
        request: aiohttp request object with query parameters

    Returns:
        web.json_response: JSON response with exams data and pagination info
    """
    try:
        # Get user role from request (set by auth_middleware)
        user_role = getattr(request, 'user_role', 'user')
        
        page = int(request.query.get("page", "1"))
        filters = {}
        for filter in ['reviewed', 'positive', 'correct']:
            value = request.query.get(filter, 'any')
            if value != 'any':
                filters[filter] = value[0].lower() == 'y' and 1 or 0
        for filter in ['region', 'status', 'search']:
            value = request.query.get(filter, 'any')
            if value != 'any':
                filters[filter] = value
        offset = (page - 1) * PAGE_SIZE
        data, total = db_get_exams(limit = PAGE_SIZE, offset = offset, **filters)
        
        # Anonymize patient data for non-admin users
        for exam in data:
            if user_role != 'admin':
                # Anonymize the patient name
                exam['patient']['name'] = extract_patient_initials(exam['patient']['name'])
                # Show only first 7 digits of patient CNP
                patient_cnp = exam['patient']['cnp']
                if patient_cnp and len(patient_cnp) > 7:
                    exam['patient']['cnp'] = patient_cnp[:7] + '...'
                elif patient_cnp:
                    exam['patient']['cnp'] = patient_cnp
                else:
                    exam['patient']['cnp'] = 'Unknown'
        # Return the response
        return web.json_response({
            "exams": data,
            "total": total,
            "pages": int(total / PAGE_SIZE) + 1,
            "filters": filters,
        })
    except Exception as e:
        logging.error(f"Exams page error: {e}")
        return web.json_response([], status = 500)


async def stats_handler(request):
    """Provide statistical data for the dashboard.

    Retrieves comprehensive statistics from the database including totals,
    regional breakdowns, temporal trends, and processing performance metrics.

    Args:
        request: aiohttp request object

    Returns:
        web.json_response: JSON response with statistical data
    """
    try:
        return web.json_response(await db_get_stats())
    except Exception as e:
        logging.error(f"Exams page error: {e}")
        return web.json_response([], status = 500)


async def config_handler(request):
    """Provide global configuration parameters to the frontend.

    Returns configuration values that the frontend may need to display
    or use for various operations.

    Args:
        request: aiohttp request object

    Returns:
        web.json_response: JSON response with configuration parameters
    """
    try:
        config = {
            "OPENAI_URL_PRIMARY": OPENAI_URL_PRIMARY,
            "OPENAI_URL_SECONDARY": OPENAI_URL_SECONDARY,
            "NTFY_URL": NTFY_URL,
            "AE_TITLE": AE_TITLE,
            "AE_PORT": AE_PORT,
            "REMOTE_AE_TITLE": REMOTE_AE_TITLE,
            "REMOTE_AE_IP": REMOTE_AE_IP,
            "REMOTE_AE_PORT": REMOTE_AE_PORT
        }
        return web.json_response(config)
    except Exception as e:
        logging.error(f"Config endpoint error: {e}")
        return web.json_response({}, status = 500)


async def regions_handler(request):
    """Provide supported regions for the frontend dropdown.

    Returns the list of regions from the database that can be used in the dashboard
    filter dropdown.

    Args:
        request: aiohttp request object

    Returns:
        web.json_response: JSON response with regions list
    """
    try:
        # Get distinct regions from the database
        regions = db_get_regions()
        return web.json_response(regions)
    except Exception as e:
        logging.error(f"Regions endpoint error: {e}")
        return web.json_response([], status = 500)


async def diagnostics_handler(request):
    """Provide distinct diagnostic summaries from radiologist reports.

    Returns the list of unique diagnostic summaries (from rad_reports.summary)
    for use in filtering or display in the frontend.

    Args:
        request: aiohttp request object

    Returns:
        web.json_response: JSON response with diagnostic summaries list
    """
    try:
        # Get distinct diagnostic summaries from the database
        query = "SELECT DISTINCT summary FROM rad_reports WHERE summary IS NOT NULL AND summary != '' ORDER BY summary"
        rows = db_execute_query(query, fetch_mode='all')
        diagnostics = [row[0] for row in rows] if rows else []
        return web.json_response(diagnostics)
    except Exception as e:
        logging.error(f"Diagnostics endpoint error: {e}")
        return web.json_response([], status = 500)


async def patients_handler(request):
    """Provide paginated patient data with optional filters.

    Retrieves patients from database with pagination and filtering options.
    Supports filtering by name and CNP search.
    Anonymizes patient data for non-admin users.

    Args:
        request: aiohttp request object with query parameters

    Returns:
        web.json_response: JSON response with patients data and pagination info
    """
    try:
        # Get user role from request (set by auth_middleware)
        user_role = getattr(request, 'user_role', 'user')
        
        page = int(request.query.get("page", "1"))
        filters = {}
        for filter in ['search']:
            value = request.query.get(filter, 'any')
            if value != 'any':
                filters[filter] = value
        offset = (page - 1) * PAGE_SIZE
        
        # Get patients with pagination
        patients, total = db_get_patients(limit=PAGE_SIZE, offset=offset, **filters)
        
        # Anonymize patient data for non-admin users
        for patient in patients:
            if user_role != 'admin':
                # Anonymize the patient name
                patient['name'] = extract_patient_initials(patient['name'])
                # Show only first 7 digits of patient CNP
                patient_cnp = patient['cnp']
                if patient_cnp and len(patient_cnp) > 7:
                    patient['cnp'] = patient_cnp[:7] + '...'
                elif patient_cnp:
                    patient['cnp'] = patient_cnp
                else:
                    patient['cnp'] = 'Unknown'
        
        return web.json_response({
            "patients": patients,
            "total": total,
            "pages": int(total / PAGE_SIZE) + 1,
            "filters": filters,
        })
    except Exception as e:
        logging.error(f"Patients page error: {e}")
        return web.json_response([], status = 500)


async def patient_handler(request):
    """Provide a single patient's data by CNP.

    Retrieves a specific patient from the database by their CNP.
    Anonymizes patient data for non-admin users.
    Includes a list of all exam UIDs for this patient.

    Args:
        request: aiohttp request object with CNP parameter

    Returns:
        web.json_response: JSON response with patient data and exam UIDs or error
    """
    try:
        # Get user role from request (set by auth_middleware)
        user_role = getattr(request, 'user_role', 'user')
        
        # Get CNP from URL parameter
        cnp = request.match_info['cnp']
        
        # Get patient from database
        patient = db_get_patient_by_cnp(cnp)
        
        if not patient:
            return web.json_response({"error": "Patient not found"}, status=404)
        
        # Get all exam UIDs for this patient
        exam_uids = db_get_patient_exam_uids(cnp)
        
        # Add exam UIDs to patient data
        patient['exams'] = exam_uids
        
        # Anonymize patient data for non-admin users
        if user_role != 'admin':
            # Anonymize the patient name
            patient['name'] = extract_patient_initials(patient['name'])
            # Show only first 7 digits of patient CNP
            patient_cnp = patient['cnp']
            if patient_cnp and len(patient_cnp) > 7:
                patient['cnp'] = patient_cnp[:7] + '...'
            elif patient_cnp:
                patient['cnp'] = patient_cnp
            else:
                patient['cnp'] = 'Unknown'
        
        return web.json_response(patient)
    except Exception as e:
        logging.error(f"Patient endpoint error: {e}")
        return web.json_response({"error": "Internal server error"}, status=500)


async def exam_handler(request):
    """Provide a single exam's data by UID.

    Retrieves a specific exam from the database by its UID, including
    all associated patient and report data.

    Args:
        request: aiohttp request object with UID parameter

    Returns:
        web.json_response: JSON response with exam data or error
    """
    try:
        # Get user role from request (set by auth_middleware)
        user_role = getattr(request, 'user_role', 'user')
        
        # Get UID from URL parameter
        uid = request.match_info['uid']
        
        # Get exam from database
        exams, _ = db_get_exams(search=uid)
        if not exams:
            return web.json_response({"error": "Exam not found"}, status=404)
        
        exam = exams[0]
        
        # Anonymize patient data for non-admin users
        if user_role != 'admin':
            # Anonymize the patient name
            exam['patient']['name'] = extract_patient_initials(exam['patient']['name'])
            # Show only first 7 digits of patient CNP
            patient_cnp = exam['patient']['cnp']
            if patient_cnp and len(patient_cnp) > 7:
                exam['patient']['cnp'] = patient_cnp[:7] + '...'
            elif patient_cnp:
                exam['patient']['cnp'] = patient_cnp
            else:
                exam['patient']['cnp'] = 'Unknown'
        # Return the exam data
        return web.json_response(exam)
    except Exception as e:
        logging.error(f"Exam endpoint error: {e}")
        return web.json_response({"error": "Internal server error"}, status=500)


async def dicom_query(request):
    """Trigger a manual DICOM query/retrieve operation.

    Allows manual triggering of DICOM query operations for specified time
    periods through the dashboard interface.

    Args:
        request: aiohttp request object with JSON body containing hours parameter

    Returns:
        web.json_response: JSON response with operation status
    """
    try:
        data = await request.json()
        hours = int(data.get('hours', 3))
        logging.info(f"Manual QueryRetrieve triggered for the last {hours} hours.")
        await query_and_retrieve(hours * 60)
        return web.json_response({'status': 'success',
                                  'message': f'Query triggered for the last {hours} hours.'})
    except Exception as e:
        logging.error(f"Error processing manual query: {e}")
        return web.json_response({'status': 'error',
                                  'message': str(e)})


async def rad_review(request):
    """Record radiologist's review of an exam as normal or abnormal.

    Updates the radiologist report with the normal/abnormal status based on
    the radiologist's assessment. Uses the authenticated username as the
    radiologist identifier.

    Args:
        request: aiohttp request object with JSON body containing:
            - uid: The unique identifier of the exam
            - normal: Whether the radiologist marked the case as normal (True) or abnormal (False)

    Returns:
        web.json_response: JSON response with review status
    """
    try:
        data = await request.json()
        # Get 'uid', 'normal', and optional 'radiologist' from request
        uid = data.get('uid')
        normal = data.get('normal', None)
        # Use authenticated username as radiologist name, fallback to 'rad' if not available
        radiologist = getattr(request, 'username', 'rad')
        
        if uid is None or normal is None:
            return web.json_response({'status': 'error', 'message': 'UID and normal status are required'}, status=400)
        
        # Update the radiologist report
        db_rad_review(uid, normal, radiologist)
        
        # Get the updated exam data
        exam_data = db_get_exam_by_uid(uid)        
        logging.info(f"Exam {uid} marked as {normal and 'normal' or 'abnormal'} by radiologist {radiologist}, which {exam_data['report']['correct'] and 'validates' or 'invalidates'} the AI report.")
        await broadcast_dashboard_update(event = "radreview", payload = exam_data)
        response = {'status': 'success'}
        return web.json_response(response)
    except Exception as e:
        logging.error(f"Error processing radiologist review: {e}")
        return web.json_response({'status': 'error', 'message': str(e)}, status=500)




async def requeue_exam(request):
    """Re-queue an exam for processing.

    Sets an exam's status to 'queued' so it will be processed again by the AI.
    Clears existing AI report data (except text for reference) to ensure fresh analysis.

    Args:
        request: aiohttp request object with JSON body containing:
            - uid: The unique identifier of the exam to re-queue

    Returns:
        web.json_response: JSON response with re-queue status
    """
    try:
        data = await request.json()
        uid = data.get('uid')
        
        if not uid:
            return web.json_response({'status': 'error', 'message': 'UID is required'}, status=400)
        
        # Re-queue the exam
        success = db_requeue_exam(uid)
        
        if success:
            logging.info(f"Exam {uid} re-queued for processing.")
            # Notify the queue
            QUEUE_EVENT.set()
            payload = {'uid': uid}
            await broadcast_dashboard_update(event="requeue", payload=payload)
            return web.json_response({'status': 'success', 'message': f'Exam {uid} re-queued'})
        else:
            return web.json_response({'status': 'error', 'message': f'Failed to re-queue exam {uid}'}, status=500)
    except Exception as e:
        logging.error(f"Error re-queuing exam: {e}")
        return web.json_response({'status': 'error', 'message': str(e)}, status=500)


async def check_report(report_text):
    """Analyze a free-text radiology report for pathological findings.

    Takes a radiology report text and sends it to the LLM for analysis
    using a specialized prompt to extract key information.

    Args:
        report_text: Radiology report text to analyze

    Returns:
        dict: Analysis results with pathologic, severity, and summary
    """
    try:
        logging.info(f"Report check request received with report length: {len(report_text)} characters")
        
        if not report_text:
            logging.warning("Report check request failed: no report text provided")
            return {'error': 'No report text provided'}
        
        # Prepare the request headers
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json',
        }
        
        # Prepare the JSON data
        payload = {
            "model": MODEL_NAME,
            "stream": False,
            "keep_alive": 1800,
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": CHK_PROMPT}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": report_text}
                    ]
                }
            ]
        }
        
        logging.debug(f"Sending report to OpenAI API with model: {MODEL_NAME}")
        
        async with aiohttp.ClientSession() as session:
            result = await send_to_openai(session, headers, payload)
            if not result:
                logging.error("Failed to get response from AI service")
                return {'error': 'Failed to get response from AI service'}
            
            response_text = result["choices"][0]["message"]["content"].strip()
            logging.debug(f"Raw AI response: {response_text}")
            
            # Clean up markdown code fences if present
            response_text = re.sub(r"^```(?:json)?\s*", "", response_text, flags=re.IGNORECASE | re.MULTILINE)
            response_text = re.sub(r"\s*```$", "", response_text, flags=re.MULTILINE)
            
            try:
                parsed_response = json.loads(response_text)
                logging.info(f"AI responded: {parsed_response}")
                
                # Validate required fields
                if "pathologic" not in parsed_response or "severity" not in parsed_response or "summary" not in parsed_response:
                    raise ValueError("Missing required fields in AI response")
                
                # Validate pathologic field
                if parsed_response["pathologic"] not in ["yes", "no"]:
                    raise ValueError("Invalid pathologic value in AI response")
                
                # Validate severity field
                if not isinstance(parsed_response["severity"], int) or parsed_response["severity"] < 0 or parsed_response["severity"] > 10:
                    raise ValueError("Invalid severity value in AI response")
                
                # Validate summary field
                if not isinstance(parsed_response["summary"], str):
                    raise ValueError("Invalid summary value in AI response")
                
                logging.info(f"AI analysis completed: severity {parsed_response['severity']}, {parsed_response['pathologic'] and 'pathologic' or 'non-pathologic'}: {parsed_response['summary']}")
                return parsed_response
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse AI response as JSON: {response_text}")
                return {'error': 'Failed to parse AI response', 'response': response_text}
            except ValueError as e:
                logging.error(f"Invalid AI response format: {e}")
                return {'error': f'Invalid AI response format: {str(e)}', 'response': response_text}
    except Exception as e:
        logging.error(f"Error processing report check request: {e}")
        return {'error': 'Internal server error'}

async def check_report_handler(request):
    """Analyze a free-text radiology report for pathological findings.

    Takes a radiology report text and sends it to the LLM for analysis
    using a specialized prompt to extract key information.

    Args:
        request: aiohttp request object with JSON body containing report text

    Returns:
        web.json_response: JSON response with analysis results
    """
    try:
        data = await request.json()
        report_text = data.get('report', '').strip()
        
        result = await check_report(report_text)
        
        # Check if there was an error
        if 'error' in result:
            status = 500 if result['error'] != 'No report text provided' else 400
            return web.json_response(result, status=status)
        
        return web.json_response(result)
    except Exception as e:
        logging.error(f"Error processing report check request: {e}")
        return web.json_response({'error': 'Internal server error'}, status=500)


@web.middleware
async def auth_middleware(request, handler):
    """Basic authentication middleware for API endpoints.

    Implements HTTP Basic authentication for all API endpoints except
    static files and OPTIONS requests. Validates credentials against
    configured users and stores user role in request.

    Args:
        request: aiohttp request object
        handler: Request handler function

    Returns:
        Response from the handler if authenticated, or HTTP 401 if not
    """
    # Skip auth for static files and OPTIONS requests
    if request.path.startswith('/static/') or request.path.startswith('/images/') or request.method == 'OPTIONS':
        return await handler(request)
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Basic '):
        raise web.HTTPUnauthorized(
            text = "401: Authentication required",
            headers = {'WWW-Authenticate': 'Basic realm="XRayVision"'})
    try:
        credentials = base64.b64decode(auth_header[6:]).decode('utf-8')
        username, password = credentials.split(':', 1)
        user_info = USERS.get(username)
        if not user_info or user_info['password'] != password:
            logging.warning(f"Invalid authentication for user: {username}")
            raise ValueError("Invalid authentication")
        # Store user role in request for later use
        request.user_role = user_info['role']
    except (ValueError, UnicodeDecodeError) as e:
        raise web.HTTPUnauthorized(
            text = "401: Invalid authentication",
            headers = {'WWW-Authenticate': 'Basic realm="XRayVision"'})
    return await handler(request)


async def broadcast_dashboard_update(event = None, payload = None, client = None):
    """Broadcast dashboard updates to all connected WebSocket clients.

    Sends real-time updates to dashboard clients including queue status,
    processing information, statistics, and OpenAI health status.

    Args:
        event: Optional event name for specific update types
        payload: Optional data payload for the event
        client: Optional specific client to send update to (instead of all)
    """
    # Check if there are any clients
    if not (websocket_clients or client):
        return
    # Update the queue size
    dashboard['queue_size'] = db_get_queue_size()
    # Get error statistics
    error_stats = db_get_error_stats()
    dashboard['error_count'] = error_stats['error']
    dashboard['ignore_count'] = error_stats['ignore']
    # Get the count of successfully processed exams in the last week
    dashboard['success_count'] = db_get_weekly_processed_count()
    # Create a list of clients
    if client:
        clients = [client,]
    else:
        clients = websocket_clients.copy()
    # Create the json object
    data = {}
    if event:
        data['event'] = {'name': event, 'payload': payload}
    data['dashboard'] = dashboard
    data['openai'] = {'url': active_openai_url,
                      'health': {
                        'pri': health_status.get(OPENAI_URL_PRIMARY,  False),
                        'sec': health_status.get(OPENAI_URL_SECONDARY, False)
                       }
                     }
    data['timings'] = timings
    if NO_QUERY:
        data['next_query'] = 'Disabled'
    elif next_query:
        data['next_query'] = next_query.strftime('%Y-%m-%d %H:%M:%S')
    # Send the update to all clients
    for client in clients:
        # Send the update to the client
        try:
            await client.send_json(data)
        except Exception as e:
            logging.error(f"Error sending update to WebSocket client: {e}")
            websocket_clients.remove(client)


# Notification operations
async def send_ntfy_notification(uid, report, info):
    """Send notification to ntfy.sh with image and report"""
    if not ENABLE_NTFY:
        logging.info("NTFY notifications are disabled")
        return

    try:
        # Construct image URL
        image_url = f"https://xray.eridu.eu.org/static/{uid}.png"
        # Create headers and message body
        message = f"Positive finding in {info['exam']['region']} study\nPatient: {info['patient']['name']}\nReport: {report}"
        headers = {
            "Title": "XRayVision Alert - Positive Finding",
            "Tags": "warning,skull",
            "Priority": "4",
            "Attach": image_url
        }

        # Post the notification
        async with aiohttp.ClientSession() as session:
            async with session.post(
                NTFY_URL,
                data=message,
                headers=headers
            ) as resp:
                if resp.status == 200:
                    logging.info("Successfully sent ntfy notification")
                else:
                    logging.error(f"Notification failed with status {resp.status}: {await resp.text()}")
    except Exception as e:
        logging.error(f"Failed to send ntfy notification: {e}")


# API operations
def validate_romanian_cnp(patient_cnp):
    """
    Validate Romanian personal identification number (CNP) format and checksum.

    Romanian personal IDs (CNP) have 13 digits with the following structure:
    - Position 1: Gender/Sector (1-8 for born 1900-2099, 9 for foreign
      residents)
    - Positions 2-3: Year of birth (00-99)
    - Positions 4-5: Month of birth (01-12)
    - Positions 6-7: Day of birth (01-31)
    - Positions 8-9: County code (01-52, 99)
    - Positions 10-12: Serial number (001-999)
    - Position 13: Checksum digit

    Args:
        patient_cnp: Personal identification number as string

    Returns:
        dict: Dictionary with validation result and parsed information if valid
              {
                  'valid': bool,
                  'birth_date': datetime object (if valid),
                  'age': int (current age in years, if valid),
                  'sex': str ('M' or 'F', if valid),
                  'county': int (county code, if valid)
              }
    """
    try:
        # Ensure we have a string and clean it
        pid = str(patient_cnp).strip()
        # Check if it's exactly 13 digits
        if not pid or len(pid) != 13 or not pid.isdigit():
            return {'valid': False}
        # Extract components
        gender_digit = int(pid[0])
        year = int(pid[1:3])
        month = int(pid[3:5])
        day = int(pid[5:7])
        county = int(pid[7:9])
        serial = int(pid[9:12])
        checksum_digit = int(pid[12])
        # Validate gender digit (1-9)
        if gender_digit < 1 or gender_digit > 9:
            return {'valid': False}
        # Validate date components
        # Determine century based on gender digit
        if gender_digit in [1, 2]:
            full_year = 1900 + year
        elif gender_digit in [3, 4]:
            full_year = 1800 + year
        elif gender_digit in [5, 6]:
            full_year = 2000 + year
        elif gender_digit in [7, 8]:
            full_year = 2000 + year  # For people born after 2000
        elif gender_digit == 9:
            full_year = 1900 + year  # Foreign residents
        else:
            return {'valid': False}
        # Validate month (1-12)
        if month < 1 or month > 12:
            return {'valid': False}
        # Validate day (1-31)
        if day < 1 or day > 31:
            return {'valid': False}
        # More precise date validation
        try:
            birth_date = datetime(full_year, month, day)
        except ValueError:
            return {'valid': False}
        # Validate county code (01-52 excluding 47-50, 70-79, 90-99)
        if not ((1 <= county <= 52 and not (47 <= county <= 50)) or (70 <= county <= 79) or (90 <= county <= 99)):
            return {'valid': False}
        # Get county name
        county_name = county_names.get(county, "Unknown")
        # Validate checksum using the official algorithm
        # Weights for each digit position
        weights = [2, 7, 9, 1, 4, 6, 3, 5, 8, 2, 7, 9]
        # Calculate weighted sum
        weighted_sum = sum(int(pid[i]) * weights[i] for i in range(12))
        # Calculate checksum
        checksum = weighted_sum % 11
        if checksum == 10:
            checksum = 1
        # Compare with provided checksum digit
        if checksum != checksum_digit:
            return {'valid': False}
        
        # Calculate current age
        today = datetime.now()
        age = today.year - birth_date.year
        # Adjust if birthday hasn't occurred this year
        if (today.month, today.day) < (birth_date.month, birth_date.day):
            age -= 1
        
        # Determine sex
        sex = 'M' if gender_digit % 2 == 1 else 'F'
        
        # Return validation result with parsed information
        return {
            'valid': True,
            'birth_date': birth_date,
            'age': age,
            'sex': sex,
            'county': county
        }
    except Exception as e:
        logging.debug(f"Error validating Romanian CNP {patient_cnp}: {e}")
        return {'valid': False}


def compute_age_from_cnp(patient_cnp):
    """
    Compute patient age based on Romanian personal identification number.

    Romanian personal IDs have the format:
    - First digit: 1/2 for 1900s, 5/6 for 2000s, etc.
    - Next 6 digits: YYMMDD (birth date)

    Args:
        patient_cnp: Personal identification number as string

    Returns:
        int: Age in years, or -1 if unable to compute
    """
    # First validate the Romanian ID format and get parsed information
    result = validate_romanian_cnp(patient_cnp)
    if not result['valid']:
        return -1
    # Return the computed age
    return result['age']


def contains_any_word(string, *words):
    """
    Check if any of the specified words are present in the given string.

    Args:
        string: String to search in
        *words: Variable number of words to search for

    Returns:
        bool: True if any word is found in the string, False otherwise
    """
    return any(i in string for i in words)


def identify_anatomic_region(info):
    """
    Identify the anatomic region and appropriate question based on protocol name.

    Maps DICOM protocol names to anatomic regions and formulates region-specific
    questions for the AI to analyze. Uses pattern matching to handle variations
    in naming conventions.

    Args:
        info: Dictionary containing exam information with protocol name

    Returns:
        tuple: (region, question) where region is the identified anatomic region
               and question is the region-specific query for AI analysis
    """
    desc = info["exam"]["protocol"].lower()

    # Check each region rule from config
    for region_key, keywords in REGION_RULES.items():
        if contains_any_word(desc, *keywords):
            region = region_key
            break
    else:
        # Fallback
        region = desc

    # Get question from config or use fallback
    question = REGION_QUESTIONS.get(region, "Is there anything abnormal")

    # Return the region and the question
    return region, question



def identify_imaging_projection(info):
    """
    Identify the imaging projection based on protocol name.

    Determines if the X-ray view is frontal (AP/PA), lateral, or oblique
    based on keywords in the protocol name.

    Args:
        info: Dictionary containing exam information with protocol name

    Returns:
        str: Identified projection ('frontal', 'lateral', 'oblique', or '')
    """
    desc = info["exam"]["protocol"].lower()
    if contains_any_word(desc, "a.p.", "p.a.", "d.v.", "v.d.", "d.p"):
        projection = "frontal"
    elif contains_any_word(desc, "lat.", "pr."):
        projection = "lateral"
    elif contains_any_word(desc, "oblic"):
        projection = "oblique"
    else:
        # Fallback
        projection = ""
    # Return the projection
    return projection


def determine_patient_gender_description(info):
    """
    Determine patient gender description based on DICOM sex field.

    Maps DICOM patient sex codes to descriptive terms for use in AI prompts.

    Args:
        info: Dictionary containing patient information with sex field

    Returns:
        str: Gender description ('boy', 'girl', or 'child')
    """
    patient_sex = info["patient"].get("sex", "").lower()
    if "m" in patient_sex:
        gender = "boy"
    elif "f" in patient_sex:
        gender = "girl"
    else:
        # Fallback
        gender = "child"
    # Return the gender
    return gender


async def get_fhir_patient(session, cnp):
    """
    Search for a patient in FHIR system by CNP.

    Args:
        session: aiohttp ClientSession instance
        cnp: Patient CNP

    Returns:
        dict or None: Patient data from FHIR if successful, None otherwise
    """
    try:
        # Use basic authentication
        auth = aiohttp.BasicAuth(FHIR_USERNAME, FHIR_PASSWORD)
        
        url = f"{FHIR_URL}/fhir/Patient"
        params = {'q': cnp}
        
        async with session.get(url, auth=auth, params=params, timeout=30) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get('resourceType') == 'Patient':
                    # Single patient returned
                    return data
                # Multiple patients returned
                logging.error(f"FHIR patient search error: multiple patients found for CNP {cnp}")
            else:
                logging.warning(f"FHIR patient search failed with status {resp.status}")
    except Exception as e:
        logging.error(f"FHIR patient search error: {e}")
    return None

async def get_fhir_imagingstudies(session, patient_id, exam_datetime):
    """
    Search for imaging studies for a patient in FHIR system.

    Args:
        session: aiohttp ClientSession instance
        patient_id: Patient ID from HIS
        exam_datetime: Exam datetime to search around

    Returns:
        list: List of imaging studies from FHIR (exactly one study) or empty list
    """
    try:
        # Use basic authentication
        auth = aiohttp.BasicAuth(FHIR_USERNAME, FHIR_PASSWORD)
        
        url = f"{FHIR_URL}/fhir/Observation"
        params = {
            'patient': patient_id,
            'type': 'radio',
            'dt': exam_datetime
        }
        
        async with session.get(url, auth=auth, params=params, timeout=30) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get('resourceType') == 'Bundle' and 'entry' in data:
                    studies = [entry['resource'] for entry in data['entry'] if entry['resource'].get('resourceType') == 'Observation']
                    # We need exactly one study
                    if len(studies) == 1:
                        return studies
                    elif len(studies) > 1:
                        logging.warning(f"FHIR imaging studies search returned {len(studies)} studies, expected exactly one")
                    # Return empty list if no studies or more than one
                    return []
            else:
                logging.warning(f"FHIR imaging studies search failed with status {resp.status}")
    except Exception as e:
        logging.error(f"FHIR imaging studies search error: {e}")
    return []

async def get_fhir_diagnosticreport(session, report_id):
    """
    Get a diagnostic report from FHIR system.

    Args:
        session: aiohttp ClientSession instance
        report_id: Report ID

    Returns:
        dict or None: Diagnostic report from FHIR if successful, None otherwise
    """
    try:
        # Use basic authentication
        auth = aiohttp.BasicAuth(FHIR_USERNAME, FHIR_PASSWORD)
        
        url = f"{FHIR_URL}/fhir/DiagnosticReport/{report_id}"
        
        async with session.get(url, auth=auth, timeout=30) as resp:
            if resp.status == 200:
                data = await resp.json()
                # Ensure the resource type is DiagnosticReport
                if data.get('resourceType') == 'DiagnosticReport':
                    return data
                else:
                    logging.warning(f"FHIR diagnostic report has incorrect resource type: {data.get('resourceType')}")
            else:
                logging.warning(f"FHIR diagnostic report failed with status {resp.status}")
    except Exception as e:
        logging.error(f"FHIR diagnostic report error: {e}")
    return None

async def send_to_openai(session, headers, payload):
    """
    Send a request to the currently active OpenAI API endpoint.

    Attempts to send a POST request to the active OpenAI endpoint with the
    provided headers and payload. Handles HTTP errors and exceptions.

    Args:
        session: aiohttp ClientSession instance
        headers: HTTP headers for the request
        payload: JSON payload containing the request data

    Returns:
        dict or None: JSON response from API if successful, None otherwise
    """
    try:
        async with session.post(active_openai_url, headers = headers, json = payload, timeout = 300) as resp:
            if resp.status == 200:
                return await resp.json()
            logging.warning(f"{active_openai_url} failed with status {resp.status}")
    except Exception as e:
        logging.error(f"{active_openai_url} request error: {e}")
    # Failed
    return None


async def send_exam_to_openai(exam, max_retries = 3):
    """
    Send an exam's PNG image to the OpenAI API for analysis.

    Processes the exam by identifying region/projection, preparing AI prompts,
    encoding the image, sending requests with retries, parsing responses,
    storing results in the database, and sending notifications for positive
    findings. Implements exponential backoff for retry attempts.

    Args:
        exam: Dictionary containing exam information and metadata
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        bool: True if successfully processed, False otherwise
    """
    try:
        # Read the PNG file
        with open(os.path.join(IMAGES_DIR, f"{exam['uid']}.png"), 'rb') as f:
            image_bytes = f.read()
        # Identify the region
        region, question = identify_anatomic_region(exam)
        # Filter on specific region
        if not region in REGIONS:
            logging.info(f"Ignoring {exam['uid']} with {region} x-ray.")
            db_set_status(exam['uid'], 'ignore')
            return False
        # Identify the prjection, gender and age
        projection = identify_imaging_projection(exam)
        gender = determine_patient_gender_description(exam)
        age = exam["patient"]["age"]
        if age > 1:
            txtAge = f"{age} years old"
        elif age > 0:
            txtAge = f"{age} year old"
        elif age == 0:
            txtAge = "newborn"
        else:
            txtAge = ""
        # Update exam info
        exam['exam'].update({'region': region, 'projection': projection})
        # Get the subject of the study and the studied region
        subject = " ".join([txtAge, gender])
        if region:
            anatomy = " ".join([projection, region])
        else:
            anatomy = ""
        # Get previous reports for the same patient and region
        previous_reports = db_get_previous_reports(exam['patient']['cnp'], region, months=3)

        # Create the prompt
        prompt = USR_PROMPT.format(question=question, anatomy=anatomy, subject=subject)

        # Append previous reports if any exist
        if previous_reports:
            prompt += "\n\nPRIOR STUDIES:"
            for i, (report, date) in enumerate(previous_reports, 1):
                prompt += f"\n\n[{date}] {report}"
            prompt += "\n\nCompare to prior studies. Note new, stable, resolved, or progressive findings with dates."
        prompt += "\n\nIMPORTANT: Also identify any other lesions or abnormalities beyond the primary clinical question. Output JSON only."

        logging.debug(f"Prompt: {prompt}")
        logging.info(f"Processing {exam['uid']} with {region} x-ray.")
        if exam['report']['ai']['text']:
            json_report = {'short': exam['report']['ai']['short'],
                           'report': exam['report']['ai']['text']}
            exam['report']['json'] = json.dumps(json_report)
            logging.info(f"Previous report: {exam['report']['json']}")
        # Base64 encode the PNG to comply with OpenAI Vision API
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        image_url = f"data:image/png;base64,{image_b64}"
        # Prepare the request headers
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json',
        }
        # Prepare the JSON data
        data = {
            "model": MODEL_NAME,
            "timings_per_token": True,
            "min_p": 0.05,
            "top_k": 40,
            "top_p": 0.95,
            "temperature": 0.6,
            "cache_prompt": True,
            "stream": False,
            "keep_alive": 1800,
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYS_PROMPT}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ]
        }
        if 'json' in exam['report']:
            data['messages'].append({'role': 'assistant', 'content': exam['report']['json']})
            data['messages'].append({'role': 'user', 'content': REV_PROMPT})
        # Up to 3 attempts with exponential backoff (2s, 4s, 8s delays).
        attempt = 1
        while attempt <= max_retries:
            try:
                # Start timing
                start_time = asyncio.get_event_loop().time()
                async with aiohttp.ClientSession() as session:
                    result = await send_to_openai(session, headers, data)
                    if not result:
                        break
                    response = result["choices"][0]["message"]["content"].strip()
                    # Clean up markdown code fences (```json ... ```, ``` ... ```, etc.)
                    response = re.sub(r"^```(?:json)?\s*", "", response, flags = re.IGNORECASE | re.MULTILINE)
                    response = re.sub(r"\s*```$", "", response, flags = re.MULTILINE)
                    # Clean up any text before '{'
                    response = re.sub(r"^[^{]*{", "{", response, flags = re.IGNORECASE | re.MULTILINE)
                    try:
                        parsed = json.loads(response)
                        short = parsed["short"].strip().lower()
                        report = parsed["report"].strip()
                        confidence = parsed.get("confidence", 0)
                        if short not in ("yes", "no") or not report:
                            raise ValueError("Invalid json format in OpenAI response")
                    except Exception as e:
                        logging.error(f"Rejected malformed OpenAI response: {e}")
                        logging.error(response)
                        break
                    logging.info(f"OpenAI API response for {exam['uid']}: [{short.upper()}] {report} (confidence: {confidence})")
                    # Save to exams database
                    is_positive = short == "yes"
                    # Calculate timing statistics
                    global timings
                    end_time = asyncio.get_event_loop().time()
                    processing_time = end_time - start_time  # In seconds
                    timings['total'] = int(processing_time * 1000)  # Convert to milliseconds
                    if timings['average'] > 0:
                        timings['average'] = int((3 * timings['average'] + timings['total']) / 4)
                    else:
                        timings['average'] = timings['total']
                    # Update the AI report with the processing time
                    query = "UPDATE ai_reports SET latency = ? WHERE uid = ?"
                    params = (processing_time, exam['uid'])
                    db_execute_query_retry(query, params)
                    # Save to exams database
                    db_add_exam(exam, report = report, positive = is_positive, confidence = confidence)
                    # Send notification for positive cases
                    if is_positive:
                        try:
                            await send_ntfy_notification(exam['uid'], report, exam)
                        except Exception as e:
                            logging.error(f"Failed to send ntfy notification: {e}")
                    # Notify the dashboard frontend to reload first page
                    await broadcast_dashboard_update(event = "new_exam", payload = {'uid': exam['uid'], 'positive': is_positive, 'reviewed': exam['report']['ai'].get('reviewed', False)})
                    # Success
                    return True
            except Exception as e:
                logging.warning(f"Error uploading {exam['uid']} (attempt {attempt}): {e}")
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
                attempt += 1
        # Failure after max_retries
        db_set_status(exam['uid'], 'error')
        QUEUE_EVENT.clear()
        logging.error(f"Failed to process {exam['uid']} after {attempt} attempts.")
        await broadcast_dashboard_update()
        return False
    except Exception as e:
        logging.error(f"Critical error in send_exam_to_openai for {exam['uid']}: {e}")
        db_set_status(exam['uid'], 'error')
        await broadcast_dashboard_update()
        return False


# Threads
async def start_dashboard():
    """
    Start the dashboard web server with all routes and middleware.

    Configures and starts the aiohttp web server that serves the dashboard
    frontend, API endpoints, and static files. Sets up authentication
    middleware and all required routes.
    """
    global web_server
    app = web.Application(middlewares = [auth_middleware])
    app.router.add_get('/', serve_dashboard_page)
    app.router.add_get('/stats', serve_stats_page)
    app.router.add_get('/about', serve_about_page)
    app.router.add_get('/check', serve_check_page)
    app.router.add_get('/favicon.ico', serve_favicon)
    app.router.add_get('/ws', websocket_handler)
    
    # API endpoints - Data retrieval
    app.router.add_get('/api/exams', exams_handler)
    app.router.add_get('/api/exams/{uid}', exam_handler)
    app.router.add_get('/api/patients', patients_handler)
    app.router.add_get('/api/patients/{cnp}', patient_handler)
    app.router.add_get('/api/stats', stats_handler)
    app.router.add_get('/api/regions', regions_handler)
    app.router.add_get('/api/diagnostics', diagnostics_handler)
    app.router.add_get('/api/config', config_handler)
    
    # API endpoints - Actions
    app.router.add_post('/api/radreview', rad_review)
    app.router.add_post('/api/requeue', requeue_exam)
    app.router.add_post('/api/dicomquery', dicom_query)
    app.router.add_post('/api/check', check_report_handler)
    
    # API endpoints - Metadata
    app.router.add_get('/api/spec', serve_api_spec)
    
    # Static file serving
    app.router.add_static('/images/', path = IMAGES_DIR, name = 'images')
    app.router.add_static('/static/', path = STATIC_DIR, name = 'static')
    web_server = web.AppRunner(app)
    await web_server.setup()
    site = web.TCPSite(web_server, '0.0.0.0', DASHBOARD_PORT)
    await site.start()
    logging.info(f"Dashboard available at http://localhost:{DASHBOARD_PORT}")


async def relay_to_openai_loop():
    """
    Main processing loop that sends queued exams to the OpenAI API.

    This is the core processing function that:
    1. Continuously monitors the database for queued exams
    2. Processes one exam at a time to avoid overwhelming the AI service
    3. Updates dashboard status during processing
    4. Handles success/failure cases and cleanup
    5. Implements proper error handling and status updates

    The loop waits on a QUEUE_EVENT when there's nothing to process,
    which gets signaled when new items are added to the queue.
    """
    while True:
        # Get one file from queue
        exams, total = db_get_exams(limit = 1, status = ['queued', 'requeue', 'check'])
        # Wait here if there are no items in queue or there is no OpenAI server
        if not exams or active_openai_url is None:
            QUEUE_EVENT.clear()
            await QUEUE_EVENT.wait()
            continue
        # Get only one exam, if any
        (exam,) = exams
        # The DICOM file name
        dicom_file = os.path.join(IMAGES_DIR, f"{exam['uid']}.dcm")
        try:
            # Set the status
            db_set_status(exam['uid'], "processing")
            # Update the dashboard
            dashboard['queue_size'] = total
            dashboard['processing'] = extract_patient_initials(exam['patient']['name'])
            await broadcast_dashboard_update()
            
            # Check the exam status and process accordingly
            exam_status = exam['exam']['status']
            if exam_status in ['queued', 'requeue']:
                # Send to AI for processing
                result = await send_exam_to_openai(exam)
                # Check the result
                if result:
                    # Set the status
                    db_set_status(exam['uid'], "done")
                    # Remove the DICOM file
                    if not KEEP_DICOM:
                        try:
                            os.remove(dicom_file)
                            logging.info(f"DICOM file {dicom_file} deleted after processing.")
                        except Exception as e:
                            logging.warning(f"Error removing DICOM file {dicom_file}: {e}")
                    else:
                        logging.debug(f"Keeping DICOM file: {dicom_file}")
                else:
                    # Error already set in send_exam_to_openai
                    pass
            elif exam_status == 'check':
                # Process FHIR report with LLM
                await process_fhir_report_with_llm(exam['uid'])
                # Set the status to done
                db_set_status(exam['uid'], "done")
        except Exception as e:
            logging.error(f"Unexpected error processing {exam['uid']}: {e}")
            db_set_status(exam['uid'], "error")
        finally:
            dashboard['processing'] = None
            await broadcast_dashboard_update()


async def openai_health_check():
    """
    Periodically check the health status of OpenAI API endpoints.

    Tests both primary and secondary OpenAI endpoints every 5 minutes,
    updates health status tracking, selects the active endpoint based
    on health status, and signals the processing queue when endpoints
    become available.
    """
    global active_openai_url
    while True:
        for url in [OPENAI_URL_PRIMARY, OPENAI_URL_SECONDARY]:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url.replace("/chat/completions", "/models"), timeout = 5) as resp:
                        health_status[url] = (resp.status == 200)
                        logging.info(f"Health check {url} → {resp.status}")
            except Exception as e:
                health_status[url] = False
                logging.warning(f"Health check failed for {url}: {e}")

        if health_status.get(OPENAI_URL_PRIMARY):
            active_openai_url = OPENAI_URL_PRIMARY
            logging.info("Using primary OpenAI backend.")
        elif health_status.get(OPENAI_URL_SECONDARY):
            active_openai_url = OPENAI_URL_SECONDARY
            logging.info("Using secondary OpenAI backend.")
        else:
            active_openai_url = None
            logging.error("No OpenAI backend is currently healthy")
        # Signal the queue
        if active_openai_url:
            QUEUE_EVENT.set()
        # WebSocket broadcast
        await broadcast_dashboard_update()
        # Sleep for 5 minutes
        await asyncio.sleep(300)

async def fhir_loop():
    """
    Periodically check the health status of FHIR API endpoint and process exams.

    Tests FHIR endpoint health and processes exams without radiologist reports.
    """
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                # Test FHIR connectivity using the proper metadata endpoint
                # Health check does not require authentication
                async with session.get(f"{FHIR_URL}/fhir/Metadata", timeout=10) as resp:
                    health_status[FHIR_URL] = resp.status == 200
                    logging.debug(f"FHIR check {FHIR_URL} → {resp.status}")
                
                if health_status[FHIR_URL]:
                    # Process exams without radiologist reports
                    await process_exams_without_rad_reports(session)
        except Exception as e:
            health_status[FHIR_URL] = False
            logging.warning(f"Health check failed for FHIR: {e}")
        
        # WebSocket broadcast
        await broadcast_dashboard_update()
        
        # Random delay between 1 and 2 minutes
        delay = random.randint(60, 120)
        await asyncio.sleep(delay)

async def process_fhir_report_with_llm(exam_uid):
    """
    Process a FHIR diagnostic report with the LLM and update the database.

    Args:
        exam_uid: The exam UID to process
    """
    # Get the radiologist report from the database
    rad_report = db_get_exam_rad_report(exam_uid)
    
    if not rad_report:
        logging.warning(f"No radiologist report found for exam {exam_uid}")
        return
    
    # If the report has already been assessed (positive > -1), skip processing
    if rad_report.get('positive', -1) > -1:
        logging.debug(f"Radiologist report for exam {exam_uid} already assessed, skipping LLM processing")
        return
    
    # Extract the report text
    report_text = rad_report.get('text', "").strip()
    if not report_text:
        logging.warning(f"No text in radiologist report for exam {exam_uid}")
        return
    
    # Log the current status before sending to LLM
    logging.info(f"Sending FHIR report for exam {exam_uid} to LLM for analysis")
    
    # Use LLM to analyze the diagnostic report and fill positive, severity, and summary fields
    start_time = asyncio.get_event_loop().time()
    analysis_result = await check_report(report_text)
    end_time = asyncio.get_event_loop().time()
    processing_time = end_time - start_time  # In seconds
    
    # Set default values in case of analysis failure
    positive = -1
    severity = -1
    summary = ''
    
    # Extract values from analysis result if successful
    if 'error' not in analysis_result:
        try:
            positive = 1 if analysis_result['pathologic'] == 'yes' else 0
            severity = analysis_result['severity']
            summary = analysis_result['summary']
        except Exception as e:
            logging.warning(f"Could not extract analysis results from LLM: {e}")
    else:
        logging.warning(f"LLM failed for exam {exam_uid}: {analysis_result['error']}")
        processing_time = -1  # Set to -1 if failed
    
    # Update the radiologist report in our database
    db_update_rad_report(
        uid=exam_uid,
        positive=positive,
        severity=severity,
        summary=summary,
        model=MODEL_NAME,
        latency=processing_time
    )
    logging.info(f"Updated FHIR report for exam {exam_uid} with LLM analysis results")

async def process_exams_without_rad_reports(session):
    """
    Process exams that don't have radiologist reports yet.

    This function identifies exams without radiologist reports, finds the
    corresponding patient in HIS, and retrieves the radiologist report.
    """
    # Get the oldest exam without a radiologist report
    exam = db_get_exam_without_rad_report()
    if not exam:
        return
    
    # Extract necessary information
    patient_cnp = exam['patient']['cnp']
    exam_datetime = exam['exam']['created']
    exam_uid = exam['uid']
    
    # Get patient from database
    patient_id = exam['patient']['id']
    
    # If patient ID is not known, search for it in FHIR
    if not patient_id:
        fhir_patient = await get_fhir_patient(session, patient_cnp)
        if fhir_patient and 'id' in fhir_patient:
            patient_id = fhir_patient['id']
            # Update patient ID in database
            db_update_patient_id(patient_cnp, patient_id)
    # If still no patient ID, log and skip
    if not patient_id:
        logging.warning(f"Could not find FHIR patient for CNP {patient_cnp}, skipping exam {exam_uid}")
        return
    
    # If we have patient ID, search for imaging studies
    studies = await get_fhir_imagingstudies(session, patient_id, exam_datetime)
    if not studies:
        logging.info(f"No imaging studies found for exam {exam_uid}")
        return
    elif len(studies) > 1:
         logging.info(f"Multiple close imaging studies found for exam {exam_uid}, skipping.")
         return

    # Get the single study
    study = studies[0]
    if not 'id' in study:
        logging.warning(f"Imaging study for exam {exam_uid} has no ID, skipping.")
        return
    
    # If exactly one study found, get its diagnostic report
    report = await get_fhir_diagnosticreport(session, study['id'])
    if report and 'conclusion' in report and report['conclusion']:
        # Extract radiologist name from resultsInterpreter if available
        radiologist = 'rad'  # Default value
        try:
            if 'resultsInterpreter' in report and len(report['resultsInterpreter']) > 0:
                interpreter = report['resultsInterpreter'][0]
                if 'display' in interpreter:
                    radiologist = interpreter['display']
        except Exception as e:
            logging.warning(f"Could not extract radiologist name from FHIR report: {e}")
        
        # Extract justification from extensions if available
        justification = ''  # Default value
        try:
            if 'extension' in report:
                for ext in report['extension']:
                    if 'diagnostic-report-reason' in ext.get('url', ''):
                        if 'extension' in ext:
                            for nested_ext in ext['extension']:
                                if nested_ext.get('url') == 'text' and 'valueString' in nested_ext:
                                    justification = nested_ext['valueString']
                                    break
        except Exception as e:
            logging.warning(f"Could not extract justification from FHIR report: {e}")

        # Add the radiologist report to our database
        db_add_rad_report(
            uid=exam_uid,
            report_id=study['id'],
            report_text=report['conclusion'],
            positive=-1,
            severity=-1,
            summary="",
            report_type='radio',
            radiologist=radiologist,
            justification=justification,
            model=MODEL_NAME,
            latency=-1
        )

        # Set the exam status to 'check' for LLM processing in queue
        db_set_status(exam_uid, "check")
    else:
        logging.debug(f"No conclusion found in diagnostic report for exam {exam_uid}")

async def query_retrieve_loop():
    """
    Periodically query the remote DICOM server for new studies.

    Runs an infinite loop that queries the remote PACS for new CR studies
    every 15 minutes (900 seconds). Can be disabled with the --no-query flag.
    Updates the next_query timestamp for dashboard display.
    """
    if NO_QUERY:
        logging.warning(f"Automatic Query/Retrieve disabled.")
    while not NO_QUERY:
        await query_and_retrieve()
        current_time = datetime.now()
        global next_query
        next_query = current_time + timedelta(seconds = 900)
        logging.info(f"Next Query/Retrieve at {next_query.strftime('%Y-%m-%d %H:%M:%S')}")
        await asyncio.sleep(900)


async def maintenance_loop():
    """
    Perform daily maintenance tasks including database cleanup and backup.

    Runs an infinite loop that performs daily maintenance operations:
    1. Purges old ignored/error records and their associated files
    2. Creates a timestamped backup of the database
    Waits 24 hours between runs.
    """
    while True:
        # Purge old ignored/error records
        db_purge_ignored_errors()

        # Create database backup
        try:
            db_backup()
        except Exception as e:
            logging.error(f"Database backup failed: {e}")

        # Wait for 24 hours
        await asyncio.sleep(86400)


def start_dicom_server():
    """
    Start the DICOM Storage SCP (Service Class Provider) server.

    Configures and starts the pynetdicom AE server that listens for incoming
    DICOM C-STORE requests. Supports Computed Radiography and Digital X-Ray
    storage SOP classes. Runs in a separate thread to avoid blocking the
    main asyncio event loop.
    """
    global dicom_server
    dicom_server = AE(ae_title = AE_TITLE)
    # Accept everything
    #ae.supported_contexts = StoragePresentationContexts
    # Accept only XRays
    dicom_server.add_supported_context(ComputedRadiographyImageStorage)
    dicom_server.add_supported_context(DigitalXRayImageStorageForPresentation)
    # C-Store handler
    handlers = [(evt.EVT_C_STORE, dicom_store)]
    logging.info(f"Starting DICOM server on port {AE_PORT} with AE Title '{AE_TITLE}'...")
    dicom_server.start_server(("0.0.0.0", AE_PORT), evt_handlers = handlers, block = False)


async def stop_servers():
    """
    Stop all servers gracefully during shutdown.

    Attempts to gracefully shutdown both the DICOM server and web server,
    handling any exceptions that may occur during the shutdown process.
    """
    global dicom_server, web_server
    # Stop DICOM server
    if dicom_server:
        try:
            dicom_server.shutdown()
            logging.info("DICOM server stopped.")
        except Exception as e:
            logging.error(f"Error stopping DICOM server: {e}")
    # Stop web server
    if web_server:
        try:
            await web_server.cleanup()
            logging.info("Web server stopped.")
        except Exception as e:
            logging.error(f"Error stopping web server: {e}")


def process_dicom_file(dicom_file, uid):
    """
    Process a DICOM file by extracting metadata, converting to PNG, and adding to queue.

    This helper function handles the common logic between dicom_store() and
    load_existing_dicom_files() to avoid code duplication.

    Args:
        dicom_file: Path to the DICOM file
        uid: Unique identifier for the exam
    """
    try:
        # Get the dataset
        ds = dcmread(dicom_file)
        # Get some info for queueing
        try:
            info = extract_dicom_metadata(ds)
        except Exception as e:
            logging.error(f"Error getting info {dicom_file}: {e}")
            db_set_status(uid, "error")
            return
        # Try to convert to PNG
        png_file = None
        try:
            png_file = convert_dicom_to_png(dicom_file)
        except Exception as e:
            logging.error(f"Error converting DICOM file {dicom_file}: {e}")
            db_set_status(uid, "error")
            return
        # Check the result
        if png_file:
            # Add to processing queue
            db_add_exam(info)
            # Notify the queue
            QUEUE_EVENT.set()
        else:
            # Set error status if no PNG was created
            db_set_status(uid, "error")
    except Exception as e:
        logging.error(f"Error processing DICOM file {dicom_file}: {e}")
        db_set_status(uid, "error")

def handle_error(e, context="", default_return=None, raise_on_error=False):
    """Unified error handling wrapper.

    Provides consistent error handling across the application with optional
    exception re-raising capabilities.

    Args:
        e (Exception): The exception that occurred
        context (str): Context information about where the error occurred
        default_return: Default value to return on error
        raise_on_error (bool): Whether to re-raise the exception

    Returns:
        The default_return value or re-raises the exception
    """
    error_msg = f"Error{f' in {context}' if context else ''}: {e}"
    logging.error(error_msg)

    if raise_on_error:
        raise e

    return default_return


async def main():
    """
    Main application entry point and orchestrator.

    Initializes the database, loads existing exams, starts all server
    components (DICOM, web dashboard, AI processing, health checks,
    query/retrieve, maintenance), loads existing DICOM files if requested,
    and manages the asyncio event loop. Handles graceful shutdown on
    KeyboardInterrupt.
    """
    # Main event loop
    global MAIN_LOOP
    MAIN_LOOP = asyncio.get_running_loop()
    # Init the database if not found
    if not os.path.exists(DB_FILE):
        logging.info("SQLite database not found. Creating a new one...")
        db_init()
    else:
        logging.info("SQLite database found.")
    # Print some data
    logging.info(f"Python SQLite version: {sqlite3.version}")
    logging.info(f"SQLite library version: {sqlite3.sqlite_version}")
    # Load exams
    exams, total = db_get_exams(status = 'done')
    logging.info(f"Loaded {len(exams)} exams from a total of {total}.")
    tasks = []
    # Start the DICOM server in a separate thread
    dicom_task = asyncio.create_task(asyncio.to_thread(start_dicom_server))
    tasks.append(dicom_task)
    # Start the tasks
    tasks.append(asyncio.create_task(start_dashboard()))
    tasks.append(asyncio.create_task(openai_health_check()))
    tasks.append(asyncio.create_task(relay_to_openai_loop()))
    tasks.append(asyncio.create_task(query_retrieve_loop()))
    tasks.append(asyncio.create_task(maintenance_loop()))
    tasks.append(asyncio.create_task(fhir_loop()))
    # Preload the existing dicom files
    if LOAD_DICOM:
        await load_existing_dicom_files()
    try:
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logging.info("Main task cancelled. Shutting down...")
        # Cancel all tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        # Wait for tasks to finish cancellation
        await asyncio.gather(*tasks, return_exceptions=True)


# Command run
if __name__ == '__main__':
    # Need to process the arguments
    parser = argparse.ArgumentParser(description = "XRayVision - Async DICOM processor with OpenAI and WebSocket dashboard")
    parser.add_argument("--keep-dicom", action = "store_true", default=KEEP_DICOM, help = "Do not delete .dcm files after conversion")
    parser.add_argument("--load-dicom", action = "store_true", default=LOAD_DICOM, help = "Load existing .dcm files in queue")
    parser.add_argument("--no-query", action = "store_true", default=NO_QUERY, help = "Do not query the DICOM server automatically")
    parser.add_argument("--enable-ntfy", action = "store_true", default=ENABLE_NTFY, help = "Enable ntfy.sh notifications")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name to use for analysis")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set logging level")
    args = parser.parse_args()
    # Store in globals
    KEEP_DICOM = args.keep_dicom
    LOAD_DICOM = args.load_dicom
    NO_QUERY = args.no_query
    ENABLE_NTFY = args.enable_ntfy
    MODEL_NAME = args.model
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Run
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("XRayVision stopped by user. Shutting down.")
    finally:
        # Stop all servers
        try:
            asyncio.run(stop_servers())
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
        logging.shutdown()
