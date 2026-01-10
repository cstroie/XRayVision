#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# XRayVision - Async DICOM processor with AI and WebSocket dashboard.
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
    Verification,
    ComputedRadiographyImageStorage,
    DigitalXRayImageStorageForPresentation,
    PatientRootQueryRetrieveInformationModelFind,
    PatientRootQueryRetrieveInformationModelMove,
    PatientRootQueryRetrieveInformationModelGet
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
        'REMOTE_AE_PORT': '104',
        'RETRIEVAL_METHOD': 'C-MOVE'
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
        'ENABLE_NTFY': 'False',
        'QUERY_INTERVAL': '300',
        'SEVERITY_THRESHOLD': '5'
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
        logging.debug("Local configuration loaded from local.cfg")
except Exception as e:
    logging.debug("Using default configuration values")

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
RETRIEVAL_METHOD = config.get('dicom', 'RETRIEVAL_METHOD')
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
  "confidence": integer from 0 to 100,
  "severity": integer from 0 to 10,
  "summary": "diagnosis in 1-3 words"
}

CRITICAL RULES:
- Output ONLY the JSON object - no additional text, explanations, or apologies before or after
- The "short" field must be exactly "yes" or "no" (lowercase, in quotes)
- "yes" means pathological findings are present
- "no" means no significant findings detected
- The "confidence" field must be a number between 0-100 (no quotes)
- The "severity" field must be a number between 0-10 (no quotes), where 0 is normal and 10 is critical
- The "summary" field must be a brief diagnosis in 1-3 words, focusing on major category classifications (e.g., "pneumonia", "fracture", "normal", "interstitial infiltrate")
- Use double quotes for all keys and string values
- Properly escape special characters in the report string

EXAMPLES:

Example 1 - Chest X-ray with pneumonia:
Input: Chest X-ray, patient with cough and fever
Output: {"short": "yes", "report": "Consolidation in the right lower lobe consistent with pneumonia. No pleural effusion or pneumothorax. Heart size normal.", "confidence": 92, "severity": 6, "summary": "pneumonia"}

Example 2 - Normal chest X-ray:
Input: Chest X-ray, routine screening
Output: {"short": "no", "report": "Clear lung fields bilaterally. No consolidation, pleural effusion, or pneumothorax. Cardiac silhouette within normal limits. No acute bony abnormalities.", "confidence": 95, "severity": 0, "summary": "normal"}

Example 3 - Abdominal X-ray with uncertain findings:
Input: Abdominal X-ray, abdominal pain
Output: {"short": "yes", "report": "Dilated small bowel loops measuring up to 3.5 cm with air-fluid levels, concerning for possible small bowel obstruction. No free air under the diaphragm. Limited assessment of solid organs on plain film.", "confidence": 78, "severity": 7, "summary": "bowel obstruction"}

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

SEVERITY SCORING:
- 0: Normal findings
- 1-3: Minimal abnormalities, no immediate clinical concern
- 4-6: Moderate findings requiring clinical correlation
- 7-8: Significant abnormalities requiring prompt attention
- 9-10: Critical findings requiring immediate intervention

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
ANALYZE EACH SENTENCE SEPARATELY and identify any pathological findings, even if other aspects are described as normal.

OUTPUT FORMAT (JSON):
{
  "pathologic": "yes/no",
  "severity": 1-10,
  "summary": "1-3 words"
}

RULES:
- "pathologic": "yes" if ANY pathological finding exists, otherwise "no"
- "severity": 1=minimal, 5=moderate, 10=critical/urgent
- "summary": diagnosis in 1-3 words, focusing on major category classifications (e.g., "fracture", "pneumonia", "interstitial infiltrate")
- If everything is normal: {"pathologic": "no", "severity": 0, "summary": "normal"}
- CRITICAL: Analyze each sentence separately - if ANY sentence describes a pathological finding, mark as pathologic
- Ignore spelling errors
- Note: In Romanian reports, "fără" and "fara" mean "no" or "without"
- Note: In Romanian reports, "SCD" means "costo-diaphragmatic sinuses" 
- Note: In Romanian reports, "liber" means "clear" or "free"
- Respond ONLY with the JSON, without additional text

EXAMPLES:

Report: "Hazy opacity in the left mid lung field, possibly representing consolidation or infiltrate."
Response: {"pathologic": "yes", "severity": 6, "summary": "pneumonia"}

Report: "No pathological changes. Heart of normal size."
Response: {"pathologic": "no", "severity": 0, "summary": "normal"}

Report: "Fără semne de fractură sau leziuni osteolitice."
Response: {"pathologic": "no", "severity": 0, "summary": "normal"}

Report: "SCD libere, fără lichid pleural."
Response: {"pathologic": "no", "severity": 0, "summary": "normal"}

Report: "Proces de condensare paracardiac dreapta. SCD libere. Cord normal"
Response: {"pathologic": "yes", "severity": 7, "summary": "pneumonia"}
""")

ANA_PROMPT = ("""
You are a senior radiologist providing detailed analysis of radiology reports.

TASK: Perform a three-pass critical analysis of the radiology report:

FIRST PASS - Overview and Context:
- Identify the main clinical topic and purpose of the report
- Determine the primary findings and overall conclusion

SECOND PASS - Detailed Content Analysis:
- Summarize the main points and key findings
- Identify supporting evidence and clinical observations
- Evaluate if the conclusions logically follow from the findings

THIRD PASS - Critical Evaluation:
- Identify and challenge every statement and assumption
- Point out any implicit assumptions or missing information
- Evaluate potential issues with interpretations or missing citations to standard practices

OUTPUT FORMAT (JSON):
{
  "first_pass": {
    "topic": "main clinical topic",
    "purpose": "purpose of the examination",
    "primary_findings": "overall primary findings"
  },
  "second_pass": {
    "main_points": ["key finding 1", "key finding 2"],
    "supporting_evidence": ["evidence 1", "evidence 2"],
    "conclusions_valid": true/false
  },
  "third_pass": {
    "assumptions": ["assumption 1", "assumption 2"],
    "missing_info": ["missing information 1", "missing information 2"],
    "critique": "detailed critique of the report"
  },
  "overall_assessment": "overall quality and completeness assessment"
}

RULES:
- Provide detailed, professional radiological analysis
- Be factual and constructive in your critique
- Focus on clinical relevance and report quality
- Use clear, concise language
- Respond ONLY with the JSON, without additional text
- IMPORTANT: Properly escape all special characters in JSON strings, especially double quotes (") should be escaped as (\")
- Ensure all JSON keys and string values are properly quoted with double quotes
- Do not include any text before or after the JSON object
""")

TRN_PROMPT = ("""
You are a professional medical translator specializing in radiology reports.

TASK: Translate the Romanian radiology report into English.

OUTPUT FORMAT:
[English translation of the report]

RULES:
- Translate the entire report text from Romanian to English
- Maintain all medical terminology and anatomical references
- Preserve the original meaning and clinical context
- Use professional medical English terminology
- Keep the same structure and formatting as the original
- Do not add any explanations, comments, or additional text
- Respond ONLY with the translation text, no JSON, no formatting, no additional content
- Do not include any text before or after the translation

EXAMPLES:

Romanian: "SCD libere, fără lichid pleural."
English: Clear costo-diaphragmatic sinuses, no pleural effusion.

Romanian: "Proces de condensare paracardiac dreapta."
English: Right paracardiac consolidation process.

Romanian: "Fără semne de fractură sau leziuni osteolitice."
English: No signs of fracture or osteolytic lesions.
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

# External API health
active_openai_url = None  # Currently active AI API endpoint
health_status = {
    OPENAI_URL_PRIMARY: False,  # Health status of primary AI endpoint
    OPENAI_URL_SECONDARY: False,  # Health status of secondary AI endpoint
    FHIR_URL: False  # Health status of FHIR endpoint
}
# AI timings
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
ENABLE_HIS = config.getboolean('processing', 'ENABLE_HIS')   # Whether to enable HIS/FHIR integration
QUERY_INTERVAL = config.getint('processing', 'QUERY_INTERVAL')  # Base interval for query/retrieve in seconds
SEVERITY_THRESHOLD = config.getint('processing', 'SEVERITY_THRESHOLD')  # Severity threshold for correctness calculation

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
    'queue_size': 0,        # Number of exams waiting in the processing queue (queued + requeue)
    'check_queue_size': 0,  # Number of exams queued for FHIR report checking
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
        - birthdate (TEXT): Patient birth date (YYYY-MM-DD format)
        - sex (TEXT): Patient sex ('M', 'F', or 'O')
    
    exams:
        - uid (TEXT, PRIMARY KEY): Unique exam identifier (SOP Instance UID)
        - cnp (TEXT, FOREIGN KEY): References patients.cnp
        - id (TEXT): service request ID from HIS
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
    with sqlite3.connect(DB_FILE, isolation_level=None) as conn:
        # Configure SQLite for concurrent access
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('PRAGMA cache_size = 10000')
        conn.execute('PRAGMA temp_store = MEMORY')
        conn.execute('PRAGMA mmap_size = 268435456')  # 256MB
        
        # Create tables within a transaction
        try:
            conn.execute('BEGIN IMMEDIATE')
            
            # Patients table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS patients (
                    cnp TEXT PRIMARY KEY,
                    id TEXT,
                    name TEXT,
                    birthdate TEXT,
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
                    severity INTEGER DEFAULT -1 CHECK(severity BETWEEN -1 AND 10),
                    summary TEXT,
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
                    text_en TEXT,
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
            
            conn.commit()
            logging.info("Initialized SQLite database with normalized schema.")
        except Exception as e:
            conn.rollback()
            logging.error(f"Failed to initialize database: {e}")
            raise


def db_execute_query(query: str, params: tuple = (), fetch_mode: str = 'all') -> Optional[list]:
    """Execute a database query and return results.

    Executes a parameterized SQL query with proper transaction isolation
    for concurrent operations.

    Args:
        query (str): SQL query to execute
        params (tuple): Query parameters
        fetch_mode (str): 'all', 'one', or 'none' for fetchall(), fetchone(), or no fetch

    Returns:
        Query results based on fetch_mode, or None on error
    """
    with sqlite3.connect(DB_FILE, isolation_level=None) as conn:
        # Configure SQLite for concurrent access
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.execute('PRAGMA foreign_keys = ON')
        
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
            conn.rollback()
            return handle_error(e, "database query execution", None, raise_on_error=False)


def db_execute_query_retry(query: str, params: tuple = (), max_retries: int = 5) -> Optional[int]:
    """Execute a database query with retry logic.

    Executes a database query with exponential backoff retry logic in case
    of failures, with proper transaction isolation for concurrent operations.

    Args:
        query (str): SQL query to execute
        params (tuple): Query parameters
        max_retries (int): Maximum number of retry attempts

    Returns:
        Number of affected rows or None on error
    """
    with sqlite3.connect(DB_FILE, isolation_level=None) as conn:
        # Configure SQLite for concurrent access
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.execute('PRAGMA foreign_keys = ON')
        
        for attempt in range(max_retries):
            try:
                conn.execute('BEGIN IMMEDIATE')
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                return cursor.rowcount
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    # Use sync sleep for synchronous function
                    import time
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                    continue
                conn.rollback()
                return handle_error(e, "database query with retry", None, raise_on_error=False)
            except Exception as e:
                conn.rollback()
                return handle_error(e, "database query with retry", None, raise_on_error=False)
    return None


# Cache for db_analyze results
_db_analyze_cache = {}


def db_analyze(table_name):
    """
    Analyze a table to get its primary key and columns using PRAGMA table_info.
    
    Args:
        table_name: Name of the table to analyze
        
    Returns:
        tuple: (primary_key, columns) where primary_key is the name of the 
               primary key column and columns is a list of all column names
    """
    # Check cache first
    if table_name in _db_analyze_cache:
        return _db_analyze_cache[table_name]
    
    query = f"PRAGMA table_info({table_name})"
    rows = db_execute_query(query, fetch_mode='all')
    
    if not rows:
        return None, []
    
    primary_key = None
    columns = []
    
    for row in rows:
        # row format: (cid, name, type, notnull, dflt_value, pk)
        cid, name, type, notnull, dflt_value, pk = row
        columns.append(name)
        if pk:  # pk is 1 if this column is part of the primary key
            primary_key = name
    
    # Cache the result
    result = (primary_key, columns)
    _db_analyze_cache[table_name] = result
    return result

def db_unpack_result(result: list, keys: list) -> dict:
    """
    Unpack a database result list into a dictionary using provided keys.

    Args:
        result: List of values from a database query result
        keys: List of keys to map to the values

    Returns:
        dict: Dictionary with keys mapped to corresponding values
    """
    if not result or not keys:
        return {}
    return dict(zip(keys, result))


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


def db_create_select_query(table_name, *columns, where=None, order_by=None, asc=True, limit=None):
    """
    Convenience function to build SELECT query strings.

    Args:
        table_name: Name of the table to select from
        *columns: Variable number of column names (use '*' for all columns)
        where: Optional WHERE clause (without the WHERE keyword)
        order_by: Optional column to order by
        asc: Boolean indicating ascending (True) or descending (False) order
        limit: Optional limit on number of rows returned

    Returns:
        str: Formatted SQL query string
    """
    if not columns:
        columns_str = '*'
    else:
        columns_str = ', '.join(columns)
    
    query = f'SELECT {columns_str} FROM {table_name}'
    if where:
        query += f' WHERE {where}'
    if order_by:
        query += f' ORDER BY {order_by}'
        if not asc:
            query += ' DESC'
    if limit:
        query += f' LIMIT {limit}'
    return query


def db_select(table_name, columns=None, where_clause=None, where_params=None, limit=None, order_by=None, asc=True):
    """
    Convenience function to select records from a table and return as list of dictionaries.

    Args:
        table_name: Name of the table to select from
        columns: List of column names to select (None for all columns)
        where_clause: Optional WHERE clause (without the WHERE keyword)
        where_params: Parameters for the WHERE clause
        limit: Optional limit on number of rows returned
        order_by: Optional column to order by
        asc: Boolean indicating ascending (True) or descending (False) order

    Returns:
        list: List of dictionaries representing the selected rows
    """
    # Use all columns if none specified
    if columns is None:
        # Get all columns from table schema
        _, all_columns = db_analyze(table_name)
        columns = all_columns
    
    # Build query
    query = db_create_select_query(table_name, *columns, where=where_clause, order_by=order_by, asc=asc, limit=limit)
    
    # Execute query
    params = where_params if where_params else ()
    rows = db_execute_query(query, params, fetch_mode='all')
    
    # Convert to list of dictionaries
    if rows:
        return [db_unpack_result(row, columns) for row in rows]
    return []


def db_count(table_name, where_clause=None, where_params=None):
    """
    Convenience function to count records in a table.

    Args:
        table_name: Name of the table to count records in
        where_clause: Optional WHERE clause (without the WHERE keyword)
        where_params: Parameters for the WHERE clause

    Returns:
        int: Number of records matching the criteria
    """
    query = f"SELECT COUNT(*) FROM {table_name}"
    params = ()
    
    if where_clause:
        query += f" WHERE {where_clause}"
        params = where_params if where_params else ()
    
    result = db_execute_query(query, params, fetch_mode='one')
    return result[0] if result else 0


def db_update(table_name, where_clause, where_params, **kwargs):
    """
    Convenience function to update records in a table.

    Args:
        table_name: Name of the table to update
        where_clause: WHERE clause (without the WHERE keyword)
        where_params: Parameters for the WHERE clause
        **kwargs: Column-value pairs to update

    Returns:
        int: Number of rows affected
    """
    if not kwargs:
        return 0
    
    # Build SET clause
    set_columns = list(kwargs.keys())
    set_values = list(kwargs.values())
    set_clause = ', '.join([f'{col} = ?' for col in set_columns])
    
    # Build query
    query = f'UPDATE {table_name} SET {set_clause} WHERE {where_clause}'
    params = set_values + list(where_params)
    
    return db_execute_query_retry(query, tuple(params))


def db_insert(table_name, **kwargs):
    """
    Convenience function to insert records into a table.

    Args:
        table_name: Name of the table to insert into
        **kwargs: Column-value pairs to insert

    Returns:
        int: Number of rows affected
    """
    if not kwargs:
        return 0
    
    # Build query using db_create_insert_query
    columns = list(kwargs.keys())
    values = list(kwargs.values())
    query = db_create_insert_query(table_name, *columns)
    params = tuple(values)
    
    return db_execute_query_retry(query, params)


def db_select_one(table_name, pk_value):
    """
    Convenience function to get a single record from a table using its primary key.

    Args:
        table_name: Name of the table to select from
        pk_value: Value of the primary key to search for

    Returns:
        dict: Record data or None if not found
    """
    # Get primary key and all columns for the table
    primary_key, columns = db_analyze(table_name)
    
    if not primary_key or not columns:
        return None
    
    # Create query using the primary key
    where_clause = f"{primary_key} = ?"
    query = db_create_select_query(table_name, *columns, where=where_clause)
    params = (pk_value,)
    
    result = db_execute_query(query, params, fetch_mode='one')
    if result:
        return db_unpack_result(result, columns)
    return None


def db_add_patient(cnp, id, name, birthdate, sex):
    """
    Add a new patient to the database or update existing patient information.

    Args:
        cnp: Romanian personal identification number (primary key)
        id: Patient ID from hospital system
        name: Patient full name
        birthdate: Patient birth date (YYYY-MM-DD format)
        sex: Patient sex ('M', 'F', or 'O')
    """
    query = db_create_insert_query('patients', 'cnp', 'id', 'name', 'birthdate', 'sex')
    params = (cnp, id, name, birthdate, sex)
    return db_execute_query_retry(query, params)


def db_add_ai_report(uid, report_text, positive, confidence, model, latency, severity=None, summary=None):
    """
    Add or update an AI report entry in the database.

    Args:
        uid: Exam unique identifier
        report_text: AI-generated report content
        positive: AI prediction result (True/False)
        confidence: AI confidence score (0-100)
        model: Name of the model used
        latency: Processing time in seconds
        severity: Severity score (0-10, -1 if not assessed)
        summary: Brief summary of findings
    """
    db_insert('ai_reports',
              uid=uid,
              text=report_text,
              positive=int(positive),
              confidence=confidence if confidence is not None else -1,
              severity=severity if severity is not None else -1,
              summary=summary,
              model=model,
              latency=latency)


def db_add_rad_report(uid, report_id, report_text, positive, severity, summary, report_type, radiologist, justification, model, latency, text_en=None):
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
        text_en: English translation of the report (optional)
    """
    db_insert('rad_reports',
              uid=uid,
              id=report_id,
              text=report_text,
              text_en=text_en,
              positive=positive,
              severity=severity,
              summary=summary,
              type=report_type,
              radiologist=radiologist,
              justification=justification,
              model=model,
              latency=latency)

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
    db_update('rad_reports', 'uid = ?', (uid,), 
              positive=positive, 
              severity=severity, 
              summary=summary, 
              model=model, 
              latency=latency)

def db_get_exams_without_rad_report():
    """
    Get all exams for a patient that don't have radiologist reports yet.
    
    First identifies a patient with at least one exam without a radiologist report,
    then retrieves all exams for that patient without radiologist reports.
    Newer exams have a higher chance of being selected. If no exam is found in the initial
    time window, the window is doubled (up to 52 weeks) and tried once more.

    Returns:
        dict: Dictionary with patient info and list of exams, or empty dict if none found
    """
    # Randomly choose number of weeks (1-52) for initial search
    weeks = random.randint(1, 52)
    
    # Try three - first with random weeks, then double if no results
    for attempt in range(4):
        # Calculate cutoff date
        # TODO Temporary fix for selecting older exams 
        cutoff_date = datetime.now() - timedelta(weeks=52-weeks)
        cutoff_date_str = cutoff_date.strftime('%Y-%m-%d %H:%M:%S')
        
        # First, find a patient with at least one exam without a radiologist report
        patient_query = """
            SELECT DISTINCT p.cnp, p.name, p.id, p.birthdate, p.sex
            FROM exams e
            INNER JOIN patients p ON e.cnp = p.cnp
            LEFT JOIN rad_reports rr ON e.uid = rr.uid
            WHERE (rr.severity IS NULL OR rr.severity = -1)
            AND (rr.id IS NULL OR rr.id > 0)
            AND e.status = 'done'
            AND e.created >= ?
            ORDER BY RANDOM()
            LIMIT 1
        """
        patient_row = db_execute_query(patient_query, (cutoff_date_str,), fetch_mode='one')
        
        if patient_row:
            patient_cnp, patient_name, patient_id, patient_birthdate, patient_sex = patient_row
            
            # Now get all exams for this patient without radiologist reports
            exams_query = """
                SELECT 
                    e.uid, e.created, e.protocol, e.region, e.status, e.type, e.study, e.series, e.id
                FROM exams e
                LEFT JOIN rad_reports rr ON e.uid = rr.uid
                WHERE e.cnp = ?
                AND (rr.severity IS NULL OR rr.severity = -1)
                AND e.status = 'done'
                ORDER BY e.created DESC
            """
            exam_rows = db_execute_query(exams_query, (patient_cnp,), fetch_mode='all')
            
            if exam_rows:
                # Calculate age from birthdate if available
                patient_age = -1
                if patient_birthdate:
                    try:
                        birth_date = datetime.strptime(patient_birthdate, "%Y-%m-%d")
                        today = datetime.now()
                        patient_age = today.year - birth_date.year
                        if (today.month, today.day) < (birth_date.month, birth_date.day):
                            patient_age -= 1
                    except ValueError:
                        patient_age = -1
                        
                result = {
                    'patient': {
                        'name': patient_name,
                        'cnp': patient_cnp,
                        'id': patient_id,
                        'age': patient_age,
                        'birthdate': patient_birthdate,
                        'sex': patient_sex,
                    },
                    'exams': []
                }
                
                for row in exam_rows:
                    # Unpack row into named variables for better readability
                    (uid, exam_created, exam_protocol, exam_region, exam_status, exam_type, exam_study, exam_series, exam_id) = row
                    
                    result['exams'].append({
                        'uid': uid,
                        'created': exam_created,
                        'protocol': exam_protocol,
                        'region': exam_region,
                        'status': exam_status,
                        'type': exam_type,
                        'study': exam_study,
                        'series': exam_series,
                        'id': exam_id,
                    })
                return result
        
        # If no exam found, double the interval for second attempt (max 52 weeks)
        if attempt == 0:
            weeks = min(weeks * 2, 52)
    
    # No exams found after several attempts
    return {}

def db_update_patient_id(cnp, patient_id):
    """
    Update patient ID in the database.

    Args:
        cnp: Patient CNP
        patient_id: Patient ID from HIS
    """
    db_update('patients', 'cnp = ?', (cnp,), id=patient_id)


def db_add_exam(info):
    """
    Add or update an exam entry in the database.

    This function handles queuing new exams for processing. It sets status to 'queued'
    and stores exam metadata. Patient information is stored in the patients table.
    If report data is provided, it also creates an entry in the ai_reports table.
    If justification is provided, it creates an entry in the rad_reports table.

    Args:
        info: Dictionary containing exam metadata (uid, patient info, exam details)
    """
    # Add or update patient information
    patient = info["patient"]
    db_add_patient(
        patient["cnp"],
        patient.get("id",""),
        patient["name"],
        patient.get("birthdate", None),
        patient["sex"]
    )
    
    # Set status to queued for new exams
    status = 'queued'
    
    # Insert into database
    exam = info["exam"]
    db_insert('exams',
              uid=info['uid'],
              cnp=patient["cnp"],
              id=exam.get("id",""),
              created=exam['created'],
              protocol=exam["protocol"],
              region=exam['region'],
              type=exam.get("type", "CR"),
              status=status,
              study=exam.get("study"),
              series=exam.get("series"))


def db_get_exams(limit = PAGE_SIZE, offset = 0, **filters):
    """
    Load exams from the database with optional filters and pagination.

    Retrieves exams with associated patient information, AI reports, and radiologist
    reports. Calculates correctness based on agreement between AI and radiologist
    predictions. This function performs complex JOIN operations across multiple tables
    and supports extensive filtering capabilities for the dashboard interface.

    The function builds dynamic SQL queries based on provided filters and uses
    parameterized queries to prevent SQL injection. It calculates correctness metrics
    by comparing AI and radiologist severity scores against the configured threshold.

    Args:
        limit (int): Maximum number of exams to return (default: PAGE_SIZE)
        offset (int): Number of exams to skip for pagination (default: 0)
        **filters: Optional filters for querying exams:
            - reviewed (int): Filter by review status (0/1) - reviewed if severity > -1
            - positive (int): Filter by AI prediction (0/1) based on severity threshold
            - correct (int): Filter by correctness status (0/1) - agreement between AI and radiologist
            - region (str): Filter by anatomic region (case-insensitive partial match)
            - status (str or list): Filter by processing status (case-insensitive exact
              match or list of statuses)
            - search (str): Filter by patient name, CNP, patient ID, or UID (case-insensitive
              partial match for name/CNP/ID, exact for UID)
            - diagnostic (str): Filter by radiologist diagnostic summary (case-insensitive exact match)
            - radiologist (str): Filter by radiologist name (case-insensitive exact match)
            - uid (str): Filter by exam UID (exact match)
            - cnp (str): Filter by patient CNP (exact match)
            - severity (str): Filter by radiologist severity with interval notation:
                - "3-6": Severity between 3 and 6 (inclusive)
                - "-8": Severity from 0 to 8 (inclusive)
                - "2-": Severity from 2 to 10 (inclusive)
                - "5": Exact severity of 5

    Returns:
        tuple: (exams_list, total_count) where:
            - exams_list (list): List of exam dictionaries containing:
                * uid: Exam unique identifier
                * patient: Patient information (name, cnp, age, birthdate, sex, id)
                * exam: Exam details (created, date, time, protocol, region, status, type, study, series, id)
                * report: Report data with AI and radiologist findings, correctness metrics
            - total_count (int): Total number of exams matching the filters (for pagination)

    Database Schema:
        This function JOINs four tables:
        - exams (e): Core exam information with status tracking
        - patients (p): Patient demographics linked by CNP
        - ai_reports (ar): AI-generated reports linked by exam UID
        - rad_reports (rr): Radiologist reports linked by exam UID

    Correctness Calculation:
        - correct = 1: True positives (both AI and radiologist positive) or 
                     True negatives (both AI and radiologist negative)
        - correct = 0: False positives (AI positive, radiologist negative) or
                     False negatives (AI negative, radiologist positive)
        - correct = -1: Not reviewed (radiologist severity = -1 or NULL)

    Performance Considerations:
        - Uses parameterized queries to prevent SQL injection
        - Leverages database indexes on status, region, cnp, and created columns
        - Applies LIMIT/OFFSET for efficient pagination
        - Calculates age dynamically from birthdate for display
    """
    conditions = []
    params = []

    # Build dynamic WHERE conditions based on provided filters
    # Each condition is parameterized to prevent SQL injection
    if 'reviewed' in filters:
        if filters['reviewed'] == 1:
            conditions.append("rr.severity > -1")
        else:
            conditions.append("(rr.severity = -1 OR rr.severity IS NULL)")
    if 'positive' in filters:
        conditions.append("ar.severity >= ?")
        params.append(SEVERITY_THRESHOLD if filters['positive'] == 1 else 0)
    if 'correct' in filters:
        if filters['correct'] == 1:
            # Correct predictions (TP or TN)
            conditions.append("((rr.severity = -1 OR rr.severity IS NULL) OR (ar.severity >= ? AND rr.severity >= ?) OR (ar.severity < ? AND rr.severity < ?))")
            params.extend([SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD])
        else:
            # Incorrect predictions (FP or FN)
            conditions.append("((ar.severity >= ? AND rr.severity < ? AND rr.severity > -1) OR (ar.severity < ? AND rr.severity >= ?))")
            params.extend([SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD])
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
        conditions.append("(LOWER(p.name) LIKE ? OR LOWER(p.cnp) LIKE ? OR LOWER(p.id) LIKE ? OR e.uid LIKE ?)")
        search_term = f"%{filters['search']}%"
        params.extend([search_term, search_term, search_term, search_term])
    if 'diagnostic' in filters:
        conditions.append("LOWER(rr.summary) = LOWER(?)")
        params.append(filters['diagnostic'])
    if 'radiologist' in filters:
        conditions.append("LOWER(rr.radiologist) = LOWER(?)")
        params.append(filters['radiologist'])
    if 'uid' in filters:
        conditions.append("e.uid = LOWER(?)")
        params.append(filters['uid'])
    if 'cnp' in filters:
        conditions.append("p.cnp = ?")
        params.append(filters['cnp'])
    if 'severity' in filters:
        severity_value = str(filters['severity']).strip()
        # Handle interval notation: "3-6", "-8", "2-"
        if '-' in severity_value and severity_value != '-':
            parts = severity_value.split('-', 1)
            try:
                if parts[0] == '':  # "-8" format (0 to 8)
                    upper = int(parts[1])
                    conditions.append("rr.severity >= ? AND rr.severity <= ?")
                    params.extend([0, upper])
                elif parts[1] == '':  # "2-" format (2 to 10)
                    lower = int(parts[0])
                    conditions.append("rr.severity >= ? AND rr.severity <= ?")
                    params.extend([lower, 10])
                else:  # "3-6" format (3 to 6)
                    lower = int(parts[0])
                    upper = int(parts[1])
                    conditions.append("rr.severity >= ? AND rr.severity <= ?")
                    params.extend([lower, upper])
            except ValueError:
                # If parsing fails, treat as exact value
                try:
                    exact_val = int(severity_value)
                    conditions.append("rr.severity = ?")
                    params.append(exact_val)
                except ValueError:
                    pass  # Invalid severity value, ignore filter
        else:
            # Handle single number
            try:
                exact_val = int(severity_value)
                conditions.append("rr.severity = ?")
                params.append(exact_val)
            except ValueError:
                pass  # Invalid severity value, ignore filter

    # Build WHERE clause
    where = ""
    if conditions:
        where = "WHERE " + " AND ".join(conditions)

    # Apply the limits (pagination)
    query = f"""
        SELECT
            e.uid, e.created, e.protocol, e.region, e.status, e.type, e.study, e.series, e.id,
            p.name, p.cnp, p.id, p.birthdate, p.sex,
            ar.created, ar.text, ar.updated, ar.confidence, ar.severity, ar.summary, ar.model, ar.latency,
            rr.text, rr.text_en, rr.severity, rr.summary, rr.created, rr.updated, rr.id, rr.type, rr.radiologist, rr.justification, rr.model, rr.latency,
            CASE
                WHEN (rr.severity = -1 OR rr.severity IS NULL) THEN -1
                WHEN (ar.severity >= ? AND rr.severity >= ?) OR (ar.severity < ? AND rr.severity < ?) THEN 1
                ELSE 0
            END AS correct,
            CASE
                WHEN rr.severity > -1 THEN 1
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
    all_params = [SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD] + params + [limit, offset]

    # Get the exams
    exams = []
    rows = db_execute_query(query, tuple(all_params), fetch_mode='all')
    if rows:
        for row in rows:
            # Unpack row into named variables for better readability
            (uid, exam_created, exam_protocol, exam_region, exam_status, exam_type, exam_study, exam_series, exam_id,
             patient_name, patient_cnp, patient_id, patient_birthdate, patient_sex,
             ai_created, ai_text, ai_updated, ai_confidence, ai_severity, ai_summary, ai_model, ai_latency,
             rad_text, rad_text_en, rad_severity, rad_summary, rad_created, rad_updated, rad_id, rad_type, rad_radiologist, rad_justification, rad_model, rad_latency,
             correct, reviewed) = row
                
            dt = datetime.strptime(exam_created, "%Y-%m-%d %H:%M:%S")
            # Calculate age from birthdate if available
            patient_age = -1
            if patient_birthdate:
                try:
                    birth_date = datetime.strptime(patient_birthdate, "%Y-%m-%d")
                    today = datetime.now()
                    patient_age = today.year - birth_date.year
                    if (today.month, today.day) < (birth_date.month, birth_date.day):
                        patient_age -= 1
                except ValueError:
                    patient_age = -1
                    
            exams.append({
                'uid': uid,
                'patient': {
                    'name': patient_name,
                    'cnp': patient_cnp,
                    'id': patient_id,
                    'age': patient_age,
                    'birthdate': patient_birthdate,
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
                        'short': 'yes' if ai_severity is not None and ai_severity >= SEVERITY_THRESHOLD else 'no',
                        'created': ai_created,
                        'updated': ai_updated,
                        'positive': ai_severity is not None and ai_severity >= SEVERITY_THRESHOLD,
                        'confidence': ai_confidence,
                        'severity': ai_severity,
                        'summary': ai_summary,
                        'model': ai_model,
                        'latency': ai_latency,
                    },
                    'rad': {
                        'text': rad_text,
                        'text_en': rad_text_en,
                        'positive': rad_severity is not None and rad_severity >= SEVERITY_THRESHOLD,
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
    total_row = db_execute_query(count_query, tuple(params), fetch_mode='one')
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
        AND ar.severity >= 0
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
    results = db_select('exams', ['status'], where_clause='uid = ? AND status IN (?, ?, ?, ?)', 
                       where_params=(uid, 'done', 'queued', 'requeue', 'processing'))
    return len(results) > 0


def db_check_study_exists(study_uid):
    """
    Check if a study is already in the database.

    Args:
        study_uid: Study Instance UID to check
        series_uid: Unused parameter for compatibility

    Returns:
        bool: True if study exists in the database
    """
    # Check for any exam with this study UID
    results = db_select('exams', ['uid'], where_clause='study = ?', 
                       where_params=(study_uid,))
    return len(results) > 0


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
            SUM(CASE WHEN rr.severity > -1 THEN 1 ELSE 0 END) AS reviewed
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
            SUM(CASE WHEN (ar.severity >= ? AND rr.severity >= ?) THEN 1 ELSE 0 END) AS tpos,
            SUM(CASE WHEN (ar.severity < ? AND rr.severity < ? AND rr.severity > -1) THEN 1 ELSE 0 END) AS tneg,
            SUM(CASE WHEN (ar.severity >= ? AND rr.severity < ? AND rr.severity > -1) THEN 1 ELSE 0 END) AS fpos,
            SUM(CASE WHEN (ar.severity < ? AND rr.severity >= ?) THEN 1 ELSE 0 END) AS fneg
        FROM exams e
        LEFT JOIN ai_reports ar ON e.uid = ar.uid
        LEFT JOIN rad_reports rr ON e.uid = rr.uid
        WHERE e.status LIKE 'done'
          AND ar.severity IS NOT NULL;
    """
    metrics_row = db_execute_query(query, (SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD), fetch_mode='one')
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

    # Get processing time statistics (last day only) - using AI report latency
    query = """
        SELECT
            AVG(CAST(ar.latency AS REAL)) AS avg_processing_time,
            COUNT(*) * 1.0 / (SUM(CAST(ar.latency AS REAL)) + 1) AS throughput
        FROM exams e
        LEFT JOIN ai_reports ar ON e.uid = ar.uid
        WHERE e.status LIKE 'done'
          AND ar.latency IS NOT NULL
          AND ar.latency >= 0
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
                SUM(CASE WHEN rr.severity > -1 THEN 1 ELSE 0 END) AS reviewed,
                SUM(CASE WHEN ar.severity >= ? THEN 1 ELSE 0 END) AS positive,
                SUM(CASE WHEN (ar.severity >= ? AND rr.severity >= ?) THEN 1 ELSE 0 END) AS tpos,
                SUM(CASE WHEN (ar.severity < ? AND rr.severity < ? AND rr.severity > -1) THEN 1 ELSE 0 END) AS tneg,
                SUM(CASE WHEN (ar.severity >= ? AND rr.severity < ? AND rr.severity > -1) THEN 1 ELSE 0 END) AS fpos,
                SUM(CASE WHEN (ar.severity < ? AND rr.severity >= ?) THEN 1 ELSE 0 END) AS fneg
        FROM exams e
        LEFT JOIN ai_reports ar ON e.uid = ar.uid
        LEFT JOIN rad_reports rr ON e.uid = rr.uid
        WHERE e.status LIKE 'done'
          AND ar.severity IS NOT NULL
        GROUP BY e.region
    """
    region_data = db_execute_query(query, (SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD), fetch_mode='all')
    if region_data:
        for row in region_data:
            (region, total, reviewed, positive, tpos, tneg, fpos, fneg) = row
            region = region or 'unknown'
            stats["region"][region] = {
                "total": total,
                "reviewed": reviewed,
                "positive": positive,
                "correct": tpos + tneg,
                "wrong": fpos + fneg,
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
               SUM(CASE WHEN ar.severity >= ? THEN 1 ELSE 0 END) as positive
        FROM exams e
        LEFT JOIN ai_reports ar ON e.uid = ar.uid
        WHERE e.status LIKE 'done'
          AND e.created >= date('now', '-30 days')
        GROUP BY DATE(e.created), e.region
        ORDER BY date
    """
    trends_data = db_execute_query(query, (SEVERITY_THRESHOLD,), fetch_mode='all')
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
               SUM(CASE WHEN ar.severity >= ? THEN 1 ELSE 0 END) as positive
        FROM exams e
        LEFT JOIN ai_reports ar ON e.uid = ar.uid
        WHERE e.status LIKE 'done'
          AND e.created >= date('now', '-12 months')
        GROUP BY strftime('%Y-%m', e.created), e.region
        ORDER BY month
    """
    monthly_trends_data = db_execute_query(query, (SEVERITY_THRESHOLD,), fetch_mode='all')
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
    return db_count('exams', where_clause="status IN (?, ?)", where_params=('queued', 'requeue'))


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


def db_get_ai_report(uid):
    """
    Get AI report for a specific exam.

    Args:
        uid: Unique identifier of the exam

    Returns:
        dict: Report data or None if not found
    """
    result = db_select_one('ai_reports', uid)
    if result:
        # Ensure severity and summary fields are present even if None
        if 'severity' not in result:
            result['severity'] = None
        if 'summary' not in result:
            result['summary'] = None
    return result


def db_get_rad_report(uid):
    """
    Get radiologist report for a specific exam.

    Args:
        uid: Unique identifier of the exam

    Returns:
        dict: Report data or None if not found
    """
    return db_select_one('rad_reports', uid)


def db_have_rad_reports(uid):
    """
    Check if a radiologist report already exists for a given exam UID.

    Args:
        uid: Unique identifier of the exam

    Returns:
        bool: True if a radiologist report exists for this UID, False otherwise
    """
    check_query = "SELECT 1 FROM rad_reports WHERE uid = ?"
    check_params = (uid,)
    result = db_execute_query(check_query, check_params, fetch_mode='one')
    return result is not None


def db_get_patient_by_cnp(cnp):
    """
    Get patient information by CNP.

    Args:
        cnp: Romanian personal identification number

    Returns:
        dict: Patient data or None if not found
    """
    return db_select_one('patients', cnp)


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
        SELECT cnp, id, name, birthdate, sex
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
            # Unpack row into named variables for better readability
            (cnp, id, name, birthdate, sex) = row
            # Calculate age from birthdate if available
            age = -1
            if birthdate:
                try:
                    birth_date = datetime.strptime(birthdate, "%Y-%m-%d")
                    today = datetime.now()
                    age = today.year - birth_date.year
                    if (today.month, today.day) < (birth_date.month, birth_date.day):
                        age -= 1
                except ValueError:
                    age = -1
                    
            patients.append({
                'cnp': cnp,
                'id': id,
                'name': name,
                'age': age,
                'birthdate': birthdate,
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


def db_rad_review(uid, normal, radiologist=''):
    """
    Update radiologist report with normal/abnormal status.

    When a radiologist reviews a case, they indicate if the finding is normal (negative)
    or abnormal (positive). This function updates the radiologist report with that information.
    If no report entry exists for this UID, a new one is created with default values.

    Args:
        uid: The unique identifier of the exam
        normal: Whether the radiologist marked the case as normal (True) or abnormal (False)
        radiologist: Name/identifier of the radiologist (default: '')

    Returns:
        None
    """
    positive = 0 if normal else 1
    
    # Check if a row already exists for this UID
    result = db_select_one('rad_reports', uid)
    
    if result:
        # Row exists, update it
        db_update('rad_reports', 'uid = ?', (uid,), positive=positive, radiologist=radiologist)
    else:
        # Row doesn't exist, insert a new one
        db_insert('rad_reports', uid=uid, positive=positive, radiologist=radiologist)


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

    This function sets the exam status to 'requeue' so it will be processed again.
    It clears most existing AI report data but preserves the text for reference.

    Args:
        uid: Unique identifier of the exam to re-queue

    Returns:
        bool: True if successfully re-queued, False otherwise
    """
    try:
        # Set the status to requeue
        db_set_status(uid, 'requeue')
        
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
async def query_and_retrieve(minutes=60):
    """
    Query and Retrieve new studies from the remote DICOM server.

    This function implements the DICOM Query/Retrieve workflow by:
    1. Establishing a C-FIND association with the remote PACS
    2. Querying for CR (Computed Radiography) studies within a time window
    3. Handling time ranges that cross midnight by splitting into two queries
    4. Processing each found study by either C-MOVE or C-GET retrieval
    5. Skipping studies already in the local database
    6. Properly releasing the DICOM association

    The function uses pynetdicom library for DICOM network operations and
    handles complex time range calculations to ensure complete study retrieval.

    Args:
        minutes (int): Number of minutes to look back for new studies (default: 60)
        
    DICOM Workflow:
        1. C-FIND: Query remote PACS for studies matching criteria
        2. Study Filtering: Skip studies already in local database
        3. C-MOVE/C-GET: Request study transfer based on configuration
        4. Association Management: Proper establishment and release of connections

    Time Range Handling:
        When the query period crosses midnight (e.g., 23:00-01:00), the function
        automatically splits the query into two separate time ranges to comply
        with DICOM time range format limitations:
        - First query: From start time to 23:59:59
        - Second query: From 00:00:00 to end time

    Retrieval Methods:
        - C-MOVE: Server pushes studies to configured AE title
        - C-GET: Server sends studies over the same association

    Error Handling:
        - Logs association failures
        - Continues processing other studies on individual study errors
        - Properly releases associations even on errors
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
                    # Check if this study is already in our database
                    if db_check_study_exists(study_instance_uid):
                        logging.info(f"Skipping Study {study_instance_uid} - already in database")
                        continue
                    logging.info(f"Found Study {study_instance_uid}")
                    if RETRIEVAL_METHOD.upper() == 'C-GET':
                        await send_c_get(ae, study_instance_uid)
                    else:
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


async def send_c_get(ae, study_instance_uid):
    """
    Request a study to be sent from the remote PACS over the same association.

    Sends a C-GET request to the remote DICOM server to transfer a specific
    study (identified by Study Instance UID) directly over the current association.

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
        responses = assoc.send_c_get(
            ds,
            PatientRootQueryRetrieveInformationModelGet
        )
        # Release the association
        assoc.release()
    else:
        logging.error("Could not establish C-GET association.")


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
    
    # Validate SOP Instance UID
    if 'SOPInstanceUID' not in ds or not ds.SOPInstanceUID or ds.SOPInstanceUID == 'NO_UID':
        logging.error("Invalid or missing SOP Instance UID in received DICOM file")
        return 0x0110  # Processing failure
    
    uid = f"{ds.SOPInstanceUID}"
    # Check if already processed
    if db_check_already_processed(uid):
        # Check if the existing exam record is missing study or series information
        existing_exam = db_select_one('exams', uid)
        if existing_exam and (not existing_exam.get('study') or not existing_exam.get('series')):
            # Extract study and series from the new DICOM
            study_uid = str(ds.StudyInstanceUID) if 'StudyInstanceUID' in ds else None
            series_uid = str(ds.SeriesInstanceUID) if 'SeriesInstanceUID' in ds else None
            
            # Update the existing record with study and series if missing
            update_fields = {}
            if study_uid and not existing_exam.get('study'):
                update_fields['study'] = study_uid
            if series_uid and not existing_exam.get('series'):
                update_fields['series'] = series_uid
            
            if update_fields:
                db_update('exams', 'uid = ?', (uid,), **update_fields)
                logging.debug(f"Updated study/series info for exam {uid}")
        
        logging.debug(f"Skipping already processed image {uid}")
    elif ds.Modality == "CR":
        # Check the Modality
        dicom_file = os.path.join(IMAGES_DIR, f"{uid}.dcm")
        # Save the DICOM file
        ds.save_as(dicom_file, enforce_file_format = True)
        logging.debug(f"DICOM file saved to {dicom_file}")
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
                logging.debug(f"Skipping already processed image {uid}")
            else:
                logging.debug(f"Adding {uid} into processing queue...")
                full_path = os.path.join(IMAGES_DIR, dicom_file)
                # Process the DICOM file
                process_dicom_file(full_path, uid)
    # At the end, update the dashboard
    await broadcast_dashboard_update()


def extract_dicom_metadata(ds):
    """
    Extract relevant information from a DICOM dataset.

    Parses patient demographics, exam details, and timestamps from a DICOM dataset.
    Handles missing or malformed data gracefully with fallback values. This function
    performs several key operations:
    
    1. Patient Age Calculation: Computes age from birth date or Romanian CNP
    2. Timestamp Processing: Extracts and validates study/series timestamps
    3. Anatomic Region Identification: Maps protocol names to standardized regions
    4. Gender Normalization: Ensures consistent gender representation
    5. Data Validation: Handles missing or malformed DICOM fields gracefully

    The function prioritizes data quality and consistency, using multiple fallback
    mechanisms when primary data sources are unavailable or invalid.

    Args:
        ds (Dataset): pydicom Dataset object containing DICOM file metadata

    Returns:
        dict: Dictionary containing structured exam information with the following keys:
            - uid (str): SOP Instance UID - unique exam identifier
            - patient (dict): Patient information including:
                * name (str): Patient name from DICOM
                * cnp (str): Patient ID (often Romanian CNP)
                * age (int): Computed age in years (-1 if unavailable)
                * birthdate (str): Birth date in YYYY-MM-DD format (None if unavailable)
                * sex (str): Normalized gender ('M', 'F', or 'O')
                * county (str, optional): County code from CNP validation
            - exam (dict): Exam details including:
                * protocol (str): Imaging protocol name
                * created (str): Exam timestamp in YYYY-MM-DD HH:MM:SS format
                * region (str): Identified anatomic region
                * study (str): Study Instance UID
                * series (str): Series Instance UID
                * id (None): Placeholder for service request ID

    Data Extraction Process:
        1. Birth Date Processing:
           - Extracts PatientBirthDate and validates format
           - Calculates age from birth date
           - Falls back to CNP-based age calculation if birth date invalid
        2. Timestamp Validation:
           - Uses SeriesDate/SeriesTime for exam timestamp
           - Falls back to current time if DICOM timestamps missing/invalid
        3. Region Identification:
           - Maps ProtocolName to standardized anatomic regions
           - Uses configurable region mapping rules
        4. Gender Normalization:
           - Validates PatientSex values
           - Extracts gender from CNP if DICOM field invalid
        5. Error Handling:
           - Gracefully handles missing/invalid DICOM fields
           - Provides sensible default values
           - Logs extraction errors for debugging
    """
    age = -1
    birthdate = None
    county = None
    if 'PatientBirthDate' in ds and ds.PatientBirthDate:
        try:
            birthdate = str(ds.PatientBirthDate)
            # Validate the format (should be YYYYMMDD)
            if len(birthdate) == 8:
                birthdate = f"{birthdate[:4]}-{birthdate[4:6]}-{birthdate[6:8]}"
                # Calculate age from birthdate
                birth_date = datetime.strptime(birthdate, "%Y-%m-%d")
                today = datetime.now()
                age = today.year - birth_date.year
                if (today.month, today.day) < (birth_date.month, birth_date.day):
                    age -= 1
        except Exception as e:
            logging.error(f"Cannot parse birth date: {e}")
            birthdate = None
            age = -1
    elif 'PatientID' in ds:
        # Try to compute birthdate and age from PatientID (CNP) if available
        cnp_result = validate_romanian_cnp(ds.PatientID)
        if cnp_result['valid']:
            birthdate = cnp_result['birth_date'].strftime("%Y-%m-%d")
            age = cnp_result['age']
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

    # Identify the region from the protocol name
    region, _ = identify_anatomic_region(str(ds.ProtocolName))

    info = {
        'uid': str(ds.SOPInstanceUID),
        'patient': {
            'name':  str(ds.PatientName),
            'cnp':   str(ds.PatientID),
            'age':   age,
            'birthdate': birthdate,
            'sex':   str(ds.PatientSex),
        },
        'exam': {
            'protocol': str(ds.ProtocolName),
            'created':  created,
            'region':   region,
            'study':    str(ds.StudyInstanceUID) if 'StudyInstanceUID' in ds else None,
            'series':   str(ds.SeriesInstanceUID) if 'SeriesInstanceUID' in ds else None,
            'id':       None,
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
            # Set birthdate if not already set
            if not info['patient']['birthdate']:
                info['patient']['birthdate'] = result['birth_date'].strftime("%Y-%m-%d")
                info['patient']['age'] = result['age']
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


def extract_radiologist_initials(name):
    """
    Extract initials from a radiologist name, keeping "Dr." prefix.

    Args:
        name: Radiologist name string

    Returns:
        str: Radiologist name with "Dr." prefix and initials
    """
    if not name or not isinstance(name, str):
        return "Dr. NoName"
    # Check if name starts with "Dr." (case insensitive)
    if name.lower().startswith("dr."):
        # Remove "Dr." prefix and extract initials from the rest
        name_without_dr = name[3:].strip()
        if not name_without_dr:
            return "Dr. NoName"
        # Split by spaces, hyphens, and carets and take first letter of each part
        parts = re.split(r'[-^ ]', name_without_dr)
        initials = ''.join([part[0] + '.' for part in parts if part])
        return "Dr. " + initials.upper() if initials else "Dr. NoName"
    else:
        # No "Dr." prefix, just extract initials
        parts = re.split(r'[-^ ]', name)
        initials = ''.join([part[0] + '.' for part in parts if part])
        return "Dr. " + initials.upper() if initials else "Dr. NoName"


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


def convert_dicom_to_png(dicom_file, max_size = 896):
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
        max_size: Maximum dimension for the output image (default: 896)

    Returns:
        str: Path to the saved PNG file
    """
    # Check if PNG file already exists
    base_name = os.path.splitext(os.path.basename(dicom_file))[0]
    png_file = os.path.join(IMAGES_DIR, f"{base_name}.png")
    if os.path.exists(png_file):
        logging.debug(f"PNG file already exists: {png_file}")
        return png_file
        
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
        image = apply_gamma_correction(image)
        # Save the PNG file
        cv2.imwrite(png_file, image)
        logging.debug(f"Converted PNG saved to {png_file}")
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


async def serve_radiologists_page(request):
    """Serve the radiologists HTML page.

    Args:
        request: aiohttp request object

    Returns:
        web.FileResponse: Radiologists HTML file response
    """
    return web.FileResponse(path=os.path.join(STATIC_DIR, "radiologists.html"))


async def serve_diagnostics_page(request):
    """Serve the diagnostics HTML page.

    Args:
        request: aiohttp request object

    Returns:
        web.FileResponse: Diagnostics HTML file response
    """
    return web.FileResponse(path=os.path.join(STATIC_DIR, "diagnostics.html"))


async def serve_insights_page(request):
    """Serve the insights HTML page.

    Args:
        request: aiohttp request object

    Returns:
        web.FileResponse: Insights HTML file response
    """
    return web.FileResponse(path=os.path.join(STATIC_DIR, "insights.html"))


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
    
    # Add client to the set
    websocket_clients.add(ws)
    
    try:
        # Send connection notification
        await broadcast_dashboard_update(event="connected", payload={'address': request.remote}, client=ws)
        logging.info(f"Dashboard connected via WebSocket from {request.remote}")
        
        # Handle incoming messages
        async for msg in ws:
            # Currently we don't process incoming messages, but this could be extended
            # to handle client requests or commands
            pass
            
    except asyncio.CancelledError:
        # Handle task cancellation gracefully
        logging.debug(f"WebSocket connection cancelled for {request.remote}")
        raise
    except Exception as e:
        # Log any unexpected errors
        logging.error(f"WebSocket error for {request.remote}: {e}")
    finally:
        # Ensure client is removed from the set, even if an exception occurs
        try:
            if ws in websocket_clients:
                websocket_clients.remove(ws)
        except KeyError:
            # Client was already removed, which is fine
            pass
        finally:
            # Close the WebSocket connection
            try:
                await ws.close()
            except Exception as e:
                logging.debug(f"Error closing WebSocket for {request.remote}: {e}")
            
            logging.info(f"Dashboard WebSocket disconnected from {request.remote}")
    
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
        for filter in ['positive', 'correct', 'reviewed']:
            value = request.query.get(filter, 'any')
            if value != 'any':
                filters[filter] = value[0].lower() == 'y' and 1 or 0
        for filter in ['region', 'status', 'search', 'diagnostic', 'radiologist']:
            value = request.query.get(filter, 'any')
            if value != 'any':
                if filter == 'status':
                    # Handle status as a list if it contains commas
                    if ',' in value:
                        filters[filter] = [s.strip().lower() for s in value.split(',')]
                    else:
                        filters[filter] = value.lower()
                else:
                    filters[filter] = value
        # Handle severity filter with comparison operator
        severity_value = request.query.get('severity', 'any')
        severity_op = request.query.get('severity_op', 'any')
        if severity_value != 'any' and severity_op != 'any':
            try:
                filters['severity'] = int(severity_value)
                if severity_op in ['equal', 'lower', 'higher']:
                    filters['severity_op'] = severity_op
            except ValueError:
                pass  # Ignore invalid severity value
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
                # Anonymize the radiologist name
                if 'radiologist' in exam['report']['rad']:
                    exam['report']['rad']['radiologist'] = extract_radiologist_initials(exam['report']['rad']['radiologist'])
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
    """Provide distinct diagnostic summaries from radiologist reports along with their report counts.

    Returns the list of unique diagnostic summaries (from rad_reports.summary) and their report counts
    for use in filtering or display in the frontend.

    Args:
        request: aiohttp request object

    Returns:
        web.json_response: JSON response with diagnostic summaries and report counts in {'summary': 3} format
    """
    try:
        # Get distinct diagnostic summaries and their report counts from the database
        query = """
            SELECT summary, COUNT(*) as report_count 
            FROM rad_reports 
            WHERE summary IS NOT NULL AND summary != '' 
            GROUP BY summary 
            ORDER BY summary
        """
        rows = db_execute_query(query, fetch_mode='all')
        diagnostics = {summary: count for summary, count in rows} if rows else {}
        return web.json_response(diagnostics)
    except Exception as e:
        logging.error(f"Diagnostics endpoint error: {e}")
        return web.json_response({}, status = 500)


async def diagnostics_monthly_trends_handler(request):
    """Provide monthly trends data showing top 10 diagnostics for each month over the last 12 months.

    Returns top 10 diagnostics with their counts for each month over the last 12 months
    for use in the diagnostics trends chart.

    Args:
        request: aiohttp request object

    Returns:
        web.json_response: JSON response with monthly trends data
    """
    try:
        # Get top 10 diagnostics for each month over the last 12 months
        trends_query = """
            SELECT 
                strftime('%Y-%m', e.created) as month,
                rr.summary,
                COUNT(*) as report_count
            FROM rad_reports rr
            JOIN exams e ON rr.uid = e.uid
            WHERE rr.summary IS NOT NULL AND rr.summary != ''
            AND e.created >= date('now', '-12 months')
            GROUP BY strftime('%Y-%m', e.created), rr.summary
            ORDER BY month, report_count DESC
        """
        trends_rows = db_execute_query(trends_query, fetch_mode='all')
        
        # Process the data to get top 10 diagnostics per month
        monthly_trends = {}
        if trends_rows:
            # Group by month first
            monthly_data = {}
            for row in trends_rows:
                month, diagnostic, count = row
                if month not in monthly_data:
                    monthly_data[month] = []
                monthly_data[month].append({
                    'diagnostic': diagnostic,
                    'count': count
                })
            
            # For each month, take top 10 diagnostics
            for month, diagnostics in monthly_data.items():
                # Sort by count descending and take top 10
                top_diagnostics = sorted(diagnostics, key=lambda x: x['count'], reverse=True)[:10]
                monthly_trends[month] = top_diagnostics
        
        # Format the response
        result = {
            'trends': monthly_trends
        }
        
        return web.json_response(result)
    except Exception as e:
        logging.error(f"Diagnostics monthly trends endpoint error: {e}")
        return web.json_response({}, status = 500)


async def diagnostics_stats_handler(request):
    """Provide detailed statistics for diagnostics.

    Returns detailed statistics for each diagnostic including:
    - Report counts
    - Average severity scores
    - AI validation rates
    - Associated regions
    - Temporal trends

    Args:
        request: aiohttp request object

    Returns:
        web.json_response: JSON response with detailed diagnostic statistics
    """
    try:
        # Get detailed diagnostic statistics with correlations
        query = """
            SELECT 
                rr.summary,
                COUNT(*) as report_count,
                AVG(CAST(rr.severity AS FLOAT)) as avg_severity,
                SUM(CASE WHEN (ar.severity >= ? AND rr.severity >= ?) OR (ar.severity < ? AND rr.severity < ?) THEN 1 ELSE 0 END) as correct_predictions,
                COUNT(CASE WHEN ar.severity IS NOT NULL THEN 1 END) as ai_compared,
                GROUP_CONCAT(e.region, ', ') as regions,
                MIN(e.created) as first_seen,
                MAX(e.created) as last_seen
            FROM rad_reports rr
            LEFT JOIN exams e ON rr.uid = e.uid
            LEFT JOIN ai_reports ar ON e.uid = ar.uid
            WHERE rr.summary IS NOT NULL AND rr.summary != ''
            GROUP BY rr.summary
            ORDER BY report_count DESC
        """
        rows = db_execute_query(query, (SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD), fetch_mode='all')
        
        diagnostic_stats = {}
        if rows:
            for row in rows:
                summary, report_count, avg_severity, correct_predictions, ai_compared, regions, first_seen, last_seen = row
                
                # Process regions to get frequency
                region_freq = {}
                if regions:
                    for region in regions.split(', '):
                        region = region.strip().lower()
                        if region:
                            region_freq[region] = region_freq.get(region, 0) + 1
                
                # Sort regions by frequency
                sorted_regions = sorted(region_freq.items(), key=lambda x: x[1], reverse=True)
                top_regions = dict(sorted_regions[:5])  # Top 5 regions
                
                # Calculate accuracy if AI comparisons exist
                accuracy = 0
                if ai_compared and ai_compared > 0:
                    accuracy = round((correct_predictions / ai_compared) * 100, 1)
                
                diagnostic_stats[summary] = {
                    'report_count': report_count,
                    'avg_severity': round(avg_severity, 1) if avg_severity else 0,
                    'ai_accuracy': accuracy,
                    'ai_compared': ai_compared,
                    'top_regions': top_regions,
                    'first_seen': first_seen,
                    'last_seen': last_seen
                }
        
        return web.json_response(diagnostic_stats)
    except Exception as e:
        logging.error(f"Diagnostic stats endpoint error: {e}")
        return web.json_response({}, status = 500)


async def db_get_processing_times_by_region():
    """Get processing time analysis by region.
    
    Returns:
        list: List of tuples containing (region, avg_processing_time, exam_count)
    """
    query = """
        SELECT 
            e.region,
            AVG(CAST(ar.latency AS FLOAT)) as avg_processing_time,
            COUNT(*) as exam_count
        FROM exams e
        LEFT JOIN ai_reports ar ON e.uid = ar.uid
        WHERE e.status = 'done' 
        AND ar.latency IS NOT NULL 
        AND ar.latency >= 0
        GROUP BY e.region
        ORDER BY avg_processing_time DESC
    """
    return db_execute_query(query, fetch_mode='all')

async def db_get_rad_severity_distribution():
    """Get severity distribution for radiologist reports.
    
    Returns:
        list: List of tuples containing (severity, count)
    """
    query = """
        SELECT 
            severity,
            COUNT(*) as count
        FROM rad_reports
        WHERE severity >= 0
        GROUP BY severity
        ORDER BY severity
    """
    return db_execute_query(query, fetch_mode='all')

async def db_get_ai_severity_distribution():
    """Get severity distribution for AI reports.
    
    Returns:
        list: List of tuples containing (severity, count)
    """
    query = """
        SELECT 
            severity,
            COUNT(*) as count
        FROM ai_reports
        WHERE severity >= 0
        GROUP BY severity
        ORDER BY severity
    """
    return db_execute_query(query, fetch_mode='all')

async def db_get_severity_differences():
    """Get severity differences between AI and radiologist reports.
    
    Returns:
        list: List of tuples containing (severity_diff, count)
    """
    query = """
        SELECT 
            CAST(ar.severity AS INTEGER) - CAST(rr.severity AS INTEGER) as severity_diff,
            COUNT(*) as count
        FROM exams e
        JOIN ai_reports ar ON e.uid = ar.uid
        JOIN rad_reports rr ON e.uid = rr.uid
        WHERE e.status = 'done'
        AND ar.severity >= 0
        AND rr.severity >= 0
        GROUP BY severity_diff
        ORDER BY severity_diff
    """
    return db_execute_query(query, fetch_mode='all')

async def db_get_age_distribution_insights(severity_threshold):
    """Get patient demographics insights by age group.
    
    Args:
        severity_threshold: Threshold for positive findings
        
    Returns:
        list: List of tuples containing (age_group, total_exams, positive_findings)
    """
    query = """
        SELECT 
            CASE 
                WHEN p.birthdate IS NULL THEN 'Unknown'
                WHEN CAST((julianday(e.created) - julianday(p.birthdate)) / 365.25 AS INTEGER) < 0 THEN 'Unknown'
                WHEN CAST((julianday(e.created) - julianday(p.birthdate)) / 365.25 AS INTEGER) <= 2 THEN '0-2'
                WHEN CAST((julianday(e.created) - julianday(p.birthdate)) / 365.25 AS INTEGER) <= 4 THEN '2-4'
                WHEN CAST((julianday(e.created) - julianday(p.birthdate)) / 365.25 AS INTEGER) <= 6 THEN '4-6'
                WHEN CAST((julianday(e.created) - julianday(p.birthdate)) / 365.25 AS INTEGER) <= 8 THEN '6-8'
                WHEN CAST((julianday(e.created) - julianday(p.birthdate)) / 365.25 AS INTEGER) <= 10 THEN '8-10'
                WHEN CAST((julianday(e.created) - julianday(p.birthdate)) / 365.25 AS INTEGER) <= 12 THEN '10-12'
                WHEN CAST((julianday(e.created) - julianday(p.birthdate)) / 365.25 AS INTEGER) <= 14 THEN '12-14'
                WHEN CAST((julianday(e.created) - julianday(p.birthdate)) / 365.25 AS INTEGER) <= 16 THEN '14-16'
                WHEN CAST((julianday(e.created) - julianday(p.birthdate)) / 365.25 AS INTEGER) <= 18 THEN '16-18'
                ELSE '> 18'
            END as age_group,
            COUNT(*) as total_exams,
            SUM(CASE WHEN rr.severity >= ? THEN 1 ELSE 0 END) as positive_findings
        FROM patients p
        JOIN exams e ON p.cnp = e.cnp
        JOIN rad_reports rr ON e.uid = rr.uid
        WHERE p.birthdate IS NOT NULL
        GROUP BY age_group
        HAVING age_group != 'Unknown'
        ORDER BY 
            CASE age_group
                WHEN '0-2' THEN 1
                WHEN '2-4' THEN 2
                WHEN '4-6' THEN 3
                WHEN '6-8' THEN 4
                WHEN '8-10' THEN 5
                WHEN '10-12' THEN 6
                WHEN '12-14' THEN 7
                WHEN '14-16' THEN 8
                WHEN '16-18' THEN 9
                WHEN '> 18' THEN 10
                ELSE 11
            END
    """
    return db_execute_query(query, (severity_threshold,), fetch_mode='all')

async def db_get_hourly_patterns():
    """Get temporal patterns by hour of day.
    
    Returns:
        list: List of tuples containing (hour, exam_count)
    """
    query = """
        SELECT 
            CAST(strftime('%H', created) AS INTEGER) as hour,
            COUNT(*) as exam_count
        FROM exams
        WHERE status = 'done'
        GROUP BY hour
        ORDER BY hour
    """
    return db_execute_query(query, fetch_mode='all')

async def db_get_requeue_analysis():
    """Get re-queue analysis data.
    
    Returns:
        tuple: Tuple containing (total_requeued, avg_latency_improvement) or None
    """
    query = """
        SELECT 
            COUNT(*) as total_requeued,
            AVG(CAST(ai2.latency AS FLOAT) - CAST(ai1.latency AS FLOAT)) as avg_latency_improvement
        FROM exams e
        JOIN ai_reports ai1 ON e.uid = ai1.uid
        JOIN ai_reports ai2 ON e.uid = ai2.uid
        WHERE e.status = 'done'
        AND ai1.created < ai2.created
    """
    return db_execute_query(query, fetch_mode='one')

async def db_get_radiologist_metrics():
    """Get radiologist consistency metrics.
    
    Returns:
        list: List of tuples containing (radiologist, reports_count, avg_severity, unique_exams)
    """
    query = """
        SELECT 
            radiologist,
            COUNT(*) as reports_count,
            AVG(CAST(severity AS FLOAT)) as avg_severity,
            COUNT(DISTINCT uid) as unique_exams
        FROM rad_reports
        WHERE radiologist IS NOT NULL AND radiologist != ''
        GROUP BY radiologist
        HAVING reports_count > 5
        ORDER BY reports_count DESC
    """
    return db_execute_query(query, fetch_mode='all')

async def insights_handler(request):
    """Provide advanced insights and correlations from the database.

    Returns various advanced statistics including:
    - Processing time analysis by region
    - Severity distribution
    - Patient demographics insights
    - Temporal patterns
    - Re-queue analysis
    - Radiologist consistency metrics
    - AI vs Radiologist severity differences

    Args:
        request: aiohttp request object

    Returns:
        web.json_response: JSON response with advanced insights
    """
    try:
        insights = {}
        
        # 1. Processing time analysis by region
        rows = await db_get_processing_times_by_region()
        insights['processing_times'] = {}
        if rows:
            for row in rows:
                region, avg_time, count = row
                insights['processing_times'][region] = {
                    'avg_time': round(avg_time, 2),
                    'count': count
                }
        
        # 2. Severity distribution for radiologist reports
        rows = await db_get_rad_severity_distribution()
        insights['rad_severity_distribution'] = {}
        if rows:
            for row in rows:
                severity, count = row
                insights['rad_severity_distribution'][str(severity)] = count
                
        # 2b. Severity distribution for AI reports
        rows = await db_get_ai_severity_distribution()
        insights['ai_severity_distribution'] = {}
        if rows:
            for row in rows:
                severity, count = row
                insights['ai_severity_distribution'][str(severity)] = count
        
        # 2c. Severity differences between AI and radiologist reports
        rows = await db_get_severity_differences()
        insights['severity_differences'] = {}
        if rows:
            for row in rows:
                diff, count = row
                insights['severity_differences'][str(diff)] = count
        
        # 3. Patient demographics insights (positive findings by age group)
        rows = await db_get_age_distribution_insights(SEVERITY_THRESHOLD)
        insights['age_distribution'] = {}
        if rows:
            for row in rows:
                age_group, total_exams, positive_findings = row
                insights['age_distribution'][age_group] = {
                    'total_exams': total_exams,
                    'positive_findings': positive_findings,
                    'positive_rate': round((positive_findings / total_exams) * 100, 1) if total_exams > 0 else 0
                }
        
        # 4. Temporal patterns (exams by hour of day)
        rows = await db_get_hourly_patterns()
        insights['hourly_patterns'] = {}
        if rows:
            for row in rows:
                hour, count = row
                insights['hourly_patterns'][str(hour)] = count
        
        # 5. Re-queue analysis
        row = await db_get_requeue_analysis()
        if row:
            total_requeued, avg_improvement = row
            insights['requeue_analysis'] = {
                'total_requeued': total_requeued or 0,
                'avg_latency_improvement': round(avg_improvement, 2) if avg_improvement else 0
            }
        
        # 6. Radiologist consistency (if we have multiple reports for same exam)
        rows = await db_get_radiologist_metrics()
        insights['radiologist_metrics'] = {}
        if rows:
            for row in rows:
                radiologist, reports_count, avg_severity, unique_exams = row
                insights['radiologist_metrics'][radiologist] = {
                    'reports_count': reports_count,
                    'avg_severity': round(avg_severity, 1) if avg_severity else 0,
                    'unique_exams': unique_exams,
                    'avg_reports_per_exam': round(reports_count / unique_exams, 1) if unique_exams > 0 else 0
                }
        
        return web.json_response(insights)
    except Exception as e:
        logging.error(f"Insights endpoint error: {e}")
        return web.json_response({}, status = 500)


async def radiologists_handler(request):
    """Provide distinct radiologist names from radiologist reports along with their report counts.

    Returns the list of unique radiologist names (from rad_reports.radiologist) and their report counts
    for use in filtering or display in the frontend.

    Args:
        request: aiohttp request object

    Returns:
        web.json_response: JSON response with radiologist names and report counts in {'dr. X Y': 3} format
    """
    try:
        # Get distinct radiologist names and their report counts from the database
        query = """
            SELECT radiologist, COUNT(*) as report_count 
            FROM rad_reports 
            WHERE radiologist IS NOT NULL AND radiologist != '' 
            GROUP BY radiologist 
            ORDER BY radiologist
        """
        rows = db_execute_query(query, fetch_mode='all')
        radiologists = {radiologist: count for radiologist, count in rows} if rows else {}
        return web.json_response(radiologists)
    except Exception as e:
        logging.error(f"Radiologists endpoint error: {e}")
        return web.json_response({}, status = 500)


async def radiologist_stats_handler(request):
    """Provide detailed statistics for radiologists.

    Returns detailed statistics for each radiologist including:
    - Report counts
    - Average severity scores
    - AI validation rates
    - Preferred diagnostics

    Args:
        request: aiohttp request object

    Returns:
        web.json_response: JSON response with detailed radiologist statistics
    """
    try:
        # Get user role from request (set by auth_middleware)
        user_role = getattr(request, 'user_role', 'user')
        
        # Get detailed radiologist statistics
        query = """
            SELECT 
                rr.radiologist,
                COUNT(*) as report_count,
                AVG(CAST(rr.severity AS FLOAT)) as avg_severity,
                SUM(CASE WHEN (ar.severity >= ? AND rr.severity >= ?) OR (ar.severity < ? AND rr.severity < ?) THEN 1 ELSE 0 END) as correct_predictions,
                COUNT(CASE WHEN ar.severity >= 0 THEN 1 END) as ai_compared,
                GROUP_CONCAT(rr.summary, ', ') as all_diagnostics
            FROM rad_reports rr
            LEFT JOIN exams e ON rr.uid = e.uid
            LEFT JOIN ai_reports ar ON e.uid = ar.uid
            WHERE rr.radiologist IS NOT NULL AND rr.radiologist != ''
            GROUP BY rr.radiologist
            ORDER BY report_count DESC
        """
        rows = db_execute_query(query, (SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD), fetch_mode='all')
        
        radiologist_stats = {}
        if rows:
            for row in rows:
                radiologist, report_count, avg_severity, correct_predictions, ai_compared, all_diagnostics = row
                
                # Process diagnostics to get frequency
                diagnostics = {}
                if all_diagnostics:
                    for diag in all_diagnostics.split(', '):
                        diag = diag.strip().lower()
                        if diag:
                            diagnostics[diag] = diagnostics.get(diag, 0) + 1
                
                # Sort diagnostics by frequency
                sorted_diagnostics = sorted(diagnostics.items(), key=lambda x: x[1], reverse=True)
                top_diagnostics = dict(sorted_diagnostics[:5])  # Top 5 diagnostics
                
                # Calculate accuracy if AI comparisons exist
                accuracy = 0
                if ai_compared and ai_compared > 0:
                    accuracy = round((correct_predictions / ai_compared) * 100, 1)
                
                radiologist_stats[radiologist] = {
                    'report_count': report_count,
                    'avg_severity': round(avg_severity, 1) if avg_severity else 0,
                    'ai_accuracy': accuracy,
                    'ai_compared': ai_compared,
                    'top_diagnostics': top_diagnostics
                }
        
        return web.json_response(radiologist_stats)
    except Exception as e:
        logging.error(f"Radiologist stats endpoint error: {e}")
        return web.json_response({}, status = 500)


async def radiologists_monthly_trends_handler(request):
    """Provide monthly trends data showing top 10 radiologists for each month over the last 12 months.

    Returns top 10 radiologists with their report counts for each month over the last 12 months
    for use in the radiologists trends chart.

    Args:
        request: aiohttp request object

    Returns:
        web.json_response: JSON response with monthly trends data
    """
    try:
        # Get top 10 radiologists for each month over the last 12 months
        trends_query = """
            SELECT 
                strftime('%Y-%m', e.created) as month,
                rr.radiologist,
                COUNT(*) as report_count
            FROM rad_reports rr
            JOIN exams e ON rr.uid = e.uid
            WHERE rr.radiologist IS NOT NULL AND rr.radiologist != ''
            AND e.created >= date('now', '-12 months')
            GROUP BY strftime('%Y-%m', e.created), rr.radiologist
            ORDER BY month, report_count DESC
        """
        trends_rows = db_execute_query(trends_query, fetch_mode='all')
        
        # Process the data to get top 10 radiologists per month
        monthly_trends = {}
        if trends_rows:
            # Group by month first
            monthly_data = {}
            for row in trends_rows:
                month, radiologist, count = row
                if month not in monthly_data:
                    monthly_data[month] = []
                monthly_data[month].append({
                    'radiologist': radiologist,
                    'count': count
                })
            
            # For each month, take top 10 radiologists
            for month, radiologists in monthly_data.items():
                # Sort by count descending and take top 10
                top_radiologists = sorted(radiologists, key=lambda x: x['count'], reverse=True)[:10]
                monthly_trends[month] = top_radiologists
        
        # Format the response
        result = {
            'trends': monthly_trends
        }
        
        return web.json_response(result)
    except Exception as e:
        logging.error(f"Radiologists monthly trends endpoint error: {e}")
        return web.json_response({}, status = 500)


async def severity_handler(request):
    """Provide severity levels and their report counts.

    Returns a dictionary mapping severity levels (0-10) to their report counts
    for use in filtering or display in the frontend.

    Args:
        request: aiohttp request object

    Returns:
        web.json_response: JSON response with severity levels and report counts in {severity: count} format
    """
    try:
        # Get severity levels and their report counts from the database
        query = """
            SELECT severity, COUNT(*) as report_count 
            FROM rad_reports 
            WHERE severity IS NOT NULL AND severity >= 0
            GROUP BY severity 
            ORDER BY severity
        """
        rows = db_execute_query(query, fetch_mode='all')
        severity_counts = {str(severity): count for severity, count in rows} if rows else {}
        return web.json_response(severity_counts)
    except Exception as e:
        logging.error(f"Severity endpoint error: {e}")
        return web.json_response({}, status = 500)


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
        exams, _ = db_get_exams(limit=1, uid=uid)
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
            # Anonymize the radiologist name
            if 'radiologist' in exam['report']['rad']:
                exam['report']['rad']['radiologist'] = extract_radiologist_initials(exam['report']['rad']['radiologist'])
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
        logging.debug(f"Manual QueryRetrieve triggered for the last {hours} hours.")
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
        # Use authenticated username as radiologist name, fallback to '' if not available
        radiologist = getattr(request, 'username', '')
        
        if uid is None or normal is None:
            return web.json_response({'status': 'error', 'message': 'UID and normal status are required'}, status=400)
        
        # Update the radiologist report
        db_rad_review(uid, normal, radiologist)
        
        # Get the updated exam data
        exam_data = db_get_exams(limit=1, uid=uid)
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


async def get_report_handler(request):
    """Retrieve radiologist report for an exam from FHIR and queue for LLM processing.

    This function retrieves the radiologist report for a specific exam from the FHIR system
    and prepares it for LLM analysis. It performs the following steps:
    1. Gets exam details from the database
    2. Retrieves patient ID from FHIR if not already known
    3. Processes the exam to fetch the radiologist report from FHIR
    4. Sets the exam status to 'check' to trigger LLM processing
    
    Args:
        request: aiohttp request object with JSON body containing:
            - uid: The unique identifier of the exam to process

    Returns:
        web.json_response: JSON response with retrieval status
    """
    try:
        data = await request.json()
        uid = data.get('uid')
        
        if not uid:
            return web.json_response({'status': 'error', 'message': 'UID is required'}, status=400)
        
        # Get exam details from database
        exams, _ = db_get_exams(limit=1, uid=uid)
        if not exams:
            return web.json_response({'status': 'error', 'message': 'Exam not found'}, status=404)
        
        exam = exams[0]
        
        # Return early to client
        response = web.json_response({'status': 'success', 'message': f'Report retrieval started for exam {uid}'})
        
        # Process FHIR report asynchronously
        async def async_process():
            try:
                # If patient ID is not known, search for it in FHIR
                if not exam['patient']['id']:
                    async with aiohttp.ClientSession() as session:
                        # Format patient name as "last_name first_name" for FHIR search
                        formatted_name = await format_patient_name_for_fhir(exam['patient']['name'])
                        patient_id = await get_patient_id_from_fhir(session, exam['patient']['cnp'], formatted_name)
                        if patient_id:
                            exam['patient']['id'] = patient_id

                if exam['patient']['id']:
                    current_exam = exam['exam']
                    current_exam['uid'] = uid
                    # Process the exam immediately using existing patient ID
                    async with aiohttp.ClientSession() as session:
                        await process_single_exam_without_rad_report(session, current_exam, exam['patient']['id'])
         
                # Notify the queue
                QUEUE_EVENT.set()
                payload = {'uid': uid}
                await broadcast_dashboard_update(event="radreport", payload=payload)
            except Exception as e:
                logging.error(f"Error processing radiologist report for exam {uid}: {e}")
        
        # Start asynchronous processing
        asyncio.create_task(async_process())
        
        return response
    except Exception as e:
        logging.error(f"Error checking radiologist report: {e}")
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
        logging.debug(f"Report check request received with report length: {len(report_text)} characters")
        
        if not report_text:
            logging.warning("Report check request failed: no report text provided")
            return {'error': 'No report text provided'}
        
        # Add space after punctuation marks to properly separate phrases
        processed_report_text = re.sub(r'([.!?])(?=\S)', r'\1 ', report_text)
        
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
                        {"type": "text", "text": processed_report_text}
                    ]
                }
            ]
        }
        
        logging.debug(f"Sending report to AI API with model: {MODEL_NAME}")
        
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
                logging.debug(f"AI responded: {parsed_response}")
                    
                # Handle case where AI returns an array instead of single object
                if isinstance(parsed_response, list):
                    if len(parsed_response) == 0:
                        raise ValueError("Empty array response from AI")
                    # Take the first valid entry from the array
                    parsed_response = parsed_response[0]
                    logging.debug(f"Extracted first entry from array: {parsed_response}")
                    
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
                    
                logging.debug(f"AI analysis completed: severity {parsed_response['severity']}, {parsed_response['pathologic'] and 'pathologic' or 'non-pathologic'}: {parsed_response['summary']}")
                return parsed_response
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse AI response as JSON: {response_text}")
                return {'error': 'Failed to parse AI response', 'response': response_text}
            except ValueError as e:
                logging.error(f"Invalid AI response format: {e} ({response_text})")
                return {'error': f'Invalid AI response format: {str(e)}', 'response': response_text}
    except Exception as e:
        logging.error(f"Error processing report check request: {e}")
        return {'error': 'Internal server error'}


async def check_ai_report_and_update(uid):
    """Send AI report text to CHECK prompt and update database with severity/summary.

    Takes an AI report from the database, sends it to the LLM for analysis using
    the CHECK prompt, and updates the database with the extracted severity score
    and summary.

    Args:
        uid: Exam unique identifier

    Returns:
        bool: True if successfully processed and updated, False otherwise
    """
    try:
        # Get the AI report from database
        ai_report = db_get_ai_report(uid)
        if not ai_report or not ai_report.get('text'):
            logging.warning(f"No AI report text found for exam {uid}")
            return False
            
        # Extract the report text
        report_text = ai_report['text']
        
        # Summarize the AI report
        logging.info(f"Summarizing AI report for exam {uid}")
        analysis_result = await check_report(report_text)
        
        # Check if analysis was successful
        if 'error' in analysis_result:
            logging.error(f"AI check failed for exam {uid}: {analysis_result['error']}")
            return False
            
        # Extract values from analysis result
        try:
            positive = 1 if analysis_result['pathologic'] == 'yes' else 0
            severity = analysis_result['severity']
            summary = analysis_result['summary'].lower()
        except Exception as e:
            logging.error(f"Could not extract analysis results for exam {uid}: {e}")
            return False
        
        # Update the AI report in database with severity and summary
        db_update('ai_reports', 'uid = ?', (uid,),
                  positive=positive,
                  severity=severity,
                  summary=summary)
        
        logging.info(f"Updated AI report for exam {uid} with severity {severity} and summary '{summary}'")
        return True
        
    except Exception as e:
        logging.error(f"Error processing CHECK prompt for exam {uid}: {e}")
        return False


async def translate_report(report_text):
    """Translate a Romanian radiology report to English using the LLM.

    Takes a radiology report text and sends it to the LLM for translation
    using a specialized prompt.

    Args:
        report_text: Romanian radiology report text to translate

    Returns:
        str: English translation of the report, or None if translation failed
    """
    try:
        logging.debug(f"Translation request received with report length: {len(report_text)} characters")

        if not report_text:
            logging.warning("Translation request failed: no report text provided")
            return None

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
                    "content": [{"type": "text", "text": TRN_PROMPT}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": report_text}
                    ]
                }
            ]
        }

        logging.debug(f"Sending report to AI API with model: {MODEL_NAME} for translation")

        async with aiohttp.ClientSession() as session:
            result = await send_to_openai(session, headers, payload)
            if not result:
                logging.error("Failed to get response from AI service for translation")
                return None

            response_text = result["choices"][0]["message"]["content"].strip()
            logging.debug(f"Raw AI translation response: {response_text}")

            # Clean up markdown code fences if present
            response_text = response_text.replace('\n', ' ')
            response_text = re.sub(r"^```(?:json)?\s*", "", response_text, flags=re.IGNORECASE | re.MULTILINE)
            response_text = re.sub(r"\s*```$", "", response_text, flags=re.MULTILINE)

            # For translation, we expect simple text response, not JSON
            # Just return the cleaned response text directly
            if response_text:
                logging.info(f"Translation completed: {response_text[:50]}...")
                return response_text
            else:
                logging.warning("Empty translation response received")
                return None
    except Exception as e:
        logging.error(f"Error processing translation request: {e}")
        return None

async def check_rad_report_and_update(uid):
    """Send radiologist report text to CHECK prompt and update database with severity/summary.

    Takes a radiologist report from the database, sends it to the LLM for analysis using
    the CHECK prompt, and updates the database with the extracted severity score
    and summary. Also translates the report from Romanian to English.

    Args:
        uid: Exam unique identifier

    Returns:
        bool: True if successfully processed and updated, False otherwise
    """
    try:
        # Get the radiologist report from database
        rad_report = db_get_rad_report(uid)
        if not rad_report or not rad_report.get('text'):
            logging.warning(f"No radiologist report text found for exam {uid}")
            return False

        # Extract the report text
        report_text = rad_report['text']

        # Translate the report from Romanian to English
        logging.info(f"Translating radiologist report for exam {uid}")
        translation = await translate_report(report_text)
        if translation:
            logging.info(f"Translation successful for exam {uid}")
        else:
            logging.warning(f"Translation failed for exam {uid}")

        # Summarize the radiologist report
        logging.info(f"Summarizing radiologist report for exam {uid}")
        start_time = asyncio.get_event_loop().time()
        analysis_result = await check_report(report_text)
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time  # In seconds

        # Check if analysis was successful
        if 'error' in analysis_result:
            logging.error(f"AI check failed for exam {uid}: {analysis_result['error']}")
            return False

        # Extract values from analysis result
        try:
            # Validate that all required fields are present
            if 'pathologic' not in analysis_result or 'severity' not in analysis_result or 'summary' not in analysis_result:
                logging.error(f"AI check response missing required fields for exam {uid}: {list(analysis_result.keys())}")
                return False

            positive = 1 if analysis_result['pathologic'] == 'yes' else 0
            severity = analysis_result['severity']
            summary = analysis_result['summary'].lower()
        except Exception as e:
            logging.error(f"Could not extract analysis results for exam {uid}: {e}")
            return False

        # Update the radiologist report in database with severity, summary, latency, and translation
        update_fields = {
            'positive': positive,
            'severity': severity,
            'summary': summary,
            'model': MODEL_NAME,
            'latency': int(processing_time)
        }

        # Add translation if successful
        if translation:
            update_fields['text_en'] = translation

        db_update('rad_reports', 'uid = ?', (uid,), **update_fields)

        logging.info(f"Updated radiologist report for exam {uid} with severity {severity}, summary '{summary}', latency {int(processing_time)}s")
        if translation:
            logging.info(f"Added English translation for exam {uid}")
        return True

    except Exception as e:
        logging.error(f"Error processing CHECK prompt for exam {uid}: {e}")
        return False

async def detailed_analysis_report(report_text):
    """Perform detailed three-pass analysis of a radiology report.

    Takes a radiology report text and sends it to the LLM for detailed analysis
    using the ANA_PROMPT to extract comprehensive insights.

    Args:
        report_text: Radiology report text to analyze

    Returns:
        dict: Detailed analysis results with three-pass structure
    """
    try:
        logging.debug(f"Detailed analysis request received with report length: {len(report_text)} characters")
        
        if not report_text:
            logging.warning("Detailed analysis request failed: no report text provided")
            return {'error': 'No report text provided'}
        
        # Add space after punctuation marks to properly separate phrases
        processed_report_text = re.sub(r'([.!?])(?=\S)', r'\1 ', report_text)
        
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
                    "content": [{"type": "text", "text": ANA_PROMPT}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": processed_report_text}
                    ]
                }
            ]
        }
        
        logging.debug(f"Sending report to AI API with model: {MODEL_NAME} for detailed analysis")
        
        async with aiohttp.ClientSession() as session:
            result = await send_to_openai(session, headers, payload)
            if not result:
                logging.error("Failed to get response from AI service")
                return {'error': 'Failed to get response from AI service'}
            
            response_text = result["choices"][0]["message"]["content"].strip()
            logging.debug(f"Raw AI response: {response_text}")
            
            # Log the response text for debugging before cleaning
            logging.debug(f"AI response before cleaning: {repr(response_text)}")
            
            # Clean up markdown code fences if present
            response_text = re.sub(r"^```(?:json)?\s*", "", response_text, flags=re.IGNORECASE | re.MULTILINE)
            response_text = re.sub(r"\s*```$", "", response_text, flags=re.MULTILINE)
            
            # Log the response text after cleaning
            logging.debug(f"AI response after cleaning: {repr(response_text)}")
            
            try:
                parsed_response = json.loads(response_text)
                logging.debug(f"AI detailed analysis completed")
                logging.debug(f"Parsed response keys: {list(parsed_response.keys())}")
                    
                # Handle case where AI returns an array instead of single object
                if isinstance(parsed_response, list):
                    if len(parsed_response) == 0:
                        raise ValueError("Empty array response from AI")
                    # Take the first valid entry from the array
                    parsed_response = parsed_response[0]
                    logging.debug(f"Extracted first entry from array: {parsed_response}")
                    logging.debug(f"Parsed response keys: {list(parsed_response.keys())}")
                    
                return parsed_response
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse AI response as JSON: {response_text}")
                logging.error(f"JSON decode error: {str(e)}")
                logging.error(f"Response length: {len(response_text)}")
                return {'error': 'Failed to parse AI response', 'response': response_text}
            except ValueError as e:
                logging.error(f"Invalid AI response format: {e} ({response_text})")
                return {'error': f'Invalid AI response format: {str(e)}', 'response': response_text}
    except Exception as e:
        logging.error(f"Error processing detailed analysis request: {e}")
        logging.exception("Full traceback:")
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


async def detailed_analysis_handler(request):
    """Perform detailed three-pass analysis of a radiology report.

    Takes a radiology report text and sends it to the LLM for detailed analysis
    using the ANA_PROMPT to extract comprehensive insights.

    Args:
        request: aiohttp request object with JSON body containing report text

    Returns:
        web.json_response: JSON response with detailed analysis results
    """
    try:
        data = await request.json()
        report_text = data.get('report', '').strip()

        result = await detailed_analysis_report(report_text)

        # Check if there was an error
        if 'error' in result:
            status = 500 if result['error'] != 'No report text provided' else 400
            return web.json_response(result, status=status)

        return web.json_response(result)
    except Exception as e:
        logging.error(f"Error processing detailed analysis request: {e}")
        return web.json_response({'error': 'Internal server error'}, status=500)

async def translate_handler(request):
    """Translate a Romanian radiology report to English.

    Takes a radiology report text and sends it to the LLM for translation
    using the TRN_PROMPT.

    Args:
        request: aiohttp request object with JSON body containing report text

    Returns:
        web.json_response: JSON response with translation results
    """
    try:
        # Get report text from request
        data = await request.json()
        report_text = data.get('report', '').strip()
        # Translate the report
        result = await translate_report(report_text)
        # Check if there was an error
        if result is None:
            return web.json_response({'error': 'Translation failed'}, status=500)
        return web.json_response({'translation': result})
    except Exception as e:
        logging.error(f"Error processing translation request: {e}")
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
        # Store user role and username in request for later use
        request.user_role = user_info['role']
        request.username = username
    except (ValueError, UnicodeDecodeError) as e:
        raise web.HTTPUnauthorized(
            text = "401: Invalid authentication",
            headers = {'WWW-Authenticate': 'Basic realm="XRayVision"'})
    return await handler(request)


async def broadcast_dashboard_update(event = None, payload = None, client = None):
    """Broadcast dashboard updates to all connected WebSocket clients.

    Sends real-time updates to dashboard clients including queue status,
    processing information, statistics, and AI health status.

    Args:
        event: Optional event name for specific update types
        payload: Optional data payload for the event
        client: Optional specific client to send update to (instead of all)
    """
    # Check if there are any clients
    if not (websocket_clients or client):
        return
    # Update the queue sizes
    dashboard['queue_size'] = db_count('exams', where_clause="status IN (?, ?)", where_params=('queued', 'requeue'))
    dashboard['check_queue_size'] = db_count('exams', where_clause="status = ?", where_params=('check',))
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
        logging.debug("NTFY notifications are disabled")
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
                    logging.debug("Successfully sent ntfy notification")
                else:
                    logging.warning(f"Notification failed with status {resp.status}: {await resp.text()}")
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
    checksum_digit = int(pid[12])
    
    # Validate gender digit (1-9)
    if gender_digit < 1 or gender_digit > 9:
        return {'valid': False}
    
    # Validate date components
    # Determine century based on gender digit
    century_map = {1: 1900, 2: 1900, 3: 1800, 4: 1800, 5: 2000, 6: 2000, 7: 2000, 8: 2000, 9: 1900}
    if gender_digit not in century_map:
        return {'valid': False}
    
    full_year = century_map[gender_digit] + year
    
    # Validate month (1-12) and day (1-31) with precise date validation
    try:
        birth_date = datetime(full_year, month, day)
    except ValueError:
        return {'valid': False}
    
    # Validate county code (01-52 excluding 47-50, 70-79, 90-99)
    valid_counties = set(range(1, 47)) | set(range(51, 53)) | set(range(70, 80)) | set(range(90, 100))
    if county not in valid_counties:
        return {'valid': False}
    
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
        info: Dictionary containing exam information with protocol name or string with protocol name

    Returns:
        tuple: (region, question) where region is the identified anatomic region
               and question is the region-specific query for AI analysis
    """
    # Handle both string and dict inputs
    if isinstance(info, str):
        desc = info.lower()
    else:
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


async def format_patient_name_for_fhir(dicom_name):
    """
    Format DICOM patient name as "last_name first_name" for FHIR search.
    
    Args:
        dicom_name: Patient name in DICOM format (Last^First^Middle)
        
    Returns:
        str: Formatted patient name as "last_name first_name"
    """
    if not dicom_name or not isinstance(dicom_name, str):
        return ""
    
    # Convert DICOM name format (Last^First^Middle) to "last_name first_name"
    if '^' in dicom_name:
        name_parts = dicom_name.split('^')
        # Extract last name (first part) and first name (second part)
        last_name = name_parts[0].strip() if len(name_parts) > 0 else ""
        first_name = name_parts[1].strip() if len(name_parts) > 1 else ""
        middle_name = name_parts[2].strip() if len(name_parts) > 2 else ""
        
        # Format as "last_name first_name middle_name" if the parts exist
        return f"{last_name} {first_name} {middle_name}".strip()
    else:
        return dicom_name.strip()


async def get_fhir_patient(session, cnp, patient_name=None):
    """
    Search for a patient in FHIR system by CNP, and if not found, by name.

    Args:
        session: aiohttp ClientSession instance
        cnp: Patient CNP
        patient_name: Patient full name (optional)

    Returns:
        dict or None: Patient data from FHIR if successful, None otherwise
    """
    logging.info(f"Starting FHIR patient search for CNP: {cnp}")
    try:
        # Use basic authentication
        auth = aiohttp.BasicAuth(FHIR_USERNAME, FHIR_PASSWORD)
        
        # First, try searching by CNP
        url = f"{FHIR_URL}/fhir/Patient"
        params = {'q': cnp}
        
        logging.debug(f"Sending FHIR patient search by CNP request to {url} with params {params}")
        async with session.get(url, auth=auth, params=params, timeout=30) as resp:
            logging.debug(f"Received FHIR patient search by CNP response with status {resp.status}")
            if resp.status == 200:
                data = await resp.json()
                logging.debug(f"FHIR patient search by CNP returned resourceType: {data.get('resourceType')}")
                if data.get('resourceType') == 'Patient':
                    # Single patient returned
                    logging.info(f"Found single patient by CNP {cnp}")
                    return data
                elif data.get('resourceType') == 'Bundle' and 'entry' in data:
                    # Multiple patients returned in a bundle
                    patients = []
                    for entry in data['entry']:
                        if 'resource' in entry and entry['resource'].get('resourceType') == 'Patient':
                            patients.append(entry['resource'])
                    if patients:
                        logging.info(f"Found {len(patients)} patients in bundle for CNP {cnp}")
                        # Validate CNP before proceeding
                        cnp_result = validate_romanian_cnp(cnp)
                        if not cnp_result['valid']:
                            logging.warning(f"Invalid CNP {cnp}, skipping patient selection")
                            return None
                        # Log warning about multiple patients
                        logging.info(f"Multiple patients found for CNP {cnp}, selecting the one with the greatest ID")
                        # Sort patients by ID (assuming IDs are numeric or comparable)
                        # and select the one with the greatest ID
                        patients.sort(key=lambda p: p.get('id', ''), reverse=True)
                        logging.info(f"Selected patient with ID {patients[0].get('id')} for CNP {cnp}")
                        return patients[0]
                    else:
                        logging.warning(f"FHIR patient search error: no valid patients found in bundle for CNP {cnp}")
                elif data.get('resourceType') == 'OperationOutcome':
                    # Handle OperationOutcome responses (typically errors)
                    issues = data.get('issue', [])
                    error_details = '; '.join([f"{issue.get('severity', 'unknown')}: {issue.get('diagnostics', issue.get('details', {}).get('text', 'no details'))}" for issue in issues])
                    logging.debug(f"FHIR patient search returned OperationOutcome for CNP {cnp}: {error_details}")
                    # Check if all issues are just informational - if so, we should still try name search
                    all_info = all(issue.get('severity', '').lower() == 'information' for issue in issues)
                    if not all_info:
                        # If there are non-informational issues, don't proceed to name search
                        logging.info(f"Non-informational issues found for CNP {cnp}, not proceeding to name search")
                        return None
                    else:
                        logging.info(f"Only informational issues found for CNP {cnp}, will proceed to name search")
                else:
                    logging.error(f"FHIR patient search error: unexpected response format for CNP {cnp}")
            else:
                logging.warning(f"FHIR patient search by CNP failed with status {resp.status}")
    except Exception as e:
        logging.error(f"FHIR patient search by CNP error: {e}")
    
    # If CNP search failed or returned only informational messages and patient_name is provided, try searching by name
    if patient_name:
        logging.info(f"Proceeding to name search for patient: {patient_name}")
        try:
            # Format patient name as "last_name first_name" for FHIR search if it's in DICOM format
            if '^' in patient_name:
                formatted_name = await format_patient_name_for_fhir(patient_name)
            else:
                formatted_name = patient_name.strip()
            
            if formatted_name:
                logging.info(f"Retrying FHIR patient search by name: {formatted_name}")
                params = {'q': formatted_name}
                
                logging.debug(f"Sending FHIR patient search by name request to {url} with params {params}")
                async with session.get(url, auth=auth, params=params, timeout=30) as resp:
                    logging.debug(f"Received FHIR patient search by name response with status {resp.status}")
                    if resp.status == 200:
                        data = await resp.json()
                        logging.debug(f"FHIR patient search by name returned resourceType: {data.get('resourceType')}")
                        if data.get('resourceType') == 'Patient':
                            # Single patient returned
                            logging.info(f"Found single patient by name '{formatted_name}'")
                            return data
                        elif data.get('resourceType') == 'Bundle' and 'entry' in data:
                            # Multiple patients returned in a bundle
                            # Log warning about multiple patients
                            logging.warning(f"Multiple patients found for name {formatted_name}, selecting no one")
                        elif data.get('resourceType') == 'OperationOutcome':
                            # Handle OperationOutcome responses (typically errors)
                            issues = data.get('issue', [])
                            error_details = '; '.join([f"{issue.get('severity', 'unknown')}: {issue.get('diagnostics', issue.get('details', {}).get('text', 'no details'))}" for issue in issues])
                            logging.warning(f"FHIR patient search returned OperationOutcome for name '{formatted_name}': {error_details}")
                        else:
                            logging.error(f"FHIR patient search by name error: unexpected response format for name '{formatted_name}'")
                    else:
                        logging.warning(f"FHIR patient search by name failed with status {resp.status}")
            else:
                logging.warning("Patient name is empty, skipping name search")
        except Exception as e:
            logging.error(f"FHIR patient search by name error: {e}")
    else:
        logging.info("No patient name provided, skipping name search")
    
    # If both searches failed, return None
    logging.info(f"FHIR patient search completed for CNP {cnp}, no patient found")
    return None

async def get_fhir_servicerequests(session, patient_id, exam_datetime, exam_type, exam_region):
    """
    Search for service requests for a patient in FHIR system.

    Args:
        session: aiohttp ClientSession instance
        patient_id: Patient ID from HIS
        exam_datetime: Exam datetime to search around
        exam_region: Exam region to filter by
        exam_type: Exam type to filter by (default: 'radio')

    Returns:
        list: List of service requests from FHIR (exactly one study) or empty list
    """
    try:
        # Use basic authentication
        auth = aiohttp.BasicAuth(FHIR_USERNAME, FHIR_PASSWORD)
        
        url = f"{FHIR_URL}/fhir/ServiceRequest"
        params = {
            'patient': patient_id,
            'dt': exam_datetime
        }
        if exam_type:
            params['type'] = exam_type
        if exam_region:
            params['region'] = exam_region
        
        # Try without full=yes parameter
        async with session.get(url, auth=auth, params=params, timeout=30) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get('resourceType') == 'Bundle' and 'entry' in data:
                    srv_reqs = []
                    for entry in data['entry']:
                        if 'resource' in entry and entry['resource'].get('resourceType') == 'ServiceRequest':
                            # Only add resources that have an 'id' field
                            if 'id' in entry['resource']:
                                srv_reqs.append(entry['resource'])
                            else:
                                logging.warning("FHIR service request resource missing 'id' field")
                    # We need exactly one study
                    if len(srv_reqs) == 1:
                        return srv_reqs
                    elif len(srv_reqs) > 1:
                        logging.info(f"FHIR service requests search returned {len(srv_reqs)} service requests, expected exactly one")
                    # Return empty list if no service requests or more than one
                    return []
                elif data.get('resourceType') == 'OperationOutcome':
                    # Handle OperationOutcome responses (typically errors)
                    issues = data.get('issue', [])
                    error_details = '; '.join([f"{issue.get('severity', 'unknown')}: {issue.get('diagnostics', issue.get('details', {}).get('text', 'no details'))}" for issue in issues])
                    logging.debug(f"FHIR service requests search returned OperationOutcome: {error_details}")
                    return []
                else:
                    logging.error(f"FHIR service requests search error: unexpected response format")
                    return []
            else:
                logging.warning(f"FHIR service requests search failed with status {resp.status}")
    except Exception as e:
        logging.error(f"FHIR service requests search error: {e}")
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
                # Check if response is an OperationOutcome (error)
                if data.get('resourceType') == 'OperationOutcome':
                    # Handle OperationOutcome responses (typically errors)
                    issues = data.get('issue', [])
                    error_details = '; '.join([f"{issue.get('severity', 'unknown')}: {issue.get('diagnostics', issue.get('details', {}).get('text', 'no details'))}" for issue in issues])
                    logging.warning(f"FHIR diagnostic report returned OperationOutcome: {error_details}")
                    return None
                # Ensure the resource type is DiagnosticReport
                elif data.get('resourceType') == 'DiagnosticReport':
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
    Send a request to the currently active AI API endpoint.

    Attempts to send a POST request to the active AI endpoint with the
    provided headers and payload. Handles HTTP errors and exceptions.

    Args:
        session: aiohttp ClientSession instance
        headers: HTTP headers for the request
        payload: JSON payload containing the request data

    Returns:
        dict or None: JSON response from API if successful, None otherwise
    """
    if not active_openai_url:
        logging.error("No active AI URL configured")
        return None
        
    try:
        async with session.post(active_openai_url, headers = headers, json = payload, timeout = 300) as resp:
            if resp.status == 200:
                return await resp.json()
            logging.warning(f"{active_openai_url} failed with status {resp.status}")
    except Exception as e:
        logging.error(f"{active_openai_url} request error: {e}")
    # Failed
    return None


async def update_patient_info_from_fhir(exam):
    """
    Try to get additional patient information from FHIR before processing.
    
    Args:
        exam: Dictionary containing exam information and metadata
        
    Returns:
        None
    """
    # Check if HIS integration is enabled
    if not ENABLE_HIS:
        return
        
    patient_cnp = exam['patient']['cnp']
    patient_name = exam['patient']['name']
    patient_birthdate = exam['patient']['birthdate']
    if patient_cnp and (not exam['patient']['id'] or not patient_birthdate or patient_birthdate == -1):
        async with aiohttp.ClientSession() as session:
            # Get patient information from FHIR (first by CNP, then by name if CNP fails)
            fhir_patient = await get_fhir_patient(session, patient_cnp, patient_name)
            if fhir_patient:
                # Update patient ID if found
                if 'id' in fhir_patient:
                    exam['patient']['id'] = fhir_patient['id']
                    # Update in database
                    db_update_patient_id(patient_cnp, fhir_patient['id'])
                
                # Update patient birthdate if not already known
                if (not patient_birthdate or patient_birthdate == -1) and 'birthDate' in fhir_patient:
                    try:
                        birthdate = fhir_patient['birthDate']
                        # Validate the format (should be YYYY-MM-DD)
                        if len(birthdate) == 10 and birthdate[4] == '-' and birthdate[7] == '-':
                            exam['patient']['birthdate'] = birthdate
                            # Calculate age from birthdate
                            birth_date = datetime.strptime(birthdate, "%Y-%m-%d")
                            today = datetime.now()
                            age = today.year - birth_date.year
                            if (today.month, today.day) < (birth_date.month, birth_date.day):
                                age -= 1
                            exam['patient']['age'] = age
                            # Update in database
                            db_update('patients', 'cnp = ?', (patient_cnp,), birthdate=birthdate)
                    except Exception as e:
                        logging.error(f"Error parsing birthdate from FHIR for patient {patient_cnp}: {e}")


async def prepare_exam_data(exam):
    """
    Prepare exam data for AI processing by identifying region, projection, etc.
    
    Args:
        exam: Dictionary containing exam information and metadata
        
    Returns:
        tuple: (region, question, subject, anatomy, image_bytes) or (None, None, None, None, None) if exam should be ignored
    """
    # Read the PNG file
    with open(os.path.join(IMAGES_DIR, f"{exam['uid']}.png"), 'rb') as f:
        image_bytes = f.read()
    # Identify the region
    region, question = identify_anatomic_region(exam)
    # Filter on specific region
    if not region in REGIONS:
        logging.info(f"Ignoring {exam['uid']} with {region} x-ray.")
        db_set_status(exam['uid'], 'ignore')
        return None, None, None, None, None
    # Identify the projection, gender and age
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
        
    return region, question, subject, anatomy, image_bytes


def create_ai_prompt(exam, region, question, subject, anatomy):
    """
    Create the AI prompt for the exam.
    
    Args:
        exam: Dictionary containing exam information and metadata
        region: Anatomic region
        question: Clinical question
        subject: Patient description
        anatomy: Anatomic region description
        
    Returns:
        str: Formatted prompt for AI
    """
    # Get previous reports for the same patient and region
    previous_reports = db_get_previous_reports(exam['patient']['cnp'], region, months=3)

    # Create the prompt
    prompt = USR_PROMPT.format(question=question, anatomy=anatomy, subject=subject)

    # Add justification if available
    if exam['report']['rad'].get('justification'):
        prompt += f"\n\nCLINICAL INFORMATION: {exam['report']['rad']['justification']}"

    # Append previous reports if any exist (limit to 3 most recent)
    if previous_reports:
        prompt += "\n\nPRIOR STUDIES:"
        # Limit to at most 3 previous reports
        for i, (report, date) in enumerate(previous_reports[:3], 1):
            prompt += f"\n\n[{date}] {report}"
        prompt += "\n\nCompare to prior studies. Note any new, stable, resolved, or progressive findings with dates."
    prompt += "\n\nIMPORTANT: Also identify any other lesions or abnormalities beyond the primary clinical question. Output JSON only."
    
    return prompt


def prepare_ai_request_data(prompt, image_bytes):
    """
    Prepare the request data for sending to AI API.
    
    Args:
        prompt: Formatted prompt for AI
        image_bytes: Image data as bytes
        
    Returns:
        tuple: (headers, data) for the AI API request
    """
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
    
    return headers, data


def process_ai_response(response_text, exam_uid):
    """
    Process the AI response and extract relevant information.
    
    Args:
        response_text: Raw response text from AI
        exam_uid: Exam unique identifier
        
    Returns:
        tuple: (short, report, confidence, severity, summary) or (None, None, None, None, None) if parsing failed
    """
    # Clean up markdown code fences (```json ... ```, ``` ... ```, etc.)
    response_text = re.sub(r"^```(?:json)?\s*", "", response_text, flags = re.IGNORECASE | re.MULTILINE)
    response_text = re.sub(r"\s*```$", "", response_text, flags = re.MULTILINE)
    # Clean up any text before '{'
    response_text = re.sub(r"^[^{]*{", "{", response_text, flags = re.IGNORECASE | re.MULTILINE)
    try:
        parsed = json.loads(response_text)
        short = parsed["short"].strip().lower()
        report = parsed["report"].strip()
        confidence = parsed.get("confidence", 0)
        severity = parsed.get("severity", -1)
        summary = parsed.get("summary", "").strip()
        if short not in ("yes", "no") or not report:
            raise ValueError("Invalid json format in AI response")
        return short, report, confidence, severity, summary
    except Exception as e:
        logging.error(f"Rejected malformed AI response: {e}")
        logging.error(response_text)
        return None, None, None, None, None


async def handle_ai_success(exam, short, report, confidence, severity, summary, processing_time, response_model=None):
    """
    Handle successful AI processing by updating database and sending notifications.
    
    Args:
        exam: Dictionary containing exam information and metadata
        short: AI response short field ("yes"/"no")
        report: AI generated report text
        confidence: AI confidence score
        severity: AI severity score
        summary: AI summary text
        processing_time: Time taken to process the exam
        response_model: Actual model name from API response
        
    Returns:
        bool: True if successful
    """
    # Save to exams database
    is_positive = short == "yes"
    # Save to exams database with processing time
    db_add_exam(exam)
    # If report is provided, add it to ai_reports table
    if report is not None:
        db_add_ai_report(exam['uid'], report, is_positive, confidence, response_model, int(processing_time), severity if severity is not None else -1, summary)
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


async def send_exam_to_openai(exam, max_retries = 3):
    """
    Send an exam's PNG image to the AI API for analysis.

    This is the core AI processing function that handles the complete workflow:
    1. Updates patient information from FHIR system
    2. Prepares exam data (region, projection, gender, age)
    3. Filters exams by supported regions
    4. Creates AI prompts with clinical context and prior reports
    5. Encodes images for AI analysis
    6. Sends requests with exponential backoff retries
    7. Parses and validates AI responses
    8. Stores results in the database
    9. Sends notifications for positive findings
    10. Updates dashboard with processing status

    The function implements robust error handling with automatic retries and
    proper status updates in the database for both success and failure cases.

    Args:
        exam (dict): Dictionary containing exam information and metadata including:
            - uid: Unique exam identifier
            - patient: Patient information (name, cnp, age, sex)
            - exam: Exam details (protocol, created timestamp, study/series UIDs)
            - report: Previous report data if reprocessing
        max_retries (int): Maximum number of retry attempts (default: 3)

    Returns:
        bool: True if successfully processed, False otherwise

    Processing Flow:
        1. FHIR Integration: Update patient info from hospital system
        2. Data Preparation: Extract region, projection, subject description
        3. Region Filtering: Only process exams from supported anatomic regions
        4. Prompt Engineering: Create context-rich prompts with clinical info
        5. Image Encoding: Convert PNG to base64 for AI API transmission
        6. Retry Logic: Exponential backoff (2s, 4s, 8s delays) on failures
        7. Response Parsing: Validate and extract AI-generated findings
        8. Database Storage: Save results with processing timing metrics
        9. Notification: Alert for positive findings via ntfy.sh
        10. Dashboard Update: Broadcast processing completion status
    """
    try:
        # Try to get additional patient and exam information from FHIR before processing
        await update_patient_info_from_fhir(exam)
                            
        # Prepare exam data
        region, question, subject, anatomy, image_bytes = await prepare_exam_data(exam)
        if region is None:  # Exam should be ignored
            return False
            
        # Create the prompt
        prompt = create_ai_prompt(exam, region, question, subject, anatomy)
        
        logging.debug(f"Prompt: {prompt}")
        logging.info(f"Processing {exam['uid']} with {region} x-ray.")
        if exam['report']['ai']['text']:
            json_report = {'short': exam['report']['ai']['short'],
                           'report': exam['report']['ai']['text']}
            exam['report']['json'] = json.dumps(json_report)
            logging.info(f"Previous report: {exam['report']['json']}")
            
        # Prepare request data
        headers, data = prepare_ai_request_data(prompt, image_bytes)
        
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
                    response_text = result["choices"][0]["message"]["content"].strip()
                    # Extract the actual model name from the API response
                    response_model = result.get("model", MODEL_NAME)
                    
                    # Process AI response
                    short, report, confidence, severity, summary = process_ai_response(response_text, exam['uid'])
                    if short is None:  # Parsing failed
                        break
                        
                    logging.info(f"AI API response for {exam['uid']}: [{short.upper()}] {report} (confidence: {confidence}, severity: {severity}, summary: {summary})")
                    
                    # Calculate timing statistics
                    global timings
                    end_time = asyncio.get_event_loop().time()
                    processing_time = end_time - start_time  # In seconds
                    timings['total'] = int(processing_time * 1000)  # Convert to milliseconds
                    if timings['average'] > 0:
                        timings['average'] = int((3 * timings['average'] + timings['total']) / 4)
                    else:
                        timings['average'] = timings['total']
                    
                    # Handle success
                    success = await handle_ai_success(exam, short, report, confidence, severity, summary, processing_time, response_model)
                    if success:
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
    app.router.add_get('/stats/radiologists', serve_radiologists_page)
    app.router.add_get('/stats/diagnostics', serve_diagnostics_page)
    app.router.add_get('/stats/insights', serve_insights_page)
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
    app.router.add_get('/api/diagnostics/monthly_trends', diagnostics_monthly_trends_handler)
    app.router.add_get('/api/radiologists', radiologists_handler)
    app.router.add_get('/api/stats/radiologists', radiologist_stats_handler)
    app.router.add_get('/api/stats/radiologists/monthly_trends', radiologists_monthly_trends_handler)
    app.router.add_get('/api/stats/diagnostics', diagnostics_stats_handler)
    app.router.add_get('/api/stats/insights', insights_handler)
    app.router.add_get('/api/severity', severity_handler)
    app.router.add_get('/api/config', config_handler)
    
    # API endpoints - Actions
    app.router.add_post('/api/dicomquery', dicom_query)
    app.router.add_post('/api/radreview', rad_review)
    app.router.add_post('/api/requeue', requeue_exam)
    app.router.add_post('/api/getrad', get_report_handler)
    app.router.add_post('/api/check', check_report_handler)
    app.router.add_post('/api/analyse', detailed_analysis_handler)
    app.router.add_post('/api/translate', translate_handler)
    
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
    Main processing loop that sends queued exams to the AI API.

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
        # Wait here if there are no items in queue or there is no AI server
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
                            if os.path.exists(dicom_file):
                                os.remove(dicom_file)
                                logging.debug(f"DICOM file {dicom_file} deleted after processing.")
                            else:
                                logging.debug(f"DICOM file {dicom_file} not found, skipping deletion.")
                        except Exception as e:
                            logging.warning(f"Error removing DICOM file {dicom_file}: {e}")
                    else:
                        logging.debug(f"Keeping DICOM file: {dicom_file}")
                else:
                    # Error already set in send_exam_to_openai
                    pass
            elif exam_status == 'check':
                # Check if AI report already has a summary
                ai_report = db_get_ai_report(exam['uid'])
                if ai_report and not ai_report.get('summary'):
                    # Process AI report with LLM to generate summary
                    await check_ai_report_and_update(exam['uid'])
                # Process FHIR report with LLM
                rad_check_success = await check_rad_report_and_update(exam['uid'])        
                # Notify dashboard of the update only if successful
                if rad_check_success:
                    await broadcast_dashboard_update(event="radcheck", payload={'uid': exam['uid']})
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
    Periodically check the health status of AI API endpoints.

    Tests both primary and secondary AI endpoints every 5 minutes,
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
                        logging.debug(f"Health check {url} → {resp.status}")
            except Exception as e:
                health_status[url] = False
                logging.debug(f"Health check failed for {url}: {e}")

        if health_status.get(OPENAI_URL_PRIMARY):
            active_openai_url = OPENAI_URL_PRIMARY
            logging.info("Using primary AI backend.")
        elif health_status.get(OPENAI_URL_SECONDARY):
            active_openai_url = OPENAI_URL_SECONDARY
            logging.info("Using secondary AI backend.")
        else:
            active_openai_url = None
            logging.error("No AI backend is currently healthy")
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
        # Check if HIS integration is enabled
        if not ENABLE_HIS:
            await asyncio.sleep(60)  # Check again in a minute
            continue
            
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
        
        # Random delay between 1 and 5 minutes
        delay = random.randint(30, 120)
        await asyncio.sleep(delay)

async def find_service_request(session, exam_uid, patient_id, exam_datetime, exam_type='radio', exam_region=''):
    """
    Find service request for an exam in FHIR system.

    Args:
        session: aiohttp ClientSession instance
        exam_uid: Exam unique identifier
        patient_id: Patient ID in HIS
        exam_datetime: Exam datetime
        exam_region: Exam region
        exam_type: Exam type (default: 'radio')

    Returns:
        dict or None: Study resource if found, None otherwise
    """
    # Search for service requests
    srv_reqs = await get_fhir_servicerequests(session, patient_id, exam_datetime, exam_type, exam_region)
    if not srv_reqs:
        logging.warning(f"No service requests found for exam {exam_uid}")
        return None
    elif len(srv_reqs) > 1:
        logging.info(f"Multiple close service requests found for exam {exam_uid}, skipping.")
        return None

    # Get the single request
    req = srv_reqs[0]
    if 'id' not in req:
        logging.warning(f"Service request for exam {exam_uid} has no ID, skipping.")
        return None
    
    # Return the service request
    return req

async def extract_report_data(report, exam_uid, exam_type = "radio", exam_region = ""):
    """
    Extract report text and radiologist name from FHIR diagnostic report.

    This function handles FHIR diagnostic reports that may contain multiple presented forms
    and selects the one that matches the expected exam region and type. It also extracts metadata
    like the radiologist name.

    Args:
        report (dict): FHIR diagnostic report resource containing presented forms and metadata
        exam_uid (str): Exam unique identifier for logging and error tracking
        exam_type (str, optional): Expected exam type to match in presented forms. Defaults to "radio"
        exam_region (str, optional): Expected anatomic region to match in presented forms. Defaults to ""

    Returns:
        tuple: (report_text, radiologist) where:
            - report_text (str): Extracted report text content, or None if extraction fails
            - radiologist (str): Radiologist name from resultsInterpreter, or empty string if not found
            Returns (None, None) if extraction fails
    """
    # Handle multiple presentedForm items by finding the one with the matching region and type
    report_text = None
    presented_form = None
    
    if len(report['presentedForm']) == 1:
        # Single presented form - use it directly
        presented_form = report['presentedForm'][0]
    else:
        # Multiple presented forms - find the one matching both the exam region and type
        logging.info(f"Found {len(report['presentedForm'])} items in presentedForm for '{exam_type}' exam {exam_uid}, looking for region '{exam_region}'")
        for form in report['presentedForm']:
            type_match = form.get('type', '').lower() == exam_type.lower()
            region_match = form.get('region', '').lower() == exam_region.lower()
            if region_match and type_match:
                presented_form = form
                break
        
        # If still no matching form found, log and return None
        if not presented_form:
            logging.warning(f"No presentedForm found with region '{exam_region}' for exam {exam_uid}")
            return None, None
    
    # Extract the report text from the selected presented form
    report_text = presented_form.get('data', '').strip()
    if not report_text:
        logging.warning(f"No data found in presentedForm for exam {exam_uid}")
        return None, None
        
    # Extract radiologist name from resultsInterpreter if available
    radiologist = ''  # Default value
    try:
        if 'resultsInterpreter' in report and len(report['resultsInterpreter']) > 0:
            interpreter = report['resultsInterpreter'][0]
            if 'display' in interpreter:
                radiologist = interpreter['display']
    except Exception as e:
        logging.warning(f"Could not extract radiologist name from FHIR report: {e}")

    # Return the extracted report text and radiologist name
    return report_text, radiologist

def translate_exam_type_to_fhir(exam_type):
    """
    Translate database exam type to FHIR-compatible values.
    
    Args:
        exam_type (str): Exam type from database (Modality)
        
    Returns:
        str: FHIR-compatible exam type
    """
    translation_map = {
        'CR': 'radio',
        'DX': 'radio',
        'CT': 'ct',
        'MR': 'irm',
        'US': 'eco',
        'RF': 'rads'
    }
    return translation_map.get(exam_type.upper(), 'radio')

async def process_single_exam_without_rad_report(session, exam, patient_id):
    """
    Process a single exam that doesn't have a radiologist report yet.

    This function retrieves the radiologist report for a specific exam from the FHIR system
    and prepares it for LLM analysis. It performs the following steps:
    1. Checks if service request ID is already in database
    2. If not found, finds the corresponding service request in FHIR
    3. Retrieves the diagnostic report
    4. Extracts report data (text, radiologist name, justification)
    5. Updates the local database with the report information
    6. Sets the exam status to 'check' to trigger LLM processing

    Args:
        session (aiohttp.ClientSession): Active HTTP session for FHIR API calls
        exam (dict): Dictionary containing exam information including uid, created timestamp, and region
        patient_id (str): Patient ID in the Hospital Information System (HIS)
    """
    # Check if HIS integration is enabled
    if not ENABLE_HIS:
        return
        
    exam_uid = exam['uid']
    exam_datetime = exam['created']
    exam_type = translate_exam_type_to_fhir(exam.get('type', 'radio'))
    exam_region = exam.get('region', '')
    
    # If the exam region is not in our supported regions, try to identify it again from the report text
    if exam_region not in REGIONS:
        # Try to identify the region from the report text
        identified_region, _ = identify_anatomic_region(exam.get('protocol', ''))
        if identified_region in REGIONS:
            logging.info(f"Re-identified region for exam {exam_uid}: {identified_region}")
            exam_region = identified_region
            # Update the region in the exams table
            db_update('exams', 'uid = ?', (exam_uid,), region=exam_region)
        else:
            logging.warning(f"Could not identify valid region for exam {exam_uid} from report text")

    # Check if we already have the service request ID in the database
    rad_report = db_get_rad_report(exam_uid)
    srv_req = None

    if rad_report and rad_report.get('id'):
        try:
            # Convert to int for comparison to avoid string vs int comparison errors
            service_id = int(rad_report['id'])
            if service_id > 0:
                # We already have the service request ID, use it directly
                srv_req = {'id': service_id}
                logging.info(f"Using existing service request ID {srv_req['id']} for exam {exam_uid}")
        except (ValueError, TypeError):
            # If conversion fails, treat as if no valid ID exists
            pass

    # Find service request in FHIR if not already found
    if not srv_req:
        srv_req = await find_service_request(session, exam_uid, patient_id, exam_datetime, exam_type, exam_region)

    # If no service request found, log and return
    if not srv_req or 'id' not in srv_req:
        # Check if exam is older than 1 month
        exam_date = datetime.strptime(exam_datetime, "%Y-%m-%d %H:%M:%S")
        one_month_ago = datetime.now() - timedelta(days=30)
        is_old_exam = exam_date < one_month_ago

        # Only insert/update rad report if exam is older than 1 month
        if is_old_exam:
            # If we don't have a record yet, insert a negative ID to mark as not found
            if rad_report:
                # Update existing record with negative ID
                db_update('rad_reports', 'uid = ?', (exam_uid,), id=-1)
                logging.info(f"Updated report for exam {exam_uid} to mark service request as not found")
            else:
                # Insert a negative service request ID into our database to mark as not found
                db_insert('rad_reports', uid=exam_uid, id=-1)
                logging.info(f"Service request missing for exam {exam_uid}")
        else:
            logging.info(f"Service request missing for recent exam {exam_uid}, skipping rad report creation")

        # Return if no service request found
        return
    # Extract justification from supportingInfo if available
    justification = ''
    try:
        # First try supportingInfo
        if 'supportingInfo' in srv_req and isinstance(srv_req['supportingInfo'], list) and len(srv_req['supportingInfo']) > 0:
            supporting_info = srv_req['supportingInfo'][0]
            if isinstance(supporting_info, dict) and 'display' in supporting_info and isinstance(supporting_info['display'], str):
                justification = supporting_info['display']
                logging.debug(f"Extracted justification from supportingInfo: {justification}")

        # If no justification found, try reason array
        if not justification and 'reason' in srv_req and isinstance(srv_req['reason'], list) and len(srv_req['reason']) > 0:
            reason = srv_req['reason'][0]
            if isinstance(reason, dict) and 'display' in reason and isinstance(reason['display'], str):
                justification = reason['display']
                logging.debug(f"Extracted justification from reason: {justification}")

        # If still no justification, log what we found in the service request
        if not justification:
            logging.debug(f"No justification found in service request. Available fields: {list(srv_req.keys())}")
            if 'supportingInfo' in srv_req:
                logging.debug(f"supportingInfo content: {srv_req['supportingInfo']}")
            if 'reason' in srv_req:
                logging.debug(f"reason content: {srv_req['reason']}")

    except Exception as e:
        logging.warning(f"Error extracting justification from service request: {e}")
        logging.debug(f"Service request structure: {srv_req}")
    
    # Get diagnostic report first
    report = await get_fhir_diagnosticreport(session, srv_req['id'])
    if not report or 'presentedForm' not in report or not report['presentedForm']:
        logging.debug(f"No presentedForm found in diagnostic report for exam {exam_uid}")
        return

    # Extract report data
    report_text, radiologist = await extract_report_data(report, exam_uid, exam_type=exam_type, exam_region=exam_region)
    if not report_text:
        return

    # Log the retrieved report
    logging.debug(f"Retrieved radiologist report for exam {exam_uid}: {' '.join(report_text.split()[:10])}...")
    
    # Insert or update the radiologist report in our database with all fields
    db_insert('rad_reports',
            uid=exam_uid,
            id=srv_req['id'],
            text=report_text,
            radiologist=radiologist,
            positive=-1,
            severity=-1,
            summary='',
            type=exam_type,
            justification=justification,
            model=MODEL_NAME,
            latency=-1)
    logging.debug(f"Saving the service request id {srv_req['id']} for {exam_type} exam {exam_uid} with justification: {justification}")

    # Set the exam status to 'check' for LLM processing in queue
    db_set_status(exam_uid, "check")
    # Notify the queue
    QUEUE_EVENT.set()

async def get_patient_id_from_fhir(session, patient_cnp, patient_name=None):
    """
    Get patient ID from FHIR system by CNP, and if not found, by name.

    Args:
        session: aiohttp ClientSession instance
        patient_cnp: Patient CNP
        patient_name: Patient full name (optional)

    Returns:
        str or None: Patient ID from FHIR if successful, None otherwise
    """
    fhir_patient = await get_fhir_patient(session, patient_cnp, patient_name)
    if fhir_patient and 'id' in fhir_patient:
        patient_id = fhir_patient['id']
        # Update patient ID in database
        db_update_patient_id(patient_cnp, patient_id)
        return patient_id
    return None


async def process_exams_without_rad_reports(session):
    """
    Process exams that don't have radiologist reports yet.

    This function identifies exams without radiologist reports, finds the
    corresponding patient in HIS, and retrieves the radiologist report.
    """
    # Get exams for a patient without radiologist reports
    result = db_get_exams_without_rad_report()
    if not result or not result.get('exams'):
        return
    
    # Extract necessary information from the first exam
    patient_cnp = result['patient']['cnp']
    patient_id = result['patient']['id']
    exams = result['exams']
    
    # If patient ID is not known, search for it in FHIR
    patient_name = result['patient']['name']
    if not patient_id:
        patient_id = await get_patient_id_from_fhir(session, patient_cnp, patient_name)
    # If still no patient ID, log and skip
    if not patient_id:
        logging.warning(f"Could not find FHIR patient for CNP {patient_cnp} or name '{patient_name}', skipping exams")
        return
    
    # Process each exam for this patient
    for exam in exams:
        await process_single_exam_without_rad_report(session, exam, patient_id)

async def query_retrieve_loop():
    """
    Periodically query the remote DICOM server for new studies.

    Runs an infinite loop that queries the remote PACS for new CR studies
    at a configurable interval with +/- 30% random variation.
    Can be disabled with the --no-query flag.
    Updates the next_query timestamp for dashboard display.
    """
    if NO_QUERY:
        logging.warning(f"Automatic Query/Retrieve disabled.")
    while not NO_QUERY:
        await query_and_retrieve()
        # Calculate delay with +/- 30% variation
        variation = QUERY_INTERVAL * 0.3
        min_delay = max(1, int(QUERY_INTERVAL - variation))
        max_delay = int(QUERY_INTERVAL + variation)
        delay = random.randint(min_delay, max_delay)
        current_time = datetime.now()
        global next_query
        next_query = current_time + timedelta(seconds = delay)
        logging.debug(f"Next Query/Retrieve at {next_query.strftime('%Y-%m-%d %H:%M:%S')} (in {delay} seconds)")
        await asyncio.sleep(delay)


async def translate_existing_reports():
    """
    Translate existing radiologist reports that don't have English translations yet.

    This function finds all radiologist reports without English translations
    and translates them using the LLM.
    """
    try:
        # Get all exams with radiologist reports that don't have translations
        query = """
            SELECT uid, text
            FROM rad_reports
            WHERE text IS NOT NULL
            AND (text_en IS NULL OR text_en = '')
            AND severity > -1
        """
        rows = db_execute_query(query, fetch_mode='all')

        if not rows:
            logging.info("No reports found that need translation")
            return

        logging.info(f"Found {len(rows)} reports to translate")
        # Process each report
        for row in rows:
            uid, report_text = row
            try:
                logging.info(f"Translating report for exam {uid}")
                translation = await translate_report(report_text)

                if translation:
                    # Update the database with the translation
                    db_update('rad_reports', 'uid = ?', (uid,), text_en=translation)
                    logging.info(f"Successfully translated and updated exam {uid}")
                else:
                    logging.warning(f"Translation failed for exam {uid}")

                # Small delay to avoid overwhelming the AI service
                await asyncio.sleep(10)

            except Exception as e:
                logging.error(f"Error translating report for exam {uid}: {e}")

        logging.info("Translation of existing reports completed")

    except Exception as e:
        logging.error(f"Error in translate_existing_reports: {e}")

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
    # Add verification service for C-ECHO
    dicom_server.add_supported_context(Verification)
    # Accept only XRays
    dicom_server.add_supported_context(ComputedRadiographyImageStorage)
    dicom_server.add_supported_context(DigitalXRayImageStorageForPresentation)
    # C-Store handler
    handlers = [
        (evt.EVT_C_STORE, dicom_store),
        (evt.EVT_C_ECHO, lambda event: 0x0000)  # Simple C-ECHO handler that always succeeds
    ]
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
            # Remove the exam entry from the database
            db_execute_query_retry("DELETE FROM exams WHERE uid = ?", (uid,))
            # Remove the DICOM file
            try:
                os.remove(dicom_file)
                logging.info(f"Removed DICOM file {dicom_file} due to metadata extraction error")
            except Exception as rm_err:
                logging.error(f"Failed to remove DICOM file {dicom_file}: {rm_err}")
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

    # Reset any exams stuck in 'processing' status back to 'queued'
    reset_count = db_update('exams', "status = ?", ('processing',), status='queued')
    if reset_count and reset_count > 0:
        logging.info(f"Reset {reset_count} exams from 'processing' to 'queued' status")
        # Signal the queue to process these reset exams
        QUEUE_EVENT.set()

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
    if TRANSLATE_EXISTING:
        tasks.append(asyncio.create_task(translate_existing_reports()))
    # Preload the existing dicom files
    if LOAD_DICOM:
        await load_existing_dicom_files()
        # Query for studies from the last hour on startup
        await query_and_retrieve(60)
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
    parser = argparse.ArgumentParser(description = "XRayVision - Async DICOM processor with AI and WebSocket dashboard")
    parser.add_argument("--keep-dicom", action = "store_true", default=KEEP_DICOM, help = "Do not delete .dcm files after conversion")
    parser.add_argument("--load-dicom", action = "store_true", default=LOAD_DICOM, help = "Load existing .dcm files in queue")
    parser.add_argument("--no-query", action = "store_true", default=NO_QUERY, help = "Do not query the DICOM server automatically")
    parser.add_argument("--enable-ntfy", action = "store_true", default=ENABLE_NTFY, help = "Enable ntfy.sh notifications")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name to use for analysis")
    parser.add_argument("--retrieval-method", type=str, choices=['C-MOVE', 'C-GET'], default=RETRIEVAL_METHOD, help="DICOM retrieval method")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set logging level")
    parser.add_argument("--translate-existing", action = "store_true", help = "Translate existing radiologist reports without English translations")
    args = parser.parse_args()
    # Store in globals
    KEEP_DICOM = args.keep_dicom
    LOAD_DICOM = args.load_dicom
    NO_QUERY = args.no_query
    ENABLE_NTFY = args.enable_ntfy
    MODEL_NAME = args.model
    RETRIEVAL_METHOD = args.retrieval_method
    TRANSLATE_EXISTING = args.translate_existing
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
