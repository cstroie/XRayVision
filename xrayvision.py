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

import argparse
import asyncio
import base64
import csv
import glob
import io
import json
import logging
import math
import os
import re
import signal
import sqlite3
import random
import time
from datetime import datetime, timedelta
from typing import Optional

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(message)s',
    handlers=[
        logging.FileHandler("xrayvision.log"),
        logging.StreamHandler()
    ]
)
logging.getLogger('aiohttp').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('pynetdicom').setLevel(logging.WARNING)
logging.getLogger('pydicom').setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.INFO)

# Separate audit log for security and clinical actions
audit_logger = logging.getLogger('xrayvision.audit')
audit_logger.setLevel(logging.INFO)
audit_logger.propagate = False
_audit_handler = logging.FileHandler("xrayvision_audit.log")
_audit_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)8s | %(message)s'))
audit_logger.addHandler(_audit_handler)

import configparser

APP_NAME    = 'XRayVision'
APP_VERSION = '1.0'
USER_AGENT  = f'{APP_NAME}/{APP_VERSION}'

DEFAULT_CONFIG = {
    'general': {
        'XRAYVISION_DB_PATH': 'xrayvision.db',
        'XRAYVISION_BACKUP_DIR': 'backup',
        'BACKUP_MAX_FILES': '30'
    },
    'dicom': {
        'AE_TITLE': 'XRAYVISION',
        'AE_PORT': '4010',
        'REMOTE_AE_TITLE': 'DICOM_SERVER',
        'REMOTE_AE_IP': '192.168.1.1',
        'DICOM_MODALITIES': 'CR,DX',
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
        'NTFY_URL': 'https://ntfy.sh/xrayvision-alerts',
        'NTFY_IMAGE_BASE_URL': ''
    },
    'processing': {
        'PAGE_SIZE': '10',
        'KEEP_DICOM': 'False',
        'LOAD_DICOM': 'False',
        'NO_QUERY': 'False',
        'ENABLE_NTFY': 'False',
        'ENABLE_HIS': 'True',
        'QUERY_INTERVAL': '300',
        'SEVERITY_THRESHOLD': '5'
    },
    'fhir': {
        'FHIR_URL': 'http://127.0.0.1:44660',
        'FHIR_USERNAME': 'hipocrate',
        'FHIR_PASSWORD': 'hipocrate'
    }
}

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

MEDICAL_ACRONYMS = {
    "AD": "right atrium",
    "AP": "antero-posterior",
    "APP": "pathological personal history",
    "ATI": "intensive care unit",
    "CT": "computed tomography",
    "CTL": "cervico-toraco-lumbar",
    "CV": "venous catheter",
    "CVC": "central venous catheter",
    "DD": "dorsal decubitus",
    "DR": "right side",
    "DVP": "ventriculo-peritoneal derivation",
    "FID": "right iliac fossa",
    "FIS": "left iliac fossa",
    "IOT": "tracheal tube",
    "LID": "inferior right lobe",
    "LIS": "inferior left lobe",
    "LSD": "superior right lobe",
    "LSS": "superior left lobe",
    "NZG": "naso-gastric",
    "PAI": "interstitial pneumonia",
    "PN": "nasal bones",
    "PU": "pielo-uretheral",
    "RD": "right kidney",
    "RG": "xray",
    "RMN": "MRI",
    "RP": "pleural effusion",
    "RS": "left kidney",
    "RVU": "vesico-uretheral reflux",
    "SAF": "paranasal sinuses",
    "SCD": "costo-diaphramatic angles",
    "SF": "frontal sinus",
    "SM": "maxilar sinus",
    "SNG": "naso-gastric catheter",
    "STG": "left",
    "TCC": "cranio-cerebral trauma",
    "UPU": "ER",
    "VCI": "inferior vena cava",
    "VCS": "superior vena cava",
    "VP": "portal vein",
    "VS": "left ventricle",
    "VU": "bladder"
}

config = configparser.ConfigParser()
config.read_dict(DEFAULT_CONFIG)
try:
    config.read('xrayvision.cfg')
    logging.info("Configuration loaded from xrayvision.cfg")
    local_config_files = config.read('local.cfg')
    if local_config_files:
        logging.debug("Local configuration loaded from local.cfg")
except Exception as e:
    logging.error(f"Failed to load configuration file: {e}; using default values")

USERS = {}
if 'users' in config:
    for user in config['users']:
        try:
            password, role = config.get('users', user).split(',', 1)
            USERS[user.strip()] = {
                'password': password.strip(),
                'role': role.strip()
            }
        except ValueError:
            logging.error(f"Malformed user entry '{user}' in config — expected 'password,role'")

OPENAI_URL_PRIMARY = config.get('openai', 'OPENAI_URL_PRIMARY')
OPENAI_URL_SECONDARY = config.get('openai', 'OPENAI_URL_SECONDARY')
OPENAI_API_KEY = config.get('openai', 'OPENAI_API_KEY')
NTFY_URL = config.get('notifications', 'NTFY_URL')
NTFY_IMAGE_BASE_URL = config.get('notifications', 'NTFY_IMAGE_BASE_URL')
IMAGES_DIR = 'images'
STATIC_DIR = 'static'
DB_FILE = config.get('general', 'XRAYVISION_DB_PATH')
BACKUP_DIR = config.get('general', 'XRAYVISION_BACKUP_DIR')
BACKUP_MAX_FILES = config.getint('general', 'BACKUP_MAX_FILES', fallback=30)
MODEL_NAME = config.get('openai', 'MODEL_NAME')
AE_TITLE = config.get('dicom', 'AE_TITLE')
REMOTE_AE_TITLE = config.get('dicom', 'REMOTE_AE_TITLE')
REMOTE_AE_IP = config.get('dicom', 'REMOTE_AE_IP')
RETRIEVAL_METHOD = config.get('dicom', 'RETRIEVAL_METHOD')
DICOM_MODALITIES = [m.strip().upper() for m in config.get('dicom', 'DICOM_MODALITIES', fallback='CR,DX').split(',')]
FHIR_URL = config.get('fhir', 'FHIR_URL')
FHIR_USERNAME = config.get('fhir', 'FHIR_USERNAME')
FHIR_PASSWORD = config.get('fhir', 'FHIR_PASSWORD')
try:
    DASHBOARD_PORT = config.getint('dashboard', 'DASHBOARD_PORT')
    AE_PORT = config.getint('dicom', 'AE_PORT')
    REMOTE_AE_PORT = config.getint('dicom', 'REMOTE_AE_PORT')
except ValueError as e:
    logging.error(f"Invalid integer in configuration: {e}")
    raise SystemExit(1)

def load_prompts():
    prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')
    prompts = {}
    
    prompt_files = {
        'REP_PROMPT': 'rep_prompt.txt',
        'USR_PROMPT': 'usr_prompt.txt',
        'REV_PROMPT': 'rev_prompt.txt',
        'CHK_PROMPT': 'chk_prompt.txt',
        'ANA_PROMPT': 'ana_prompt.txt',
        'TRN_PROMPT': 'trn_prompt.txt',
    }
    
    for prompt_name, filename in prompt_files.items():
        filepath = os.path.join(prompts_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                prompts[prompt_name] = f.read()
            logging.debug(f"Loaded prompt: {prompt_name}")
        except FileNotFoundError:
            logging.error(f"Prompt file not found: {filepath}")
            prompts[prompt_name] = ""
        except Exception as e:
            logging.error(f"Error loading prompt {prompt_name}: {e}")
            prompts[prompt_name] = ""
    
    return prompts

PROMPTS = load_prompts()

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

MAIN_LOOP = None
QUEUE_EVENT = asyncio.Event()
next_query = None

dicom_server = None
web_server = None

active_openai_url = None  # Currently active AI API endpoint; None until health check passes
health_status = {
    OPENAI_URL_PRIMARY: False,
    OPENAI_URL_SECONDARY: False,
    FHIR_URL: False
}
timings = {
    'examination': 0,
    'translation': 0,
    'checking': 0,
    'analysis': 0
}

TRANSLATE_EXISTING = False
try:
    PAGE_SIZE = config.getint('processing', 'PAGE_SIZE')
    KEEP_DICOM = config.getboolean('processing', 'KEEP_DICOM')
    LOAD_DICOM = config.getboolean('processing', 'LOAD_DICOM')
    NO_QUERY = config.getboolean('processing', 'NO_QUERY')
    ENABLE_NTFY = config.getboolean('processing', 'ENABLE_NTFY')
    ENABLE_HIS = config.getboolean('processing', 'ENABLE_HIS')
    QUERY_INTERVAL = config.getint('processing', 'QUERY_INTERVAL')
    SEVERITY_THRESHOLD = config.getint('processing', 'SEVERITY_THRESHOLD')
except ValueError as e:
    logging.error(f"Invalid value in [processing] configuration: {e}")
    raise SystemExit(1)

REGION_RULES = {}
if 'regions' in config:
    for key in config['regions']:
        REGION_RULES[key] = [word.strip() for word in config['regions'][key].split(',')]
else:
    logging.warning("No [regions] section in configuration; region detection disabled")

REGION_QUESTIONS = {}
if 'questions' in config:
    for key in config['questions']:
        REGION_QUESTIONS[key] = config['questions'][key]
else:
    logging.warning("No [questions] section in configuration; region questions disabled")

# Maps internal sub-region names (e.g. 'occipital') to the HIS/FHIR region name ('skull') for service request lookup
REGION_FHIR_MAP = {}
if 'region_fhir_map' in config:
    for key in config['region_fhir_map']:
        REGION_FHIR_MAP[key] = config['region_fhir_map'][key]

# Items delimited by '|' because template values contain commas
REGION_TEMPLATES = {}
if 'templates' in config:
    for key in config['templates']:
        items = [t.strip() for t in config['templates'][key].split('|') if t.strip()]
        REGION_TEMPLATES[key] = items
else:
    logging.warning("No [templates] section in configuration; region reporting templates disabled")

REGIONS = []
if 'supported_regions' in config:
    for key in config['supported_regions']:
        try:
            if config.getboolean('supported_regions', key):
                REGIONS.append(key)
        except ValueError:
            logging.error(f"Invalid boolean for supported_regions.{key}; skipping")
else:
    logging.warning("No [supported_regions] section in configuration; all regions will be ignored")

dashboard = {
    'queue_size': 0,
    'check_queue_size': 0,
    'processing': None,
    'success_count': 0,
    'error_count': 0,
    'ignore_count': 0
}





def handle_error(e, context="", default_return=None, raise_on_error=False):
    error_msg = f"Error{f' in {context}' if context else ''}: {e}"
    logging.error(error_msg)

    if raise_on_error:
        raise e

    return default_return

# Database operations
def _db_connect() -> sqlite3.Connection:
    # WAL mode is a persistent DB-level setting set once in db_init; other PRAGMAs are per-connection
    conn = sqlite3.connect(DB_FILE, isolation_level=None)
    conn.execute('PRAGMA synchronous = NORMAL')
    conn.execute('PRAGMA foreign_keys = ON')
    conn.execute('PRAGMA cache_size = 10000')
    conn.execute('PRAGMA temp_store = MEMORY')
    conn.execute('PRAGMA mmap_size = 268435456')  # 256 MB
    return conn


def db_init():
    with sqlite3.connect(DB_FILE, isolation_level=None) as conn:
        # WAL must be set once at database creation, not per-connection
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('PRAGMA cache_size = 10000')
        conn.execute('PRAGMA temp_store = MEMORY')
        conn.execute('PRAGMA mmap_size = 268435456')  # 256 MB
        try:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS patients (
                    cnp TEXT PRIMARY KEY,
                    id TEXT,
                    name TEXT,
                    birthdate TEXT,
                    sex TEXT CHECK(sex IN ('M', 'F', 'O'))
                )
            ''')
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
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_rad_reports_radiologist
                ON rad_reports(radiologist)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_rad_reports_summary
                ON rad_reports(summary)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_rad_reports_severity
                ON rad_reports(severity)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_ai_reports_severity
                ON ai_reports(severity)
            ''')

            conn.commit()
            logging.info("Initialized SQLite database with normalized schema.")
        except Exception as e:
            conn.rollback()
            logging.error(f"Failed to initialize database: {e}")
            raise


def db_execute_query(query: str, params: tuple = (), fetch_mode: str = 'all') -> Optional[list]:
    with _db_connect() as conn:
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            if fetch_mode == 'all':
                return cursor.fetchall()
            elif fetch_mode == 'one':
                return cursor.fetchone()
        except Exception as e:
            conn.rollback()
            return handle_error(e, "database query execution", None, raise_on_error=False)


def db_execute_query_retry(query: str, params: tuple = (), max_retries: int = 5) -> Optional[int]:
    with _db_connect() as conn:
        for attempt in range(max_retries):
            try:
                conn.execute('BEGIN IMMEDIATE')
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                return cursor.rowcount
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))
                    continue
                conn.rollback()
                return handle_error(e, "database query with retry", None, raise_on_error=False)
            except Exception as e:
                conn.rollback()
                return handle_error(e, "database query with retry", None, raise_on_error=False)
    return None


_db_analyze_cache = {}
_translation_cache = {}
websocket_clients = set()


def clear_db_analyze_cache():
    global _db_analyze_cache
    cache_size = len(_db_analyze_cache)
    _db_analyze_cache.clear()
    if cache_size > 0:
        logging.debug(f"Cleared db_analyze_cache ({cache_size} entries)")


def clear_translation_cache():
    global _translation_cache
    cache_size = len(_translation_cache)
    _translation_cache.clear()
    if cache_size > 0:
        logging.debug(f"Cleared translation_cache ({cache_size} entries)")



def db_analyze(table_name):
    if table_name in _db_analyze_cache:
        return _db_analyze_cache[table_name]

    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', table_name):  # guard against SQL injection via table name
        logging.error(f"db_analyze: invalid table name '{table_name}'")
        return None, []

    query = f"PRAGMA table_info({table_name})"
    rows = db_execute_query(query, fetch_mode='all')
    
    if not rows:
        return None, []
    
    primary_key = None
    columns = []
    
    for row in rows:
        cid, name, type, notnull, dflt_value, pk = row
        columns.append(name)
        if pk:
            primary_key = name
    result = (primary_key, columns)
    _db_analyze_cache[table_name] = result
    return result

def db_unpack_result(result: list, keys: list) -> dict:
    if not result or not keys:
        return {}
    return dict(zip(keys, result))


def db_create_insert_query(table_name, *columns):
    placeholders = ', '.join(['?'] * len(columns))
    columns_str = ', '.join(columns)
    return f'INSERT OR REPLACE INTO {table_name} ({columns_str}) VALUES ({placeholders})'


def db_create_select_query(table_name, *columns, where=None, order_by=None, asc=True, limit=None):
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
    if columns is None:
        _, all_columns = db_analyze(table_name)
        columns = all_columns
    query = db_create_select_query(table_name, *columns, where=where_clause, order_by=order_by, asc=asc, limit=limit)
    params = where_params if where_params else ()
    rows = db_execute_query(query, params, fetch_mode='all')
    if rows:
        return [db_unpack_result(row, columns) for row in rows]
    return []


def db_count(table_name, where_clause=None, where_params=None):
    query = f"SELECT COUNT(*) FROM {table_name}"
    params = ()
    if where_clause:
        query += f" WHERE {where_clause}"
        params = where_params if where_params else ()
    
    result = db_execute_query(query, params, fetch_mode='one')
    return result[0] if result else 0


def db_update(table_name, where_clause, where_params, **kwargs):
    if not kwargs:
        return 0
    set_columns = list(kwargs.keys())
    set_values = list(kwargs.values())
    set_clause = ', '.join([f'{col} = ?' for col in set_columns])
    query = f'UPDATE {table_name} SET {set_clause} WHERE {where_clause}'
    params = set_values + list(where_params)
    
    return db_execute_query_retry(query, tuple(params))


def db_insert(table_name, **kwargs):
    if not kwargs:
        return 0
    columns = list(kwargs.keys())
    values = list(kwargs.values())
    query = db_create_insert_query(table_name, *columns)
    return db_execute_query_retry(query, tuple(values))


def db_select_one(table_name, pk_value):
    primary_key, columns = db_analyze(table_name)
    if not primary_key or not columns:
        return None
    where_clause = f"{primary_key} = ?"
    query = db_create_select_query(table_name, *columns, where=where_clause)
    result = db_execute_query(query, (pk_value,), fetch_mode='one')
    if result:
        return db_unpack_result(result, columns)
    return None


def db_add_patient(cnp, id, name, birthdate, sex):
    query = db_create_insert_query('patients', 'cnp', 'id', 'name', 'birthdate', 'sex')
    params = (cnp, id, name, birthdate, sex)
    return db_execute_query_retry(query, params)


def db_get_exams_without_rad_report():
    """Pick one patient with pending FHIR reports; doubles the search window on miss (up to 52 weeks)."""
    weeks = random.randint(1, 52)
    for attempt in range(4):
        cutoff_date = datetime.now() - timedelta(weeks=weeks)
        cutoff_date_str = cutoff_date.strftime('%Y-%m-%d %H:%M:%S')
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
        # rr.id IS NULL means no record; rr.id = -1 is the sentinel for "permanently not found" (excluded)
        patient_row = db_execute_query(patient_query, (cutoff_date_str,), fetch_mode='one')
        if patient_row:
            patient_cnp, patient_name, patient_id, patient_birthdate, patient_sex = patient_row
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
        if attempt == 0:
            weeks = min(weeks * 2, 52)
    return {}

def db_update_patient_id(cnp, patient_id):
    db_update('patients', 'cnp = ?', (cnp,), id=patient_id)


def db_add_exam(info):
    patient = info["patient"]
    db_add_patient(
        patient["cnp"],
        patient.get("id",""),
        patient["name"],
        patient.get("birthdate", None),
        patient["sex"]
    )
    exam = info["exam"]
    db_insert('exams',
              uid=info['uid'],
              cnp=patient["cnp"],
              id=exam.get("id",""),
              created=exam['created'],
              protocol=exam["protocol"],
              region=exam['region'],
              type=exam.get("type", "CR"),
              status='queued',
              study=exam.get("study"),
              series=exam.get("series"))


def db_get_exams(limit = PAGE_SIZE, offset = 0, **filters):
    conditions = []
    params = []
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
            conditions.append("((ar.severity >= ? AND rr.severity >= ?) OR (ar.severity < ? AND rr.severity < ?))")
            params.extend([SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD])
        else:
            conditions.append("((ar.severity >= ? AND rr.severity < ? AND rr.severity > -1) OR (ar.severity < ? AND rr.severity >= ?))")
            params.extend([SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD])
    if 'region' in filters:
        conditions.append("LOWER(e.region) LIKE ?")
        params.append(f"%{filters['region'].lower()}%")
    if 'status' in filters:
        status_value = filters['status']
        if isinstance(status_value, list):
            placeholders = ','.join(['?'] * len(status_value))
            conditions.append(f"LOWER(e.status) IN ({placeholders})")
            params.extend([s.lower() for s in status_value])
        else:
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
        conditions.append("e.uid = ?")
        params.append(filters['uid'])
    if 'cnp' in filters:
        conditions.append("p.cnp = ?")
        params.append(filters['cnp'])
    if 'severity' in filters:
        severity_value = str(filters['severity']).strip()
        # Interval notation: "3-6", "-8" (up to 8), "2-" (2 and above)
        if '-' in severity_value and severity_value != '-':
            parts = severity_value.split('-', 1)
            try:
                if parts[0] == '':
                    upper = int(parts[1])
                    conditions.append("rr.severity >= ? AND rr.severity <= ?")
                    params.extend([0, upper])
                elif parts[1] == '':
                    lower = int(parts[0])
                    conditions.append("rr.severity >= ? AND rr.severity <= ?")
                    params.extend([lower, 10])
                else:
                    lower = int(parts[0])
                    upper = int(parts[1])
                    conditions.append("rr.severity >= ? AND rr.severity <= ?")
                    params.extend([lower, upper])
            except ValueError:
                try:
                    exact_val = int(severity_value)
                    conditions.append("rr.severity = ?")
                    params.append(exact_val)
                except ValueError:
                    pass
        else:
            try:
                exact_val = int(severity_value)
                conditions.append("rr.severity = ?")
                params.append(exact_val)
            except ValueError:
                pass

    if 'confidence' in filters:
        conf_value = str(filters['confidence']).strip()
        # Interval notation: "80-", "-49", "50-79"
        if '-' in conf_value and conf_value != '-':
            parts = conf_value.split('-', 1)
            try:
                if parts[0] == '':
                    upper = int(parts[1])
                    conditions.append("ar.confidence >= 0 AND ar.confidence <= ?")
                    params.append(upper)
                elif parts[1] == '':
                    lower = int(parts[0])
                    conditions.append("ar.confidence >= ?")
                    params.append(lower)
                else:
                    lower = int(parts[0])
                    upper = int(parts[1])
                    conditions.append("ar.confidence >= ? AND ar.confidence <= ?")
                    params.extend([lower, upper])
            except ValueError:
                pass
        else:
            try:
                exact_val = int(conf_value)
                conditions.append("ar.confidence = ?")
                params.append(exact_val)
            except ValueError:
                pass

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
                
            try:
                dt = datetime.strptime(exam_created, "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                dt = datetime.min
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
        AND ar.text != ''
        AND length(ar.text) >= 30
        AND ar.text NOT LIKE '%ROLE%'
        AND ar.text NOT LIKE '%TASK%'
        AND ar.text NOT LIKE '%ASSESS IN ORDER%'
        AND ar.text NOT LIKE '%OUTPUT CONSTRAINTS%'
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
    results = db_select('exams', ['uid'], where_clause='study = ?',
                       where_params=(study_uid,))
    return len(results) > 0


def db_get_stats():
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
    
    query = """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN rr.severity > -1 THEN 1 ELSE 0 END) AS reviewed
        FROM exams e
        LEFT JOIN rad_reports rr ON e.uid = rr.uid
        WHERE e.status = 'done'
    """
    row = db_execute_query(query, fetch_mode='one')
    if row:
        (total, reviewed) = row
        stats["total"] = total
        stats["reviewed"] = reviewed or 0

    query = """
        SELECT
            SUM(CASE WHEN (ar.severity >= ? AND rr.severity >= ?) THEN 1 ELSE 0 END) AS tpos,
            SUM(CASE WHEN (ar.severity < ? AND rr.severity < ? AND rr.severity > -1) THEN 1 ELSE 0 END) AS tneg,
            SUM(CASE WHEN (ar.severity >= ? AND rr.severity < ? AND rr.severity > -1) THEN 1 ELSE 0 END) AS fpos,
            SUM(CASE WHEN (ar.severity < ? AND rr.severity >= ?) THEN 1 ELSE 0 END) AS fneg
        FROM exams e
        LEFT JOIN ai_reports ar ON e.uid = ar.uid
        LEFT JOIN rad_reports rr ON e.uid = rr.uid
        WHERE e.status = 'done'
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

        denominator = math.sqrt((tpos + fpos) * (tpos + fneg) * (tneg + fpos) * (tneg + fneg))
        if denominator == 0:
            stats["mcc"] = 0.0
        else:
            mcc = (tpos * tneg - fpos * fneg) / denominator
            stats["mcc"] = round(mcc, 2)

    query = """
        SELECT
            AVG(CAST(ar.latency AS REAL)) AS avg_processing_time,
            COUNT(*) * 1.0 / (SUM(CAST(ar.latency AS REAL)) + 1) AS throughput
        FROM exams e
        LEFT JOIN ai_reports ar ON e.uid = ar.uid
        WHERE e.status = 'done'
          AND ar.latency IS NOT NULL
          AND ar.latency >= 0
          AND e.created >= datetime('now', '-1 days')
    """
    timing_row = db_execute_query(query, fetch_mode='one')
    if timing_row and timing_row[0] is not None:
        (avg_processing_time, throughput) = timing_row
        stats["avg_processing_time"] = round(avg_processing_time, 2)
        stats["throughput"] = round(throughput * 3600, 2)  # exams per hour

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
        WHERE e.status = 'done'
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
            if (tpos + fpos) != 0:
                stats["region"][region]["ppv"] = int(100.0 * tpos / (tpos + fpos))
            if (tneg + fneg) != 0:
                stats["region"][region]["pnv"] = int(100.0 * tneg / (tneg + fneg))
            if (tpos + fneg) != 0:
                stats["region"][region]["snsi"] = int(100.0 * tpos / (tpos + fneg))
            if (tneg + fpos) != 0:
                stats["region"][region]["spci"] = int(100.0 * tneg / (tneg + fpos))

            tp = tpos or 0
            tn = tneg or 0
            fp = fpos or 0
            fn = fneg or 0
            denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            if denominator == 0:
                stats["region"][region]["mcc"] = 0.0
            else:
                mcc = (tp * tn - fp * fn) / denominator
                stats["region"][region]["mcc"] = round(mcc, 2)

    query = """
        SELECT DATE(e.created) as date,
               e.region,
               COUNT(*) as total,
               SUM(CASE WHEN ar.severity >= ? THEN 1 ELSE 0 END) as positive
        FROM exams e
        LEFT JOIN ai_reports ar ON e.uid = ar.uid
        WHERE e.status = 'done'
          AND e.created >= date('now', '-30 days')
        GROUP BY DATE(e.created), e.region
        ORDER BY date
    """
    trends_data = db_execute_query(query, (SEVERITY_THRESHOLD,), fetch_mode='all')
    if trends_data:
        for row in trends_data:
            (date, region, total, positive) = row
            if region not in stats["trends"]:
                stats["trends"][region] = []
            stats["trends"][region].append({
                "date": date,
                "total": total,
                "positive": positive
            })

    query = """
        SELECT strftime('%Y-%m', e.created) as month,
               e.region,
               COUNT(*) as total,
               SUM(CASE WHEN ar.severity >= ? THEN 1 ELSE 0 END) as positive
        FROM exams e
        LEFT JOIN ai_reports ar ON e.uid = ar.uid
        WHERE e.status = 'done'
          AND e.created >= date('now', '-12 months')
        GROUP BY strftime('%Y-%m', e.created), e.region
        ORDER BY month
    """
    monthly_trends_data = db_execute_query(query, (SEVERITY_THRESHOLD,), fetch_mode='all')
    if monthly_trends_data:
        for row in monthly_trends_data:
            (month, region, total, positive) = row
            if region not in stats["monthly_trends"]:
                stats["monthly_trends"][region] = []
            stats["monthly_trends"][region].append({
                "month": month,
                "total": total,
                "positive": positive
            })

    stats["accuracy_drift"] = []
    query = """
        SELECT strftime('%Y-%m', e.created) as month,
               COUNT(*) as total,
               SUM(CASE WHEN rr.severity > -1 THEN 1 ELSE 0 END) as reviewed,
               SUM(CASE WHEN (ar.severity >= ? AND rr.severity >= ?)
                           OR (ar.severity < ? AND rr.severity < ? AND rr.severity > -1)
                        THEN 1 ELSE 0 END) as correct
        FROM exams e
        LEFT JOIN ai_reports ar ON e.uid = ar.uid
        LEFT JOIN rad_reports rr ON e.uid = rr.uid
        WHERE e.status = 'done'
        GROUP BY month
        ORDER BY month
    """
    drift_data = db_execute_query(
        query,
        (SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD, SEVERITY_THRESHOLD),
        fetch_mode='all'
    )
    if drift_data:
        for row in drift_data:
            month, total, reviewed, correct = row
            accuracy = round((correct or 0) / reviewed * 100, 1) if reviewed else None
            stats["accuracy_drift"].append({
                "month": month,
                "total": total,
                "reviewed": reviewed,
                "correct": correct or 0,
                "accuracy": accuracy,
            })

    return stats


def db_get_queue_size():
    return db_count('exams', where_clause="status IN (?, ?)", where_params=('queued', 'requeue'))


def db_get_error_stats():
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
    result = db_select_one('ai_reports', uid)
    if result:
        if 'severity' not in result:
            result['severity'] = None
        if 'summary' not in result:
            result['summary'] = None
    return result


def db_get_rad_report(uid):
    return db_select_one('rad_reports', uid)


def db_have_rad_reports(uid):
    result = db_execute_query("SELECT 1 FROM rad_reports WHERE uid = ?", (uid,), fetch_mode='one')
    return result is not None


def db_get_patient_by_cnp(cnp):
    return db_select_one('patients', cnp)


def db_get_patient_exam_uids(cnp):
    rows = db_execute_query("SELECT uid FROM exams WHERE cnp = ? ORDER BY created DESC", (cnp,), fetch_mode='all')
    return [uid for (uid,) in rows] if rows else []


def db_get_regions():
    rows = db_execute_query("SELECT DISTINCT region FROM exams WHERE region IS NOT NULL AND region != '' AND status = 'done' ORDER BY region", fetch_mode='all')
    return [region for (region,) in rows] if rows else []


def db_get_patients(limit=PAGE_SIZE, offset=0, **filters):
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
            (cnp, id, name, birthdate, sex) = row
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
    count_query = "SELECT COUNT(*) FROM patients"
    count_params = []
    if conditions:
        count_query += ' WHERE ' + " AND ".join(conditions)
        count_params = params[:-2]
    total_row = db_execute_query(count_query, count_params, fetch_mode='one')
    total = total_row[0] if total_row else 0
    return patients, total


def db_purge_ignored_errors():
    ai_query = '''
        DELETE FROM ai_reports
        WHERE uid IN (
            SELECT uid FROM exams
            WHERE status IN ('ignore', 'error')
            AND created < datetime('now', '-7 days')
        )
    '''
    db_execute_query_retry(ai_query)
    rad_query = '''
        DELETE FROM rad_reports
        WHERE uid IN (
            SELECT uid FROM exams
            WHERE status IN ('ignore', 'error')
            AND created < datetime('now', '-7 days')
        )
    '''
    db_execute_query_retry(rad_query)
    uid_query = '''
        SELECT uid FROM exams
        WHERE status IN ('ignore', 'error')
        AND created < datetime('now', '-7 days')
    '''
    uid_rows = db_execute_query(uid_query, fetch_mode='all')
    deleted_uids = [row[0] for row in uid_rows] if uid_rows else []
    exam_query = '''
        DELETE FROM exams
        WHERE status IN ('ignore', 'error')
        AND created < datetime('now', '-7 days')
    '''
    deleted_count = db_execute_query_retry(exam_query)
    for uid in deleted_uids:
        for ext in ('dcm', 'png'):
            file_path = os.path.join(IMAGES_DIR, f"{uid}.{ext}")
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass
    logging.info(f"Purged {deleted_count or 0} old records from database and their files.")
    return deleted_count


def db_backup():
    try:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(BACKUP_DIR, f"xrayvision_{timestamp}.db")
        with sqlite3.connect(DB_FILE) as conn:
            with sqlite3.connect(backup_path) as backup_conn:
                conn.backup(backup_conn)
        logging.info(f"Database backed up to {backup_path}")
        try:
            backups = sorted(glob.glob(os.path.join(BACKUP_DIR, 'xrayvision_*.db')))
            while len(backups) > BACKUP_MAX_FILES:
                oldest = backups.pop(0)
                os.remove(oldest)
                logging.info(f"Removed old backup: {oldest}")
        except Exception as rot_e:
            logging.warning(f"Backup rotation failed: {rot_e}")
        return backup_path
    except Exception as e:
        logging.error(f"Failed to create database backup: {e}")
        return None


def db_rad_review(uid, normal, radiologist=''):
    positive = 0 if normal else 1
    result = db_select_one('rad_reports', uid)
    if result:
        db_update('rad_reports', 'uid = ?', (uid,), positive=positive, radiologist=radiologist)
    else:
        db_insert('rad_reports', uid=uid, positive=positive, radiologist=radiologist)


def db_set_status(uid, status):
    db_execute_query_retry("UPDATE exams SET status = ? WHERE uid = ?", (status, uid))
    return status


def db_requeue_exam(uid):
    try:
        db_set_status(uid, 'requeue')
        # Preserve existing text so the review prompt can reference it
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


def db_get_processing_times_by_region():
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

def db_get_rad_severity_distribution():
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

def db_get_ai_severity_distribution():
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

def db_get_severity_differences():
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

def db_get_age_distribution_insights(severity_threshold):
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

def db_get_hourly_patterns():
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

def db_get_requeue_analysis():
    # uid is PK in ai_reports; updated > created indicates at least one requeue
    query = """
        SELECT
            COUNT(*) as total_requeued,
            AVG(CAST(ar.latency AS FLOAT)) as avg_latency
        FROM exams e
        JOIN ai_reports ar ON e.uid = ar.uid
        WHERE e.status = 'done'
        AND ar.updated > ar.created
    """
    return db_execute_query(query, fetch_mode='one')

def db_get_radiologist_metrics():
    query = """
        SELECT 
            radiologist,
            COUNT(*) as reports_count,
            AVG(CAST(severity AS FLOAT)) as avg_severity,
            COUNT(DISTINCT uid) as unique_exams
        FROM rad_reports
        WHERE radiologist IS NOT NULL AND radiologist != ''
        GROUP BY radiologist
        HAVING COUNT(*) > 5
        ORDER BY reports_count DESC
    """
    return db_execute_query(query, fetch_mode='all')

# DICOM network operations
async def query_and_retrieve(minutes=60):
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
        try:
            current_time = datetime.now()
            past_time = current_time - timedelta(minutes=minutes)
            # DICOM time ranges cannot wrap around midnight; split into two queries when they cross it
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
            for modality in DICOM_MODALITIES:
                for study_date, time_range in queries:
                    ds = Dataset()
                    ds.QueryRetrieveLevel = "STUDY"
                    ds.StudyDate = study_date
                    ds.StudyTime = time_range
                    ds.Modality = modality
                    responses = assoc.send_c_find(
                        ds,
                        PatientRootQueryRetrieveInformationModelFind
                    )
                    for (status, identifier) in responses:
                        if status and status.Status in (0xFF00, 0xFF01):
                            study_instance_uid = identifier.StudyInstanceUID
                            if db_check_study_exists(study_instance_uid):
                                logging.info(f"Skipping Study {study_instance_uid} - already in database")
                                continue
                            logging.info(f"Found Study {study_instance_uid}")
                            if RETRIEVAL_METHOD.upper() == 'C-GET':
                                send_c_get(ae, study_instance_uid)
                            else:
                                send_c_move(ae, study_instance_uid)
        except Exception as e:
            logging.error(f"Error during QueryRetrieve: {e}")
        finally:
            assoc.release()
    else:
        logging.error("Could not establish QueryRetrieve association.")

def send_c_move(ae, study_instance_uid):
    assoc = ae.associate(REMOTE_AE_IP, REMOTE_AE_PORT, ae_title=REMOTE_AE_TITLE)
    if assoc.is_established:
        ds = Dataset()
        ds.QueryRetrieveLevel = "STUDY"
        ds.StudyInstanceUID = study_instance_uid
        try:
            for status, _ in assoc.send_c_move(
                ds,
                AE_TITLE,
                PatientRootQueryRetrieveInformationModelMove
            ):
                if status:
                    logging.debug(f"C-MOVE {study_instance_uid} status: 0x{status.Status:04X}")
                else:
                    logging.warning(f"C-MOVE {study_instance_uid}: no status returned (connection may have failed)")
        except Exception as e:
            logging.error(f"C-MOVE {study_instance_uid} failed: {e}")
        finally:
            assoc.release()
    else:
        logging.error("Could not establish C-MOVE association.")


def send_c_get(ae, study_instance_uid):
    assoc = ae.associate(REMOTE_AE_IP, REMOTE_AE_PORT, ae_title=REMOTE_AE_TITLE)
    if assoc.is_established:
        ds = Dataset()
        ds.QueryRetrieveLevel = "STUDY"
        ds.StudyInstanceUID = study_instance_uid
        try:
            for status, _ in assoc.send_c_get(
                ds,
                PatientRootQueryRetrieveInformationModelGet
            ):
                if status:
                    logging.debug(f"C-GET {study_instance_uid} status: 0x{status.Status:04X}")
                else:
                    logging.warning(f"C-GET {study_instance_uid}: no status returned (connection may have failed)")
        except Exception as e:
            logging.error(f"C-GET {study_instance_uid} failed: {e}")
        finally:
            assoc.release()
    else:
        logging.error("Could not establish C-GET association.")


def dicom_store(event):
    """C-STORE callback: validates, saves, converts, and queues the received study."""
    ds = event.dataset
    ds.file_meta = event.file_meta
    if 'SOPInstanceUID' not in ds or not ds.SOPInstanceUID or ds.SOPInstanceUID == 'NO_UID':
        logging.error("Invalid or missing SOP Instance UID in received DICOM file")
        return 0x0110  # Processing failure

    uid = f"{ds.SOPInstanceUID}"
    # DICOM UIDs contain only digits and dots; reject anything else to prevent path traversal
    if not re.fullmatch(r'[\d.]+', uid):
        logging.error(f"Rejected DICOM file with non-standard SOP Instance UID: {uid!r}")
        return 0x0110  # Processing failure
    if db_check_already_processed(uid):
        # Check if the existing exam record is missing study or series information
        existing_exam = db_select_one('exams', uid)
        if existing_exam and (not existing_exam.get('study') or not existing_exam.get('series')):
            study_uid = str(ds.StudyInstanceUID) if 'StudyInstanceUID' in ds else None
            series_uid = str(ds.SeriesInstanceUID) if 'SeriesInstanceUID' in ds else None
            update_fields = {}
            if study_uid and not existing_exam.get('study'):
                update_fields['study'] = study_uid
            if series_uid and not existing_exam.get('series'):
                update_fields['series'] = series_uid
            
            if update_fields:
                db_update('exams', 'uid = ?', (uid,), **update_fields)
                logging.debug(f"Updated study/series info for exam {uid}")
        
        logging.debug(f"Skipping already processed image {uid}")
    elif ds.Modality in DICOM_MODALITIES:
        dicom_file = os.path.join(IMAGES_DIR, f"{uid}.dcm")
        try:
            ds.save_as(dicom_file, enforce_file_format=True)
        except Exception as e:
            logging.error(f"Failed to save DICOM file {dicom_file}: {e}")
            return 0x0110  # file not stored
        logging.debug(f"DICOM file saved to {dicom_file}")
        process_dicom_file(dicom_file, uid)
        asyncio.run_coroutine_threadsafe(broadcast_dashboard_update(), MAIN_LOOP)
    else:
        logging.debug(f"Received {ds.Modality} study {uid} — modality not supported, discarding")
    return 0x0000


# DICOM files operations
async def load_existing_dicom_files():
    for dicom_file in os.listdir(IMAGES_DIR):
        uid, ext = os.path.splitext(os.path.basename(dicom_file.lower()))
        if ext == '.dcm':
            if db_check_already_processed(uid):
                logging.debug(f"Skipping already processed image {uid}")
            else:
                logging.debug(f"Adding {uid} into processing queue...")
                full_path = os.path.join(IMAGES_DIR, dicom_file)
                await asyncio.to_thread(process_dicom_file, full_path, uid)
    await broadcast_dashboard_update()


def extract_dicom_metadata(ds):
    age = -1
    birthdate = None
    county = None
    if 'PatientBirthDate' in ds and ds.PatientBirthDate:
        try:
            birthdate = str(ds.PatientBirthDate)
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
        cnp_result = validate_romanian_cnp(ds.PatientID)
        if cnp_result['valid']:
            birthdate = cnp_result['birth_date'].strftime("%Y-%m-%d")
            age = cnp_result['age']
            county = cnp_result['county']
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if 'SeriesDate' in ds and 'SeriesTime' in ds and \
        str(ds.SeriesDate) and str(ds.SeriesTime) and \
        len(str(ds.SeriesDate)) == 8 and len(str(ds.SeriesTime)) >= 6:
        try:
            dt = datetime.strptime(f'{str(ds.SeriesDate)} {str(ds.SeriesTime)[:6]}', "%Y%m%d %H%M%S")
            created = dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            created = now
    else:
        created = now

    protocol_name = str(ds.ProtocolName) if 'ProtocolName' in ds and ds.ProtocolName else ''
    region, _ = identify_anatomic_region(protocol_name)

    info = {
        'uid': str(ds.SOPInstanceUID),
        'patient': {
            'name':  str(ds.PatientName) if 'PatientName' in ds else '',
            'cnp':   str(ds.PatientID) if 'PatientID' in ds else '',
            'age':   age,
            'birthdate': birthdate,
            'sex':   str(ds.PatientSex) if 'PatientSex' in ds else '',
        },
        'exam': {
            'protocol': protocol_name,
            'created':  created,
            'region':   region,
            'study':    str(ds.StudyInstanceUID) if 'StudyInstanceUID' in ds else None,
            'series':   str(ds.SeriesInstanceUID) if 'SeriesInstanceUID' in ds else None,
            'id':       None,
        }
    }
    if county is not None:
        info['patient']['county'] = county
    if not info['patient']['sex'] in ['M', 'F', 'O']:
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
    return info



def process_dicom_file(dicom_file, uid):
    """Shared logic for dicom_store and load_existing_dicom_files; must run in a thread (PIL/CV2 blocks)."""
    try:
        ds = dcmread(dicom_file)
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
        png_file = None
        try:
            png_file = convert_dicom_to_png(dicom_file)
        except Exception as e:
            logging.error(f"Error converting DICOM file {dicom_file}: {e}")
            db_set_status(uid, "error")
            return
        if png_file:
            db_add_exam(info)
            QUEUE_EVENT.set()
        else:
            db_set_status(uid, "error")
    except Exception as e:
        logging.error(f"Error processing DICOM file {dicom_file}: {e}")
        db_set_status(uid, "error")
def extract_patient_initials(name):
    if not name or not isinstance(name, str):
        return "NoName"
    parts = re.split(r'[-^ ]', name)
    initials = ''.join([part[0] + '.' for part in parts if part])
    return initials.upper() if initials else "NoName"


def extract_radiologist_initials(name):
    if not name or not isinstance(name, str):
        return "Dr. NoName"
    if name.lower().startswith("dr."):
        name_without_dr = name[3:].strip()
        if not name_without_dr:
            return "Dr. NoName"
        parts = re.split(r'[-^ ]', name_without_dr)
        initials = ''.join([part[0] + '.' for part in parts if part])
        return "Dr. " + initials.upper() if initials else "Dr. NoName"
    else:
        parts = re.split(r'[-^ ]', name)
        initials = ''.join([part[0] + '.' for part in parts if part])
        return "Dr. " + initials.upper() if initials else "Dr. NoName"


# Image processing operations
def apply_gamma_correction(image, gamma = 1.2):
    if gamma is None:
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mid = 0.5
        median = np.median(image)
        if median <= 0:
            logging.debug("Image median is zero, using gamma=1.0 (identity)")
            median = mid * 255  # results in gamma = 1.0
        gamma = math.log(mid * 255) / math.log(median)
        logging.debug(f"Calculated gamma is {gamma:.2f}")
    if gamma == 0:
        logging.debug("Gamma is zero, using gamma=1.0 (identity)")
        gamma = 1.0
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def convert_dicom_to_png(dicom_file, max_size = 896):
    base_name = os.path.splitext(os.path.basename(dicom_file))[0]
    png_file = os.path.join(IMAGES_DIR, f"{base_name}.png")
    if os.path.exists(png_file):
        logging.debug(f"PNG file already exists: {png_file}")
        return png_file
        
    try:
        ds = dcmread(dicom_file)
        if 'PixelData' not in ds:
            raise ValueError(f"DICOM file {dicom_file} has no pixel data!")
        image = ds.pixel_array.astype(np.float32)
        height, width = image.shape[:2]
        if max(height, width) > max_size:
            if height > width:
                new_height = max_size
                new_width = int(width * (max_size / height))
            else:
                new_width = max_size
                new_height = int(height * (max_size / width))
            image = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_AREA)
        # Clip to 1st–99th percentile to remove outliers
        minval = np.percentile(image, 1)
        maxval = np.percentile(image, 99)
        image = np.clip(image, minval, maxval)
        img_min = image.min()
        img_max = image.max()
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min) * 255.0
        else:
            image = np.zeros_like(image)
        image = image.astype(np.uint8)
        image = apply_gamma_correction(image)
        if not cv2.imwrite(png_file, image):
            raise IOError(f"cv2.imwrite failed to write {png_file}")
        logging.debug(f"Converted PNG saved to {png_file}")
        return png_file
    except Exception as e:
        logging.error(f"Error converting DICOM to PNG: {e}")
        raise




# Domain helpers

def validate_romanian_cnp(patient_cnp):
    """Validate a Romanian CNP (13-digit personal ID) and return parsed fields, or {'valid': False}."""
    pid = str(patient_cnp).strip()
    if not pid or len(pid) != 13 or not pid.isdigit():
        return {'valid': False}
    
    gender_digit = int(pid[0])
    year = int(pid[1:3])
    month = int(pid[3:5])
    day = int(pid[5:7])
    county = int(pid[7:9])
    checksum_digit = int(pid[12])
    if gender_digit < 1 or gender_digit > 9:
        return {'valid': False}
    century_map = {1: 1900, 2: 1900, 3: 1800, 4: 1800, 5: 2000, 6: 2000, 7: 2000, 8: 2000, 9: 1900}
    if gender_digit not in century_map:
        return {'valid': False}
    full_year = century_map[gender_digit] + year
    try:
        birth_date = datetime(full_year, month, day)
    except ValueError:
        return {'valid': False}
    valid_counties = set(range(1, 47)) | set(range(51, 53)) | set(range(70, 80)) | set(range(90, 100))
    if county not in valid_counties:
        return {'valid': False}
    weights = [2, 7, 9, 1, 4, 6, 3, 5, 8, 2, 7, 9]
    weighted_sum = sum(int(pid[i]) * weights[i] for i in range(12))
    checksum = weighted_sum % 11
    if checksum == 10:
        checksum = 1
    if checksum != checksum_digit:
        return {'valid': False}
    today = datetime.now()
    age = today.year - birth_date.year
    if (today.month, today.day) < (birth_date.month, birth_date.day):
        age -= 1
    sex = 'M' if gender_digit % 2 == 1 else 'F'
    return {
        'valid': True,
        'birth_date': birth_date,
        'age': age,
        'sex': sex,
        'county': county
    }


def compute_age_from_cnp(patient_cnp):
    result = validate_romanian_cnp(patient_cnp)
    if not result['valid']:
        return -1
    return result['age']


def contains_any_word(string, *words):
    return any(i in string for i in words)


def identify_anatomic_region(info):
    if isinstance(info, str):
        desc = info.lower()
    else:
        desc = info["exam"]["protocol"].lower()
    for region_key, keywords in REGION_RULES.items():
        if contains_any_word(desc, *keywords):
            region = region_key
            break
    else:
        # No keyword matched — store empty string rather than the full protocol string
        region = ''
    question = REGION_QUESTIONS.get(region, "Is there anything abnormal")
    return region, question



def identify_imaging_projection(info):
    desc = (info if isinstance(info, str) else info["exam"]["protocol"]).lower()
    if contains_any_word(desc, "a.p.", "p.a.", "d.v.", "v.d.", "d.p"):
        return "frontal"
    elif contains_any_word(desc, "lat.", "pr."):
        return "lateral"
    elif contains_any_word(desc, "oblic"):
        return "oblique"
    return ""


def determine_patient_gender_description(info):
    patient_sex = info["patient"].get("sex", "").lower()
    if "m" in patient_sex:
        return "boy"
    elif "f" in patient_sex:
        return "girl"
    return "child"




# FHIR integration

def format_patient_name_for_fhir(dicom_name):
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
        async with session.get(url, auth=auth, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
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
                formatted_name = format_patient_name_for_fhir(patient_name)
            else:
                formatted_name = patient_name.strip()
            
            if formatted_name:
                logging.info(f"Retrying FHIR patient search by name: {formatted_name}")
                params = {'q': formatted_name}
                
                logging.debug(f"Sending FHIR patient search by name request to {url} with params {params}")
                async with session.get(url, auth=auth, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
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

async def search_fhir_servicerequests(session, patient_id, exam_datetime, exam_type, exam_region):
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
        async with session.get(url, auth=auth, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
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
                logging.debug(f"FHIR service requests search failed with status {resp.status}")
    except Exception as e:
        logging.error(f"FHIR service requests search error: {e}")
    return []

async def get_fhir_servicerequest(session, request_id):
    try:
        auth = aiohttp.BasicAuth(FHIR_USERNAME, FHIR_PASSWORD)
        url = f"{FHIR_URL}/fhir/ServiceRequest/{request_id}"

        async with session.get(url, auth=auth, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get('resourceType') == 'OperationOutcome':
                    issues = data.get('issue', [])
                    error_details = '; '.join([f"{issue.get('severity', 'unknown')}: {issue.get('diagnostics', issue.get('details', {}).get('text', 'no details'))}" for issue in issues])
                    logging.warning(f"FHIR service request returned OperationOutcome: {error_details}")
                    return None
                elif data.get('resourceType') == 'ServiceRequest':
                    return data
                else:
                    logging.warning(f"FHIR service request has incorrect resource type: {data.get('resourceType')}")
            else:
                logging.warning(f"FHIR service request failed with status {resp.status}")
    except Exception as e:
        logging.error(f"FHIR service request error: {e}")
    return None

async def get_fhir_diagnosticreport(session, report_id):
    try:
        auth = aiohttp.BasicAuth(FHIR_USERNAME, FHIR_PASSWORD)
        url = f"{FHIR_URL}/fhir/DiagnosticReport/{report_id}"

        async with session.get(url, auth=auth, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get('resourceType') == 'OperationOutcome':
                    issues = data.get('issue', [])
                    error_details = '; '.join([f"{issue.get('severity', 'unknown')}: {issue.get('diagnostics', issue.get('details', {}).get('text', 'no details'))}" for issue in issues])
                    logging.warning(f"FHIR diagnostic report returned OperationOutcome: {error_details}")
                    return None
                elif data.get('resourceType') == 'DiagnosticReport':
                    return data
                else:
                    logging.warning(f"FHIR diagnostic report has incorrect resource type: {data.get('resourceType')}")
            else:
                logging.warning(f"FHIR diagnostic report failed with status {resp.status}")
    except Exception as e:
        logging.error(f"FHIR diagnostic report error: {e}")
    return None

async def find_service_request(session, exam_uid, patient_id, exam_datetime, exam_type='radio', exam_region=''):
    srv_reqs = await search_fhir_servicerequests(session, patient_id, exam_datetime, exam_type, exam_region)
    if not srv_reqs:
        logging.debug(f"No service requests found for exam {exam_uid}")
        return None

    req = srv_reqs[0]
    if 'id' not in req:
        logging.warning(f"Service request for exam {exam_uid} has no ID, skipping.")
        return None

    return req

async def extract_report_data(report, exam_uid, exam_type = "radio", exam_region = ""):
    report_text = None
    presented_form = None

    if not isinstance(report.get('presentedForm'), list) or not report['presentedForm']:
        logging.warning(f"FHIR DiagnosticReport for exam {exam_uid} has no presentedForm")
        return None, None

    if len(report['presentedForm']) == 1:
        presented_form = report['presentedForm'][0]
    else:
        logging.info(f"Found {len(report['presentedForm'])} items in presentedForm for '{exam_type}' exam {exam_uid}, looking for region '{exam_region}'")
        for form in report['presentedForm']:
            type_match = form.get('type', '').lower() == exam_type.lower()
            region_match = form.get('region', '').lower() == exam_region.lower()
            if region_match and type_match:
                presented_form = form
                break

        if not presented_form:
            logging.warning(f"No presentedForm found with region '{exam_region}' for exam {exam_uid}")
            return None, None

    report_text = presented_form.get('data', '')
    if not isinstance(report_text, str):
        logging.warning(f"FHIR presentedForm data is not a string for exam {exam_uid}")
        return None, None
    report_text = report_text.strip()
    if not report_text:
        logging.warning(f"No data found in presentedForm for exam {exam_uid}")
        return None, None

    radiologist = ''
    try:
        radiologist = presented_form.get('validator', '') or ''
    except Exception as e:
        logging.warning(f"Could not extract radiologist name from FHIR report: {e}")

    return report_text, radiologist

def translate_exam_type_to_fhir(exam_type):
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
    if not ENABLE_HIS:
        return
        
    exam_uid = exam['uid']
    exam_datetime = exam['created']
    exam_type = translate_exam_type_to_fhir(exam.get('type') or 'radio')
    exam_region = exam.get('region', '')
    fhir_region = REGION_FHIR_MAP.get(exam_region, exam_region)

    if exam_region not in REGIONS:
        identified_region, _ = identify_anatomic_region(exam.get('protocol', ''))
        if identified_region in REGIONS:
            logging.info(f"Re-identified region for exam {exam_uid}: {identified_region}")
            exam_region = identified_region
            db_update('exams', 'uid = ?', (exam_uid,), region=exam_region)
        else:
            logging.warning(f"Could not identify valid region for exam {exam_uid} from report text")

    rad_report = db_get_rad_report(exam_uid)
    srv_req = None

    if rad_report and rad_report.get('id'):
        try:
            service_id = int(rad_report['id'])
            if service_id > 0:
                srv_req = await get_fhir_servicerequest(session, service_id)
                if srv_req:
                    logging.info(f"Retrieved service request ID {srv_req['id']} for exam {exam_uid}")
                else:
                    logging.warning(f"Failed to retrieve service request ID {service_id} for exam {exam_uid}")
        except (ValueError, TypeError):
            pass

    if not srv_req:
        srv_req = await find_service_request(session, exam_uid, patient_id, exam_datetime, exam_type, fhir_region)

    if not srv_req or 'id' not in srv_req:
        try:
            exam_date = datetime.strptime(exam_datetime, "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            logging.warning(f"Cannot parse exam datetime '{exam_datetime}' for exam {exam_uid}, treating as recent")
            return
        one_month_ago = datetime.now() - timedelta(days=30)
        is_old_exam = exam_date < one_month_ago

        if is_old_exam:
            if rad_report:
                db_update('rad_reports', 'uid = ?', (exam_uid,), id=-1)
                logging.info(f"Updated report for exam {exam_uid} to mark service request as not found")
            else:
                db_insert('rad_reports', uid=exam_uid, id=-1)
                logging.info(f"Service request missing for exam {exam_uid}")
        else:
            logging.info(f"Service request missing for recent exam {exam_uid}, skipping rad report creation")

        return

    justification = ''
    try:
        if 'supportingInfo' in srv_req and isinstance(srv_req['supportingInfo'], list) and len(srv_req['supportingInfo']) > 0:
            supporting_info = srv_req['supportingInfo'][0]
            if isinstance(supporting_info, dict) and 'display' in supporting_info and isinstance(supporting_info['display'], str):
                justification = supporting_info['display']

        if not justification and 'reason' in srv_req and isinstance(srv_req['reason'], list) and len(srv_req['reason']) > 0:
            reason = srv_req['reason'][0]
            if isinstance(reason, dict) and 'display' in reason and isinstance(reason['display'], str):
                justification = reason['display']

        if not justification:
            logging.debug(f"No justification found in service request {srv_req['id']}. Available fields: {list(srv_req.keys())}")
    except Exception as e:
        logging.warning(f"Error extracting justification from service request: {e}")

    report = await get_fhir_diagnosticreport(session, srv_req['id'])
    if not report or 'presentedForm' not in report or not report['presentedForm']:
        logging.debug(f"No presentedForm found in diagnostic report for exam {exam_uid}")
        return

    report_text, radiologist = await extract_report_data(report, exam_uid, exam_type=exam_type, exam_region=fhir_region)
    if not report_text:
        return

    logging.debug(f"Retrieved radiologist report for exam {exam_uid}: {' '.join(report_text.split()[:10])}...")

    if rad_report:
        db_update('rad_reports', 'uid = ?', (exam_uid,),
            id=srv_req['id'],
            text=report_text,
            radiologist=radiologist,
            justification=justification,
            type=exam_type,
            model=MODEL_NAME)
    else:
        db_insert('rad_reports',
            uid=exam_uid,
            id=srv_req['id'],
            text=report_text,
            radiologist=radiologist,
            summary=None,
            type=exam_type,
            justification=justification,
            model=MODEL_NAME)
    logging.debug(f"Saving the service request id {srv_req['id']} for {exam_type} exam {exam_uid} with justification: {justification}")

    db_set_status(exam_uid, "check")
    QUEUE_EVENT.set()

async def get_patient_id_from_fhir(session, patient_cnp, patient_name=None):
    fhir_patient = await get_fhir_patient(session, patient_cnp, patient_name)
    if fhir_patient and 'id' in fhir_patient:
        patient_id = fhir_patient['id']
        db_update_patient_id(patient_cnp, patient_id)
        return patient_id
    return None


async def process_exams_without_rad_reports(session):
    result = db_get_exams_without_rad_report()
    if not result or not result.get('exams'):
        return

    patient_cnp = result['patient']['cnp']
    patient_id = result['patient']['id']
    exams = result['exams']

    patient_name = result['patient']['name']
    if not patient_id:
        if not validate_romanian_cnp(patient_cnp).get('valid'):
            logging.warning(f"Invalid CNP '{patient_cnp}' for patient '{patient_name}', marking exams as unresolvable")
            for exam in exams:
                exam_uid = exam['uid']
                existing = db_select_one('rad_reports', exam_uid)
                if existing:
                    db_update('rad_reports', 'uid = ?', (exam_uid,), id=-1)
                else:
                    db_insert('rad_reports', uid=exam_uid, id=-1)
            return
        patient_id = await get_patient_id_from_fhir(session, patient_cnp, patient_name)
    # If still no patient ID, only mark unresolvable if the exams are old enough
    if not patient_id:
        one_week_ago = datetime.now() - timedelta(weeks=1)
        recent = any(
            datetime.strptime(e['created'][:19], '%Y-%m-%d %H:%M:%S') > one_week_ago
            for e in exams if e.get('created')
        )
        if recent:
            logging.debug(f"Could not find FHIR patient for CNP {patient_cnp}, exam is recent — will retry later")
            return
        logging.warning(f"Could not find FHIR patient for CNP {patient_cnp} or name '{patient_name}', marking exams as unresolvable")
        for exam in exams:
            exam_uid = exam['uid']
            existing = db_select_one('rad_reports', exam_uid)
            if existing:
                db_update('rad_reports', 'uid = ?', (exam_uid,), id=-1)
            else:
                db_insert('rad_reports', uid=exam_uid, id=-1)
        return
    
    # Process each exam for this patient
    for exam in exams:
        await process_single_exam_without_rad_report(session, exam, patient_id)



# AI pipeline

async def check_report(report_text):
    try:
        if not report_text:
            logging.warning("Report check request failed: no report text provided")
            return {'error': 'No report text provided'}

        logging.debug(f"Report check request received with report length: {len(report_text.split())} words")

        processed_report_text = re.sub(r'([.!?])(?=\S)', r'\1 ', report_text)

        acronym_pattern = re.compile(r'\b[A-Z]{2,}\b')
        found_acronyms = acronym_pattern.findall(processed_report_text)
        used_acronyms = [acro for acro in found_acronyms if acro in MEDICAL_ACRONYMS]
        acronym_list = "\n".join([f"- {acronym}: {MEDICAL_ACRONYMS[acronym]}" for acronym in used_acronyms])

        SYSTEM_PROMPT = PROMPTS['CHK_PROMPT'].strip()
        if used_acronyms:
            SYSTEM_PROMPT += f"\n\nMEDICAL ACRONYMS\n{acronym_list}"
            logging.debug(f"Added acronym list to check prompt:\n{acronym_list}")

        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json',
        }

        payload = {
            "model": MODEL_NAME,
            "timings_per_token": True,
            "cache_prompt": True,
            "stream": False,
            "keep_alive": 1800,
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}]
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

        start_time = asyncio.get_running_loop().time()
        async with aiohttp.ClientSession(headers={'User-Agent': USER_AGENT}) as session:
            result = await send_to_openai(session, headers, payload)
            global timings
            end_time = asyncio.get_running_loop().time()
            processing_time = int((end_time - start_time) * 1000)
            if timings['checking'] > 0:
                timings['checking'] = int((3 * timings['checking'] + processing_time) / 4)
            else:
                timings['checking'] = processing_time

            if not result:
                logging.error("Failed to get response from AI")
                return {'error': 'Failed to get response from AI'}

            response_text = result["choices"][0]["message"]["content"].strip()
            logging.debug(f"Raw AI check response: {response_text}")

            # chk_prompt.txt only requires "ONLY valid JSON", not fenced JSON --
            # fall back to the raw text when no ```json ... ``` fence is present,
            # rather than discarding a valid bare-JSON response.
            fenced_matches = re.findall(r'```json\s*({.*?})\s*```', response_text, re.DOTALL)
            if fenced_matches:
                response_text = fenced_matches[-1]

            try:
                parsed_response = json.loads(response_text) if response_text else None
                logging.debug(f"AI responded: {parsed_response}")

                if isinstance(parsed_response, list):
                    if len(parsed_response) == 0:
                        raise ValueError("Empty array response from AI")
                    parsed_response = parsed_response[0]

                if "pathologic" not in parsed_response or "severity" not in parsed_response or "summary" not in parsed_response:
                    raise ValueError("Missing required fields in AI response")

                if parsed_response["pathologic"] not in ["yes", "no"]:
                    raise ValueError("Invalid pathologic value in AI response")

                if not isinstance(parsed_response["severity"], int) or parsed_response["severity"] < 0 or parsed_response["severity"] > 10:
                    raise ValueError("Invalid severity value in AI response")

                if not isinstance(parsed_response["summary"], str):
                    raise ValueError("Invalid summary value in AI response")
                else:
                    parsed_response["summary"] = parsed_response["summary"].strip().lower()

                return parsed_response
            except json.JSONDecodeError as e:
                logging.debug(f"Initial JSON parsing failed, trying to extract JSON from response: {response_text}")

                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        parsed_response = json.loads(json_match.group(0))
                        return parsed_response
                    except json.JSONDecodeError as json_e:
                        logging.error(f"Failed to parse extracted JSON: {json_e}")
                        return {'error': 'Failed to parse AI response', 'response': response_text}
                else:
                    logging.error(f"Failed to parse AI response as JSON: {response_text}")
                    return {'error': 'Failed to parse AI response', 'response': response_text}
            except ValueError as e:
                logging.error(f"Invalid AI response format: {e} ({response_text})")
                return {'error': f'Invalid AI response format: {str(e)}', 'response': response_text}
    except Exception as e:
        logging.error(f"Error processing report check request: {e}")
        return {'error': 'Internal server error'}


async def check_ai_report_and_update(uid):
    try:
        ai_report = db_get_ai_report(uid)
        if not ai_report or not ai_report.get('text'):
            logging.warning(f"No AI report text found for exam {uid}")
            return False

        findings = ai_report['text']
        impression = ai_report.get('summary', None)
        if impression:
            report_text = f"FINDINGS: {findings}\n\nIMPRESSION: {impression}"
        else:
            report_text = findings

        logging.info(f"Summarizing AI report for exam {uid}")
        analysis_result = await check_report(report_text)

        if 'error' in analysis_result:
            logging.error(f"AI check failed for exam {uid}: {analysis_result['error']}")
            return False

        db_update('ai_reports', 'uid = ?', (uid,),
                    positive=1 if analysis_result['pathologic'] == 'yes' else 0,
                    severity=analysis_result['severity'],
                    summary=analysis_result['summary'])

        logging.info(f"Updated AI report for exam {uid} with severity {analysis_result['severity']} and summary '{analysis_result['summary']}'")
        return True
        
    except Exception as e:
        logging.error(f"Error processing CHECK prompt for exam {uid}: {e}")
        return False


async def translate_report(report_text):
    try:
        if not report_text:
            logging.warning("Translation request failed: no report text provided")
            return None

        logging.debug(f"Translation request received: {' '.join(report_text.split()[:10])}...")

        report_hash = hash(report_text)
        if report_hash in _translation_cache:
            return _translation_cache[report_hash]

        report_text = re.sub(r'([.])(?=\S)', r'\1 ', report_text)

        acronym_pattern = re.compile(r'\b[A-Z]{2,}\b')
        found_acronyms = acronym_pattern.findall(report_text)
        used_acronyms = [acro for acro in found_acronyms if acro in MEDICAL_ACRONYMS]
        acronym_list = "\n".join([f"- {acronym}: {MEDICAL_ACRONYMS[acronym]}" for acronym in used_acronyms])

        SYSTEM_PROMPT = PROMPTS['TRN_PROMPT'].strip()
        if used_acronyms:
            SYSTEM_PROMPT += f"\n\nMEDICAL ACRONYMS\n{acronym_list}"
            logging.debug(f"Added acronym list to translation prompt:\n{acronym_list}")

        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json',
        }

        payload = {
            "model": MODEL_NAME,
            "timings_per_token": True,
            "cache_prompt": True,
            "stream": False,
            "keep_alive": 1800,
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}]
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

        start_time = asyncio.get_running_loop().time()
        async with aiohttp.ClientSession(headers={'User-Agent': USER_AGENT}) as session:
            result = await send_to_openai(session, headers, payload)
            global timings
            end_time = asyncio.get_running_loop().time()
            processing_time = int((end_time - start_time) * 1000)
            if timings['translation'] > 0:
                timings['translation'] = int((3 * timings['translation'] + processing_time) / 4)
            else:
                timings['translation'] = processing_time

            if not result:
                logging.error("Failed to get response from AI service for translation")
                return None

            response_text = result["choices"][0]["message"]["content"].strip()
            logging.debug(f"Raw AI translation response: {response_text}")

            response_text = re.findall(r'```text\s*([^`]*?)\s*```', response_text, re.DOTALL)
            if response_text:
                response_text = response_text[-1]

            if not response_text:
                logging.warning("Empty translation response received")
                return None

            _translation_cache[report_hash] = response_text
            if len(_translation_cache) > 500:
                oldest_key = next(iter(_translation_cache))
                del _translation_cache[oldest_key]

            logging.info(f"Translation: {' '.join(response_text.split()[:10])}...")
            return response_text
    except Exception as e:
        logging.error(f"Error processing translation request: {e}")
        return None


def expand_medical_acronyms(text):
    if not text or not isinstance(text, str):
        return text, []

    expanded_text = text
    found_acronyms = []

    sorted_acronyms = sorted(MEDICAL_ACRONYMS.keys(), key=len, reverse=True)  # longest first to avoid partial matches
    for acronym in sorted_acronyms:
        pattern = r'\b' + re.escape(acronym) + r'\b'
        translation = MEDICAL_ACRONYMS[acronym]
        if re.search(pattern, expanded_text):
            found_acronyms.append(acronym)
            expanded_text = re.sub(pattern, translation, expanded_text)
    return expanded_text, found_acronyms

def validate_translation(source_text, translated_text):
    if not translated_text or not translated_text.strip():
        return False, "Translation is empty or None"

    if translated_text.strip() == source_text.strip():
        return False, "Translation is identical to source text - no translation performed"

    if len(translated_text.strip()) < 10:
        return False, f"Translation is too short ({len(translated_text)} characters)"

    # Multi-word phrases only — single words like "error", "unable", "sorry" appear legitimately in medical reports.
    placeholder_patterns = [
        r"\b(no translation available)\b",
        r"\b(could not translate)\b",
        r"\b(translation failed)\b",
        r"\b(i am unable to)\b",
        r"\b(i cannot translate)\b",
        r"\b(i('m| am) sorry)\b",
    ]

    for pattern in placeholder_patterns:
        if re.search(pattern, translated_text, re.IGNORECASE):
            return False, f"Translation contains error text: {translated_text}"

    expanded_translation, found_acronyms = expand_medical_acronyms(translated_text)

    romanian_acronym_pattern = r'\b[A-Z]{2,}\b'
    untranslated_acronyms = re.findall(romanian_acronym_pattern, expanded_translation)
    untranslated_acronyms = [acro for acro in untranslated_acronyms if acro not in MEDICAL_ACRONYMS]
    if untranslated_acronyms:
        logging.debug(f"Found medical acronyms in translation: {', '.join(untranslated_acronyms)}")

    return True, expanded_translation

async def check_rad_report_and_update(uid):
    try:
        rad_report = db_get_rad_report(uid)
        if not rad_report or not rad_report.get('text'):
            logging.warning(f"No radiologist report text found for exam {uid}")
            return False

        report_text = rad_report['text']

        if not active_openai_url:
            logging.debug(f"Skipping translation for exam {uid}: AI service not reachable")
            return False
        logging.info(f"Translating radiologist report for exam {uid}")
        translation = await translate_report(report_text)
        if translation:
            is_valid, message = validate_translation(report_text, translation)
            if not is_valid:
                logging.warning(f"Translation validation failed for exam {uid}: {message}")
                translation = None
            else:
                logging.info(f"Translation successful for exam {uid}: {' '.join(message.split()[:10])}...")
                translation = message
        else:
            logging.warning(f"Translation failed for exam {uid}")

        logging.info(f"Summarizing radiologist report for exam {uid}")
        start_time = asyncio.get_running_loop().time()
        analysis_result = await check_report(report_text)
        end_time = asyncio.get_running_loop().time()
        processing_time = int((end_time - start_time) * 1000)

        if 'error' in analysis_result:
            logging.error(f"Check failed for exam {uid}: {analysis_result['error']}")
            return False

        try:
            if 'pathologic' not in analysis_result or 'severity' not in analysis_result or 'summary' not in analysis_result:
                logging.error(f"Check response missing required fields for exam {uid}: {list(analysis_result.keys())}")
                return False
            positive = 1 if analysis_result['pathologic'] == 'yes' else 0
            severity = analysis_result['severity']
            summary = analysis_result['summary'].lower()
        except Exception as e:
            logging.error(f"Could not extract analysis results for exam {uid}: {e}")
            return False

        update_fields = {
            'positive': positive,
            'severity': severity,
            'summary': summary,
            'model': MODEL_NAME,
            'latency': int(processing_time)
        }
        if translation:
            update_fields['text_en'] = translation

        db_update('rad_reports', 'uid = ?', (uid,), **update_fields)
        logging.info(f"Updated radiologist report for exam {uid} with severity {severity}, summary '{summary}', latency {processing_time}ms")
        if translation:
            logging.info(f"Added English translation for exam {uid}")
        return True

    except Exception as e:
        logging.error(f"Error processing CHECK prompt for exam {uid}: {e}")
        return False

async def detailed_analysis_report(report_text):
    try:
        if not report_text:
            logging.warning("Detailed analysis request failed: no report text provided")
            return {'error': 'No report text provided'}

        logging.debug(f"Detailed analysis request received ({len(report_text.split())} words)")

        processed_report_text = re.sub(r'([.!?])(?=\S)', r'\1 ', report_text)

        acronym_pattern = re.compile(r'\b[A-Z]{2,}\b')
        found_acronyms = acronym_pattern.findall(report_text)
        used_acronyms = [acro for acro in found_acronyms if acro in MEDICAL_ACRONYMS]
        acronym_list = "\n".join([f"- {acronym}: {MEDICAL_ACRONYMS[acronym]}" for acronym in used_acronyms])

        SYSTEM_PROMPT = PROMPTS['ANA_PROMPT'].strip()
        if used_acronyms:
            SYSTEM_PROMPT += f"\n\nMEDICAL ACRONYMS\n{acronym_list}"
            logging.debug(f"Added acronym list to detailed analysis prompt:\n{acronym_list}")

        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json',
        }

        payload = {
            "model": MODEL_NAME,
            "timings_per_token": True,
            "cache_prompt": True,
            "stream": False,
            "keep_alive": 1800,
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}]
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

        start_time = asyncio.get_running_loop().time()
        async with aiohttp.ClientSession(headers={'User-Agent': USER_AGENT}) as session:
            result = await send_to_openai(session, headers, payload)
            global timings
            end_time = asyncio.get_running_loop().time()
            processing_time = int((end_time - start_time) * 1000)
            if timings['analysis'] > 0:
                timings['analysis'] = int((3 * timings['analysis'] + processing_time) / 4)
            else:
                timings['analysis'] = processing_time

            if not result:
                logging.error("Failed to get response from AI service")
                return {'error': 'Failed to get response from AI service'}

            response_text = result["choices"][0]["message"]["content"].strip()
            logging.debug(f"Raw AI detailed analysis response: {response_text}")

            response_text = re.sub(r"^```(?:json)?\s*", "", response_text, flags=re.IGNORECASE | re.MULTILINE)
            response_text = re.sub(r"\s*```$", "", response_text, flags=re.MULTILINE)

            logging.debug(f"AI response after cleaning: {repr(response_text)}")

            try:
                parsed_response = json.loads(response_text)

                if isinstance(parsed_response, list):
                    if len(parsed_response) == 0:
                        raise ValueError("Empty array response from AI")
                    parsed_response = parsed_response[0]
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


async def send_to_openai(session, headers, payload):
    if not active_openai_url:
        logging.error("No active AI URL configured")
        return None
        
    try:
        async with session.post(active_openai_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            if resp.status == 200:
                return await resp.json()
            logging.warning(f"{active_openai_url} failed with status {resp.status}")
    except Exception as e:
        logging.error(f"{active_openai_url} request error: {e}")
    return None


async def update_patient_info_from_fhir(exam):
    # Check if HIS integration is enabled
    if not ENABLE_HIS:
        return
        
    patient_cnp = exam['patient']['cnp']
    patient_name = exam['patient']['name']
    patient_birthdate = exam['patient']['birthdate']
    if patient_cnp and (not exam['patient']['id'] or not patient_birthdate or patient_birthdate == -1):
        async with aiohttp.ClientSession(headers={'User-Agent': USER_AGENT}) as session:
            fhir_patient = await get_fhir_patient(session, patient_cnp, patient_name)
            if fhir_patient:
                if 'id' in fhir_patient:
                    exam['patient']['id'] = fhir_patient['id']
                    db_update_patient_id(patient_cnp, fhir_patient['id'])

                if (not patient_birthdate or patient_birthdate == -1) and 'birthDate' in fhir_patient:
                    try:
                        birthdate = fhir_patient['birthDate']
                        if len(birthdate) == 10 and birthdate[4] == '-' and birthdate[7] == '-':
                            exam['patient']['birthdate'] = birthdate
                            birth_date = datetime.strptime(birthdate, "%Y-%m-%d")
                            today = datetime.now()
                            age = today.year - birth_date.year
                            if (today.month, today.day) < (birth_date.month, birth_date.day):
                                age -= 1
                            exam['patient']['age'] = age
                            db_update('patients', 'cnp = ?', (patient_cnp,), birthdate=birthdate)
                    except Exception as e:
                        logging.error(f"Error parsing birthdate from FHIR for patient {patient_cnp}: {e}")


def prepare_exam_data(exam):
    with open(os.path.join(IMAGES_DIR, f"{exam['uid']}.png"), 'rb') as f:
        image_bytes = f.read()
    region, question = identify_anatomic_region(exam)
    if not region in REGIONS:
        logging.info(f"Ignoring {exam['uid']} with {region} x-ray.")
        db_set_status(exam['uid'], 'ignore')
        return None, None, None, None, None
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
    exam['exam'].update({'region': region, 'projection': projection})
    subject = " ".join([txtAge, gender])
    anatomy = " ".join([projection, region]) if region else ""
    return region, question, subject, anatomy.strip(), image_bytes


def create_exam_prompt(exam, region, question, subject, anatomy):
    has_ai_report = (
        'ai' in exam.get('report', {})
        and exam['report']['ai'].get('text')
    )

    previous_reports = []
    if not has_ai_report:
        previous_reports = db_get_previous_reports(
            exam['patient']['cnp'],
            region,
            months=3
        ) or []

    prompt_lines = []

    justification = exam.get('report', {}).get('rad', {}).get('justification')
    if justification:
        prompt_lines.extend([
            "CLINICAL INFORMATION",
            justification.strip(),
            ""
        ])

    if previous_reports:
        prompt_lines.append("PRIOR STUDIES")
        for report, date in previous_reports[:3]:
            prompt_lines.append(f"- {date}: {report}")
        prompt_lines.append("")

    prompt_lines.extend([
        "TASK",
        PROMPTS['USR_PROMPT'].format(
            question=question,
            anatomy=anatomy,
            subject=subject
        ).strip()
    ])

    template_items = REGION_TEMPLATES.get(region, [])
    if template_items:
        prompt_lines.append("")
        prompt_lines.append("ASSESS IN ORDER")
        for item in template_items:
            prompt_lines.append(f"- {item}")

    if previous_reports:
        prompt_lines.append(
            "When relevant, describe interval change compared to prior studies "
            "(new, stable, improved, or resolved findings)."
        )

    return "\n".join(prompt_lines)

def prepare_ai_request_data(prompt, image_bytes):
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    image_url = f"data:image/png;base64,{image_b64}"
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json',
    }
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
                "content": [{"type": "text", "text": PROMPTS['REP_PROMPT'].strip()}]
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


def parse_ai_report_text(report_text, impression_max_words=3):
    """Split a raw AI report into (findings, impression).

    Impression is discarded (set to None) when longer than
    impression_max_words, matching the terse-summary convention the rest of
    the pipeline expects from IMPRESSION lines.
    """
    findings = None
    impression = None
    findings_match = re.search(r'FINDINGS:(.*?)(IMPRESSION:|$)', report_text, re.DOTALL)
    impression_match = re.search(r'IMPRESSION:(.*)', report_text, re.DOTALL)

    if findings_match and impression_match:
        findings = findings_match.group(1).strip()
        impression = impression_match.group(1).strip()
        if impression and len(impression.split()) > impression_max_words:
            impression = None
    elif findings_match:
        findings = findings_match.group(1).strip()
        impression = None
    else:
        findings = report_text
        impression = None

    return findings, impression


async def send_exam_to_openai(exam, max_retries = 3):
    try:
        await update_patient_info_from_fhir(exam)

        region, question, subject, anatomy, image_bytes = await asyncio.to_thread(prepare_exam_data, exam)
        if region is None:
            return False

        prompt = create_exam_prompt(exam, region, question, subject, anatomy)

        prior_ai_report = (exam.get('report') or {}).get('ai') or {}
        prior_ai_text = prior_ai_report.get('text') if isinstance(prior_ai_report, dict) else None
        if prior_ai_text:
            try:
                parsed = json.loads(prior_ai_text)
                if isinstance(parsed, dict) and 'report' in parsed:
                    prior_ai_text = parsed['report']
            except (json.JSONDecodeError, TypeError):
                pass
        if prior_ai_text and (len(prior_ai_text) < 30 or any(k in prior_ai_text for k in ('ROLE', 'TASK', 'ASSESS IN ORDER', 'OUTPUT CONSTRAINTS'))):
            prior_ai_text = None
        if prior_ai_text:
            logging.info(f"Previous report: {prior_ai_text}")
            # MedGemma is tuned for single-turn use, not multi-turn chat, so the
            # revision request is folded into one user turn rather than faked
            # via assistant/user message roles.
            prompt = f"{prompt}\n\nPREVIOUS REPORT:\n{prior_ai_text}\n\n{PROMPTS['REV_PROMPT'].strip()}"

        logging.debug(f"Prompt: {prompt}")
        logging.info(f"Processing {exam['uid']} with {region} x-ray.")

        headers, data = prepare_ai_request_data(prompt, image_bytes)

        # Up to 3 attempts with exponential backoff (2s, 4s, 8s delays).
        attempt = 1
        async with aiohttp.ClientSession(headers={'User-Agent': USER_AGENT}) as session:
            while attempt <= max_retries:
                try:
                    start_time = asyncio.get_running_loop().time()
                    result = await send_to_openai(session, headers, data)

                    global timings
                    end_time = asyncio.get_running_loop().time()
                    processing_time = int((end_time - start_time) * 1000)
                    if timings['examination'] > 0:
                        timings['examination'] = int((3 * timings['examination'] + processing_time) / 4)
                    else:
                        timings['examination'] = processing_time

                    if not result:
                        break
                    response_text = result["choices"][0]["message"]["content"]
                    response_model = result.get("model", MODEL_NAME)
                    
                    # Process AI response - extract report text
                    report = response_text.strip()
                    if not report:
                        logging.error(f"Empty AI response for exam {exam['uid']}")
                        raise ValueError("Empty AI response")

                    logging.info(f"AI report for {exam['uid']}: {' '.join(report.split()[:10])}...")

                    findings, impression = parse_ai_report_text(report)

                    db_insert('ai_reports',
                        uid=exam['uid'],
                        text=findings,
                        summary=impression,
                        model=response_model,
                        latency=int(processing_time))

                    await check_ai_report_and_update(exam['uid'])

                    updated_report = db_get_ai_report(exam['uid'])
                    severity = updated_report.get('severity', -1) if updated_report else -1
                    is_positive = severity >= SEVERITY_THRESHOLD
                    ai_report = (exam.get('report') or {}).get('ai') or {}
                    reviewed = ai_report.get('reviewed', False) if isinstance(ai_report, dict) else False
                    await broadcast_dashboard_update(event = "new_exam", payload = {'uid': exam['uid'], 'positive': is_positive, 'reviewed': reviewed, 'severity': severity})
                    if is_positive:
                        try:
                            await send_ntfy_notification(exam['uid'], report, exam)
                        except Exception as e:
                            logging.error(f"Failed to send ntfy notification: {e}")
                    return True

                except Exception as e:
                    logging.warning(f"Error uploading {exam['uid']} (attempt {attempt}): {e}")
                    await asyncio.sleep(2 ** attempt)
                    attempt += 1

        db_set_status(exam['uid'], 'error')
        QUEUE_EVENT.clear()
        logging.error(f"Failed to process {exam['uid']} after {attempt} attempts.")
        await broadcast_dashboard_update(event="error", payload={'uid': exam['uid'], 'reason': 'max_retries'})
        return False
    except Exception as e:
        logging.error(f"Critical error for {exam['uid']}: {e}")
        db_set_status(exam['uid'], 'error')
        await broadcast_dashboard_update(event="error", payload={'uid': exam['uid'], 'reason': str(e)})
        return False


# Threads
# WebSocket and WebServer operations
async def serve_dashboard_page(request):
    return web.FileResponse(path=os.path.join(STATIC_DIR, "dashboard.html"))

async def serve_stats_page(request):
    return web.FileResponse(path=os.path.join(STATIC_DIR, "stats.html"))

async def serve_about_page(request):
    return web.FileResponse(path=os.path.join(STATIC_DIR, "about.html"))


async def serve_radiologists_page(request):
    return web.FileResponse(path=os.path.join(STATIC_DIR, "radiologists.html"))


async def serve_diagnostics_page(request):
    return web.FileResponse(path=os.path.join(STATIC_DIR, "diagnostics.html"))


async def serve_insights_page(request):
    return web.FileResponse(path=os.path.join(STATIC_DIR, "insights.html"))


async def serve_check_page(request):
    return web.FileResponse(path=os.path.join(STATIC_DIR, "check.html"))

async def serve_favicon(request):
    return web.FileResponse(path=os.path.join(STATIC_DIR, "favicon.ico"))


async def serve_api_spec(request):
    spec_path = os.path.join(STATIC_DIR, "spec.json")
    with open(spec_path, 'r') as f:
        spec = json.load(f)
    server_url = f"{request.scheme}://{request.host}"
    spec['servers'][0]['url'] = server_url
    return web.json_response(spec)


async def websocket_handler(request):
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)

    websocket_clients.add(ws)

    try:
        await broadcast_dashboard_update(event="connected", payload={'address': request.remote}, client=ws)
        logging.info(f"Dashboard connected via WebSocket from {request.remote}")

        async for msg in ws:
            pass

    except asyncio.CancelledError:
        logging.debug(f"WebSocket connection cancelled for {request.remote}")
        raise
    except Exception as e:
        logging.error(f"WebSocket error for {request.remote}: {e}")
    finally:
        websocket_clients.discard(ws)
        try:
            await ws.close()
        except Exception as e:
            logging.debug(f"Error closing WebSocket for {request.remote}: {e}")
        
        logging.info(f"Dashboard WebSocket disconnected from {request.remote}")
    
    return ws


async def exams_handler(request):
    try:
        user_role = getattr(request, 'user_role', 'user')
        
        try:
            page = max(1, int(request.query.get("page", "1")))
        except ValueError:
            return web.json_response({"error": "Invalid page parameter"}, status=400)
        filters = {}
        for filter in ['positive', 'correct', 'reviewed']:
            value = request.query.get(filter, 'any')
            if value != 'any':
                filters[filter] = 1 if value.lower().startswith('y') else 0
        for filter in ['region', 'status', 'search', 'diagnostic', 'radiologist']:
            value = request.query.get(filter, 'any')
            if value != 'any':
                if filter == 'status':
                    if ',' in value:
                        filters[filter] = [s.strip().lower() for s in value.split(',')]
                    else:
                        filters[filter] = value.lower()
                else:
                    filters[filter] = value
        severity_value = request.query.get('severity', 'any')
        severity_op = request.query.get('severity_op', 'any')
        if severity_value != 'any' and severity_op != 'any':
            try:
                filters['severity'] = int(severity_value)
                if severity_op in ['equal', 'lower', 'higher']:
                    filters['severity_op'] = severity_op
            except ValueError:
                pass
        confidence_value = request.query.get('confidence', 'any')
        if confidence_value != 'any' and confidence_value:
            filters['confidence'] = confidence_value
        offset = (page - 1) * PAGE_SIZE
        data, total = db_get_exams(limit = PAGE_SIZE, offset = offset, **filters)

        for exam in data:
            if user_role != 'admin':
                exam['patient']['name'] = extract_patient_initials(exam['patient']['name'])
                patient_cnp = exam['patient']['cnp']
                if patient_cnp and len(patient_cnp) > 7:
                    exam['patient']['cnp'] = patient_cnp[:7] + '...'
                elif patient_cnp:
                    exam['patient']['cnp'] = patient_cnp
                else:
                    exam['patient']['cnp'] = 'Unknown'
                if 'radiologist' in exam['report']['rad']:
                    exam['report']['rad']['radiologist'] = extract_radiologist_initials(exam['report']['rad']['radiologist'])
        return web.json_response({
            "exams": data,
            "total": total,
            "pages": math.ceil(total / PAGE_SIZE) if total else 1,
            "filters": filters,
        })
    except Exception as e:
        logging.error(f"Exams page error: {e}")
        return web.json_response([], status = 500)


async def csv_export_handler(request):
    try:
        user_role = getattr(request, 'user_role', 'user')
        filters = {}
        for f in ['positive', 'correct', 'reviewed']:
            value = request.query.get(f, 'any')
            if value != 'any':
                filters[f] = 1 if value.lower().startswith('y') else 0
        for f in ['region', 'status', 'search', 'diagnostic', 'radiologist']:
            value = request.query.get(f, 'any')
            if value != 'any':
                filters[f] = value
        confidence_value = request.query.get('confidence', 'any')
        if confidence_value != 'any' and confidence_value:
            filters['confidence'] = confidence_value
        data, _ = db_get_exams(limit=10000, offset=0, **filters)

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            'uid', 'patient_name', 'cnp', 'age', 'sex',
            'exam_date', 'region', 'status', 'modality', 'protocol',
            'ai_positive', 'ai_severity', 'ai_confidence',
            'rad_positive', 'rad_severity', 'rad_diagnostic', 'radiologist',
            'correct',
        ])
        for exam in data:
            p = exam.get('patient', {})
            e = exam.get('exam', {})
            ai = exam.get('report', {}).get('ai', {})
            rad = exam.get('report', {}).get('rad', {})
            name = p.get('name', '')
            rad_name = rad.get('radiologist', '')
            if user_role != 'admin':
                name = extract_patient_initials(name)
                rad_name = extract_radiologist_initials(rad_name)
            writer.writerow([
                exam.get('uid', ''),
                name,
                p.get('cnp', ''),
                p.get('age', ''),
                p.get('sex', ''),
                e.get('date', ''),
                e.get('region', ''),
                e.get('status', ''),
                e.get('type', ''),
                e.get('protocol', ''),
                ai.get('positive', ''),
                ai.get('severity', ''),
                ai.get('confidence', ''),
                rad.get('positive', ''),
                rad.get('severity', ''),
                rad.get('summary', ''),
                rad_name,
                exam.get('report', {}).get('correct', ''),
            ])
        filename = f"xrayvision_{datetime.now().strftime('%Y%m%d')}.csv"
        return web.Response(
            body=output.getvalue(),
            content_type='text/csv',
            headers={'Content-Disposition': f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        logging.error(f"CSV export error: {e}")
        return web.json_response({"error": "Export failed"}, status=500)


async def stats_handler(request):
    try:
        return web.json_response(await asyncio.to_thread(db_get_stats))
    except Exception as e:
        logging.error(f"Exams page error: {e}")
        return web.json_response([], status = 500)


async def config_handler(request):
    try:
        user_role = getattr(request, 'user_role', 'user')
        config = {
            "MODEL_NAME": MODEL_NAME,
            "AE_TITLE": AE_TITLE,
            "AE_PORT": AE_PORT,
            "DASHBOARD_PORT": DASHBOARD_PORT,
            "USER_ROLE": user_role,
        }
        # Infrastructure addresses only exposed to admins
        if user_role == 'admin':
            config.update({
                "OPENAI_URL_PRIMARY": OPENAI_URL_PRIMARY,
                "OPENAI_URL_SECONDARY": OPENAI_URL_SECONDARY,
                "NTFY_URL": NTFY_URL,
                "REMOTE_AE_TITLE": REMOTE_AE_TITLE,
                "REMOTE_AE_IP": REMOTE_AE_IP,
                "REMOTE_AE_PORT": REMOTE_AE_PORT,
            })
        return web.json_response(config)
    except Exception as e:
        logging.error(f"Config endpoint error: {e}")
        return web.json_response({}, status = 500)


async def regions_handler(request):
    try:
        regions = db_get_regions()
        if request.rel_url.query.get('detail') == '1':
            detail = []
            for r in regions:
                detail.append({
                    'region': r,
                    'question': REGION_QUESTIONS.get(r, ''),
                    'template': REGION_TEMPLATES.get(r, []),
                    'supported': r in REGIONS,
                })
            return web.json_response(detail)
        return web.json_response(regions)
    except Exception as e:
        logging.error(f"Regions endpoint error: {e}")
        return web.json_response([], status = 500)


async def diagnostics_handler(request):
    try:
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
    try:
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
        
        monthly_trends = {}
        if trends_rows:
            monthly_data = {}
            for row in trends_rows:
                month, diagnostic, count = row
                if month not in monthly_data:
                    monthly_data[month] = []
                monthly_data[month].append({'diagnostic': diagnostic, 'count': count})

            for month, diagnostics in monthly_data.items():
                monthly_trends[month] = sorted(diagnostics, key=lambda x: x['count'], reverse=True)[:10]

        return web.json_response({'trends': monthly_trends})
    except Exception as e:
        logging.error(f"Diagnostics monthly trends endpoint error: {e}")
        return web.json_response({}, status = 500)


async def diagnostics_stats_handler(request):
    try:
        query = """
            SELECT 
                rr.summary,
                COUNT(*) as report_count,
                AVG(CAST(rr.severity AS FLOAT)) as avg_severity,
                SUM(CASE WHEN (ar.severity >= ? AND rr.severity >= ?) OR (ar.severity < ? AND rr.severity < ?) THEN 1 ELSE 0 END) as correct_predictions,
                COUNT(CASE WHEN ar.severity IS NOT NULL THEN 1 END) as ai_compared,
                GROUP_CONCAT(e.region, '||') as regions,
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

                region_freq = {}
                if regions:
                    for region in regions.split('||'):
                        region = region.strip().lower()
                        if region:
                            region_freq[region] = region_freq.get(region, 0) + 1

                top_regions = dict(sorted(region_freq.items(), key=lambda x: x[1], reverse=True)[:5])

                accuracy = 0
                correct_predictions = correct_predictions or 0
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




async def cleanup_dead_websocket_clients():
    global websocket_clients
    dead_clients = []

    for client in list(websocket_clients):
        try:
            if client._protocol is None:  # protocol is None when the connection is closed
                dead_clients.append(client)
        except (AttributeError, RuntimeError):
            dead_clients.append(client)

    for client in dead_clients:
        try:
            websocket_clients.discard(client)
        except Exception as e:
            logging.debug(f"Error removing dead WebSocket client: {e}")

    if dead_clients:
        logging.debug(f"Cleaned up {len(dead_clients)} dead WebSocket clients")

# Per-IP request timestamps for rate limiting: {ip: [timestamps]}
_rate_limit_store: dict = {}
# Heavy AI endpoints get a stricter limit
_RATE_LIMIT_HEAVY = {'/api/check', '/api/analyse', '/api/translate'}
_RATE_LIMIT_HEAVY_MAX = 10    # requests per minute
_RATE_LIMIT_DEFAULT_MAX = 60  # requests per minute

@web.middleware
async def rate_limit_middleware(request, handler):
    """Sliding-window per-IP rate limiter for API endpoints."""
    if not request.path.startswith('/api/'):
        return await handler(request)
    ip = request.remote
    now = asyncio.get_running_loop().time()
    window = 60.0
    limit = _RATE_LIMIT_HEAVY_MAX if request.path in _RATE_LIMIT_HEAVY else _RATE_LIMIT_DEFAULT_MAX
    timestamps = _rate_limit_store.get(ip, [])
    timestamps = [t for t in timestamps if now - t < window]
    if len(timestamps) >= limit:
        logging.warning(f"Rate limit exceeded for {ip} on {request.path}")
        return web.json_response({"error": "Too many requests"}, status=429)
    timestamps.append(now)
    _rate_limit_store[ip] = timestamps
    stale = [k for k, v in _rate_limit_store.items() if not v or now - v[-1] >= window]
    for k in stale:
        del _rate_limit_store[k]
    return await handler(request)


@web.middleware
async def auth_middleware(request, handler):
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
            audit_logger.warning(f"AUTH_FAIL user={username} ip={request.remote} path={request.path}")
            raise ValueError("Invalid authentication")
        request.user_role = user_info['role']
        request.username = username
        if not request.path.startswith('/api/') and request.path not in ('/ws', '/favicon.ico'):
            audit_logger.info(f"AUTH_OK user={username} role={user_info['role']} ip={request.remote} path={request.path}")
    except UnicodeDecodeError:
        audit_logger.warning(f"AUTH_FAIL user=<malformed> ip={request.remote} path={request.path}")
        raise web.HTTPUnauthorized(
            text = "401: Invalid authentication",
            headers = {'WWW-Authenticate': 'Basic realm="XRayVision"'})
    except ValueError:
        raise web.HTTPUnauthorized(
            text = "401: Invalid authentication",
            headers = {'WWW-Authenticate': 'Basic realm="XRayVision"'})
    return await handler(request)


async def broadcast_dashboard_update(event = None, payload = None, client = None):
    if not (websocket_clients or client):
        return
    dashboard['queue_size'] = db_count('exams', where_clause="status IN (?, ?)", where_params=('queued', 'requeue'))
    dashboard['check_queue_size'] = db_count('exams', where_clause="status = ?", where_params=('check',))
    error_stats = db_get_error_stats()
    dashboard['error_count'] = error_stats['error']
    dashboard['ignore_count'] = error_stats['ignore']
    dashboard['success_count'] = db_get_weekly_processed_count()
    clients = [client,] if client else websocket_clients.copy()
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
    for client in clients:
        try:
            await client.send_json(data)
        except Exception as e:
            logging.debug(f"Error sending update to WebSocket client: {e}")
            websocket_clients.discard(client)


# Notification operations
async def send_ntfy_notification(uid, report, info):
    if not ENABLE_NTFY:
        return

    try:
        message = f"Positive finding in {info['exam']['region']} study\nPatient: {info['patient']['name']}\nReport: {report}"
        headers = {
            "Title": "XRayVision Alert - Positive Finding",
            "Tags": "warning,skull",
            "Priority": "4",
        }
        if NTFY_IMAGE_BASE_URL:
            headers["Attach"] = f"{NTFY_IMAGE_BASE_URL.rstrip('/')}/images/{uid}.png"

        async with aiohttp.ClientSession(headers={'User-Agent': USER_AGENT}) as session:
            async with session.post(
                NTFY_URL,
                data=message,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    logging.debug("Successfully sent ntfy notification")
                else:
                    logging.warning(f"Notification failed with status {resp.status}: {await resp.text()}")
    except Exception as e:
        logging.error(f"Failed to send ntfy notification: {e}")


async def insights_handler(request):
    try:
        insights = {}

        rows = db_get_processing_times_by_region()
        insights['processing_times'] = {}
        if rows:
            for row in rows:
                region, avg_time, count = row
                insights['processing_times'][region] = {'avg_time': round(avg_time, 2), 'count': count}

        rows = db_get_rad_severity_distribution()
        insights['rad_severity_distribution'] = {}
        if rows:
            for row in rows:
                severity, count = row
                insights['rad_severity_distribution'][str(severity)] = count

        rows = db_get_ai_severity_distribution()
        insights['ai_severity_distribution'] = {}
        if rows:
            for row in rows:
                severity, count = row
                insights['ai_severity_distribution'][str(severity)] = count

        rows = db_get_severity_differences()
        insights['severity_differences'] = {}
        if rows:
            for row in rows:
                diff, count = row
                insights['severity_differences'][str(diff)] = count

        rows = db_get_age_distribution_insights(SEVERITY_THRESHOLD)
        insights['age_distribution'] = {}
        if rows:
            for row in rows:
                age_group, total_exams, positive_findings = row
                positive_findings = positive_findings or 0
                insights['age_distribution'][age_group] = {
                    'total_exams': total_exams,
                    'positive_findings': positive_findings,
                    'positive_rate': round((positive_findings / total_exams) * 100, 1) if total_exams > 0 else 0
                }

        rows = db_get_hourly_patterns()
        insights['hourly_patterns'] = {}
        if rows:
            for row in rows:
                hour, count = row
                insights['hourly_patterns'][str(hour)] = count

        row = db_get_requeue_analysis()
        if row:
            total_requeued, avg_latency = row
            insights['requeue_analysis'] = {
                'total_requeued': total_requeued or 0,
                'avg_latency': round(avg_latency, 2) if avg_latency else 0
            }

        user_role = getattr(request, 'user_role', 'user')
        rows = db_get_radiologist_metrics()
        insights['radiologist_metrics'] = {}
        if rows:
            for row in rows:
                radiologist, reports_count, avg_severity, unique_exams = row
                display_name = radiologist if user_role == 'admin' else extract_radiologist_initials(radiologist)
                insights['radiologist_metrics'][display_name] = {
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
    try:
        user_role = getattr(request, 'user_role', 'user')
        query = """
            SELECT radiologist, COUNT(*) as report_count
            FROM rad_reports
            WHERE radiologist IS NOT NULL AND radiologist != ''
            GROUP BY radiologist
            ORDER BY radiologist
        """
        rows = db_execute_query(query, fetch_mode='all')
        if rows:
            radiologists = {}
            for radiologist, count in rows:
                display_name = radiologist if user_role == 'admin' else extract_radiologist_initials(radiologist)
                radiologists[display_name] = radiologists.get(display_name, 0) + count
        else:
            radiologists = {}
        return web.json_response(radiologists)
    except Exception as e:
        logging.error(f"Radiologists endpoint error: {e}")
        return web.json_response({}, status = 500)


async def radiologist_stats_handler(request):
    try:
        user_role = getattr(request, 'user_role', 'user')
        query = """
            SELECT 
                rr.radiologist,
                COUNT(*) as report_count,
                AVG(CAST(rr.severity AS FLOAT)) as avg_severity,
                SUM(CASE WHEN (ar.severity >= ? AND rr.severity >= ?) OR (ar.severity < ? AND rr.severity < ?) THEN 1 ELSE 0 END) as correct_predictions,
                COUNT(CASE WHEN ar.severity >= 0 THEN 1 END) as ai_compared,
                GROUP_CONCAT(rr.summary, '||') as all_diagnostics
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

                diagnostics = {}
                if all_diagnostics:
                    for diag in all_diagnostics.split('||'):
                        diag = diag.strip().lower()
                        if diag:
                            diagnostics[diag] = diagnostics.get(diag, 0) + 1

                top_diagnostics = dict(sorted(diagnostics.items(), key=lambda x: x[1], reverse=True)[:5])

                accuracy = 0
                correct_predictions = correct_predictions or 0
                if ai_compared and ai_compared > 0:
                    accuracy = round((correct_predictions / ai_compared) * 100, 1)

                display_name = radiologist if user_role == 'admin' else extract_radiologist_initials(radiologist)
                radiologist_stats[display_name] = {
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
    try:
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
        
        user_role = getattr(request, 'user_role', 'user')

        monthly_trends = {}
        if trends_rows:
            monthly_data = {}
            for row in trends_rows:
                month, radiologist, count = row
                display_name = radiologist if user_role == 'admin' else extract_radiologist_initials(radiologist)
                if month not in monthly_data:
                    monthly_data[month] = []
                monthly_data[month].append({'radiologist': display_name, 'count': count})

            for month, radiologists in monthly_data.items():
                monthly_trends[month] = sorted(radiologists, key=lambda x: x['count'], reverse=True)[:10]

        return web.json_response({'trends': monthly_trends})
    except Exception as e:
        logging.error(f"Radiologists monthly trends endpoint error: {e}")
        return web.json_response({}, status = 500)


async def severity_handler(request):
    try:
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
    try:
        user_role = getattr(request, 'user_role', 'user')

        try:
            page = max(1, int(request.query.get("page", "1")))
        except ValueError:
            return web.json_response({"error": "Invalid page parameter"}, status=400)
        filters = {}
        for filter in ['search']:
            value = request.query.get(filter, 'any')
            if value != 'any':
                filters[filter] = value
        offset = (page - 1) * PAGE_SIZE
        patients, total = db_get_patients(limit=PAGE_SIZE, offset=offset, **filters)

        for patient in patients:
            if user_role != 'admin':
                patient['name'] = extract_patient_initials(patient['name'])
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
            "pages": math.ceil(total / PAGE_SIZE) if total else 1,
            "filters": filters,
        })
    except Exception as e:
        logging.error(f"Patients page error: {e}")
        return web.json_response([], status = 500)


async def patient_handler(request):
    try:
        user_role = getattr(request, 'user_role', 'user')
        cnp = request.match_info['cnp']
        patient = db_get_patient_by_cnp(cnp)

        if not patient:
            return web.json_response({"error": "Patient not found"}, status=404)

        exam_uids = db_get_patient_exam_uids(cnp)
        patient['exams'] = exam_uids

        if user_role != 'admin':
            patient['name'] = extract_patient_initials(patient['name'])
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
    try:
        user_role = getattr(request, 'user_role', 'user')
        uid = request.match_info['uid']
        exams, _ = db_get_exams(limit=1, uid=uid)
        if not exams:
            return web.json_response({"error": "Exam not found"}, status=404)

        exam = exams[0]

        if user_role != 'admin':
            exam['patient']['name'] = extract_patient_initials(exam['patient']['name'])
            patient_cnp = exam['patient']['cnp']
            if patient_cnp and len(patient_cnp) > 7:
                exam['patient']['cnp'] = patient_cnp[:7] + '...'
            elif patient_cnp:
                exam['patient']['cnp'] = patient_cnp
            else:
                exam['patient']['cnp'] = 'Unknown'
            if 'radiologist' in exam['report']['rad']:
                exam['report']['rad']['radiologist'] = extract_radiologist_initials(exam['report']['rad']['radiologist'])
        return web.json_response(exam)
    except Exception as e:
        logging.error(f"Exam endpoint error: {e}")
        return web.json_response({"error": "Internal server error"}, status=500)


async def dicom_query(request):
    try:
        data = await request.json()
        try:
            hours = max(1, min(168, int(data.get('hours', 3))))  # clamp 1–168 h (1 week)
        except (ValueError, TypeError):
            return web.json_response({'status': 'error', 'message': 'Invalid hours parameter'}, status=400)
        logging.debug(f"Manual QueryRetrieve triggered for the last {hours} hours.")
        audit_logger.info(f"DICOM_QUERY hours={hours} user={getattr(request, 'username', '')} ip={request.remote}")
        await query_and_retrieve(hours * 60)
        return web.json_response({'status': 'success',
                                  'message': f'Query triggered for the last {hours} hours.'})
    except Exception as e:
        logging.error(f"Error processing manual query: {e}")
        return web.json_response({'status': 'error',
                                  'message': str(e)})


async def rad_review(request):
    try:
        data = await request.json()
        uid = data.get('uid')
        normal = data.get('normal', None)
        radiologist = getattr(request, 'username', '')

        if not uid or normal is None:
            return web.json_response({'status': 'error', 'message': 'UID and normal status are required'}, status=400)

        db_rad_review(uid, normal, radiologist)

        exams, _ = db_get_exams(limit=1, uid=uid)
        exam_data = exams[0] if exams else {}
        verdict = 'normal' if normal else 'abnormal'
        correct = exam_data.get('report', {}).get('correct')
        validation_str = 'validates' if correct == 1 else 'invalidates' if correct == 0 else 'does not assess'
        logging.info(f"Exam {uid} marked as {verdict} by radiologist {radiologist}, which {validation_str} the AI report.")
        audit_logger.info(f"RAD_REVIEW uid={uid} verdict={verdict} radiologist={radiologist} ip={request.remote}")
        await broadcast_dashboard_update(event = "radreview", payload = exam_data)
        response = {'status': 'success'}
        return web.json_response(response)
    except Exception as e:
        logging.error(f"Error processing radiologist review: {e}")
        return web.json_response({'status': 'error', 'message': str(e)}, status=500)


async def requeue_exam(request):
    try:
        data = await request.json()
        uid = data.get('uid')

        if not uid:
            return web.json_response({'status': 'error', 'message': 'UID is required'}, status=400)

        success = db_requeue_exam(uid)

        if success:
            logging.info(f"Exam {uid} re-queued for processing.")
            audit_logger.info(f"REQUEUE uid={uid} user={getattr(request, 'username', '')} ip={request.remote}")
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
    try:
        data = await request.json()
        uid = data.get('uid')
        
        if not uid:
            return web.json_response({'status': 'error', 'message': 'UID is required'}, status=400)
        
        exams, _ = db_get_exams(limit=1, uid=uid)
        if not exams:
            return web.json_response({'status': 'error', 'message': 'Exam not found'}, status=404)

        exam = exams[0]
        response = web.json_response({'status': 'success', 'message': f'Report retrieval started for exam {uid}'})

        async def async_process():
            try:
                if not exam['patient']['id']:
                    async with aiohttp.ClientSession(headers={'User-Agent': USER_AGENT}) as session:
                        formatted_name = format_patient_name_for_fhir(exam['patient']['name'])
                        patient_id = await get_patient_id_from_fhir(session, exam['patient']['cnp'], formatted_name)
                        if patient_id:
                            exam['patient']['id'] = patient_id

                if exam['patient']['id']:
                    current_exam = exam['exam']
                    current_exam['uid'] = uid
                    async with aiohttp.ClientSession(headers={'User-Agent': USER_AGENT}) as session:
                        await process_single_exam_without_rad_report(session, current_exam, exam['patient']['id'])

                QUEUE_EVENT.set()
                await broadcast_dashboard_update(event="radreport", payload={'uid': uid})
            except Exception as e:
                logging.error(f"Error processing radiologist report for exam {uid}: {e}")

        asyncio.create_task(async_process())
        
        return response
    except Exception as e:
        logging.error(f"Error checking radiologist report: {e}")
        return web.json_response({'status': 'error', 'message': str(e)}, status=500)

async def check_report_handler(request):
    try:
        data = await request.json()
        report_text = data.get('report', '').strip()

        result = await check_report(report_text)

        if 'error' in result:
            status = 500 if result['error'] != 'No report text provided' else 400
            return web.json_response(result, status=status)
        
        return web.json_response(result)
    except Exception as e:
        logging.error(f"Error processing report check request: {e}")
        return web.json_response({'error': 'Internal server error'}, status=500)


async def detailed_analysis_handler(request):
    try:
        data = await request.json()
        report_text = data.get('report', '').strip()

        result = await detailed_analysis_report(report_text)

        if 'error' in result:
            status = 500 if result['error'] != 'No report text provided' else 400
            return web.json_response(result, status=status)

        return web.json_response(result)
    except Exception as e:
        logging.error(f"Error processing detailed analysis request: {e}")
        return web.json_response({'error': 'Internal server error'}, status=500)

async def translate_handler(request):
    try:
        data = await request.json()
        report_text = data.get('report', '').strip()
        result = await translate_report(report_text)
        if result is None:
            return web.json_response({'error': 'Translation failed'}, status=500)
        return web.json_response({'translation': result})
    except Exception as e:
        logging.error(f"Error processing translation request: {e}")
        return web.json_response({'error': 'Internal server error'}, status=500)


async def start_dashboard():
    global web_server
    app = web.Application(middlewares = [rate_limit_middleware, auth_middleware])
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
    app.router.add_get('/api/exams/export', csv_export_handler)
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




# Background loops

async def relay_to_openai_loop():
    while True:
        try:
            exams, total = db_get_exams(limit=1, status=['queued', 'requeue', 'check'])
        except Exception as e:
            logging.error(f"relay_to_openai_loop: failed to query queue: {e}")
            await asyncio.sleep(5)
            continue
        if not exams or active_openai_url is None:
            QUEUE_EVENT.clear()
            await QUEUE_EVENT.wait()
            continue
        (exam,) = exams
        dicom_file = os.path.join(IMAGES_DIR, f"{exam['uid']}.dcm")
        try:
            db_set_status(exam['uid'], "processing")
            dashboard['queue_size'] = total
            dashboard['processing'] = extract_patient_initials(exam['patient']['name'])
            await broadcast_dashboard_update(event="processing_start", payload={'uid': exam['uid'], 'patient': dashboard['processing'], 'region': exam['exam'].get('region', '')})

            exam_status = exam['exam']['status']
            if exam_status in ['queued', 'requeue']:
                result = await send_exam_to_openai(exam)
                if result:
                    db_set_status(exam['uid'], "done")
                    if not KEEP_DICOM:
                        try:
                            if os.path.exists(dicom_file):
                                os.remove(dicom_file)
                        except Exception as e:
                            logging.warning(f"Error removing DICOM file {dicom_file}: {e}")
            elif exam_status == 'check':
                ai_report = db_get_ai_report(exam['uid'])
                if ai_report and not ai_report.get('summary'):
                    await check_ai_report_and_update(exam['uid'])
                rad_check_success = await check_rad_report_and_update(exam['uid'])
                if rad_check_success:
                    ai_report = db_get_ai_report(exam['uid']) or {}
                    await broadcast_dashboard_update(event="radcheck", payload={
                        'uid': exam['uid'],
                        'positive': ai_report.get('positive', -1),
                        'severity': ai_report.get('severity', -1),
                        'summary': ai_report.get('summary', ''),
                        'confidence': ai_report.get('confidence', -1),
                    })
                db_set_status(exam['uid'], "done")
        except Exception as e:
            logging.error(f"Unexpected error processing {exam['uid']}: {e}")
            db_set_status(exam['uid'], "error")
        finally:
            dashboard['processing'] = None
            await broadcast_dashboard_update()


async def openai_health_check():
    global active_openai_url
    while True:
        for url in [OPENAI_URL_PRIMARY, OPENAI_URL_SECONDARY]:
            base_url = url.split('/v1/')[0] if '/v1/' in url else url.rstrip('/')
            models_url = f"{base_url}/v1/models"
            try:
                async with aiohttp.ClientSession(headers={'User-Agent': USER_AGENT}) as session:
                    async with session.get(models_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
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
        if active_openai_url:
            QUEUE_EVENT.set()
        await broadcast_dashboard_update()
        await asyncio.sleep(300)


async def fhir_loop():
    while True:
        if not ENABLE_HIS:
            await asyncio.sleep(60)
            continue

        try:
            async with aiohttp.ClientSession(headers={'User-Agent': USER_AGENT}) as session:
                async with session.get(f"{FHIR_URL}/fhir/Metadata", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    health_status[FHIR_URL] = resp.status == 200
                    logging.debug(f"FHIR check {FHIR_URL} → {resp.status}")
                
                if health_status[FHIR_URL]:
                    # Process exams without radiologist reports
                    await process_exams_without_rad_reports(session)
        except Exception as e:
            health_status[FHIR_URL] = False
            logging.warning(f"Health check failed for FHIR: {e}")
        
        await broadcast_dashboard_update()
        delay = random.randint(30, 120)
        await asyncio.sleep(delay)

async def query_retrieve_loop():
    if NO_QUERY:
        logging.warning(f"Automatic Query/Retrieve disabled.")
    while not NO_QUERY:
        try:
            await query_and_retrieve()
        except Exception as e:
            logging.error(f"Unhandled error in query_retrieve_loop: {e}")
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
    while active_openai_url is None:
        logging.debug("translate_existing_reports: waiting for AI service to become available...")
        await asyncio.sleep(30)

    try:
        # Get all exams with radiologist reports that don't have translations
        query = """
            SELECT uid, text
            FROM rad_reports
            WHERE text IS NOT NULL
            AND text != ''
            AND (text_en IS NULL OR text_en = '')
            LIMIT 100
        """
        rows = db_execute_query(query, fetch_mode='all')

        if not rows:
            logging.info("No reports found that need translation")
            return

        logging.info(f"Found {len(rows)} reports to translate")
        for row in rows:
            uid, report_text = row
            try:
                translation = await translate_report(report_text)

                if translation:
                    is_valid, message = validate_translation(report_text, translation)
                    if is_valid:
                        db_update('rad_reports', 'uid = ?', (uid,), text_en=message)
                        logging.info(f"Successfully translated and updated exam {uid}: {' '.join(message.split()[:10])}...")
                    else:
                        logging.warning(f"Translation validation failed for exam {uid}: {message}")
                else:
                    logging.warning(f"Translation failed for exam {uid}")

                await asyncio.sleep(10)

            except Exception as e:
                logging.error(f"Error translating report for exam {uid}: {e}")

        logging.info("Translation of existing reports completed")

    except Exception as e:
        logging.error(f"Error in translate_existing_reports: {e}")

async def maintenance_loop():
    while True:
        clear_db_analyze_cache()
        clear_translation_cache()
        await cleanup_dead_websocket_clients()

        # Offloaded — runs multiple DELETE queries
        await asyncio.to_thread(db_purge_ignored_errors)

        # Offloaded — sqlite3.backup can take seconds on large DBs
        try:
            await asyncio.to_thread(db_backup)
        except Exception as e:
            logging.error(f"Database backup failed: {e}")

        await asyncio.sleep(86400)


def start_dicom_server():
    global dicom_server
    dicom_server = AE(ae_title = AE_TITLE)
    dicom_server.add_supported_context(Verification)
    dicom_server.add_supported_context(ComputedRadiographyImageStorage)
    dicom_server.add_supported_context(DigitalXRayImageStorageForPresentation)
    handlers = [
        (evt.EVT_C_STORE, dicom_store),
        (evt.EVT_C_ECHO, lambda event: 0x0000)
    ]
    logging.info(f"Starting DICOM server on port {AE_PORT} with AE Title '{AE_TITLE}'...")
    dicom_server.start_server(("0.0.0.0", AE_PORT), evt_handlers = handlers, block = False)


async def stop_servers():
    global dicom_server, web_server, websocket_clients
    clear_db_analyze_cache()
    for client in list(websocket_clients):
        try:
            await client.close()
        except Exception as e:
            logging.debug(f"Error closing WebSocket client: {e}")
    websocket_clients.clear()
    if dicom_server:
        try:
            dicom_server.shutdown()
            logging.info("DICOM server stopped.")
        except Exception as e:
            logging.error(f"Error stopping DICOM server: {e}")
    if web_server:
        try:
            await web_server.cleanup()
            logging.info("Web server stopped.")
        except Exception as e:
            logging.error(f"Error stopping web server: {e}")



async def main():
    global MAIN_LOOP
    MAIN_LOOP = asyncio.get_running_loop()
    if not os.path.exists(DB_FILE):
        logging.info("SQLite database not found. Creating a new one...")
        db_init()
    else:
        logging.info("SQLite database found.")
    logging.info(f"Python SQLite version: {sqlite3.version}")
    logging.info(f"SQLite library version: {sqlite3.sqlite_version}")

    if any(u['password'] == 'admin' for u in USERS.values()):
        logging.warning("SECURITY: Default admin password is still in use. Update credentials in local.cfg.")
    if OPENAI_API_KEY == 'sk-your-api-key':
        logging.warning("SECURITY: Default OPENAI_API_KEY placeholder is still set. Update it in local.cfg.")

    reset_count = db_update('exams', "status = ?", ('processing',), status='queued')
    if reset_count and reset_count > 0:
        logging.info(f"Reset {reset_count} exams from 'processing' to 'queued' status")
        QUEUE_EVENT.set()
    none_count = db_update('exams', "status = ?", ('none',), status='queued')
    if none_count and none_count > 0:
        logging.info(f"Recovered {none_count} orphaned exams from 'none' to 'queued' status")
        QUEUE_EVENT.set()

    exams, total = db_get_exams(status = 'done')
    logging.info(f"Loaded {len(exams)} exams from a total of {total}.")
    tasks = []
    tasks.append(asyncio.create_task(asyncio.to_thread(start_dicom_server)))
    tasks.append(asyncio.create_task(start_dashboard()))
    tasks.append(asyncio.create_task(openai_health_check()))
    tasks.append(asyncio.create_task(relay_to_openai_loop()))
    tasks.append(asyncio.create_task(query_retrieve_loop()))
    tasks.append(asyncio.create_task(maintenance_loop()))
    tasks.append(asyncio.create_task(fhir_loop()))
    if TRANSLATE_EXISTING:
        tasks.append(asyncio.create_task(translate_existing_reports()))
    if LOAD_DICOM:
        await load_existing_dicom_files()
        await query_and_retrieve(60)

    # SIGTERM (sent by the xrayvision control script / systemd / OpenRC) and
    # SIGINT (Ctrl-C) both resolve this event instead of tearing the process
    # down immediately, so stop_servers() always runs and in-flight requests
    # get a chance to finish.
    stop_event = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        MAIN_LOOP.add_signal_handler(sig, stop_event.set)
    stop_task = asyncio.create_task(stop_event.wait())

    try:
        await asyncio.wait([stop_task, *tasks], return_when=asyncio.FIRST_COMPLETED)
        if stop_task.done():
            logging.info("Shutdown signal received. Stopping XRayVision...")
    except asyncio.CancelledError:
        logging.info("Main task cancelled. Shutting down...")
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if not stop_task.done():
            stop_task.cancel()
        await asyncio.gather(*tasks, stop_task, return_exceptions=True)
        await stop_servers()


# Command run
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "XRayVision - Async DICOM processor with AI and WebSocket dashboard")
    parser.add_argument("--keep-dicom", action = "store_true", default=KEEP_DICOM, help = "Do not delete .dcm files after conversion")
    parser.add_argument("--load-dicom", action = "store_true", default=LOAD_DICOM, help = "Load existing .dcm files in queue")
    parser.add_argument("--no-query", action = "store_true", default=NO_QUERY, help = "Do not query the DICOM server automatically")
    parser.add_argument("--enable-ntfy", action = "store_true", default=ENABLE_NTFY, help = "Enable ntfy.sh notifications")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name to use for analysis")
    parser.add_argument("--retrieval-method", type=str, choices=['C-MOVE', 'C-GET'], default=RETRIEVAL_METHOD, help="DICOM retrieval method")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set logging level")
    parser.add_argument("--translate-existing", action = "store_true", help = "Translate existing radiologist reports without English translations")
    parser.add_argument("--pidfile", metavar="PATH", help="Write the process PID to this file on startup and remove it on clean shutdown; enables the xrayvision control script to find and signal this process")
    args = parser.parse_args()
    KEEP_DICOM = args.keep_dicom
    LOAD_DICOM = args.load_dicom
    NO_QUERY = args.no_query
    ENABLE_NTFY = args.enable_ntfy
    MODEL_NAME = args.model
    RETRIEVAL_METHOD = args.retrieval_method
    TRANSLATE_EXISTING = args.translate_existing
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.pidfile:
        with open(args.pidfile, "w") as f:
            f.write(str(os.getpid()))

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("XRayVision stopped by user. Shutting down.")
    finally:
        if args.pidfile:
            try:
                os.remove(args.pidfile)
            except FileNotFoundError:
                pass
        logging.shutdown()
