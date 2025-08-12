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

import asyncio
import os
import base64
import aiohttp
import cv2
import numpy as np
import math
import json
import re
import sqlite3
import logging
from aiohttp import web
from pydicom import dcmread
from pydicom.dataset import Dataset
from pynetdicom import AE, evt, StoragePresentationContexts, QueryRetrievePresentationContexts
from pynetdicom.sop_class import ComputedRadiographyImageStorage, DigitalXRayImageStorageForPresentation, PatientRootQueryRetrieveInformationModelFind, PatientRootQueryRetrieveInformationModelMove
from datetime import datetime, timedelta

# Logger config
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s | %(levelname)8s | %(message)s',
    handlers = [
        logging.FileHandler("xrayvision.log"),
        logging.StreamHandler()
    ]
)
# Filter out noisy module logs
logging.getLogger('aiohttp').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('pynetdicom').setLevel(logging.WARNING)  # DICOM network operations
logging.getLogger('pydicom').setLevel(logging.WARNING)     # DICOM file operations

# Configuration
OPENAI_URL_PRIMARY = os.getenv("OPENAI_URL_PRIMARY", "http://192.168.3.239:8080/v1/chat/completions")
OPENAI_URL_SECONDARY = os.getenv("OPENAI_URL_SECONDARY", "http://127.0.0.1:8080/v1/chat/completions")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'sk-your-api-key')
XRAYVISION_USER = os.getenv('XRAYVISION_USER', 'admin')
XRAYVISION_PASS = os.getenv('XRAYVISION_PASS', 'admin')
NTFY_URL = os.getenv('NTFY_URL', 'https://ntfy.sh/xrayvision-alerts')
DASHBOARD_PORT = 8000
AE_TITLE = 'XRAYVISION'
AE_PORT  = 4010
REMOTE_AE_TITLE = '3DNETCLOUD'
REMOTE_AE_IP = '192.168.3.50'
REMOTE_AE_PORT = 104
IMAGES_DIR = 'images'
DB_FILE = os.path.join(IMAGES_DIR, "xrayvision.db")

SYS_PROMPT = """You are a smart radiologist working in ER. 
You only output mandatory JSON to a RESTful API, in the following format: {"short": "yes or no", "report": "REPORT"} where "yes or no" is the short answer, only "yes" and "no" being allowed, and "REPORT" is the full description of the findings, like a radiologist would write.
It is important to identify all lesions in the xray and respond with 'yes' if there is anything pathological and 'no' if there is nothing to report.
If in doubt, do not assume, stick to the facts.
Look again at the xray if you think there is something ambiguous.
The output format is JSON, keys and values require double-quotes, the keys are "short", "report", value types are escaped string, int, truth value.
No explanation or other text is allowed."""
USR_PROMPT = "{} in this {} xray of a {}? Are there any other lesions?"
REV_PROMPT = """There is something inaccurate in your report.
Analyse the xray again and look for any other possible lesions.
Do not apologize or explain yourself.
No explanation or other text is allowed. Only JSON is allowed as an answer.
Update the JSON report according to the template."""
REGIONS = ["chest", "abdominal", "nasal bones", "maxilar and frontal sinus", "clavicle"]

# Images directory
os.makedirs(IMAGES_DIR, exist_ok = True)

# Global variables
main_loop = None
websocket_clients = set()
queue_event = asyncio.Event()
next_query = None

active_openai_url = None
health_status = {
    OPENAI_URL_PRIMARY: False,
    OPENAI_URL_SECONDARY: False
}

# Global paramters
PAGE_SIZE = 10
KEEP_DICOM = False
LOAD_DICOM = False
NO_QUERY = False
ENABLE_NTFY = False

# Dashboard state
dashboard = {
    'queue_size': 0,
    'processing_file': None,
    'success_count': 0,
    'failure_count': 0
}
# OpenAI timings
timings = {'prompt': 0, 'predicted': 0, 'total': 0, 'average': 0}

# Database operations
def init_database():
    """ Initialize the database """
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS exams (
                uid TEXT PRIMARY KEY,
                name TEXT,
                id TEXT,
                age INTEGER,
                sex TEXT CHECK(sex IN ('M', 'F', 'O')),
                created TIMESTAMP,
                protocol TEXT,
                region TEXT,
                reported TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                report TEXT,
                positive INTEGER DEFAULT 0 CHECK(positive IN (0, 1)),
                valid INTEGER DEFAULT 1 CHECK(valid IN (0, 1)),
                reviewed INTEGER DEFAULT 0 CHECK(reviewed IN (0, 1)),
                status TEXT DEFAULT 'none'
            )
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_cleanup 
            ON exams(status, created)
        ''')
        logging.info("Initialized SQLite database.")

def db_get_exams(limit = PAGE_SIZE, offset = 0, **filters):
    """ Load the exams from the database, with filters and pagination """
    conditions = []
    params = []
    where = ""
    # Update the conditions
    if 'reviewed' in filters:
        conditions.append(f"reviewed = {filters['reviewed']}")
    if 'positive' in filters:
        conditions.append(f"positive = {filters['positive']}")
    if 'valid' in filters:
        conditions.append(f"valid = {filters['valid']}")
    if 'region' in filters:
        conditions.append(f"LOWER(region) LIKE '%{filters['region'].lower()}%'")
    if 'status' in filters:
        conditions.append(f"LOWER(status) = '{filters['status'].lower()}'")
    else:
        conditions.append("status = 'done'")
    if 'search' in filters:
        #conditions.append(f"(LOWER(uid) LIKE '%{filters['search']}%' OR LOWER(name) LIKE '%{filters['search']}%' OR LOWER(report) LIKE '%{filters['search']}%')")
        conditions.append(f"LOWER(name) LIKE '%{filters['search']}%'")
    if conditions:
        where = "WHERE " + " AND ".join(conditions)
    # Apply the limits (pagination)
    query = f"SELECT * FROM exams {where} ORDER BY created DESC LIMIT {limit} OFFSET {offset}"
    # Get the exams
    exams = []
    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.execute(query, params)
        for row in rows:
            dt = datetime.strptime(row[5], "%Y-%m-%d %H:%M:%S")
            exams.append({
                'uid': row[0],
                'patient': {
                    'name': row[1],
                    'id': row[2], 
                    'age': row[3],
                    'sex': row[4],
                },
                'exam': {
                    'created': row[5],
                    'date': dt.strftime('%Y%m%d'),
                    'time': dt.strftime('%H%M%S'),
                    'protocol': row[6],
                    'region': row[7],
                },
                'report': {
                    'text': row[9],
                    'short': row[10] and 'yes' or 'no',
                    'datetime': row[8],
                    'positive': bool(row[10]),
                    'valid': bool(row[11]),
                    'reviewed': bool(row[12]),
                },
                'status': row[13],
            })
        # Get the total for pagination
        count_query = 'SELECT COUNT(*) FROM exams'
        if conditions:
            count_query += ' WHERE ' + " AND ".join(conditions)
        total = conn.execute(count_query).fetchone()[0]
    return exams, total

def db_add_exam(info, report = None, positive = None):
    """ Add one row to the database """
    # Check if we have a new report or just enqueue an exam
    if report:
        # We have a new report
        poz = positive
        valid = True
        reviewed = False
        status = 'done'
        # Check if we have previous report
        if 'report' in info:
            # There is an previous also, check validity and new positivity
            if not info['report']['valid'] and info['report']['positive'] != positive:
                # It was invalid and now positivity flipped, mark it as reviewed
                reviewed = True
                logging.info(f"Exam {info['uid']} marked as reviewed and valid after being reanalyzed.")

    else:
        # Null report, just enqueue a new exam
        poz = False
        valid = True
        reviewed = False
        status = 'queued'
    # Timestamp
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Insert into database
    with sqlite3.connect(DB_FILE) as conn:
        values = (
            info['uid'],
            info["patient"]["name"],
            info["patient"]["id"],
            info["patient"]["age"],
            info["patient"]["sex"],
            info["exam"]['created'],
            info["exam"]["protocol"],
            info['exam']['region'],
            now,
            report,
            poz,
            valid,
            reviewed,
            status
        )
        conn.execute('''
            INSERT OR REPLACE INTO exams
                (uid, name, id, age, sex, created, protocol, region, reported, report, positive, valid, reviewed, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', values)

def db_check_already_processed(uid):
    """ Check if the file has already been processed, in queue or processing """
    with sqlite3.connect(DB_FILE) as conn:
        result = conn.execute(
            "SELECT status FROM exams WHERE uid = ? AND status IN ('done', 'queued', 'processing')", (uid,)
        ).fetchone()
        return result is not None

async def db_get_stats():
    """ Get statistics from database """
    stats = {
        "total": 0,
        "reviewed": 0,
        "positive": 0,
        "invalid": 0,
        "region": {},
        "trends": {},
        "monthly_trends": {},
        "avg_processing_time": 0,
        "throughput": 0
    }
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        # Get all statistics in a single query
        cursor.execute("""
            SELECT 
                COUNT(*) AS total,
                SUM(CASE WHEN reviewed = 1 THEN 1 ELSE 0 END) AS reviewed,
                SUM(CASE WHEN positive = 1 THEN 1 ELSE 0 END) AS positive,
                SUM(CASE WHEN valid = 0 THEN 1 ELSE 0 END) AS invalid
            FROM exams
            WHERE status LIKE 'done'
        """)
        row = cursor.fetchone()
        stats["total"] = row[0]
        stats["reviewed"] = row[1] or 0
        stats["positive"] = row[2] or 0
        stats["invalid"] = row[3] or 0

        # Get processing time statistics
        cursor.execute("""
            SELECT 
                AVG(CAST(strftime('%s', reported) - strftime('%s', created) AS REAL)) AS avg_processing_time,
                COUNT(*) * 1.0 / (SUM(CAST(strftime('%s', reported) - strftime('%s', created) AS REAL)) + 1) AS throughput
            FROM exams
            WHERE status LIKE 'done' AND reported IS NOT NULL AND created IS NOT NULL
        """)
        timing_row = cursor.fetchone()
        if timing_row and timing_row[0] is not None:
            stats["avg_processing_time"] = round(timing_row[0], 2)
            stats["throughput"] = round(timing_row[1] * 3600, 2)  # exams per hour

        # Totals per anatomic part
        cursor.execute("""
            SELECT region,
                    COUNT(*) AS total,
                    SUM(reviewed = 1) AS reviewed,
                    SUM(positive = 1) AS positive,
                    SUM(valid = 0) AS invalid,
                    SUM(reviewed = 1 AND positive = 1 AND valid = 1) AS tpos,
                    SUM(reviewed = 1 AND positive = 0 AND valid = 1) AS tneg,
                    SUM(reviewed = 1 AND positive = 1 AND valid = 0) AS fpos,
                    SUM(reviewed = 1 AND positive = 0 AND valid = 0) AS fneg
            FROM exams
            WHERE status LIKE 'done'
            GROUP BY region
        """)
        for row in cursor.fetchall():
            region = row[0] or 'unknown'
            stats["region"][region] = {
                "total": row[1],
                "reviewed": row[2],
                "positive": row[3],
                "invalid": row[4],
                "tpos": row[5],
                "tneg": row[6],
                "fpos": row[7],
                "fneg": row[8],
                "ppv": '-',
                "pnv": '-',
                "snsi": '-',
                "spci": '-',
            }
            if (row[5] + row[7]) != 0:
                stats["region"][region]["ppv"] = int(100.0 * row[5] / (row[5] + row[7]))
            if (row[6] + row[8])  != 0:
                stats["region"][region]["pnv"] = int(100.0 * row[6] / (row[6] + row[8]))
            if (row[5] + row[8]) != 0:
                stats["region"][region]["snsi"] = int(100.0 * row[5] / (row[5] + row[8]))
            if (row[6] + row[7]) != 0:
                stats["region"][region]["spci"] = int(100.0 * row[6] / (row[6] + row[7]))
        
        # Get temporal trends (last 30 days)
        cursor.execute("""
            SELECT DATE(created) as date,
                   region,
                   COUNT(*) as total,
                   SUM(positive = 1) as positive
            FROM exams
            WHERE status LIKE 'done'
              AND created >= date('now', '-30 days')
            GROUP BY DATE(created), region
            ORDER BY date
        """)
        trends_data = cursor.fetchall()
        
        # Process trends data into a structured format
        for row in trends_data:
            date, region, total, positive = row
            if region not in stats["trends"]:
                stats["trends"][region] = []
            stats["trends"][region].append({
                "date": date,
                "total": total,
                "positive": positive
            })
        
        # Get monthly trends (last 12 months)
        cursor.execute("""
            SELECT strftime('%Y-%m', created) as month,
                   region,
                   COUNT(*) as total,
                   SUM(positive = 1) as positive
            FROM exams
            WHERE status LIKE 'done'
              AND created >= date('now', '-12 months')
            GROUP BY strftime('%Y-%m', created), region
            ORDER BY month
        """)
        monthly_trends_data = cursor.fetchall()
        
        # Process monthly trends data into a structured format
        for row in monthly_trends_data:
            month, region, total, positive = row
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
    """ Get the queue size """
    with sqlite3.connect(DB_FILE) as conn:
        result = conn.execute("SELECT COUNT(*) FROM exams WHERE status = 'queued'").fetchone()
        return result[0]
    return 0

def db_purge_ignored_errors():
    """ Delete ignored and erroneous records older than 1 week and their associated files """
    deleted_uids = []
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute('''
            DELETE FROM exams 
            WHERE status IN ('ignore', 'error')
            AND created < datetime('now', '-1 week')
            RETURNING uid
        ''')
        deleted_uids = [row[0] for row in cursor.fetchall()]
        deleted_count = cursor.rowcount
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

def db_validate(uid, normal = True, valid = None, enqueue = False):
    """ Mark the entry as valid or invalid """
    with sqlite3.connect(DB_FILE) as conn:
        if valid is None:
            # Check if the report is positive
            result = conn.execute("SELECT positive FROM exams WHERE uid = ?", (uid,)).fetchone()
            # Valid when review matches prediction
            valid = bool(normal) != bool(result[0])
        # Update the entry
        columns = []
        params = []
        columns.append("reviewed = 1")
        columns.append("valid = ?")
        params.append(bool(valid))
        if enqueue:
            columns.append("status = 'queued'")
        params.append(uid)
        set_clause = 'SET ' + ','.join(columns)
        conn.execute(f"UPDATE exams {set_clause} WHERE uid = ?", params)
    return valid

def db_set_status(uid, status):
    """ Set the specified satatus for uid """
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(f'UPDATE exams SET status = "{status}" WHERE uid = ?', (uid,))
    # Return the status
    return status


# DICOM network operations
async def query_and_retrieve(minutes = 15):
    """ Query and Retrieve new studies """
    ae = AE(ae_title = AE_TITLE)
    ae.requested_contexts = QueryRetrievePresentationContexts
    ae.connection_timeout = 30
    # Create the association
    assoc = ae.associate(REMOTE_AE_IP, REMOTE_AE_PORT, ae_title = REMOTE_AE_TITLE)
    if assoc.is_established:
        logging.info(f"QueryRetrieve association established. Asking for studies in the last {minutes} minutes.")
        # Prepare the timespan
        current_time = datetime.now()
        past_time = current_time - timedelta(minutes = minutes)
        # Check if the time span crosses midnight, split into two queries
        if past_time.date() < current_time.date():
            date_yesterday = past_time.strftime('%Y%m%d')
            time_yesterday = f"{past_time.strftime('%H%M%S')}-235959"
            date_today = current_time.strftime('%Y%m%d')
            time_today = f"000000-{current_time.strftime('%H%M%S')}"
            queries = [(date_yesterday, time_yesterday), (date_today, time_today)]
        else:
            time_range = f"{past_time.strftime('%H%M%S')}-{current_time.strftime('%H%M%S')}"
            date_today = current_time.strftime('%Y%m%d')
            queries = [(date_today, time_range),]
        # Perform one or two queries, as needed
        for study_date, time_range in queries:
            # The query dataset
            ds = Dataset()
            ds.QueryRetrieveLevel = "STUDY"
            ds.StudyDate = study_date
            ds.StudyTime = time_range
            ds.Modality = "CR"
            # Get the responses list
            responses = assoc.send_c_find(ds, PatientRootQueryRetrieveInformationModelFind)
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
    """ Ask for a study to be sent """
    # Create the association
    assoc = ae.associate(REMOTE_AE_IP, REMOTE_AE_PORT, ae_title = REMOTE_AE_TITLE)
    if assoc.is_established:
        # The retrieval dataset
        ds = Dataset()
        ds.QueryRetrieveLevel = "STUDY"
        ds.StudyInstanceUID = study_instance_uid
        # Get the response
        responses = assoc.send_c_move(ds, AE_TITLE, PatientRootQueryRetrieveInformationModelMove)
        # Release the association
        assoc.release()
    else:
        logging.error("Could not establish C-MOVE association.")

def dicom_store(event):
    """ Callback for receiving a DICOM file """
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
        # Get some info for queueing
        try:
            info = get_dicom_info(ds)
        except Exception as e:
            logging.error(f"Error getting info {dicom_file}: {e}")
            return 0x0000
        # Try to convert to PNG
        png_file = None
        try:
            png_file = dicom_to_png(dicom_file)
        except Exception as e:
            logging.error(f"Error converting DICOM file {dicom_file}: {e}")
        # Check the result
        if png_file:
            # Add to processing queue
            db_add_exam(info)
            # Notify the queue
            queue_event.set()
            asyncio.run_coroutine_threadsafe(broadcast_dashboard_update(), main_loop)
    # Return success
    return 0x0000


# DICOM files operations
async def load_existing_dicom_files():
    """ Load existing .dcm files into queue """
    for dicom_file in os.listdir(IMAGES_DIR):
        uid, ext = os.path.splitext(os.path.basename(dicom_file.lower()))
        if ext == '.dcm':
            if db_check_already_processed(uid):
                logging.info(f"Skipping already processed image {uid}")
            else:
                logging.info(f"Adding {uid} into processing queue...")
                # Get the dataset
                try:
                    ds = dcmread(os.path.join(IMAGES_DIR, dicom_file))
                except Exception as e:
                    logging.error(f"Error reading the dataset from DICOM file {dicom_file}: {e}")
                    continue
                # Get some info for queueing
                try:
                    info = get_dicom_info(ds)
                except Exception as e:
                    logging.error(f"Error getting info {dicom_file}: {e}")
                    continue
                # Try to convert to PNG
                png_file = None
                try:
                    png_file = dicom_to_png(os.path.join(IMAGES_DIR, dicom_file))
                except Exception as e:
                    logging.error(f"Error converting DICOM file {dicom_file}: {e}")
                # Check the result
                if png_file:
                    # Add to processing queue
                    db_add_exam(info)
                    # Notify the queue
                    queue_event.set()
    # At the end, update the dashboard
    await broadcast_dashboard_update()

def get_dicom_info(ds):
    """ Read and return the info from a DICOM file """
    age = -1
    if 'PatientAge' in ds:
        age = str(ds.PatientAge).lower().replace("y", "").strip()
        try:
            age = int(age)
        except Exception as e:
            logging.error(f"Cannot convert age to number: {e}")
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
            'id':    str(ds.PatientID),
            'age':   age,
            'sex':   str(ds.PatientSex),
            'bdate': str(ds.PatientBirthDate),
        },
        'exam': {
            'protocol': str(ds.ProtocolName),
            'created': created,
            'region': str(ds.ProtocolName),
        }
    }
    # Check gender
    if not info['patient']['sex'] in ['M', 'F', 'O']:
        # Try to determine from ID
        try:
            info['patient']['sex'] = int(info['patient']['id'][0]) % 2 == 0 and 'F' or 'M'
        except:
            info['patient']['sex'] = 'O'
    # Return the dicom info
    return info

# Image processing operations
def adjust_gamma(image, gamma = 1.2):
    """ Auto-adjust gamma """
    # If gamma is None, compute it
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

def dicom_to_png(dicom_file, max_size = 800):
    """ Convert DICOM to PNG and return the PNG filename """
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
    # Clip to 1..99 percentiles
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
    image = adjust_gamma(image, None)
    # Save the PNG file
    base_name = os.path.splitext(os.path.basename(dicom_file))[0]
    png_file = os.path.join(IMAGES_DIR, f"{base_name}.png")
    cv2.imwrite(png_file, image)
    logging.info(f"Converted PNG saved to {png_file}")
    # Return the PNG file name
    return png_file


# WebSocket and WebServer operations
async def serve_dashboard_page(request):
    return web.FileResponse(path = "dashboard.html")

async def serve_stats_page(request):
    return web.FileResponse(path = "stats.html")

async def serve_about_page(request):
    return web.FileResponse(path = "about.html")

async def serve_favicon(request):
    return web.FileResponse(path = "favicon.ico")

async def websocket_handler(request):
    """ Handle each WebSocket client """
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
    """ Provide a page of exams """
    try:
        page = int(request.query.get("page", "1"))
        filters = {}
        for filter in ['reviewed', 'positive', 'valid']:
            value = request.query.get(filter, 'any')
            if value != 'any':
                filters[filter] = value[0].lower() == 'y' and 1 or 0
        for filter in ['region', 'status', 'search']:
            value = request.query.get(filter, 'any')
            if value != 'any':
                filters[filter] = value
        offset = (page - 1) * PAGE_SIZE
        data, total = db_get_exams(limit = PAGE_SIZE, offset = offset, **filters)
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
    """ Provide a page of statistics """
    try:
        return web.json_response(await db_get_stats())
    except Exception as e:
        logging.error(f"Exams page error: {e}")
        return web.json_response([], status = 500)

async def manual_query(request):
    """ Trigger a manual query/retrieve operation """
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

async def validate(request):
    """ Mark a study valid or invalid """
    data = await request.json()
    # Get 'uid' and 'normal' from request
    uid = data.get('uid')
    normal = data.get('normal', None)
    # Validate/Invalidate a study, send only the 'normal' attribute
    valid = db_validate(uid, normal)
    logging.info(f"Exam {uid} marked as {normal and 'normal' or 'abnormal'} which {valid and 'validates' or 'invalidates'} the report.")
    payload = {'uid': uid, 'valid': valid}
    await broadcast_dashboard_update(event = "validate", payload = payload)
    return web.json_response({'status': 'success'}.update(payload))

async def lookagain(request):
    """ Send an exam back to the queue """
    data = await request.json()
    # Get 'uid' and custom 'prompt' from request
    uid = data.get('uid')
    prompt = data.get('prompt', None)
    # Mark reviewed, invalid and re-enqueue
    valid = db_validate(uid, valid = False, enqueue = True)
    logging.info(f"Exam {uid} sent to the processing queue (look again).")
    # Notify the queue
    queue_event.set()
    payload = {'uid': uid, 'valid': valid}
    await broadcast_dashboard_update(event = "lookagain", payload = payload)
    return web.json_response({'status': 'success'}.update(payload))

@web.middleware
async def auth_middleware(request, handler):
    """ Basic authentication middleware """
    # Skip auth for static files and OPTIONS requests
    if request.path.startswith('/static/') or request.method == 'OPTIONS':
        return await handler(request)
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Basic '):
        raise web.HTTPUnauthorized(
            text = "401: Authentication required",
            headers = {'WWW-Authenticate': 'Basic realm="XRayVision"'})
    try:
        credentials = base64.b64decode(auth_header[6:]).decode('utf-8')
        username, password = credentials.split(':', 1)
        if username != XRAYVISION_USER or password != XRAYVISION_PASS:
            logging.warning("Invalid authentication")
            raise ValueError("Invalid authentication")
    except (ValueError, UnicodeDecodeError) as e:
        raise web.HTTPUnauthorized(
            text = "401: Invalid authentication",
            headers = {'WWW-Authenticate': 'Basic realm="XRayVision"'})
    return await handler(request)

async def broadcast_dashboard_update(event = None, payload = None, client = None):
    """ Update the dashboard for all clients """
    # Check if there are any clients
    if not (websocket_clients or client):
        return
    # Update the queue size
    dashboard['queue_size'] = db_get_queue_size()
    # Create a list of clients
    if client:
        clients = [client,]
    else:
        clients = websocket_clients.copy()
    # Create the json object
    data = {'dashboard': dashboard,
            'openai': {
                "url": active_openai_url,
                "health": {
                    'pri': health_status.get(OPENAI_URL_PRIMARY,  False),
                    'sec': health_status.get(OPENAI_URL_SECONDARY, False)
                }
            },
            'timings': timings,
    }
    if next_query:
        data['next_query'] = next_query.strftime('%Y-%m-%d %H:%M:%S')
    if event:
        data['event'] = {'name': event, 'payload': payload}
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
        async with session.post(NTFY_URL,
            data = message,
            headers = headers
        ) as resp:
            if resp.status == 200:
                logging.info("Successfully sent ntfy notification")
            else:
                logging.error(f"Notification failed with status {resp.status}: {await resp.text()}")

# AI API operations
def check_any(string, *words):
    """ Check if any of the words are present in the string """
    return any(i in string for i in words)

def get_region(info):
    """ Try to identify the region. Return the region and the question. """
    desc = info["exam"]["protocol"].lower()
    if check_any(desc, 'torace', 'pulmon',
                 'thorax'):
        region = 'chest'
        question = "Are there any lung consolidations, infitrates, opacities, pleural effusion, pneumothorax or pneumoperitoneum"
    elif check_any(desc, 'grilaj', 'coaste'):
        region = 'ribs'
        question = "Are there any ribs or clavicles fractures"
    elif check_any(desc, 'stern'):
        region = 'sternum'
        question = "Are there any fractures"
    elif check_any(desc, 'abdomen', 'abdominal'):
        region = 'abdominal'
        question = "Are there any fluid levels, free gas or metallic foreign bodies"
    elif check_any(desc, 'cap', 'craniu', 'occiput',
                   'skull'):
        region = 'skull'
        question = "Are there any fractures"
    elif check_any(desc, 'mandibula'):
        region = 'mandible'
        question = "Are there any fractures"
    elif check_any(desc, 'nazal', 'piramida'):
        region = 'nasal bones'
        question = "Are there any fractures"
    elif check_any(desc, 'sinus'):
        region = 'maxilar and frontal sinus'
        question = "Are the sinuses normally aerated or are they opaque or are there fluid levels"
    elif check_any(desc, 'col.',
                   'spine', 'dens', 'sacrat'):
        region = 'spine'
        question = "Are there any fractures or dislocations"
    elif check_any(desc, 'bazin', 'pelvis'):
        region = 'pelvis'
        question = "Are there any fractures"
    elif check_any(desc, 'clavicula',
                   'clavicle'):
        region = 'clavicle'
        question = "Are there any fractures"
    elif check_any(desc, 'humerus', 'antebrat',
                   'forearm'):
        region = 'upper limb'
        question = "Are there any fractures, dislocations or bone tumors"
    elif check_any(desc, 'pumn', 'mana', 'deget',
                   'hand', 'finger'):
        region = 'hand'
        question = "Are there any fractures, dislocations or bone tumors"
    elif check_any(desc, 'umar',
                   'shoulder'):
        region = 'shoulder'
        question = "Are there any fractures or dislocations"
    elif check_any(desc, 'cot',
                   'elbow'):
        region = 'elbow'
        question = "Are there any fractures or dislocations"
    elif check_any(desc, 'sold',
                   'hip'):
        region = 'hip'
        question = "Are there any fractures or dislocations"
    elif check_any(desc, 'femur', 'tibie', 'picior', 'gamba', 'calcai',
                   'leg', 'foot'):
        region = 'lower limb'
        question = "Are there any fractures, dislocations or bone tumors"
    elif check_any(desc, 'genunchi', 'patella',
                   'knee'):
        region = 'knee'
        question = "Are there any fractures or dislocations"
    elif check_any(desc, 'glezna', 'calcaneu',
                   'ankle'):
        region = 'ankle'
        question = "Are there any fractures or dislocations"
    else:
        # Fallback
        region = desc
        question = "Is there anything abnormal"
    # Return the region and the question
    return region, question

def get_projection(info):
    """ Try to identify the projection """
    desc = info["exam"]["protocol"].lower()
    if check_any(desc, "a.p.", "p.a.", "d.v.", "v.d.", "d.p"):
        projection = "frontal"
    elif check_any(desc, "lat.", "pr."):
        projection = "lateral"
    elif check_any(desc, "oblic"):
        projection = "oblique"
    else:
        # Fallback
        projection = ""
    # Return the projection
    return projection

def get_gender(info):
    """ Try to identify the gender """
    patient_sex = info["patient"]["sex"].lower()
    if "m" in patient_sex:
        gender = "boy"
    elif "f" in patient_sex:
        gender = "girl"
    else:
        # Fallback
        gender = "child"
    # Return the gender
    return gender

async def send_to_openai(session, headers, payload):
    """ Try the healty OpenAI API endpoint """
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
    """Send PNG to OpenAI API with retries and save response to text file."""
    # Read the PNG file
    with open(os.path.join(IMAGES_DIR, f"{exam['uid']}.png"), 'rb') as f:
        image_bytes = f.read()
    # Identify the region
    region, question = get_region(exam)
    # Filter on specific region
    if not region in REGIONS:
        logging.info(f"Ignoring {exam['uid']} with {region} x-ray.")
        db_set_status(exam['uid'], 'ignore')
        return False
    # Identify the prjection, gender and age
    projection = get_projection(exam)
    gender = get_gender(exam)
    age = exam["patient"]["age"]
    if age > 0:
        txtAge = f"{age} years old"
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
    # Create the prompt
    prompt = USR_PROMPT.format(question, anatomy, subject)
    logging.debug(f"Prompt: {prompt}")
    logging.info(f"Processing {exam['uid']} with {region} x-ray.")
    if exam['report']['text']:
        json_report = {'short': exam['report']['short'],
                       'report': exam['report']['text']}
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
        "model": "medgemma-4b-it",
        "timings_per_token": True,
        "min_p": 0.05,
        "top_k": 40,
        "top_p": 0.95,
        "temperature": 0.6,
        "cache_prompt": True,
        "stream": False,
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
                    if short not in ("yes", "no") or not report:
                        raise ValueError("Invalid json format in OpenAI response")
                except Exception as e:
                    logging.error(f"Rejected malformed OpenAI response: {e}")
                    logging.error(response)
                    break
                logging.info(f"OpenAI API response for {exam['uid']}: [{short.upper()}] {report}")
                # Update the dashboard
                dashboard['success_count'] += 1
                # Save to exams database
                is_positive = short == "yes"
                db_add_exam(exam, report = report, positive = is_positive)
                # Send notification for positive cases
                if is_positive:
                    try:
                        await send_ntfy_notification(exam['uid'], report, exam)
                    except Exception as e:
                        logging.error(f"Failed to send ntfy notification: {e}")
                # Get some timing statistics
                global timings
                timings['prompt'] = int(result['timings']['prompt_ms'])
                timings['predicted'] = int(result['timings']['predicted_ms'])
                timings['total'] = timings['prompt'] + timings['predicted']
                if timings['average'] > 0:
                    timings['average'] = int((3 * timings['average'] + timings['total']) / 4)
                else:
                    timings['average'] = timings['total']
                logging.info(f"OpenAI API response timings: last {timings['total']} ms, average {timings['average']} ms")
                # Notify the dashboard frontend to reload first page
                await broadcast_dashboard_update(event = "new_item", payload = {'uid': exam['uid'], 'positive': is_positive})
                # Success
                return True
        except Exception as e:
            logging.warning(f"Error uploading {exam['uid']} (attempt {attempt}): {e}")
            # Exponential backoff
            await asyncio.sleep(2 ** attempt)
            attempt += 1
    # Failure after max_retries
    db_set_status(exam['uid'], 'error')
    queue_event.clear()
    logging.error(f"Failed to process {exam['uid']} after {attempt} attempts.")
    dashboard['failure_count'] += 1
    await broadcast_dashboard_update()
    return False


# Threads
async def start_dashboard():
    """ Start the dashboard web server """
    app = web.Application(middlewares = [auth_middleware])
    app.router.add_get('/', serve_dashboard_page)
    app.router.add_get('/stats', serve_stats_page)
    app.router.add_get('/about', serve_about_page)
    app.router.add_get('/favicon.ico', serve_favicon)
    app.router.add_get('/ws', websocket_handler)
    app.router.add_get('/api/exams', exams_handler)
    app.router.add_get('/api/stats', stats_handler)
    app.router.add_post('/api/validate', validate)
    app.router.add_post('/api/lookagain', lookagain)
    app.router.add_post('/api/trigger_query', manual_query)
    app.router.add_static('/static/', path = IMAGES_DIR, name = 'static')
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', DASHBOARD_PORT)
    await site.start()
    logging.info(f"Dashboard available at http://localhost:{DASHBOARD_PORT}")

async def relay_to_openai_loop():
    """ Relay PNG files to OpenAI API with retries and dashboard update """
    while True:
        # Get one file from queue
        exams, total = db_get_exams(limit = 1, status = 'queued')
        # Wait here if there are no items in queue or there is no OpenAI server
        if not exams or active_openai_url is None:
            queue_event.clear()
            await queue_event.wait()
            continue
        # Get only one exam, if any
        exam = exams[0]
        # Set the status
        db_set_status(exam['uid'], "processing")
        # Update the dashboard
        dashboard['queue_size'] = total
        dashboard['processing_file'] = exam['patient']['name']
        await broadcast_dashboard_update()
        # The DICOM file name
        dicom_file = os.path.join(IMAGES_DIR, f"{exam['uid']}.dcm")
        # Try to send to AI
        result = False
        try:
            result = await send_exam_to_openai(exam)
        except Exception as e:
            logging.error(f"OpenAI error processing {exam['uid']}: {e}")
            db_set_status(exam['uid'], "error")
        finally:
            dashboard['processing_file'] = None
            await broadcast_dashboard_update()
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

async def openai_health_check():
    """ Health check coroutine """
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
            queue_event.set()
        # WebSocket broadcast
        await broadcast_dashboard_update()
        # Sleep for 5 minutes
        await asyncio.sleep(300)

async def query_retrieve_loop():
    """ Async thread to periodically poll the server for new studies """
    if NO_QUERY:
        logging.warning(f"Automatic Query/Retrieve disabled.")
    while not NO_QUERY:
        await query_and_retrieve()
        current_time = datetime.now()
        global next_query
        next_query = current_time + timedelta(seconds = 900)
        logging.info(f"Next Query/Retrieve at {next_query.strftime('%Y-%m-%d %H:%M:%S')}")
        await asyncio.sleep(900)

async def purge_ignored_errors_loop():
    """ Daily cleanup of old ignored records """
    while True:
        db_purge_ignored_errors()
        # Wait for 24 hours
        await asyncio.sleep(86400)

def start_dicom_server():
    """ Start the DICOM Storage SCP """
    ae = AE(ae_title = AE_TITLE)
    # Accept everything
    #ae.supported_contexts = StoragePresentationContexts
    # Accept only XRays
    ae.add_supported_context(ComputedRadiographyImageStorage)
    ae.add_supported_context(DigitalXRayImageStorageForPresentation)
    # C-Store handler
    handlers = [(evt.EVT_C_STORE, dicom_store)]
    logging.info(f"Starting DICOM server on port {AE_PORT} with AE Title '{AE_TITLE}'...")
    ae.start_server(("0.0.0.0", AE_PORT), evt_handlers = handlers, block = True)

async def main():
    """ The main thread """
    # Main event loop
    global main_loop
    main_loop = asyncio.get_running_loop()
    # Init the database if not found
    if not os.path.exists(DB_FILE):
        logging.info("SQLite database not found. Creating a new one...")
        init_database()
    else:
        logging.info("SQLite database found.")
    # Print some data
    logging.info(f"Python SQLite version: {sqlite3.version}")
    logging.info(f"SQLite library version: {sqlite3.sqlite_version}")
     
    # Load exams
    exams, total = db_get_exams(status = 'done')
    logging.info(f"Loaded {len(exams)} exams from a total of {total}.")

    # Start the asynchronous tasks
    asyncio.create_task(start_dashboard())
    asyncio.create_task(openai_health_check())
    asyncio.create_task(relay_to_openai_loop())
    asyncio.create_task(query_retrieve_loop())
    asyncio.create_task(purge_ignored_errors_loop())
    # Preload the existing dicom files
    if LOAD_DICOM:
        await load_existing_dicom_files()
    # Start the DICOM server
    await asyncio.get_running_loop().run_in_executor(None, start_dicom_server)


# Command run
if __name__ == '__main__':
    # Need to process the arguments
    import argparse
    parser = argparse.ArgumentParser(description = "XRayVision - Async DICOM processor with OpenAI and WebSocket dashboard")
    parser.add_argument("--keep-dicom", action = "store_true", help = "Do not delete .dcm files after conversion")
    parser.add_argument("--load-dicom", action = "store_true", help = "Load existing .dcm files in queue")
    parser.add_argument("--no-query", action = "store_true", help = "Do not query the DICOM server automatically")
    parser.add_argument("--enable-ntfy", action = "store_true", help = "Enable ntfy.sh notifications")
    args = parser.parse_args()
    # Store in globals
    KEEP_DICOM = args.keep_dicom
    LOAD_DICOM = args.load_dicom
    NO_QUERY = args.no_query
    ENABLE_NTFY = args.enable_ntfy

    # Run
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("XRayVision stopped by user. Shutting down.")

    logging.shutdown()
