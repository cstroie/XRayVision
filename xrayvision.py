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
import uuid
import base64
import aiohttp
import cv2
import numpy as np
import math
import json
import sqlite3
import logging
from aiohttp import web
from pydicom import dcmread
from pydicom.dataset import Dataset
from pynetdicom import AE, evt, StoragePresentationContexts, QueryRetrievePresentationContexts
from pynetdicom.sop_class import ComputedRadiographyImageStorage, DigitalXRayImageStorageForPresentation, PatientRootQueryRetrieveInformationModelFind, PatientRootQueryRetrieveInformationModelMove
from datetime import datetime, timedelta


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(message)s',
    handlers=[
        #logging.FileHandler("xrayvision.log"),
        logging.StreamHandler()
    ]
)

# Configuration
OPENAI_API_URL = 'http://127.0.0.1:8080/v1/chat/completions'
#OPENAI_API_URL = 'http://192.168.3.239:8080/v1/chat/completions'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'sk-your-api-key')
DASHBOARD_PORT = 8000
AE_TITLE = 'XRAYVISION'
AE_PORT  = 4010
REMOTE_AE_TITLE = '3DNETCLOUD'
REMOTE_AE_IP = '192.168.3.50'
REMOTE_AE_PORT = 104
IMAGES_DIR = 'images'
DB_FILE = os.path.join(IMAGES_DIR, "history.db")

SYS_PROMPT = "You are a smart radiologist working in ER. Respond in plaintext. Start with yes or no, then provide just one line description like a radiologist. Do not assume, stick to the facts, but look again if you are in doubt."
USR_PROMPT = "{} in this {} xray of a {}? Are there any other lesions?"
ANATOMY_LIST = ["chest", "sternum", "abdominal", "nasal bones", "maxilar and frontal sinus", "clavicle", "knee", "spine"]

os.makedirs(IMAGES_DIR, exist_ok = True)

main_loop = None  # Global variable to hold the main event loop
data_queue = asyncio.Queue()
websocket_clients = set()

# Dashboard state
dashboard = {
    'queue_size': 0,
    'processing_file': None,
    'success_count': 0,
    'failure_count': 0,
    'history': []
}

PAGE_SIZE = 10
KEEP_DICOM = False
NO_QUERY = False

# Database operations
def init_database():
    """ Initialize the database """
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS history (
                uid TEXT PRIMARY KEY,
                patName TEXT,
                patId TEXT,
                stDateTime TIMESTAMP,
                stProtocol TEXT,
                repDateTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                report TEXT,
                isPositive INTEGER DEFAULT 0,
                isWrong INTEGER DEFAULT 0
            )
        ''')
        logging.info("Initialized SQLite database.")

def db_load_history(limit = PAGE_SIZE, offset = 0):
    """ Load the history from the database """
    dashboard['history'] = []
    with sqlite3.connect(DB_FILE) as conn:
        for row in conn.execute('SELECT * FROM history ORDER BY stDateTime DESC LIMIT ? OFFSET ?', (limit, offset)):
            dt = datetime.strptime(row[3], "%Y-%m-%d %H:%M:%S")
            dashboard['history'].append({
                'uid': row[0],
                'png_file': f"{row[0]}.png",
                'meta': {
                    'patient': {'name': row[1], 'id': row[2]},
                    'series': {
                        'date': dt.strftime('%Y%m%d'),
                        'time': dt.strftime('%H%M%S'),
                        'protocol': row[4]
                    }
                },
                'report': row[6],
                'positive': bool(row[7]),
                'isWrong': bool(row[8])
            })
    return dashboard['history']

def db_get_history_count():
    """ Get total row count """
    with sqlite3.connect(DB_FILE) as conn:
        result = conn.execute('SELECT COUNT(*) FROM history').fetchone()
        return result[0] if result else 0

def db_toggle_right_wrong(uid):
    """ Toggle the right/wrong flag of a study """
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute('''
            UPDATE history SET isWrong = NOT isWrong WHERE uid = ?
        ''', (uid,))

def db_add_row(uid, metadata, report):
    """ Add one row to the database """
    poz = report.lower().startswith("yes") and 1 or 0
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if metadata["study"]["date"] and metadata["study"]["time"] and \
        len(metadata["study"]["date"]) == 8 and len(metadata["study"]["time"]) >= 6:
        try:
            dt = datetime.strptime(f'{metadata["study"]["date"]} {metadata["study"]["time"][:6]}', "%Y%m%d %H%M%S")
            stDateTime = dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            stDateTime = now
    else:
        stDateTime = now
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute('''
            INSERT OR REPLACE INTO history (uid, patName, patId, stDateTime, stProtocol, repDateTime, report, isPositive, isWrong)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT isWrong FROM history WHERE uid = ?), 0))
        ''', (uid, metadata["patient"]["name"], metadata["patient"]["id"], stDateTime, metadata["series"]["desc"], now, report, poz, uid))

def db_check_already_processed(uid):
    """ Check if the file has already been processed """
    with sqlite3.connect(DB_FILE) as conn:
        result = conn.execute(
            "SELECT 1 FROM history WHERE uid = ?", (uid,)
        ).fetchone()
        return result is not None


# DICOM network operations
async def query_and_retrieve(hours = 1):
    """ Query and Retrieve new studies """
    ae = AE(ae_title = AE_TITLE)
    ae.requested_contexts = QueryRetrievePresentationContexts
    ae.connection_timeout = 30
    # Create the association
    assoc = ae.associate(REMOTE_AE_IP, REMOTE_AE_PORT, ae_title = REMOTE_AE_TITLE)
    if assoc.is_established:
        logging.info(f"QueryRetrieve association established. Asking for studies in the last {hours} hours.")
        # Prepare the timespan
        # FIXME take care of the midnight
        current_time = datetime.now()
        past_time = current_time - timedelta(hours = hours)
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
        # Add to processing queue
        main_loop.call_soon_threadsafe(data_queue.put_nowait, uid)
        asyncio.run_coroutine_threadsafe(broadcast_dashboard_update(), main_loop)
    # Return success
    return 0x0000


# DICOM files operations
async def load_existing_dicom_files():
    """ Load existing .dcm files into queue """
    for file_name in os.listdir(IMAGES_DIR):
        uid, ext = os.path.splitext(os.path.basename(file_name.lower()))
        if ext == '.dcm':
            asyncio.create_task(data_queue.put(uid))
            logging.info(f"Adding {uid} into processing queue...")
    await broadcast_dashboard_update()


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
    # Normalize image to 0-255
    # TODO consider 5 as mininum and 250 as maximum
    image = ds.pixel_array.astype(np.float32)
    image -= image.min()
    if image.max() != 0:
        image /= image.max()
    image *= 255.0
    image = image.astype(np.uint8)
    # Adjust gamma
    image = adjust_gamma(image, None)
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
    # Save the PNG file
    base_name = os.path.splitext(os.path.basename(dicom_file))[0]
    png_file = os.path.join(IMAGES_DIR, f"{base_name}.png")
    cv2.imwrite(png_file, image)
    logging.info(f"Converted PNG saved to {png_file}")
    # Return the PNG file name and DICOM metadata
    metadata = {
        'patient': {
            'name':  str(ds.PatientName),
            'id':    str(ds.PatientID),
            'age':   str(ds.PatientAge),
            'sex':   str(ds.PatientSex),
            'bdate': str(ds.PatientBirthDate),
        },
        'series': {
            'uid':   str(ds.SeriesInstanceUID),
            'desc':  str(ds.SeriesDescription),
            'proto': str(ds.ProtocolName),
            'date':  str(ds.SeriesDate),
            'time':  str(ds.SeriesTime),
        },
        'study': {
            'uid':   str(ds.StudyInstanceUID),
            'date':  str(ds.StudyDate),
            'time':  str(ds.StudyTime),
        }
    }
    # Return the PNG file name and metadata
    return png_file, metadata


# WebSocket and WebServer operations
async def dashboard_handler(request):
    with open('dashboard.html', 'r') as f:
        content = f.read()
    return web.Response(text = content, content_type = 'text/html')

async def websocket_handler(request):
    """ Handle each WebSocket client """
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    websocket_clients.add(ws)
    await broadcast_dashboard_update(ws)
    logging.info(f"Dashboard connected via WebSocket from {request.remote}")
    try:
        async for msg in ws:
            pass
    finally:
        websocket_clients.remove(ws)
        logging.info("Dashboard WebSocket disconnected.")
    return ws

async def history_handler(request):
    """ Provide a page of history """
    try:
        page = int(request.query.get("page", "1"))
        offset = (page - 1) * PAGE_SIZE
        data = db_load_history(limit = PAGE_SIZE, offset = offset)
        total = db_get_history_count()
        return web.json_response({
            "history": data,
            "total": total
        })
    except Exception as e:
        logging.error(f"History page error: {e}")
        return web.json_response([], status = 500)

async def manual_query(request):
    """ Trigger a manual query/retrieve operation """
    try:
        data = await request.json()
        hours = int(data.get('hours', 3))
        logging.info(f"Manual QueryRetrieve triggered for the last {hours} hours.")
        await query_and_retrieve(hours)
        return web.json_response({'status': 'success',
                                  'message': f'Query triggered for the last {hours} hours.'})
    except Exception as e:
        logging.error(f"Error processing manual query: {e}")
        return web.json_response({'status': 'error',
                                  'message': str(e)})

async def toggle_right_wrong(request):
    """ Toggle the right/wrong flag of a study """
    data = await request.json()
    toggle_uid = data.get('uid')
    db_toggle_right_wrong(toggle_uid)
    db_load_history()
    await broadcast_dashboard_update()
    return web.json_response({'status': 'success', 'uid': toggle_uid})

async def broadcast_dashboard_update(client = None):
    """ Update the dashboard for all clients """
    # Check if there are any clients
    if not (websocket_clients or client):
        return
    # Update the queue size
    dashboard['queue_size'] = data_queue.qsize()
    # Create a list of clients
    if client:
        clients = [client,]
    else:
        clients = websocket_clients.copy()
    # Send the update to all clients
    for client in clients:
        # Send the udate to the client
        try:
            await client.send_json(dashboard)
        except Exception as e:
            logging.error(f"Error sending update to WebSocket client: {e}")
            websocket_clients.remove(client)


# AI API operations
def check_any(string, *words):
    """ Check if any of the words are present in the string """
    return any(i in string for i in words)

def get_anatomy(metadata):
    """ Try to identify the anatomy. Return the anatomy and the question. """
    desc = metadata["series"]["desc"].lower()
    if check_any(desc, 'torace', 'pulmon', "thorax"):
        anatomy = 'chest'
        question = "Are there any lung consolidations, hyperlucencies, infitrates, nodules, mediastinal shift, pleural effusion or pneumothorax"
    elif check_any(desc, 'grilaj', 'coaste'):
        anatomy = 'chest'
        question = "Are there any ribs or clavicles fractures"
    elif check_any(desc, 'stern'):
        anatomy = 'sternum'
        question = "Are there any fractures"
    elif check_any(desc, 'abdomen'):
        anatomy = 'abdominal'
        question = "Are there any hydroaeric levels or pneumoperitoneum"
    elif check_any(desc, 'cap', 'craniu', 'occiput'):
        anatomy = 'skull'
        question = "Are there any fractures"
    elif check_any(desc, 'mandibula'):
        anatomy = 'mandible'
        question = "Are there any fractures"
    elif check_any(desc, 'nazal'):
        anatomy = 'nasal bones'
        question = "Are there any fractures"
    elif check_any(desc, 'sinus'):
        anatomy = 'maxilar and frontal sinus'
        question = "Are there any changes in transparency of the sinuses"
    elif check_any(desc, 'col.'):
        anatomy = 'spine'
        question = "Are there any fractures or dislocations"
    elif check_any(desc, 'bazin'):
        anatomy = 'pelvis'
        question = "Are there any fractures"
    elif check_any(desc, 'clavicul'):
        anatomy = 'clavicle'
        question = "Are there any fractures"
    elif check_any(desc, 'humerus', 'antebrat'):
        anatomy = 'upper limb'
        question = "Are there any fractures, dislocations or bone tumors"
    elif check_any(desc, 'pumn', 'mana', 'deget'):
        anatomy = 'hand'
        question = "Are there any fractures, dislocations or bone tumors"
    elif check_any(desc, 'umar'):
        anatomy = 'shoulder'
        question = "Are there any fractures or dislocations"
    elif check_any(desc, 'cot'):
        anatomy = 'elbow'
        question = "Are there any fractures or dislocations"
    elif check_any(desc, 'sold'):
        anatomy = 'hip'
        question = "Are there any fractures or dislocations"
    elif check_any(desc, 'femur', 'tibie', 'glezna', 'picior', 'gamba', 'calcai'):
        anatomy = 'lower limb'
        question = "Are there any fractures, dislocations or bone tumors"
    elif check_any(desc, 'genunchi', 'patella'):
        anatomy = 'knee'
        question = "Are there any fractures or dislocations"
    else:
        # Fallback
        anatomy = desc
        question = "Is there anything abnormal"
    # Return the anatomy and the question
    return anatomy, question

def get_projection(metadata):
    """ Try to identify the projection """
    desc = metadata["series"]["desc"].lower()
    if check_any(desc, "a.p.", "p.a.", "d.v.", "v.d."):
        projection = "frontal"
    elif check_any(desc, "lat."):
        projection = "lateral"
    elif check_any(desc, "oblic"):
        projection = "oblique"
    else:
        # Fallback
        projection = ""
    # Return the projection
    return projection

def get_gender(metadata):
    """ Try to identify the gender """
    patient_sex = metadata["patient"]["sex"].lower()
    if "m" in patient_sex:
        gender = "boy"
    elif "f" in patient_sex:
        gender = "girl"
    else:
        # Fallback
        gender = "child"
    # Return the gender
    return gender

def get_age(metadata):
    """ Try to get the age """
    age = metadata["patient"]["age"].lower().replace("y", "").strip()
    if age:
        if age == "000":
            age = "newborn"
        else:
            age = age + " years old"
    else:
        age = ""
    # Return the age (string)
    return age

async def send_image_to_openai(uid, metadata, max_retries = 3):
    """Send PNG to OpenAI API with retries and save response to text file."""
    # Read the PNG file
    with open(os.path.join(IMAGES_DIR, f"{uid}.png"), 'rb') as f:
        image_bytes = f.read()
    # Prepare the prompt
    anatomy, question = get_anatomy(metadata)
    projection = get_projection(metadata)
    gender = get_gender(metadata)
    age = get_age(metadata)
    # Filter on specific anatomy
    if not anatomy in ANATOMY_LIST:
        logging.info(f"Ignoring {uid} with {anatomy} x-ray.")
        return False
    # Get the subject of the study and the studied region
    subject = " ".join([age, gender])
    if anatomy:
        region = " ".join([projection, anatomy])
    else:
        region = ""
    # Create the prompt
    prompt = USR_PROMPT.format(question, region, subject)
    logging.debug(f"Prompt: {prompt}")
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
    # Up to 3 attempts with exponential backoff (2s, 4s, 8s delays).
    attempt = 1
    while attempt <= max_retries:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(OPENAI_API_URL, headers = headers, json = data) as response:
                    result = await response.json()
                    report = result["choices"][0]["message"]["content"]
                    report = report.replace('\n', " ").replace("  ", " ").strip()
                    logging.info(f"OpenAI API response for {uid}: {report}")
                    # Update the dashboard
                    dashboard['success_count'] += 1
                    # Save to history database
                    db_add_row(uid, metadata, report)
                    # Rebuild the dashboard from database
                    db_load_history()
                    await broadcast_dashboard_update()
                    # Success
                    return True
        except Exception as e:
            logging.warning(f"Error uploading {uid} (attempt {attempt}): {e}")
            # Exponential backoff
            await asyncio.sleep(2 ** attempt)
            attempt += 1
    # Failure after max_retries
    logging.error(f"Failed to upload {uid} after {max_retries} attempts.")
    dashboard['failure_count'] += 1
    await broadcast_dashboard_update()
    return False


# Threads
async def start_dashboard():
    """ Start the dashboard web server """
    app = web.Application()
    app.router.add_get('/', dashboard_handler)
    app.router.add_get('/ws', websocket_handler)
    app.router.add_get('/history', history_handler)
    app.router.add_post('/toggle_right_wrong', toggle_right_wrong)
    app.router.add_post('/trigger_query', manual_query)
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
        uid = await data_queue.get()
        # Update the dashboard
        dashboard['processing_file'] = uid
        await broadcast_dashboard_update()
        # The DICOM file name
        dicom_file = os.path.join(IMAGES_DIR, f"{uid}.dcm")
        # Try to convert to PNG
        try:
            png_file, metadata = dicom_to_png(dicom_file)
        except Exception as e:
            logging.error(f"Error converting DICOM file {dicom_file}: {e}")
        # Try to send to AI
        try:
            await send_image_to_openai(uid, metadata)
        except Exception as e:
            logging.error(f"Unhandled error processing {uid}: {e}")
        finally:
            dashboard['processing_file'] = None
            await broadcast_dashboard_update()
            data_queue.task_done()
        # Remove the DICOM file
        if not KEEP_DICOM:
            try:
                os.remove(dicom_file)
                logging.info(f"DICOM file {dicom_file} deleted after processing.")
            except Exception as e:
                logging.warning(f"Error removing DICOM file {dicom_file}: {e}")
        else:
            logging.debug(f"Kept DICOM file: {dicom_file}")

async def query_retrieve_loop():
    """ Async thread to periodically poll the server for new studies """
    while not NO_QUERY:
        await query_and_retrieve()
        await asyncio.sleep(3600)

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
        logging.info("SQLite history database not found. Creating a new one...")
        init_database()
    else:
        logging.info("SQLite history database found.")
    # Load history
    history = db_load_history()
    logging.info(f"Loaded {len(history)} history items.")

    # Start the asynchronous tasks
    asyncio.create_task(start_dashboard())
    asyncio.create_task(relay_to_openai_loop())
    asyncio.create_task(query_retrieve_loop())
    # Preload the existing dicom files
    await load_existing_dicom_files()
    # Start the DICOM server
    await asyncio.get_running_loop().run_in_executor(None, start_dicom_server)


# Command run
if __name__ == '__main__':
    # Need to process the arguments
    import argparse
    parser = argparse.ArgumentParser(description = "XRayVision - Async DICOM processor with OpenAI and WebSocket dashboard")
    parser.add_argument("--keep-dicom", action = "store_true", help = "Do not delete .dcm files after conversion")
    parser.add_argument("--no-query", action = "store_true", help = "Do not query the DICOM server automatically")
    args = parser.parse_args()
    # Store in globals
    KEEP_DICOM = args.keep_dicom
    NO_QUERY = args.no_query

    # Run
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("XRayVision stopped by user. Shutting down.")
