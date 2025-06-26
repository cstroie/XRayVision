#!/usr/bin/env python3

import asyncio
import os
import uuid
import base64
import aiohttp
import cv2
import numpy as np
import math
from aiohttp import web
from pydicom import dcmread
from pynetdicom import AE, evt, StoragePresentationContexts
from pynetdicom.sop_class import ComputedRadiographyImageStorage, DigitalXRayImageStorageForPresentation

# Configuration
IMAGES_DIR = 'images'
OPENAI_API_URL = 'http://127.0.0.1:8080/v1/chat/completions'
OPENAI_API_KEY = 'sk-your-api-key'  # Insert your OpenAI API key
LISTEN_PORT = 4010  # Updated DICOM port
DASHBOARD_PORT = 8000  # Updated dashboard port
AE_TITLE = 'XRAYVISION'  # Updated AE Title

PROMPT = "Identify the region in xray: skull, spine, chest, abdomen, pelvis, upper and lower limb. Identify the projection: frontal or lateral, standing or laying back. The pacient is always a child, so the xray might not be perfect in exposure and projection. Check if the patient rotated. Assess carefully if there is anything abnormal pictured in the xray. Do not assume, stick to the facts. The answer should be YES or NO. If in doubt, say so. Then provide a one line description of the findings like a radiologist. Check for fractures, foreign metallic bodies, lung consolidation, lung hyperlucency, lung infitrates, lung nodules, air bronchogram, tracheal narrowing, mediastinal shift, pleural effusion, pneumothorax, cardiac silhouette, heart size reported to chest size, size of thimus, large abdominal hydroaeric levels, distended bowel loops, pneumoperitoneum, no gas in lower right abdomen suggestive to intussusception, catheters, spine curvatures, vertebral fractures, vertebral alignment, subcutaneous emphysema, skull fractures, maxilar and frontal sinus transparency."

os.makedirs(IMAGES_DIR, exist_ok=True)
data_queue = asyncio.Queue()
websocket_clients = set()

# Dashboard state
dashboard_state = {
    'queue_size': 0,
    'processing_file': None,
    'success_count': 0,
    'failure_count': 0,
    'history': []  # List of (filename, patient_name, patient_id, study_date, text)
}

MAX_HISTORY = 20

main_loop = None  # Global variable to hold the main event loop

def adjust_gamma(image, gamma = 1.2):
    # If gamma is None, compute it
    if gamma is None:
        if len(image.shape) > 2: # or image.shape[2] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # compute gamma = log(mid*255)/log(mean)
        mid = 0.5
        mean = np.median(image)
        gamma = math.log(mid * 255) / math.log(mean)
        print(f"Calculated gamma is {gamma:.2f}")
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def dicom_to_png(dicom_file, max_size = 500):
    """Convert DICOM to PNG and return PNG filename."""
    # Get the dataset
    ds = dcmread(dicom_file)
    # Check for PixelData
    if 'PixelData' not in ds:
        raise ValueError("DICOM file has no pixel data!")
    # Normalize image to 0-255
    image = ds.pixel_array.astype(np.float32)
    image -= image.min()
    if image.max() != 0:
        image /= image.max()
    image *= 255.0
    image = image.astype(np.uint8)
    # Adjust gamma
    image = adjust_gamma(image, None)
    # Convert to 3-channel if needed (for OpenAI if it expects color)
    #if len(image.shape) == 2:
    #    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
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
    print(f"Converted and resized image saved to {png_file}")
    # Return the PNG file name
    return png_file, str(ds.PatientName), str(ds.PatientID), str(ds.StudyDate)

async def broadcast_dashboard_update():
    if not websocket_clients:
        return
    update = {
        'queue_size': dashboard_state['queue_size'],
        'processing_file': dashboard_state['processing_file'],
        'success_count': dashboard_state['success_count'],
        'failure_count': dashboard_state['failure_count'],
        'history': dashboard_state['history']
    }
    for ws in websocket_clients.copy():
        try:
            await ws.send_json(update)
        except Exception as e:
            print(f"Error sending update to WebSocket client: {e}")
            websocket_clients.remove(ws)

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    websocket_clients.add(ws)
    print("Dashboard connected via WebSocket.")
    try:
        async for msg in ws:
            pass
    finally:
        websocket_clients.remove(ws)
        print("Dashboard WebSocket disconnected.")
    return ws

def handle_store(event):
    """Callback for receiving a DICOM file."""
    # Get the dataset
    ds = event.dataset
    ds.file_meta = event.file_meta
    # Save the DICOM file
    dicom_file = os.path.join(IMAGES_DIR, f"{ds.SOPInstanceUID}.dcm")
    ds.save_as(dicom_file, write_like_original = False)
    print(f"Received and saved DICOM file: {dicom_file}")
    # Schedule queue put on the main event loop
    #main_loop.call_soon_threadsafe(asyncio.create_task, data_queue.put(dicom_file))
    main_loop.call_soon_threadsafe(data_queue.put_nowait, dicom_file)
    dashboard_state['queue_size'] = data_queue.qsize()
    asyncio.run_coroutine_threadsafe(broadcast_dashboard_update(), main_loop)
    # Return success
    return 0x0000

async def send_image_to_openai(png_file, patient_name = "", patient_id = "", study_date = "", max_retries = 3):
    """Send PNG to OpenAI API with retries and save response to text file."""
    # Read the PNG file
    with open(png_file, 'rb') as f:
        image_bytes = f.read()
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
        'model': 'medgemma-4b-it',
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': PROMPT},
                    {'type': 'image_url', 'image_url': {'url': image_url}}
                ]
            }
        ]
    }
    # Up to 3 attempts with exponential backoff (2s, 4s, 8s delays).
    attempt = 0
    while attempt < max_retries:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(OPENAI_API_URL, headers = headers, json = data) as response:
                    result = await response.json()
                    text = result["choices"][0]["message"]["content"]
                    text = text.replace('\n', " ").replace("  ", " ")
                    print(f"OpenAI API response for {png_file}: {text}")
                    # Save the result to a text file
                    base_name = os.path.splitext(os.path.basename(png_file))[0]
                    text_file = os.path.join(IMAGES_DIR, f"{base_name}.txt")
                    with open(text_file, 'w') as f:
                        f.write(text)
                    print(f"Response saved to {text_file}")
                    # Update the dashboard
                    dashboard_state['success_count'] += 1
                    # Add to history (keep only last MAX_HISTORY)
                    dashboard_state['history'].insert(0, (os.path.basename(png_file), patient_name, patient_id, study_date, text))
                    dashboard_state['history'] = dashboard_state['history'][:MAX_HISTORY]
                    await broadcast_dashboard_update()
                    # Success
                    return True
        except Exception as e:
            print(f"Error uploading {png_file} (Attempt {attempt + 1}): {e}")
            # Exponential backoff
            await asyncio.sleep(2 ** attempt)
            attempt += 1
    # Failure after max_retries
    print(f"Failed to upload {png_file} after {max_retries} attempts.")
    dashboard_state['failure_count'] += 1
    await broadcast_dashboard_update()
    return False

async def relay_to_openai():
    """Relay PNG files to OpenAI API with retries and dashboard update."""
    while True:
        # Get one file from queue
        dicom_file = await data_queue.get()
        # Update the dashboard
        dashboard_state['processing_file'] = os.path.basename(dicom_file)
        dashboard_state['queue_size'] = data_queue.qsize()
        await broadcast_dashboard_update()
        # Try to convert to PNG
        try:
            png_file, patient_name, patient_id, study_date = dicom_to_png(dicom_file)
            os.remove(dicom_file)
            print(f"DICOM file {dicom_file} deleted after conversion.")
        except Exception as e:
            print(f"Error converting DICOM file {dicom_file}: {e}")
        # Try to send to AI
        try:
            await send_image_to_openai(png_file, patient_name, patient_id, study_date)
        except Exception as e:
            print(f"Unhandled error processing {png_file}: {e}")
        finally:
            dashboard_state['processing_file'] = None
            dashboard_state['queue_size'] = data_queue.qsize()
            await broadcast_dashboard_update()
            data_queue.task_done()

def start_dicom_server():
    """Start the DICOM Storage SCP."""
    ae = AE(ae_title=AE_TITLE)
    # Accept only XRays
    ae.add_supported_context(ComputedRadiographyImageStorage)
    ae.add_supported_context(DigitalXRayImageStorageForPresentation)
    # C-Store handler
    handlers = [(evt.EVT_C_STORE, handle_store)]
    print(f"Starting DICOM server on port {LISTEN_PORT} with AE Title '{AE_TITLE}'...")
    ae.start_server(("0.0.0.0", LISTEN_PORT), evt_handlers = handlers, block = True)

async def dashboard(request):
    with open('dashboard.html', 'r') as f:
        content = f.read()
    return web.Response(text = content, content_type = 'text/html')

async def start_dashboard():
    """Start the dashboard web server."""
    app = web.Application()
    app.router.add_get('/', dashboard)
    app.router.add_static('/static/', path = IMAGES_DIR, name = 'static')
    app.router.add_get('/ws', websocket_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', DASHBOARD_PORT)
    await site.start()
    print(f"Dashboard available at http://localhost:{DASHBOARD_PORT}")

async def main():
    # Store main event loop here
    global main_loop
    main_loop = asyncio.get_running_loop()  
    # Start the asynchronous tasks
    asyncio.create_task(relay_to_openai())
    asyncio.create_task(start_dashboard())
    # Start the DICOM server
    await asyncio.get_running_loop().run_in_executor(None, start_dicom_server)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped by user. Shutting down.")
