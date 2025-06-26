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

os.makedirs(IMAGES_DIR, exist_ok=True)
data_queue = asyncio.Queue()

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
    image = ds.pixel_array
    image = image.astype(np.float32)
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
    return png_file, ds.PatientName, ds.PatientID, ds.StudyDate

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
                    {'type': 'text', 'text': 'Is there anything abnormal with this chest xray? Answer yes or no, then provide a one line description like a radiologist.'},
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
    return False

async def relay_to_openai():
    """Relay PNG files to OpenAI API with retries and dashboard update."""
    while True:
        # Get one file from queue
        dicom_file = await data_queue.get()
        # Update the dashboard
        dashboard_state['processing_file'] = os.path.basename(dicom_file)
        dashboard_state['queue_size'] = data_queue.qsize()
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
    """Render the status dashboard with thumbnails, lightbox previews, and mobile responsiveness using a Tango dark theme."""
    history_html = ""
    for filename, patient_name, patient_id, study_date, response in dashboard_state['history']:
        image_path = f"/static/{filename}"
        highlight = response.strip().lower().startswith('yes')
        response_style = "color: #8ae234; font-weight: bold;" if highlight else "color: #eeeeec;"
        history_html += f"""
        <div class="blockcard">
            <a href="{image_path}" class="lightbox-link">
                <img src="{image_path}">
            </a>
            <div>
                <strong>{patient_name}</strong><br>
                <span>{patient_id}</span><br>
                <span>{study_date}</span><br>
                <span style="{response_style}">{response}</span>
            </div>
        </div>
        """

    content = f"""
    <html>
    <head>
        <meta http-equiv="refresh" content="5">
        <title>XRayVision Processing Dashboard</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #2e3436;
                color: #eeeeec;
                margin: 20px;
            }}
            h1, h2 {{
                color: #729fcf;
            }}
            .card {{
                padding: 10px;
                margin: 10px;
                border: 1px solid #729fcf;
                border-radius: 8px;
                background-color: #555753;
                display: inline-block;
                width: 200px;
                vertical-align: top;
                text-align: center;
            }}
            .blockcard {{
                padding: 10px;
                margin: 10px;
                border: 1px solid #729fcf;
                border-radius: 8px;
                background-color: #555753;
                display: block;
                max-width: 100%;
                box-sizing: border-box;
            }}
            .blockcard img {{
                height: 100px;
                border: 1px solid #729fcf;
                margin-right: 10px;
                cursor: pointer;
            }}
            .blockcard div {{
                display: inline-block;
                vertical-align: top;
                max-width: 500px;
            }}
            .lightbox {{
                display: none;
                position: fixed;
                z-index: 999;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.9);
                text-align: center;
                padding-top: 60px;
            }}
            .lightbox img {{
                max-width: 90%;
                max-height: 80%;
            }}
            .lightbox:target {{
                display: block;
            }}
            @media (max-width: 768px) {{
                .blockcard {{
                    width: 100%;
                }}
                .blockcard img {{
                    height: 80px;
                }}
            }}
        </style>
    </head>
    <body>
        <h1>📊 XRayVision Processing Dashboard</h1>
        <div class="card"><strong>Queue Size:</strong><br>{dashboard_state['queue_size']}</div>
        <div class="card"><strong>Currently Processing:</strong><br>{dashboard_state['processing_file'] or "Idle"}</div>
        <div class="card"><strong>Successful Uploads:</strong><br>{dashboard_state['success_count']}</div>
        <div class="card"><strong>Failed Uploads:</strong><br>{dashboard_state['failure_count']}</div>

        <h2>Last {MAX_HISTORY} Processed Files</h2>
        <div style="display: flex; flex-wrap: wrap; justify-content: center;">{history_html}</div>

        <div id="lightbox" class="lightbox">
            <img id="lightbox-img" src="">
        </div>

        <script>
            // Lightbox click handler
            document.querySelectorAll('.lightbox-link img').forEach(img => {{
                img.addEventListener('click', function(e) {{
                    e.preventDefault();
                    document.getElementById('lightbox-img').src = this.src;
                    document.getElementById('lightbox').style.display = 'block';
                }});
            }});
            document.getElementById('lightbox').addEventListener('click', function() {{
                this.style.display = 'none';
            }});
        </script>
    </body>
    </html>
    """
    return web.Response(text=content, content_type='text/html')

async def start_dashboard():
    """Start the dashboard web server."""
    app = web.Application()
    app.router.add_get('/', dashboard)
    app.router.add_static('/static/', path = IMAGES_DIR, name = 'static')
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
