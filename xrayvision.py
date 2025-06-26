import asyncio
import os
import uuid
import base64
import aiohttp
import cv2
import numpy as np
from aiohttp import web
from pydicom import dcmread
from pynetdicom import AE, evt, StoragePresentationContexts

# Configuration
PNG_STORAGE_DIR = 'png_files'
OPENAI_API_URL = 'https://api.openai.com/v1/chat/completions'
OPENAI_API_KEY = 'sk-your-api-key'  # Insert your OpenAI API key
LISTEN_PORT = 4010  # Updated DICOM port
DASHBOARD_PORT = 8000  # Updated dashboard port
AE_TITLE = 'XRAYVISION'  # Updated AE Title

os.makedirs(PNG_STORAGE_DIR, exist_ok=True)
data_queue = asyncio.Queue()

# Dashboard state
dashboard_state = {
    'queue_size': 0,
    'processing_file': None,
    'success_count': 0,
    'failure_count': 0,
    'history': []  # List of (filename, response_text)
}

MAX_HISTORY = 10

def dicom_to_png(dicom_file, max_size=500):
    """Convert DICOM to PNG and return PNG filename."""
    ds = dcmread(dicom_file)

    if 'PixelData' not in ds:
        raise ValueError("DICOM file has no pixel data!")

    image = ds.pixel_array
    image = image.astype(np.float32)
    image -= image.min()
    if image.max() != 0:
        image /= image.max()
    image *= 255.0
    image = image.astype(np.uint8)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    height, width = image.shape[:2]
    if max(height, width) > max_size:
        if height > width:
            new_height = max_size
            new_width = int(width * (max_size / height))
        else:
            new_width = max_size
            new_height = int(height * (max_size / width))
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    base_name = str(uuid.uuid4())
    png_file = os.path.join(PNG_STORAGE_DIR, f"{base_name}.png")
    cv2.imwrite(png_file, image)
    print(f"Converted and resized image saved to {png_file}")

    return png_file

def handle_store(event):
    """Callback for receiving a DICOM file."""
    ds = event.dataset
    ds.file_meta = event.file_meta

    dicom_file = os.path.join(PNG_STORAGE_DIR, f"{uuid.uuid4()}.dcm")
    ds.save_as(dicom_file, write_like_original=False)
    print(f"Received and saved DICOM file: {dicom_file}")

    try:
        png_file = dicom_to_png(dicom_file)
        os.remove(dicom_file)
        print(f"DICOM file {dicom_file} deleted after conversion.")
        asyncio.create_task(data_queue.put(png_file))
    except Exception as e:
        print(f"Error converting DICOM file {dicom_file}: {e}")

    return 0x0000

async def send_image_to_openai(png_file, max_retries=3):
    """Send PNG to OpenAI API with retries and save response to text file."""
    with open(png_file, 'rb') as f:
        image_bytes = f.read()

    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    image_url = f"data:image/png;base64,{image_b64}"

    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json',
        'Connection': 'close'
    }

    data = {
        'model': 'gpt-4o',
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'Analyze this image.'},
                    {'type': 'image_url', 'image_url': {'url': image_url}}
                ]
            }
        ]
    }

    attempt = 0
    while attempt < max_retries:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(OPENAI_API_URL, headers=headers, json=data) as response:
                    response_text = await response.text()
                    print(f"OpenAI API response for {png_file}: {response_text}")

                    base_name = os.path.splitext(os.path.basename(png_file))[0]
                    response_file = os.path.join(PNG_STORAGE_DIR, f"{base_name}.txt")
                    with open(response_file, 'w') as f:
                        f.write(response_text)
                    print(f"Response saved to {response_file}")

                    dashboard_state['success_count'] += 1

                    dashboard_state['history'].insert(0, (os.path.basename(png_file), response_text))
                    dashboard_state['history'] = dashboard_state['history'][:MAX_HISTORY]

                    return True  # Success
        except Exception as e:
            print(f"Error uploading {png_file} (Attempt {attempt + 1}): {e}")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
            attempt += 1

    print(f"Failed to upload {png_file} after {max_retries} attempts.")
    dashboard_state['failure_count'] += 1
    return False

async def relay_to_openai():
    """Relay PNG files to OpenAI API with retries and dashboard update."""
    while True:
        png_file = await data_queue.get()
        try:
            dashboard_state['processing_file'] = os.path.basename(png_file)
            dashboard_state['queue_size'] = data_queue.qsize()

            await send_image_to_openai(png_file)

        except Exception as e:
            print(f"Unhandled error processing {png_file}: {e}")
        finally:
            dashboard_state['processing_file'] = None
            dashboard_state['queue_size'] = data_queue.qsize()
            data_queue.task_done()

async def start_dicom_server():
    """Start the DICOM Storage SCP."""
    ae = AE(ae_title=AE_TITLE)
    ae.supported_contexts = StoragePresentationContexts
    handlers = [(evt.EVT_C_STORE, handle_store)]

    print(f"Starting DICOM server on port {LISTEN_PORT} with AE Title '{AE_TITLE}'...")
    ae.start_server(('', LISTEN_PORT), evt_handlers=handlers, block=True)

# Dashboard Server
async def dashboard(request):
    """Render the status dashboard with thumbnails and responses."""
    history_html = ""
    for filename, response in dashboard_state['history']:
        image_path = f"/static/{filename}"
        highlight = response.strip().lower().startswith('yes')
        response_style = "color: green; font-weight: bold;" if highlight else "color: black;"
        history_html += f"""
        <div class="card">
            <img src="{image_path}" width="100"><br>
            <strong>{filename}</strong><br>
            <div style="{response_style}">{response}</div>
        </div>
        """

    content = f"""
    <html>
    <head>
        <meta http-equiv="refresh" content="2">
        <title>Processing Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .card {{ padding: 10px; margin: 10px; border: 1px solid #ddd; border-radius: 8px; display: inline-block; width: 200px; vertical-align: top; }}
        </style>
    </head>
    <body>
        <h1>📊 Processing Dashboard</h1>
        <div class="card"><strong>Queue Size:</strong> {dashboard_state['queue_size']}</div>
        <div class="card"><strong>Currently Processing:</strong> {dashboard_state['processing_file'] or "Idle"}</div>
        <div class="card"><strong>Successful Uploads:</strong> {dashboard_state['success_count']}</div>
        <div class="card"><strong>Failed Uploads:</strong> {dashboard_state['failure_count']}</div>

        <h2>Last 10 Processed Files</h2>
        <div style="display: flex; flex-wrap: wrap;">{history_html}</div>
    </body>
    </html>
    """
    return web.Response(text=content, content_type='text/html')

async def start_dashboard():
    """Start the dashboard web server."""
    app = web.Application()
    app.router.add_get('/', dashboard)
    app.router.add_static('/static/', path=PNG_STORAGE_DIR, name='static')
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', DASHBOARD_PORT)
    await site.start()
    print(f"Dashboard available at http://localhost:{DASHBOARD_PORT}")

async def main():
    asyncio.create_task(relay_to_openai())
    asyncio.create_task(start_dashboard())

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, start_dicom_server)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped by user. Shutting down, meatbag.")
