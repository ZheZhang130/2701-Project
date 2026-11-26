"""Demo application. The existing ML service remains a separate process."""
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

DATA = Path(os.getenv('APP_DATA_DIR', str(Path(__file__).parent / 'data')))
DATA.mkdir(parents=True, exist_ok=True)
(DATA / 'frames').mkdir(exist_ok=True)
STATE_FILE = DATA / 'state.json'
ML_URL = os.getenv('ML_API_URL', 'http://127.0.0.1:7860').rstrip('/')
MODEL = os.getenv('ML_MODEL', 'resnet18')
CLASSES = ['Safe driving', 'Texting right', 'Phone right', 'Texting left', 'Phone left',
           'Operating radio', 'Drinking', 'Reaching behind', 'Hair / makeup', 'Talking to passenger']
state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {
    'drivers': {}, 'notifications': [], 'rule': {'window': 10, 'threshold': 5}}
lock = asyncio.Lock()
app = FastAPI(title='SafeDrive Dashboard API')
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
                   allow_methods=['*'], allow_headers=['*'])
app.mount('/media', StaticFiles(directory=DATA / 'frames'), name='media')


def save():
    temporary = STATE_FILE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state))
    temporary.replace(STATE_FILE)


def now():
    return datetime.now(timezone.utc).isoformat()


def driver_for(driver_id):
    if driver_id not in state['drivers']:
        raise HTTPException(404, 'Driver not found')
    return state['drivers'][driver_id]


def summary(driver):
    return {k: v for k, v in driver.items() if k != 'frames'} | {
        'frame_count': len(driver['frames']), 'latest': driver['frames'][-1] if driver['frames'] else None}


class DriverInput(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class Rule(BaseModel):
    window: int = Field(ge=1, le=100)
    threshold: int = Field(ge=1, le=100)


@app.get('/api/dashboard')
def dashboard():
    return {'drivers': [summary(d) for d in state['drivers'].values()],
            'notifications': state['notifications'], 'rule': state['rule'], 'classes': CLASSES,
            'model': MODEL}


@app.get('/api/health')
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f'{ML_URL}/v1/models', headers={'X-API-Key': os.getenv('SAFEDRIVE_API_KEY', '')})
            response.raise_for_status()
        return {'ml_connected': True, 'model': MODEL}
    except httpx.HTTPError as exc:
        return {'ml_connected': False, 'detail': str(exc), 'model': MODEL}


@app.post('/api/drivers')
async def add_driver(body: DriverInput):
    async with lock:
        driver = {'id': uuid4().hex, 'name': body.name, 'tagged': False, 'tagged_at': None,
                  'recent': [], 'frames': []}
        state['drivers'][driver['id']] = driver
        save()
        return summary(driver)


@app.get('/api/drivers/{driver_id}')
def history(driver_id: str):
    return driver_for(driver_id)


@app.put('/api/rule')
async def update_rule(body: Rule):
    if body.threshold > body.window:
        raise HTTPException(422, 'Threshold must be no larger than the window')
    async with lock:
        state['rule'] = body.model_dump()
        # Start fresh evidence windows; preserve tags and all frame history.
        for driver in state['drivers'].values():
            driver['recent'] = []
        save()
        return state['rule']


@app.post('/api/drivers/{driver_id}/clear')
async def clear_tag(driver_id: str):
    async with lock:
        driver = driver_for(driver_id)
        driver.update(tagged=False, tagged_at=None, recent=[])
        save()
        return summary(driver)


@app.post('/api/notifications/read')
async def read_notifications():
    async with lock:
        for notification in state['notifications']:
            notification['read'] = True
        save()
    return {'ok': True}


@app.post('/api/drivers/{driver_id}/frames')
async def classify(driver_id: str, request: Request):
    image = await request.body()
    media_type = request.headers.get('content-type', 'image/jpeg').split(';')[0]
    if media_type not in ('image/jpeg', 'image/png') or not image:
        raise HTTPException(422, 'Send raw JPEG or PNG bytes')
    # One shared lock also prevents concurrent calls from overloading the single-worker ML API.
    async with lock:
        driver = driver_for(driver_id)
        try:
            async with httpx.AsyncClient(timeout=90) as client:
                response = await client.post(f'{ML_URL}/v1/models/{MODEL}/predict', content=image,
                    headers={'X-API-Key': os.getenv('SAFEDRIVE_API_KEY', ''), 'Content-Type': media_type})
                response.raise_for_status()
                prediction = response.json()
        except httpx.HTTPError as exc:
            raise HTTPException(502, f'ML service request failed: {exc}') from exc
        label = prediction['predicted_class']
        frame_id = uuid4().hex
        filename = frame_id + ('.png' if media_type == 'image/png' else '.jpg')
        (DATA / 'frames' / filename).write_bytes(image)
        frame = {'id': frame_id, 'timestamp': now(), 'image_url': f'/media/{filename}',
                 'class_id': label, 'label': CLASSES[label], 'dangerous': label != 0,
                 'confidence': prediction['probabilities'][label], 'prediction': prediction}
        driver['frames'].append(frame)
        rule = state['rule']
        driver['recent'] = (driver['recent'] + [frame['dangerous']])[-rule['window']:]
        if sum(driver['recent']) >= rule['threshold'] and not driver['tagged']:
            driver.update(tagged=True, tagged_at=frame['timestamp'])
            state['notifications'].insert(0, {'id': uuid4().hex, 'driver_id': driver_id,
                'driver_name': driver['name'], 'timestamp': frame['timestamp'], 'read': False,
                'message': f"{sum(driver['recent'])} dangerous frames in the last {len(driver['recent'])} frames"})
        save()
        return {'frame': frame, 'driver': summary(driver)}


@app.post('/api/demo/import')
async def import_demo():
    """Import distinct synthetic frames, upgrading only old demo fixtures."""
    from datetime import timedelta
    assets = Path(__file__).parent / 'demo_assets'
    sources = json.loads((assets / 'sources.json').read_text())
    added = []
    async with lock:
        for original, filename in [('driver-sequences-contact-sheet.png', 'driver-sequences-contact-sheet.png'), ('GENERATION.md', 'demo-generation.txt')]:
            (DATA / 'frames' / filename).write_bytes((assets / original).read_bytes())
        for index, source in enumerate(sources):
            driver_id = source['id']
            old = state['drivers'].get(driver_id)
            if old and (not old.get('demo') or old.get('demo_version') == 2):
                continue
            start = datetime.now(timezone.utc) - timedelta(minutes=15 - index * 4)
            frames = []
            for number, sample in enumerate(source['frames']):
                filename = sample['file']
                (DATA / 'frames' / filename).write_bytes((assets / filename).read_bytes())
                label = sample['class_id']
                frames.append({'id': f'{driver_id}-v2-frame-{number}',
                    'timestamp': (start + timedelta(seconds=number * 2)).isoformat(),
                    'image_url': f'/media/{filename}', 'class_id': label,
                    'label': CLASSES[label], 'dangerous': label != 0, 'confidence': None,
                    'prediction': None, 'demo': True, 'source': source,
                    'note': 'Distinct AI-generated staged frame; manual annotation, not ML output.'})
            # Preserve any real uploads added to an imported profile.
            frames.extend(f for f in (old or {}).get('frames', []) if not f.get('demo'))
            recent = [f['dangerous'] for f in frames][-state['rule']['window']:]
            tagged = sum(recent) >= state['rule']['threshold']
            driver = {'id': driver_id, 'name': source['name'], 'demo': True, 'demo_version': 2,
                'tagged': tagged, 'tagged_at': frames[-1]['timestamp'] if tagged else None,
                'recent': recent, 'frames': frames, 'source': source}
            state['drivers'][driver_id] = driver
            state['notifications'] = [n for n in state['notifications']
                if not (n['driver_id'] == driver_id and n.get('demo'))]
            if tagged:
                state['notifications'].insert(0, {'id': driver_id + '-v2-alert', 'driver_id': driver_id,
                    'driver_name': driver['name'], 'timestamp': driver['tagged_at'], 'read': False,
                    'demo': True, 'message': f"Demo: {sum(recent)} dangerous frames in {len(recent)} samples"})
            added.append(driver_id)
        save()
    return {'added': added}


@app.websocket('/api/demo/heartbeat')
async def demo_heartbeat(websocket: WebSocket):
    """Keep the launcher's lease alive while at least one dashboard is connected."""
    await websocket.accept()
    heartbeat = os.getenv('DEMO_HEARTBEAT_FILE')
    try:
        while True:
            if heartbeat:
                Path(heartbeat).touch()
            await websocket.send_text('connected')
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=5)
            except asyncio.TimeoutError:
                pass
    except (WebSocketDisconnect, OSError, RuntimeError):
        pass
