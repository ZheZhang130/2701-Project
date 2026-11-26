"""Single-process HTTP adapter for the versioned experimental handover."""
import io
import json
import os
import secrets
import threading
import warnings
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from .contracts import FrameResult


@dataclass(frozen=True)
class ModelFrameResult(FrameResult):
    model_id: str


class PredictionResult(BaseModel):
    model_id: str
    model_version: str
    probabilities: list[float] = Field(min_length=10, max_length=10)
    predicted_class: int = Field(ge=0, le=9)
    frame_score: float = Field(ge=0, le=1)


IMAGE_REQUEST = {'requestBody': {'required': True, 'content': {
    media: {'schema': {'type': 'string', 'format': 'binary'}}
    for media in ('image/jpeg', 'image/png')}}}

MAX_BYTES = 4 * 1024 * 1024
MAX_PIXELS = 12_000_000
CLASS_NAMES = ['safe driving', 'texting right', 'phone right', 'texting left',
               'phone left', 'operating radio', 'drinking', 'reaching behind',
               'hair/makeup', 'talking to passenger']


def load_monitors(manifest_path, device):
    import torch
    from .advanced import AdvancedClassifier
    from .ensemble import EnsembleClassifier
    from .monitoring import DriverMonitor, MonitoringConfig
    torch.set_num_threads(4)
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    monitors = {}
    for entry in manifest['models']:
        model_id = entry['model_id']
        if model_id in monitors:
            raise ValueError('Duplicate model ID')
        bundle = (manifest_path.parent / entry['bundle']).resolve()
        if not bundle.is_relative_to(manifest_path.parent):
            raise ValueError('Bundle path escapes package')
        if entry['loader'] == 'stage3':
            monitor = DriverMonitor.load(bundle, device)
        else:
            loaders = {'advanced': AdvancedClassifier, 'ensemble': EnsembleClassifier}
            classifier = loaders[entry['loader']](bundle, device)
            monitor = DriverMonitor(classifier, MonitoringConfig(**entry['policy']))
        monitors[model_id] = monitor
    if not monitors:
        raise ValueError('No models configured')
    return monitors


def decode_image(payload):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as im:
                if im.format not in ('JPEG', 'PNG') or getattr(im, 'n_frames', 1) != 1:
                    raise ValueError('Use a single-frame JPEG or PNG')
                if im.width * im.height > MAX_PIXELS:
                    raise ValueError('Image exceeds 12 megapixels')
                im.load()
                return ImageOps.exif_transpose(im).convert('RGB')
    except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise HTTPException(422, 'Invalid image: single-frame JPEG/PNG, at most 12 megapixels') from exc


async def image_body(request):
    if request.headers.get('content-type', '').split(';')[0] not in ('image/jpeg', 'image/png'):
        raise HTTPException(415, 'Send raw image/jpeg or image/png bytes, not multipart or JSON')
    if request.headers.get('content-encoding', 'identity') != 'identity':
        raise HTTPException(415, 'Compressed HTTP request bodies are not supported')
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > MAX_BYTES:
            raise HTTPException(413, 'Image body exceeds 4 MiB')
        data.extend(chunk)
    return bytes(data)


def create_app(manifest_path=None, *, api_key=None, monitor_factory=None, device=None):
    """Factory injection lets adapter tests use the real monitor with fake predictions."""
    @asynccontextmanager
    async def lifespan(app):
        key = api_key if api_key is not None else os.environ.get('SAFEDRIVE_API_KEY', '')
        if not key or not key.strip():
            raise RuntimeError('Set a nonempty SAFEDRIVE_API_KEY')
        app.state.api_key = key
        app.state.device = device or os.environ.get('SAFEDRIVE_DEVICE', 'cpu')
        if app.state.device not in ('cpu', 'mps'):
            raise RuntimeError('Supported devices: cpu, mps. This release does not support CUDA.')
        app.state.monitors = monitor_factory() if monitor_factory else load_monitors(
            manifest_path or os.environ.get('SAFEDRIVE_MANIFEST', 'models.json'), app.state.device)
        app.state.gate = threading.Lock()
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            app.state.monitors.clear()

    app = FastAPI(title='SafeDrive model comparison', version='1.0.0', lifespan=lifespan,
                  description='Experimental frame scoring and in-memory review queues. One process only.')
    header = APIKeyHeader(name='X-API-Key', auto_error=False)

    def authorize(request: Request, supplied: str | None = Depends(header)):
        expected = getattr(request.app.state, 'api_key', '')
        if supplied is None or not secrets.compare_digest(supplied.encode(), expected.encode()):
            raise HTTPException(401, 'Invalid API key')

    def get_monitor(model_id):
        if model_id not in app.state.monitors:
            raise HTTPException(404, 'Unknown model_id; see /v1/models')
        return app.state.monitors[model_id]

    @contextmanager
    def exclusive():
        if not app.state.gate.acquire(blocking=False):
            raise HTTPException(503, 'Service busy; retry without changing frame ID or timestamp',
                                headers={'Retry-After': '1'})
        try:
            yield
        finally:
            app.state.gate.release()

    def run_frame(model_id, payload, session_id=None, frame_id=None, timestamp_ms=None):
        monitor = get_monitor(model_id)
        with exclusive():
            image = decode_image(payload)
            if session_id is None:
                result = monitor.classifier.predict([image])[0]
            else:
                try:
                    result = asdict(monitor.process_frame(image, session_id=session_id,
                                      frame_id=frame_id, timestamp_ms=timestamp_ms))
                except BufferError as exc:
                    raise HTTPException(429, str(exc)) from exc
                except ValueError as exc:
                    raise HTTPException(409, str(exc)) from exc
            return {'model_id': model_id, **result}

    @app.get('/healthz')
    def health():
        if not getattr(app.state, 'ready', False):
            raise HTTPException(503, 'Not ready')
        return {'status': 'ready'}

    @app.get('/v1/models', dependencies=[Depends(authorize)])
    def models():
        return {'release': 'handover-v1', 'device': app.state.device, 'classes': CLASS_NAMES,
                'policy_status': 'Provisional engineering defaults; not validated operating points',
                'models': [{'model_id': key, 'model_version': m.classifier.metadata['model_version'],
                            'policy': asdict(m.config)} for key, m in app.state.monitors.items()]}

    @app.post('/v1/models/{model_id}/predict', dependencies=[Depends(authorize)],
              response_model=PredictionResult, openapi_extra=IMAGE_REQUEST)
    async def predict(model_id: str, request: Request):
        get_monitor(model_id)
        return await run_in_threadpool(run_frame, model_id, await image_body(request))

    @app.post('/v1/models/{model_id}/frames', dependencies=[Depends(authorize)],
              response_model=ModelFrameResult, openapi_extra=IMAGE_REQUEST)
    async def frame(model_id: str, request: Request,
                    session_id: str = Query(min_length=1, max_length=256),
                    frame_id: str = Query(min_length=1, max_length=256),
                    timestamp_ms: int = Query(ge=0, le=2**53-1)):
        get_monitor(model_id)
        if not session_id.strip() or not frame_id.strip():
            raise HTTPException(422, 'Session/frame IDs cannot be blank')
        return await run_in_threadpool(run_frame, model_id, await image_body(request),
                                      session_id, frame_id, timestamp_ms)

    @app.post('/v1/models/{model_id}/sessions/{session_id}/close', dependencies=[Depends(authorize)])
    def close(model_id: str, session_id: str):
        monitor = get_monitor(model_id)
        with exclusive():
            event = monitor.close_session(session_id)
            return {'model_id': model_id, 'review_event': asdict(event) if event else None}

    @app.get('/v1/models/{model_id}/queue', dependencies=[Depends(authorize)])
    def queue(model_id: str):
        monitor = get_monitor(model_id)
        with exclusive():
            return {'model_id': model_id, 'events': [asdict(e) for e in monitor.queue.snapshot()],
                    'evicted_closed': monitor.queue.evicted_closed}

    @app.post('/v1/models/{model_id}/queue/{event_id}/ack', dependencies=[Depends(authorize)])
    def acknowledge(model_id: str, event_id: str):
        monitor = get_monitor(model_id)
        with exclusive():
            try:
                removed = monitor.queue.acknowledge(event_id)
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            return {'model_id': model_id, 'removed': removed}

    return app
