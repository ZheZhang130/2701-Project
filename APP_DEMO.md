# SafeDrive dashboard demo

The existing `src/`, `bundles/`, model files and inference API are unchanged.

Three separate local processes:

- React + Vite frontend: http://127.0.0.1:5173
- FastAPI app backend: http://127.0.0.1:8000 (interactive API docs at `/docs`)
- Existing ML API: http://127.0.0.1:7860

## 1. Start the existing ML service

Use the original README's Docker setup (or its native Python 3.12 setup). Use the
same API key in the ML process and application backend. For a repeatable local demo,
you can choose a local key in each terminal:

```sh
export SAFEDRIVE_API_KEY=local-demo-key
```

For Docker, from this folder:

```sh
docker build -t safedrive-handover:v1 .
docker run --rm --name safedrive -p 127.0.0.1:7860:7860 \
  -e SAFEDRIVE_API_KEY safedrive-handover:v1
```

The original README has the alternative native startup instructions. The ML runtime
and weights are not dependencies of the application backend.

## Recommended: automatic shutdown

After installing the backend and frontend dependencies below, use this single
command from the repository root instead of starting the two app servers separately:

```sh
export SAFEDRIVE_API_KEY=local-demo-key
python3 run_demo.py
```

Open http://127.0.0.1:5173 within two minutes. The launcher starts FastAPI and Vite
and stops both 60 seconds after the last dashboard tab disconnects. Closing the
in-app dashboard or quitting Codex normally closes that connection. Closing only
the chat while leaving a dashboard open keeps the demo alive. An external browser
tab also keeps it alive. A lost network connection can take additional time to be
detected by the WebSocket transport before the 60-second idle timeout begins.

The connection uses WebSockets so background-tab timer throttling doesn't trigger
shutdown. If no dashboard opens, both servers stop after two minutes. Control+C
in the launcher terminal stops both immediately; no manual process IDs or kill
commands are needed. If either server exits, the launcher stops the other.
After automatic shutdown, run the launcher again to restart; saved data remains.

This manages only the React and application FastAPI servers it starts. The separate
ML service remains independently managed and must be stopped separately if running.
Individual startup commands below bypass automatic shutdown.

## 2. Start the application backend

From the repository root, in a second terminal:

```sh
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
export SAFEDRIVE_API_KEY=local-demo-key
backend/.venv/bin/uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Python 3.9 or newer works for this application backend. Optional environment settings:
`ML_API_URL` (default `http://127.0.0.1:7860`), `ML_MODEL` (default `resnet18`),
and `APP_DATA_DIR` (default `backend/data`). Run one backend worker so its simple
in-memory state and JSON snapshot stay consistent.

## 3. Start React

In a third terminal:

```sh
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173. Vite proxies `/api` and `/media` to the application
backend. The ML API key stays on the backend.

## Demo flow

1. Add two drivers with your own names. Select the first driver.
2. Upload JPEG/PNG cabin images, choose a video file, or grant webcam access.
   For image batches, select files in intended chronological order. For strict
   control of order, upload one at a time. Use cabin views matching model training.
3. The backend submits each image to `POST /v1/models/resnet18/predict` as raw
   bytes, with `X-API-Key` and the image media type. Requests are serialized.
4. Each classified frame appears in the selected driver's history with its class,
   confidence, server receipt time and expandable image. The complete ML JSON,
   including all probabilities and model version, is retained in the backend.
5. After 5 dangerous predictions within the last 10 classified frames, the driver
   gets a persistent tag and one unread notification. The dashboard polls every
   two seconds. Click a notification or tagged driver to view their history.
6. Clear the tag to restart that driver's evidence window. Old history and
   notifications remain. A later threshold crossing creates another notification.
7. Change the threshold/window at the bottom for a shorter demo; applying a rule
   resets all evidence windows and preserves existing tags and history.

Dangerous means any class c1–c9; c0 is safe driving. There is no second ML model,
confidence gating or time smoothing in this app. The deterministic rolling-window
rule counts frames, not seconds, and can trigger before the full window fills.
Safe frames gradually push old dangerous frames out of the window. Different drivers
have independent windows. Tags do not auto-clear. Counts reflect only successful ML
responses; ML failures show an error and stop the current input operation.

Video/webcam capture sends one frame, awaits inference, then waits the configured
interval. This intentionally simple replay samples a playing video; it does not
classify every video frame. Use image batches for repeatable input. Stop allows the
in-flight classification to finish. Notifications are in-app and persist locally;
no email, push service or OS notification setup is required.

State is saved in `backend/data/state.json`; retained JPEG/PNG frames are under
`backend/data/frames/`. Restarting preserves history and alerts. For a fresh demo,
stop the backend and move `backend/data` to another location before restarting.
This is a local video-demo application; the heuristic is not safety validated.

## Checks

```sh
python3 -m unittest discover -s backend -v
npm run build --prefix frontend
```

The backend test uses controlled ML responses to check independent driver state,
threshold crossing, one notification per tag, acknowledgement, history/image
retention, clearing, rule changes, and rolling-window expiration. The actual app
always calls the external ML API; there is no synthetic prediction mode in the UI.

## Imported demo drivers

Click **Import demo drivers**, or POST `/api/demo/import`. Three fictional drivers
have six distinct AI-generated cabin-camera frames each: safe driving, texting,
phone call, drinking, reaching behind, and texting again. Within each sequence,
the driver, clothing and vehicle stay consistent while their actions change.
Each profile has five dangerous samples, so the default 5-of-10 rule tags it.
With a stricter rule, a profile may remain untagged.

These samples are synthetic, manually annotated demo fixtures. No model confidence
or prediction is fabricated. Real uploads still call the external ML API.
Importing upgrades old repeated-photo fixtures while preserving real uploaded frames;
subsequent imports skip current fixtures. Source metadata is in
`backend/demo_assets/sources.json`; the generation prompt is in
`backend/demo_assets/GENERATION.md`.
