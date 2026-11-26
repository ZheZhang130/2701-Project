# SafeDrive — three-model handover v1

This package is the inference and monitoring component for Zhe's deployment work.
It contains three frozen experimental candidates, their local weights, preprocessing,
policy configurations, HTTP adapter and tests. No dataset images are included.
The synthetic example image checks connectivity only; it cannot evaluate accuracy.

| API model ID | Candidate | Exported weights |
|---|---|---:|
| `resnet18` | Existing Stage 3 ResNet18-320 | 42.7 MiB |
| `resnet18_ensemble` | Equal probability average of the two original ResNet18 seeds | 85.4 MiB |
| `resnet50` | Primary screened ResNet50-320, seed3701 | 90.0 MiB |

The ResNet50 second seed is deliberately not substituted. None of these comparison
results validates real-driving safety or real-video alert performance. The existing
Stage 3 baseline bundle is copied unchanged. See included model-comparison reports.

## GitHub source checkout

The Git repository includes application code, model metadata, tests, documentation,
and the current synthetic demo assets. It excludes installed dependencies, runtime
driver data, secrets, unrelated homework, and trained model weights.

For the React/FastAPI dashboard setup, see [APP_DEMO.md](APP_DEMO.md).
Before starting the ML API or building its Docker image from a Git checkout,
restore these unchanged weight files from the original handover package:

- `bundles/resnet18/model.pt`
- `bundles/resnet18_ensemble/member0/model.pt`
- `bundles/resnet18_ensemble/member1/model.pt`
- `bundles/resnet50/model.pt`

Keep those files outside Git. The model metadata contains their expected checksums.
The original `SHA256SUMS` describes the handover package, so its full verification
requires the excluded original files too. Imported dashboard demo profiles can be
viewed without starting the ML service, but new image classification requires it.

## Start with Docker

Run these commands from the extracted package directory. Docker builds require
network access to install the pinned runtime; inference needs no downloads.

```sh
docker build -t safedrive-handover:v1 .
export SAFEDRIVE_API_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
docker run --rm --name safedrive -p 127.0.0.1:7860:7860 \
  -e SAFEDRIVE_API_KEY safedrive-handover:v1
```

Keep the key in the caller's environment or the hosting service's secret settings.
`GET /healthz` is public and only reports readiness. Other model endpoints require
`X-API-Key`. `/docs` and `/openapi.json` describe the API. For cloud access, Zhe owns
HTTPS, network exposure, secret management and access control. A browser frontend
should call Zhe's backend rather than embed the shared service key in public code.
No permissive CORS policy is enabled.

For Hugging Face Spaces, choose the Docker SDK, expose port 7860, and set
`SAFEDRIVE_API_KEY` as a Space secret. Zhe still needs external durable storage for
events and retained frames. Do not treat the container filesystem as the audit log.

The container is CPU-only. Native macOS execution can opt into `SAFEDRIVE_DEVICE=mps`;
Linux containers cannot use the Mac's MPS GPU. CUDA is not supported in this release.
Benchmark the actual deployment host; prior M1 Max timings do not predict cloud CPU speed.

For native Python 3.12:

```sh
python3 -m venv .venv
.venv/bin/pip install torch==2.14.0 torchvision==0.29.0
.venv/bin/pip install -r requirements-api.lock.txt
.venv/bin/pip install --no-deps --no-build-isolation .
export SAFEDRIVE_MANIFEST="$PWD/models.json"
.venv/bin/uvicorn safedrive.api:create_app --factory --host 127.0.0.1 --port 7860 --workers 1 --limit-concurrency 16
```

Set `SAFEDRIVE_API_KEY` before native startup too. A missing key or invalid model
checksum prevents readiness. `models.json` paths are resolved relative to that file.

## Make requests

```sh
curl -H "X-API-Key: $SAFEDRIVE_API_KEY" http://127.0.0.1:7860/v1/models
python3 examples/send_frame.py examples/synthetic.jpg --model resnet18 --stateless
python3 examples/send_frame.py examples/synthetic.jpg --model resnet18 --session trip-1 --frame frame-0 --timestamp-ms 0
```

Send raw JPEG/PNG bytes with `Content-Type: image/jpeg` or `image/png` (not multipart).
The service limits encoded images to 4 MiB and decoded images to 12 megapixels,
applies EXIF orientation and converts to RGB before the model's own letterbox and
normalization. Input must match the expected cabin camera view and orientation.

| Route | Purpose |
|---|---|
| `GET /v1/models` | IDs, class order, versions, device and exact monitoring policies |
| `POST /v1/models/{model_id}/predict` | Stateless ten-class prediction and raw frame score |
| `POST /v1/models/{model_id}/frames?session_id=...&frame_id=...&timestamp_ms=...` | Stateful monitoring result and optional review event |
| `POST /v1/models/{model_id}/sessions/{session_id}/close` | End a trip, release state, return its closing event if active |
| `GET /v1/models/{model_id}/queue` | Latest episode snapshots and closed-item eviction count |
| `POST /v1/models/{model_id}/queue/{event_id}/ack` | Remove a closed event after external persistence |

Frame responses add `model_id` to the existing FrameResult contract. They contain
the ten probabilities in c0–c9 order, class, raw `frame_score = 1-P(c0)`, smoothed
score, uncertainty flag, versions and optional `review_event`. The score is not a
crash probability. See `docs/model_contract.md` for exact policy/event semantics.

An event's opened/updated/closed notifications share an event ID. Zhe must persist
every returned event, including session-close responses, and retain its referenced
frames. Queue polling is an aid, not a durable audit log. The monitor stores IDs,
not images. Close inactive trips explicitly to release bounded session capacity.

## Compare models correctly

Replay the same labeled frames at the same timestamps through each model ID, in
timestamp order. Every model has its own monitor and queue; the same trip/frame IDs
can be reused across different models. Await each request before sending the next.
Do not choose a model from unlabelled demo output alone.

All three policies initially use high=.97, low=.80, 2s high duration, 1s recovery,
500ms EMA, 500ms maximum frame gap and 2s cooldown. These are explicitly provisional
engineering settings, not calibrated equivalent operating points. Ensemble and
ResNet50 policies have distinct version IDs. No cross-validation fold threshold
has been substituted into a different full-screen checkpoint.

Record both model and policy versions. Compare frame false positives and recall,
then on genuinely labeled continuous video compare event recall, review precision,
false alerts/hour, delay, representative-frame usefulness, latency and memory.
Changing thresholds requires a new policy/release and a separate evaluation record.
The final five-driver holdout remains reserved; this package does not run it.

## Process, capacity and failure contract

Run exactly ONE worker and ONE replica for this prototype. The adapter serializes
model/state operations with a nonblocking lock. It does not provide distributed
session routing or durable state; multiple workers would create incompatible copies.
Future scaling requires explicit session ownership and persistence design.

| HTTP status | Action |
|---|---|
| 401 | Supply the correct API key |
| 404 | Check model ID |
| 409 | Resolve duplicate/out-of-order frames, or close an active episode before acknowledgment |
| 413 / 415 / 422 | Correct image size, media type or input values |
| 429 | Close/release sessions or resolve full active queue capacity; blind retry is insufficient |
| 503 | Busy before processing; retry this same frame after backoff |

Each model allows 32 active sessions, 100,000 unique frames per session and a queue
of 128 episodes. Old closed queue entries can be evicted; active entries cannot.
Restart discards all session/queue state. Persist events and close trips before a
planned restart, then use new session IDs. A network timeout has an uncertain outcome:
the server may have consumed the frame. Retrying may return 409; the service does
not promise exactly-once HTTP delivery. Reconcile external records and queue snapshots.

## Verify and hand over

```sh
python3 -m unittest discover -s tests -v
```

These tests exercise the real monitoring logic with deterministic fake predictions,
including cross-model isolation, episode lifecycle, invalid input, overload and
capacity behavior. `verification.json` records separate checks against real weights
and validation images; the images themselves are not shipped. `SHA256SUMS` covers
the package contents. Verify it after extraction with `shasum -a 256 -c SHA256SUMS`.

Release validation used native macOS CPU and a clean Linux ARM64 CPU container.
The container's 28 bundled tests passed, as did 60 repository tests. All three
models passed real HTTP prediction parity on 12 development images each and a
65-frame synthetic episode replay each, including opening, recovery and acknowledgment.
This is not a Linux x86-64, CUDA, multi-user load or real-video qualification.
Run startup and acceptance checks on the actual deployment host before a demo.

Yige owns the model package, preprocessing, monitoring semantics and model tests.
Zhe owns cloud deployment, durable event/image storage, frontend integration,
request ordering, uptime, shutdown/restart handling and host-level load testing.
Other teammates can run labeled acceptance scenarios and document presentation results.
