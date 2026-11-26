# Model handoff contract, version 1.0 — Stage 3 prototype

Yige owns preprocessing, classification, monitoring state, event generation, testing,
and the artifact bundle. Zhe owns transport/API, durable event storage, image retention,
UI, deployment, access control, and process/concurrency management. Teammates assist
with acceptance testing and presentation after integration.

## Public interface

```python
from dataclasses import asdict
from safedrive.monitoring import DriverMonitor

monitor = DriverMonitor.load('artifacts/monitoring/stage3/bundle', device='mps')
result = monitor.process_frame(rgb_pil_image, session_id='trip-001',
                               frame_id='camera-frame-001', timestamp_ms=0)
payload = asdict(result)  # JSON serializable; optional review_event is nested
# Persist/upsert payload['review_event'] by event_id when non-null.
# Each later frame uses a unique frame ID and an increasing timestamp.
closed_event = monitor.close_session('trip-001')  # persist this too when non-null
queue_items = [asdict(event) for event in monitor.queue.snapshot()]
```

Use `device='cpu'` explicitly for CPU deployment. Loading uses local weights and
never downloads pretrained weights or reads training data. Install the package with
`pip install -e '.[train]'` for the current pinned inference dependencies (the extra
also includes training). The Python package and bundle are both required; weights
alone do not implement monitoring. The bundle loader verifies hashes of weights,
classifier metadata, and policy configuration. Hashes check integrity, not authenticity.
Source hashes in monitoring_bundle.json record build provenance, not runtime enforcement.

Input must be a nonempty PIL RGB image in the expected camera orientation. Session
and frame IDs are nonblank strings of at most 256 characters. Timestamps are integers
from 0 through 2^53−1 in milliseconds, strictly increasing per session. Frame IDs
must be unique per session. Invalid inputs raise ValueError before inference/state
mutation. Invalid predictions do not advance state either. Capacity exhaustion raises
BufferError; release capacity, then retry the same frame if desired. Calls must be
serialized across the monitor instance; there is no thread-safety or restart persistence
promise. Route each session to one owner. A new session must start after a restart.

`FrameResult` and `ReviewEvent` are frozen dataclasses in `safedrive.contracts`.
The result includes ten probabilities in c0..c9 order, argmax class, model and policy
versions, `frame_score = 1 − P(c0)`, smoothed score, uncertainty, and an optional event.
The score estimates non-safe-class membership; it is not a crash probability or a
calibrated risk estimate. `uncertain` means max class probability is below 0.60.
It neither opens a separate alert nor suppresses an otherwise sustained high score.
This confidence heuristic cannot reliably detect unfamiliar inputs.

## Exact policy semantics

Policy defaults are stored in `configs/monitoring.json` and copied into the bundle.
They are **provisional mechanics settings, not validated safety thresholds**.
The high threshold 0.97 is rounded from the selected model's same-validation-frame
5% FPR diagnostic (~0.96924); that frame statistic does not validate this temporal policy.
Other values are engineering starting points and have not been optimized on video.

| Setting | Default | Meaning |
|---|---:|---|
| High threshold | 0.97 | Smoothed score must be at least this high |
| Sustained high | 2,000 ms | Time from first qualifying observation to opening |
| Low threshold | 0.80 | Smoothed score must be at most this low |
| Sustained low | 1,000 ms | Time from first low observation to recovery closure |
| EMA time constant | 500 ms | Alpha = 1 − exp(−elapsed_ms / tau_ms) |
| Maximum gap | 500 ms | Larger gaps break all continuity |
| Cooldown | 2,000 ms | Earliest start of fresh high-duration evidence |
| Representatives | 5 | Highest raw-score frame references per episode |
| Queue capacity | 128 | Latest episode snapshots |
| Session capacity | 32 | Concurrent session states |
| Frames per session | 100,000 | Bounds exact frame-ID uniqueness tracking |

The first frame initializes the EMA. Later frames update it using elapsed time and
the incoming score. High-duration evidence resets when below high; low-duration
evidence resets when above low. Equality satisfies the respective threshold and
duration boundary. Values between low and high keep an active episode open.
Durations describe consecutive sampled observations; unseen behavior between samples
is unknown. Sampling cadence therefore matters even with time-aware smoothing.

A gap strictly greater than maximum gap closes an active episode at the **last
observed timestamp**, resets smoothing to the current raw score, clears trigger
history, and starts cooldown from the current timestamp. This rule also applies
when no episode is active. Exactly maximum gap is permitted. Missing time cannot
create sustained evidence. Cooldown starts at recovery closure or a gap; high
evidence does not accumulate until cooldown expires. Predictions continue throughout.

Events use a unique per-instance UUID plus sequence number. `opened`, `updated`,
and `closed` notifications share one event ID, allowing one queue item per episode.
`start_timestamp_ms` is the first qualifying high observation; `triggered_timestamp_ms`
is when the duration was met. `last_timestamp_ms` is the last observed episode frame;
`end_timestamp_ms` is populated on closure. Closure reason is recovery, gap, or
session_closed. Peak score is the maximum **raw frame score**, not the EMA.
Representative IDs include qualifying pre-trigger frames and subsequent episode
frames, sorted by raw score descending, then earliest timestamp and ID for ties.
Only the top five survive. No images or paths are stored inside the monitor.

`close_session` closes any active episode at the last observed timestamp and releases
all session state, including the frame-ID set. Unknown sessions return None. Closing
an inactive session still releases state. A reused session ID starts fresh and receives
new event IDs; distinct trip IDs are preferable for external frame storage.

## Queue and failure behavior

`monitor.queue.snapshot()` returns immutable latest event snapshots in insertion order.
Updates upsert existing items. When a new episode needs a full queue, the oldest closed
item is evicted and `evicted_closed` increments. Active items are never evicted. If all
items are active, opening raises BufferError without consuming the frame or changing
monitoring state. The owner must close an ended session or otherwise resolve capacity.

After persisting a closed item, `queue.acknowledge(event_id)` removes it. It returns
False for an unknown ID and rejects active items. The local queue is bounded working
memory, **not an audit log**. Zhe must consume every returned event (including explicit
session-close events) into durable storage to avoid losing closed history on eviction
or restart. Queue APIs are owner-only; do not externally inject event snapshots.

At the session/frame limits, calls fail explicitly instead of silently forgetting
state. At 5 FPS, 100,000 frames allow about 5.6 hours per session; close/roll over sessions
before the limit. IDs are bounded to limit deduplication memory. For long-lived traffic,
Zhe must expire abandoned sessions and persist their closing events. On replay errors,
the CLI exits with partial output: it is a diagnostic tool, not a transactional service.

Frame references require external image retention. A top-scoring early frame may remain
a representative throughout a long episode. Copy referenced frames into retained episode
storage as event updates arrive; a short rolling buffer alone is insufficient. Pre-trigger
retention must also cover smoothing/trigger delay. Zhe owns access and retention policy.

## Validation and remaining acceptance

Stage 3 tests controlled timing, spikes, sustained highs, recovery, gaps, cooldown,
interleaved sessions, capacity errors, invalid inputs, session reuse, and bounded queue
behavior. MPS integration replays all 3,470 validation images with invented timestamps
and compares every prediction with the selected model export. A deliberately constructed
safe–high–safe illustration checks opening and recovery through actual image inference.
See `reports/stage3_results.md` and machine-readable `reports/stage3_verification.json`.

Stage 2 already checked preprocessing geometry, export parity, CPU/MPS compatibility,
clean/dark/blurred frames, and driver-specific errors. Those limitations still apply.
The final five drivers remain untouched for performance evaluation. No current evidence
establishes real-video alert precision, event recall, missed-event duration, or false
alerts per hour. Genuine labeled sequences from separate drivers/cameras are needed
for those measurements and for choosing temporal settings.

Remaining before a final handoff: freeze the bundle/policy and acceptance criteria,
perform the reserved final evaluation once, test installation in a clean deployment
environment, and test Zhe's durable queue, image retention, restart, routing, and load
behavior. The working prototype is ready for that integration; it is not a validated
real-driving safety system.
