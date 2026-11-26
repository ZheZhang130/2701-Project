"""Bounded, in-process monitoring. No image storage or real-time safety guarantee."""
from collections import OrderedDict
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4

from PIL.Image import Image
from .contracts import FrameResult, ReviewEvent


@dataclass(frozen=True)
class MonitoringConfig:
    policy_version: str = 'prototype-1'
    high_threshold: float = .97
    low_threshold: float = .80
    high_duration_ms: int = 2000
    low_duration_ms: int = 1000
    smoothing_tau_ms: int = 500
    max_gap_ms: int = 500
    cooldown_ms: int = 2000
    uncertainty_threshold: float = .60
    representative_limit: int = 5
    queue_capacity: int = 128
    max_sessions: int = 32
    max_frames_per_session: int = 100000

    def __post_init__(self):
        if not isinstance(self.policy_version, str) or not self.policy_version.strip():
            raise ValueError('policy_version required')
        for name in ('high_threshold', 'low_threshold', 'uncertainty_threshold'):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1:
                raise ValueError(f'Invalid {name}')
        if not self.low_threshold < self.high_threshold:
            raise ValueError('low_threshold must be less than high_threshold')
        for name in ('high_duration_ms', 'low_duration_ms', 'smoothing_tau_ms', 'max_gap_ms',
                     'cooldown_ms', 'representative_limit', 'queue_capacity', 'max_sessions',
                     'max_frames_per_session'):
            v = getattr(self, name)
            if type(v) is not int or v < (0 if name == 'cooldown_ms' else 1):
                raise ValueError(f'Invalid {name}')


class ReviewQueue:
    """Latest snapshot per episode. Evicts oldest closed item; never an active one."""
    def __init__(self, capacity):
        self.capacity = capacity
        self._items = OrderedDict()
        self.evicted_closed = 0

    def upsert(self, event):
        if event.event_id not in self._items and len(self._items) >= self.capacity:
            victim = next((k for k, v in self._items.items() if v.status == 'closed'), None)
            if victim is None:
                raise BufferError('Review queue full of active episodes; persist/close sessions before retrying')
            del self._items[victim]
            self.evicted_closed += 1
        self._items[event.event_id] = event

    def snapshot(self):
        return tuple(self._items.values())

    def acknowledge(self, event_id):
        """Remove a persisted closed item. Active items remain until they close."""
        event = self._items.get(event_id)
        if event is None:
            return False
        if event.status != 'closed':
            raise ValueError('Cannot acknowledge an active episode')
        del self._items[event_id]
        return True


@dataclass
class _Session:
    timestamp: int | None = None
    smooth: float = 0.
    high_since: int | None = None
    low_since: int | None = None
    cooldown_until: int = 0
    refs: tuple = ()  # (raw score, timestamp, frame ID), descending quality
    event: ReviewEvent | None = None
    seen: set | None = None


class DriverMonitor:
    """Call serially; one instance owns independent states and an in-memory queue."""
    def __init__(self, classifier, config=None):
        self.classifier = classifier
        self.config = config or MonitoringConfig()
        self.queue = ReviewQueue(self.config.queue_capacity)
        self._sessions = {}
        self._instance_id = uuid4().hex
        self._event_number = 0

    @classmethod
    def load(cls, bundle_path, device='mps'):
        from .experiments import CandidateClassifier
        bundle = Path(bundle_path)
        metadata = json.loads((bundle / 'monitoring_bundle.json').read_text())
        if metadata['schema_version'] != 1:
            raise ValueError('Unsupported monitoring bundle')
        for name in ('model.pt', 'bundle.json', 'monitoring.json'):
            if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != metadata['sha256'][name]:
                raise ValueError(f'Monitoring bundle checksum mismatch: {name}')
        classifier = CandidateClassifier(bundle, device)
        if classifier.metadata['candidate']['objective'] != 'multiclass':
            raise ValueError('Monitoring requires ten-class probabilities')
        return cls(classifier, MonitoringConfig(**json.loads((bundle / 'monitoring.json').read_text())))

    def _validate_input(self, image, session_id, frame_id, timestamp_ms):
        if not isinstance(image, Image) or image.mode != 'RGB' or min(image.size) <= 0:
            raise ValueError('Expected nonempty PIL RGB image')
        for name, value in (('session_id', session_id), ('frame_id', frame_id)):
            if not isinstance(value, str) or not value.strip() or len(value) > 256:
                raise ValueError(f'{name} must contain 1–256 characters')
        if type(timestamp_ms) is not int or not 0 <= timestamp_ms <= 2**53 - 1:
            raise ValueError('timestamp_ms must be a nonnegative JSON-safe integer')
        state = self._sessions.get(session_id)
        if state is None:
            if len(self._sessions) >= self.config.max_sessions:
                raise BufferError('Session capacity reached; close unused sessions')
            return _Session(seen=set())
        if timestamp_ms <= state.timestamp or frame_id in state.seen:
            raise ValueError('Timestamps must increase and frame IDs must be unique per session')
        if len(state.seen) >= self.config.max_frames_per_session:
            raise BufferError('Session frame capacity reached; close and start a new session')
        return state

    def _refs(self, refs, score, timestamp, frame_id):
        return tuple(sorted((*refs, (score, timestamp, frame_id)),
                            key=lambda r: (-r[0], r[1], r[2]))[:self.config.representative_limit])

    @staticmethod
    def _close(state, reason):
        return replace(state.event, status='closed', end_timestamp_ms=state.timestamp,
                       closure_reason=reason)

    def process_frame(self, image, *, session_id, frame_id, timestamp_ms):
        original = self._validate_input(image, session_id, frame_id, timestamp_ms)
        prediction = self.classifier.predict([image])[0]
        try:
            p = tuple(float(x) for x in prediction['probabilities'])
            version = prediction['model_version']
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Invalid classifier output') from exc
        if (len(p) != 10 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in p)
                or abs(sum(p) - 1) > 1e-6 or not isinstance(version, str) or not version):
            raise ValueError('Invalid classifier output')
        score = 1 - p[0]
        c = self.config
        state = replace(original)  # seen remains shared but is mutated only after successful commit
        event = None
        opening = False
        gap = state.timestamp is not None and timestamp_ms - state.timestamp > c.max_gap_ms
        if gap:
            if state.event is not None:
                event = self._close(state, 'gap')  # last observed frame, not the missing interval
            state = _Session(seen=original.seen, cooldown_until=timestamp_ms + c.cooldown_ms)
        if state.timestamp is None:
            state.smooth = score
        else:
            alpha = -math.expm1(-(timestamp_ms - state.timestamp) / c.smoothing_tau_ms)
            state.smooth += alpha * (score - state.smooth)
        state.timestamp = timestamp_ms
        if state.event is None:
            if state.smooth >= c.high_threshold and timestamp_ms >= state.cooldown_until:
                if state.high_since is None:
                    state.high_since = timestamp_ms
                state.refs = self._refs(state.refs, score, timestamp_ms, frame_id)
                if timestamp_ms - state.high_since >= c.high_duration_ms:
                    opening = True
                    event = ReviewEvent(
                        event_id=f'{self._instance_id}-{self._event_number + 1}', status='opened',
                        session_id=session_id, start_timestamp_ms=state.high_since, end_timestamp_ms=None,
                        peak_score=max(r[0] for r in state.refs),
                        representative_frame_ids=tuple(r[2] for r in state.refs),
                        triggered_timestamp_ms=timestamp_ms, last_timestamp_ms=timestamp_ms,
                        model_version=version, policy_version=c.policy_version)
                    state.event = event
            else:
                state.high_since = None
                state.refs = ()
        else:
            state.refs = self._refs(state.refs, score, timestamp_ms, frame_id)
            state.event = replace(state.event, status='updated', last_timestamp_ms=timestamp_ms,
                                  peak_score=max(state.event.peak_score, score),
                                  representative_frame_ids=tuple(r[2] for r in state.refs))
            if state.smooth <= c.low_threshold:
                if state.low_since is None:
                    state.low_since = timestamp_ms
                if timestamp_ms - state.low_since >= c.low_duration_ms:
                    event = self._close(state, 'recovery')
                    state.event = None
                    state.high_since = state.low_since = None
                    state.refs = ()
                    state.cooldown_until = timestamp_ms + c.cooldown_ms
            else:
                state.low_since = None
            if event is None:
                event = state.event
        result = FrameResult(
            schema_version='1.0', model_version=version, policy_version=c.policy_version,
            session_id=session_id, frame_id=frame_id, timestamp_ms=timestamp_ms,
            predicted_class=max(range(10), key=p.__getitem__), probabilities=p,
            frame_score=score, smoothed_score=state.smooth,
            uncertain=max(p) < c.uncertainty_threshold, review_event=event)
        if event is not None:
            self.queue.upsert(event)  # capacity errors must leave session and IDs unchanged
        if opening:
            self._event_number += 1
        state.seen.add(frame_id)
        self._sessions[session_id] = state
        return result

    def close_session(self, session_id):
        state = self._sessions.pop(session_id, None)
        if state is None or state.event is None:
            return None
        event = self._close(state, 'session_closed')
        self.queue.upsert(event)
        return event
