"""Serializable deployment-facing records for the monitoring prototype."""
from dataclasses import dataclass
from typing import Literal, Protocol
from PIL.Image import Image


@dataclass(frozen=True)
class ReviewEvent:
    event_id: str
    status: Literal['opened', 'updated', 'closed']
    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int | None
    peak_score: float
    representative_frame_ids: tuple[str, ...]
    triggered_timestamp_ms: int
    last_timestamp_ms: int
    model_version: str
    policy_version: str
    closure_reason: Literal['recovery', 'gap', 'session_closed'] | None = None


@dataclass(frozen=True)
class FrameResult:
    schema_version: str
    model_version: str
    policy_version: str
    session_id: str
    frame_id: str
    timestamp_ms: int
    predicted_class: int
    probabilities: tuple[float, ...]
    frame_score: float
    smoothed_score: float
    uncertain: bool
    review_event: ReviewEvent | None


class DriverMonitorProtocol(Protocol):
    @classmethod
    def load(cls, bundle_path: str, device: str = 'mps') -> 'DriverMonitorProtocol': ...

    def process_frame(self, image: Image, *, session_id: str, frame_id: str,
                      timestamp_ms: int) -> FrameResult: ...

    def close_session(self, session_id: str) -> ReviewEvent | None: ...
