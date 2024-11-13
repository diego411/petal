from datetime import datetime
from dataclasses import dataclass
from src.entity.RecordingState import RecordingState
from typing import Optional


@dataclass
class Recording:
    id: int
    name: str
    user: int
    state: RecordingState
    sample_rate: int
    threshold: int
    start_time: Optional[datetime]
    last_update: Optional[datetime]

    @classmethod
    def from_(cls, result: tuple):
        _id, _name, _user, _state, _sample_rate, _threshold, _start_time, _last_update = result
        return cls(
            id=_id,
            name=_name,
            user=_user,
            state=RecordingState(_state),
            sample_rate=_sample_rate,
            threshold=_threshold,
            start_time=_start_time,
            last_update=_last_update
        )
