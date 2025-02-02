from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Experiment:
    id: int
    name: str
    status: str
    user: int
    created_at: datetime
    started_at: Optional[datetime]
    recording: Optional[int]

    @classmethod
    def from_(cls, result: tuple):
        _id, _name, _status, _user, _created_at, _started_at, _recording = result
        return cls(
            id=_id,
            name=_name,
            status=_status,
            user=_user,
            created_at=_created_at,
            started_at=_started_at,
            recording=_recording
        )
