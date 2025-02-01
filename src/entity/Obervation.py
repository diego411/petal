from dataclasses import dataclass
from datetime import datetime


@dataclass
class Observation:
    id: int
    label: str
    observed_at: datetime
    experiment: int

    @classmethod
    def from_(cls, result: tuple):
        _id, _label, _observed_at, _experiment = result
        return cls(
            id=_id,
            label=_label,
            observed_at=_observed_at,
            experiment=_experiment
        )
