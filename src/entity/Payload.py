from dataclasses import dataclass
from datetime import datetime


@dataclass
class Payload:
    id: int
    resource: str
    exp: datetime
