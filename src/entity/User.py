from dataclasses import dataclass


@dataclass
class User:
    id: int
    name: str

    @classmethod
    def from_(cls, result: tuple):
        _id, _name = result
        return cls(id=_id, name=_name)
