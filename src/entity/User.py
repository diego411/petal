from dataclasses import dataclass


@dataclass
class User:
    id: int
    name: str
    password_hash: str

    @classmethod
    def from_(cls, result: tuple):
        _id, _name, _password_hash = result
        return cls(id=_id, name=_name, password_hash=_password_hash)
