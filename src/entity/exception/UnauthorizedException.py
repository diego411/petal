class UnauthorizedException(Exception):
    def __init__(self, origin: str, message: str):
        self.origin = origin
        self.message = message
