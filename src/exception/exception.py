"""Legacy exception name retained for import compatibility."""


class customexception(Exception):
    def __init__(self, error_message, error_details=None):
        super().__init__(str(error_message))
