"""
This is the base recognizer class that is handed to the recognition methods
"""

from abc import ABC
from audalign.config import BaseConfig


class BaseRecognizer(ABC):
    config: BaseConfig

    def __init__(self, config: BaseConfig = None):
        """takes a config object"""
        pass

    def recognize(self, target_file: str, against: str) -> dict:
        """this recognizes"""
        pass
