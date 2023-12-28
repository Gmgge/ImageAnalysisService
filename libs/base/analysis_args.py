from argparse import Namespace
from abc import abstractmethod
from typing import Any


class AnalysisArgs(Namespace):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

