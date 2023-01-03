from abc import ABC, abstractmethod
import numpy as np


class AIPlayer(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def determine_move(self, s: np.ndarray) -> int:
        """
        s can be assumed to be in canonical form.
        """
        pass
