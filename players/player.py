from abc import ABC, abstractmethod
from lib.game import DotsAndBoxesGame

class AIPlayer(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def determine_move(self, s: DotsAndBoxesGame) -> int:
        pass
