from players.player import AIPlayer
import random
import numpy as np


class AlphaBetaPlayer(AIPlayer):

    def __init__(self):
        super().__init__("RandomPlayer")

    def determine_move(self, s: np.ndarray) -> int:
        # TODO
        pass
