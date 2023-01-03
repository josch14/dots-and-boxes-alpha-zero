from players.player import AIPlayer
import random
import numpy as np


class RandomPlayer(AIPlayer):

    def __init__(self):
        super().__init__("RandomPlayer")

    def determine_move(self, s: np.ndarray) -> int:
        valid_moves = np.where(s == 0)[0].tolist()
        move = random.choice(valid_moves)
        return move
