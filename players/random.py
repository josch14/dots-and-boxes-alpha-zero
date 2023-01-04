from lib.game import DotsAndBoxesGame
from players.player import AIPlayer
import random
import numpy as np


class RandomPlayer(AIPlayer):

    def __init__(self):
        super().__init__("RandomPlayer")

    def determine_move(self, s: DotsAndBoxesGame) -> int:
        move = random.choice(s.get_valid_moves())
        return move
