from lib.model import AZNeuralNetwork
from players.player import AIPlayer
import numpy as np


class NeuralNetworkPlayer(AIPlayer):

    def __init__(self, model: AZNeuralNetwork, name: str):
        super().__init__(name=name)
        self.model = model
        self.model.eval()

    def determine_move(self, s: np.ndarray) -> int:
        # it can be assumed that s is in canonical form
        move = self.model.determine_move(s)
        return move
