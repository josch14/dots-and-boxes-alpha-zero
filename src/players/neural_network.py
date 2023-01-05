# local import
from .player import AIPlayer
from ..game import DotsAndBoxesGame
from ..model import AZNeuralNetwork


class NeuralNetworkPlayer(AIPlayer):

    def __init__(self, model: AZNeuralNetwork, name: str):
        super().__init__(name=name)
        self.model = model
        self.model.eval()

    def determine_move(self, s: DotsAndBoxesGame) -> int:
        move = self.model.determine_move(s.get_canonical_s())
        return move
