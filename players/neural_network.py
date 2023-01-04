from lib.game import DotsAndBoxesGame
from lib.model import AZNeuralNetwork
from players.player import AIPlayer


class NeuralNetworkPlayer(AIPlayer):

    def __init__(self, model: AZNeuralNetwork, name: str):
        super().__init__(name=name)
        self.model = model
        self.model.eval()

    def determine_move(self, s: DotsAndBoxesGame) -> int:
        move = self.model.determine_move(s.get_canonical_s())
        return move
