# local import
import torch

from .player import AIPlayer
from ..game import DotsAndBoxesGame
from ..model import AZNeuralNetwork


class NeuralNetworkPlayer(AIPlayer):

    def __init__(self, model: AZNeuralNetwork, name: str, device: torch.device):
        super().__init__(name=name)
        self.model = model

        # model inference
        model.eval()
        model.to(device)

    def determine_move(self, s: DotsAndBoxesGame) -> int:
        move = self.model.determine_move(s.get_canonical_s())
        return move
