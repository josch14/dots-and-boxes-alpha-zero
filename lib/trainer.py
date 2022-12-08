from .game import DotsAndBoxesGame
from .model import AZNeuralNetwork
from .mcts import MCTS

import copy
import logging as log
from tqdm import tqdm

class Trainer():
    """
    Execute self-play + learning. 
    """

    def __init__(self, size: int = 3):
        # model
        self.game = DotsAndBoxesGame(size)

        # model that is to be trained, and opponent
        self.model = AZNeuralNetwork(
            io_dim=self.game.N_LINES, 
            hidden_dim=self.game.N_LINES
        )


    def train(
        self,
        num_epochs: int,
        num_plays: int,
        win_fraction: float
        ):
        """
        Perform numEpochs epochs of training. 
        During each epoch, a specific number of games of self-play are performed, 
        resulting in a list training examples.
        The neural network is then trained with the training examples.
        Finally, the updated neural network competes against the previous network
        and is accepted only when it wins a specific fraction of games.
        """

        prev_model = copy.deepcopy(self.model)

        for i in range(1, num_epochs+1):
            log.info(f"Progress: Epoch {i}/{num_epochs}")

            train_examples = []
            log.info(f"Performing self-play ..")


            for _ in tqdm(range(num_plays)):
                # single iteration of self-play
                training_examples = self.perform_self_play()


    def perform_self_play(self):
        """
        Performs a single episode of self-play (until the game end).
        During playing, each turn results in a training example.
        """
        mtcs = MCTS(
            model=self.model,
            game=self.game
        )