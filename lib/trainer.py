import numpy as np

from .constants import GameState, Value
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

        self.size = size

        # model that is to be trained, and opponent
        n_lines = 2*size*(size+1)
        self.model = AZNeuralNetwork(
            io_dim=n_lines,
            hidden_dim=n_lines
        )

    def train(self,
              num_epochs: int = 10,
              num_plays: int = 50,
              win_fraction: float = 0.6,
              temp_move_threshold: int = 3,
              n_simulations: int = 100):
        """
        Perform numEpochs epochs of training. 
        During each epoch, a specific number of games of self-play are performed, 
        resulting in a list training examples.
        The neural network is then trained with the training examples.
        Finally, the updated neural network competes against the previous network
        and is accepted only when it wins a specific fraction of games.
        """

        prev_model = copy.deepcopy(self.model)

        for i in range(1, num_epochs + 1):
            log.info(f"Progress: Epoch {i}/{num_epochs}")

            train_examples = []
            log.info(f"Performing self-play ..")

            for _ in tqdm(range(num_plays)):
                # single iteration of self-play
                train_examples += self.single_self_play(
                    n_simulations=n_simulations,
                    temp_move_threshold=temp_move_threshold
                )

            # TODO use training examples to train the model
    def single_self_play(self, n_simulations, temp_move_threshold):
        """
        Performs a single episode of self-play (until the game end).
        During playing, each turn results in a training example.
        """

        """
        For the first 30 moves of each game, the temperature is set to τ = 1; this
        selects moves proportionally to their visit count in MCTS, and ensures a diverse
        set of positions are encountered. For the remainder of the game, an infinitesimal
        temperature is used, τ→ 0.
        
        After MCTS has completed a round of simulations, and before it has chosen a move to play, 
        it has accumulated a visit count (N above) for each potential next move. 
        MCTS works such that good moves are eventually visited more often than bad moves, 
        and are a good indication of where to play.
        
        Normally, these counts are normalized and used as a distribution to choose the actual move. 
        But a so-called temperature parameter (τ) can be used to first exponentiate these counts: N^(1/τ). 
        They set τ=1 for the first 30 moves of the game (which has no effect) and then set it to an infinitesimal value 
        for the rest of the game (which, post-normalization, suppresses all values except the maximum).
        """

        # At each time-step t, an MCTS is executed using the previous iteration of the neural network
        # and a move is played by sampling the search probabilities
        game = DotsAndBoxesGame(self.size)
        n_moves = 0

        # dataset train, containing lists of [lines_vector, probs, v]
        train_examples = []

        # iteration over time stop t in game
        while True:
            game = game.copy() # TODO check when this is necessary
            mtcs = MCTS(
                model=self.model,
                game=game,
                n_simulations=n_simulations,
            )

            temp = 1 if n_moves < temp_move_threshold else 0
            n_moves += 1

            probs = mtcs.calculate_probabilities(temp=temp)
            # TODO Include Symmetries of current Game State

            train_examples.append([game.get_lines_vector(), probs, game.get_player_at_turn()])  # v is determined later

            # sample move from using probs, and apply move
            move = np.random.choice(
                a=list(range(game.N_LINES)),
                p=probs
            )
            game.draw_line(move)

            if not game.is_running():
                state = game.get_state()
                if state == GameState.DRAW:
                    winner = 0
                elif state == GameState.WIN_PLAYER_1:
                    winner = Value.PLAYER_1
                elif state == GameState.WIN_PLAYER_2:
                    winner = Value.PLAYER_2

                # v in {-1, 0, 1}, depending on whether the player lost, draw, or won
                for i, (_, _, player_at_turn) in enumerate(train_examples):

                    if player_at_turn == winner:
                        train_examples[i][2] = 1
                    elif winner == 0:
                        train_examples[i][2] = 0
                    else:
                        train_examples[i][2] = -1
                return train_examples
