from typing import Tuple
from sys import stdout
from tqdm import tqdm

# local import
from lib.game import DotsAndBoxesGame
from lib.model import AZNeuralNetwork


class Evaluator:
    """
    Let two neural networks play games of Dots and Boxes against each other. This is used after model training: with
    the evaluator it is determined whether the updated neural network improved by comparing it against the current best
    network.

    Attributes
    ----------
    game_size : int
        board size (width & height) of a Dots and Boxes game
    model1, model2 : AZNeuralNetwork, AZNeuralNetwork
        neural networks that are compared against each other
    n_games : int
        number of games the models play against each other
    """
    def __init__(self, game_size: int, model1: AZNeuralNetwork, model2: AZNeuralNetwork, n_games: int):
        assert game_size == model1.game_size and game_size == model2.game_size

        self.game_size = game_size
        self.model1 = model1
        self.model2 = model2
        self.n_games = n_games

    def compare(self) -> Tuple[int, int, int]:
        self.model1.eval()
        self.model2.eval()

        wins_model_1, wins_model_2, draws = 0, 0, 0

        for _ in tqdm(range(self.n_games), file=stdout):
            result = self.play_game()

            if result == 1:
                wins_model_1 += 1
            elif result == -1:
                wins_model_2 += 1
            else:
                draws += 1

        return wins_model_1, wins_model_2, draws

    def play_game(self) -> int:
        game = DotsAndBoxesGame(self.game_size)

        while game.is_running():
            s = game.get_canonical_s()
            move = self.model1.determine_move(s) if game.current_player == 1 else self.model2.determine_move(s)
            game.execute_move(move)

        return game.result
