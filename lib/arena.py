from typing import Tuple
from tqdm import tqdm

# local import
from lib.game import DotsAndBoxesGame
from lib.model import AZNeuralNetwork

class Arena: # TODO Evaluator
    """
    Evaluator. To ensure we always generate the best quality data, we evaluate each
    new neural network checkpoint against the current best network θ∗ f before using
    it for data generation. The neural network fθi is evaluated by the performance of
    an MCTS search αθi that uses fθi to evaluate leaf positions and prior probabilities
    (see Search algorithm). Each evaluation consists of 400 games, using an MCTS
    with 1,600 simulations to select each move, using an infinitesimal temperature
    τ→ 0 (that is, we deterministically select the move with maximum visit count, to
    give the strongest possible play). If the new player wins by a margin of > 55% (to
    avoid selecting on noise alone) then it becomes the best player αθ∗, and is subsequently
    used for self-play generation, and also becomes the baseline for subsequent
    comparisons.

    """
    def __init__(self,
                 game_size: int,
                 model1: AZNeuralNetwork,
                 model2: AZNeuralNetwork,
                 n_games: int):

        self.game_size = game_size
        self.model1 = model1
        self.model2 = model2
        self.n_games = n_games

    def compare(self) -> Tuple[int, int, int]:
        self.model1.eval()
        self.model2.eval()

        wins_model_1, wins_model_2, draws = 0, 0, 0

        for _ in tqdm(range(self.n_games)):
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
            lines = game.get_canonical_lines_vector()
            move = self.model1.determine_move(lines) if game.current_player == 1 else self.model2.determine_move(lines)

            assert move in game.get_valid_moves(), \
                print(f"<{move}> is not a valid move. Model should have selected a move in {game.get_valid_moves()}.")

            game.execute_move(move)

        return game.result
