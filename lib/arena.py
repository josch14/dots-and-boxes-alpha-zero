from typing import Tuple
from tqdm import tqdm

# local import
from lib.game import DotsAndBoxesGame
from lib.model import AZNeuralNetwork

class Arena:

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
            lines = game.lines_vector
            valid_moves = game.get_valid_moves()
            move = self.model1.determine_move(lines, valid_moves) if game.player_at_turn == 1 \
                else self.model2.determine_move(lines, valid_moves)

            assert move in valid_moves, \
                print(f"<{move}> is not a valid move. Model should have selected a move in {valid_moves}.")

            game.execute_move(move)

        return game.result
