from tqdm import tqdm

from lib.constants import GameResult, Value, GameState
from lib.game import DotsAndBoxesGame
from lib.model import AZNeuralNetwork


class Arena:

    def __init__(self,
                 game_size: int,
                 model1: AZNeuralNetwork,
                 model2: AZNeuralNetwork,
                 n_games: int,
                 win_fraction: float):
        """
        TODO Class Description
        """
        self.game_size = game_size
        self.model1 = model1
        self.model1.eval()
        self.model2 = model2
        self.model2.eval()
        self.n_games = n_games
        self.win_fraction = win_fraction

    def compare(self):

        wins_model1, wins_model2, draws = 0, 0, 0

        for _ in tqdm(range(self.n_games)):
            game_result = self.play_game()
            if game_result == GameResult.WIN_PLAYER_1:
                wins_model1 += 1
            elif game_result == GameResult.WIN_PLAYER_2:
                wins_model2 += 1
            else:
                draws += 1

        return wins_model1, wins_model2, draws


    def play_game(self) -> GameResult:
        game = DotsAndBoxesGame(self.game_size)

        while game.is_running():
            lines = game.get_lines_vector()
            valids = game.get_valid_moves()
            move = self.model1.determine_move(lines, valids) if game.get_player_at_turn() == Value.PLAYER_1 \
                else self.model2.determine_move(lines, valids)

            assert move in valids, \
                print(f"<{move}> is not a valid move. Model should have selected a move in {valids}.")
            game.draw_line(move)

        state = game.get_state()
        if state == GameState.DRAW:
            return GameResult.DRAW
        if state == GameState.WIN_PLAYER_1:
            return GameResult.WIN_PLAYER_1
        if state == GameState.WIN_PLAYER_2:
            return GameResult.WIN_PLAYER_2
