from typing import Tuple

# local import
from .game import DotsAndBoxesGame
from .players.player import AIPlayer


class Evaluator:
    """
    Let two players (AIs) play games of Dots-and-Boxes against each other.

    Attributes
    ----------
    game_size : int
        board size (width & height) of a Dots-and-Boxes game
    player1, player2 : AIPlayer, AIPlayer
        AI players that are compared against each other
    n_games : int
        number of games the models play against each other
    """
    def __init__(self, game_size: int, player1: AIPlayer, player2: AIPlayer, n_games: int):

        self.game_size = game_size
        self.player1 = player1
        self.player2 = player2
        self.n_games = n_games

    def compare(self) -> Tuple[int, int, int, float]:

        wins_player1, wins_player2, draws = 0, 0, 0

        print(f"Comparing {self.player1.name}:Draw:{self.player2.name} ... ")
        for _ in range(self.n_games):
            result = self.play_game()

            if result == 1:
                wins_player1 += 1
            elif result == -1:
                wins_player2 += 1
            else:
                draws += 1
        print(f"Result: {wins_player1}:{draws}:{wins_player2}")

        win_percent_player1 = wins_player1 / self.n_games
        return wins_player1, wins_player2, draws, win_percent_player1

    def play_game(self) -> int:
        game = DotsAndBoxesGame(self.game_size)

        while game.is_running():
            move = self.player1.determine_move(game) if game.current_player == 1 else self.player2.determine_move(game)
            game.execute_move(move)

        return game.result
