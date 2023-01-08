from multiprocessing import Pool
from sys import stdout
from typing import Tuple
from tqdm import tqdm

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

    def __init__(self, game_size: int, player1: AIPlayer, player2: AIPlayer, n_games: int, n_workers: int):

        self.game_size = game_size
        self.player1 = player1
        self.player2 = player2
        self.n_games = n_games
        self.n_workers = n_workers

    def compare(self) -> Tuple[int, int, int]:

        wins_player1, wins_player2, draws = 0, 0, 0

        print(f"Comparing {self.player1.name}:Draw:{self.player2.name} ... ")

        with Pool(processes=self.n_workers) as pool:
            for result in pool.istarmap(self.play_game, tqdm([()] * self.n_games, file=stdout, smoothing=0.0)):

                if result == 1:
                    wins_player1 += 1
                elif result == -1:
                    wins_player2 += 1
                else:
                    draws += 1

        print(f"Result: {wins_player1}:{draws}:{wins_player2}")

        return wins_player1, wins_player2, draws

    def play_game(self) -> int:
        game = DotsAndBoxesGame(self.game_size)

        while game.is_running():
            move = self.player1.determine_move(game) if game.current_player == 1 else self.player2.determine_move(game)
            game.execute_move(move)

        return game.result
