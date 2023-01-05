import argparse
import os
import time

from players.alpha_beta import AlphaBetaPlayer
from players.player import AIPlayer
from players.random import RandomPlayer
from utils.printer import DotsAndBoxesPrinter


def cls(): os.system("cls" if os.name == "nt" else "clear")


def main(size: int, opponent: AIPlayer):
    cls()
    game = DotsAndBoxesPrinter(size)
    print(game.state_string())
    print(game.board_string())

    while game.is_running():

        if game.current_player == 1 or opponent is None:
            # print draw request
            print("Please enter a free line number: ", end="")

            # process draw request
            while True:
                move = int(input())
                if move in game.get_valid_moves():
                    break
                print(f"Line {move} is not a valid move. Please select a move in {game.get_valid_moves()}.")
            last_move_by_player = True

        else:
            # an AI opponent is at turn
            time.sleep(1.0)
            start_time = time.time()
            move = opponent.determine_move(game)
            stopped_time = time.time() - start_time
            last_move_by_player = False

        game.execute_move(move)

        # print new game state
        cls()
        if not last_move_by_player:
            print("Computation time of opponent for previous move {0:.2f}s".format(stopped_time))
        else:
            print()
        print(game.state_string())
        print(game.board_string())


    if game.result == 1:
        print("The game is over.. You won!")
    elif game.result == -1:
        print("The game is over.. You lost :(")
    else:
        print("The game ended in a draw ..")
    print(game.state_string())


"""
Example call: 
python play_console.py --o alpha_beta --depth 3 --size 3
"""
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--opponent', type=str, default="alpha_beta", choices=["person", "random", "alpha_beta"],
                    help='Type of opponent to play against.')
parser.add_argument('-d', '--depth', type=int, default=3,
                    help='Specifies the depth of a search in case of an opponent that utilizes Alpha–beta pruning.')
parser.add_argument('-s', '--size', type=int, default=3,
                    help='Size of the Dots-and-Boxes game (in number of boxes per row and column).')
args = parser.parse_args()


if __name__ == '__main__':

    if args.opponent == "person":
        opponent = None
    elif args.opponent == "random":
        opponent = RandomPlayer()
    elif args.opponent == "alpha_beta":
        opponent = AlphaBetaPlayer(depth=args.depth)

    main(args.size, opponent)
