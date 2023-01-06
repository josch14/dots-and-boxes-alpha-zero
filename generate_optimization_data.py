import argparse
import json
import os
import time
from multiprocessing import Pool
from sys import stdout
import numpy as np
from tqdm import tqdm

from src import DotsAndBoxesGame, AlphaBetaPlayer


def main(game_size: int, n_games: int, depth: int, n_workers: int):

    game_id = 0
    n_train_examples = 0
    train_examples_per_game = {}

    start_time = time.time()
    with Pool(processes=n_workers) as pool:
        for train_examples in pool.istarmap(perform_self_play, tqdm([(game_size, depth)] * n_games, file=stdout, smoothing=0.0)):

            train_examples_per_game[game_id] = train_examples
            game_id += 1
            n_train_examples += len(train_examples)


    print("{0:d} games of self-play resulted in {1:d} new training examples (without augmentations; after {2:.2f}s).".format(
        n_games, n_train_examples, time.time() - start_time))

    return train_examples_per_game


def perform_self_play(size: int, depth: int):

    game = DotsAndBoxesGame(size)
    player1, player2 = AlphaBetaPlayer(depth=depth), AlphaBetaPlayer(depth=depth)

    train_examples_pre = []
    while game.is_running():
        move = player1.determine_move(game) if game.current_player == 1 else player2.determine_move(game)

        # do not add the first 4 moves to the dataset (i.e., if 4 lines are already drawn, then add move to dataset)
        if np.count_nonzero(game.s) >= 4:
            train_examples_pre.append([
                game.get_canonical_s(),
                move,
                game.current_player  # correct v is determined later
            ])

        game.execute_move(move)

    for i, (_, _, current_player) in enumerate(train_examples_pre):
        if current_player == game.result:
            train_examples_pre[i][2] = 1
        elif game.result == 0:
            train_examples_pre[i][2] = 0
        else:
            train_examples_pre[i][2] = -1

    train_examples = []
    for train_example in train_examples_pre:
        p = [0] * game.N_LINES
        p[train_example[1]] = 1
        train_example_dict = {
            "s": [round(e) for e in train_example[0].tolist()],
            "p": p,
            "v": train_example[2]
        }
        train_examples.append(train_example_dict)

    return train_examples


"""
Example call: 
python generate_optimization_data.py --game_size 3 --n_games 100000 --depth 3 --n_workers 8
"""
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--game_size', type=int, default=3,
                    help='Size of the Dots-and-Boxes games (in number of boxes per row and column).')
parser.add_argument('-n', '--n_games', type=int, default=10,
                    help='Number of games of self-play to perform.')
parser.add_argument('-d', '--depth', type=int, default=3,
                    help='Specifies the depth of a alpha-beta search')
parser.add_argument('-w', '--n_workers', type=int, default=8,
                    help='Number of threads during self-play of alpha-beta players.')
args = parser.parse_args()


if __name__ == '__main__':

    train_examples_per_game = main(args.game_size, args.n_games, args.depth, args.n_workers)

    save_dict = {
        "game_size": args.game_size,
        "n_games": args.n_games,
        "depth": args.depth,
        "data": train_examples_per_game
    }

    if not os.path.exists('data/'):
        os.makedirs('data/')
    with open('data/optimization_data_30000.json', 'w') as f:
        json.dump(save_dict, f)



# TODO add augmentation code
"""
import json
from sys import stdout

import numpy as np
from tqdm import tqdm

from src import DotsAndBoxesGame

if __name__ == '__main__':
    CONFIG_FILE = "resources/train_config.yaml"

    train_examples = []
    for filename in [
        'data/optimization_data_25000.json',
        'data/optimization_data_30000.json',
        'data/optimization_data_12500a.json',
        'data/optimization_data_12500b.json',
        'data/optimization_data_20000.json']:

        print("loading " + filename)
        with open(filename, 'r') as f:
            save_dict = json.load(f)

            train_examples_per_game = save_dict["data"]
            for game_id in train_examples_per_game:
                train_examples.extend(train_examples_per_game[game_id])


    train_examples = [(t["s"], t["p"], t["v"]) for t in train_examples]
    s_train, p_train, v_train = [list(t) for t in zip(*train_examples)]


    train_examples_augmented = []
    train_examples = []
    for s, p, v in tqdm(zip(s_train, p_train, v_train), file=stdout, total=len(s_train)):
        # with augmentations
        train_examples_augmented.extend(zip(
            DotsAndBoxesGame.get_rotations_and_reflections(np.asarray(s)),
            DotsAndBoxesGame.get_rotations_and_reflections(np.asarray(p)),
            [v] * 8
        ))

        # without augmentations
        train_examples.append([
            [round(e) for e in s],
            [round(e) for e in p],
            v])

    print("rounding ...")
    train_examples = [[[round(e) for e in s], [round(e) for e in p], v] for s, p, v in train_examples]
    train_examples_augmented = [[[round(e) for e in s], [round(e) for e in p], v] for s, p, v in train_examples_augmented]

    print("creating dicts ...")
    train_examples = [{"s": t[0], "p": t[1], "v": t[2]} for t in train_examples]
    train_examples_augmented = [{"s": t[0], "p": t[1], "v": t[2]} for t in train_examples_augmented]

    with open('data/optimization_data.json', 'w') as f:
        json.dump(train_examples, f)

    with open('data/optimization_data_augmented.json', 'w') as f:
        json.dump(train_examples_augmented, f)
"""