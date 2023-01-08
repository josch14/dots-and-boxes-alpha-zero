import json
import os
import time

import numpy as np

CHECKPOINT_FOLDER = "checkpoints/"
TRAIN_EXAMPLES_PATH = CHECKPOINT_FOLDER + "training_examples.json"


class Checkpoint:

    @staticmethod
    def save_data(train_examples_per_game: list):
        if not os.path.exists(CHECKPOINT_FOLDER):
            os.makedirs(CHECKPOINT_FOLDER)

        start_time = time.time()
        print("Saving training examples .. ", end="")

        save_dict = {}
        for i, train_examples in enumerate(train_examples_per_game):
            save_dict[i] = [{"s": t[0].tolist(), "p": t[1].tolist(), "v": t[2]} for t in train_examples]

        with open(TRAIN_EXAMPLES_PATH, 'w') as f:
            json.dump(save_dict, f)

        print("took {0:.2f}s".format(time.time() - start_time))



    @staticmethod
    def load_data():

        print("Loading training examples .. ", end="")
        start_time = time.time()

        with open(TRAIN_EXAMPLES_PATH, 'r') as f:
            save_dict = json.load(f)

        train_examples_per_game = []
        for game_id in save_dict:
            train_examples = [(np.array(t["s"]), np.array(t["p"]), t["v"]) for t in save_dict[game_id]]
            train_examples_per_game.append(train_examples)

        print("took {0:.2f}s".format(time.time() - start_time))

        return train_examples_per_game
