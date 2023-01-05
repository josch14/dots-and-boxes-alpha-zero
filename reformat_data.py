import json



if __name__ == '__main__':

    with open('data/optimization_data_12500b.json', 'r') as f:
        old_save_dict = json.load(f)

        train_examples_per_game = {}
        old_train_examples_per_game = old_save_dict["data"]

        for game_id in old_train_examples_per_game:
            train_examples = []
            for old_train_example_dict in old_train_examples_per_game[game_id]:
                zero_list = [0] * len(old_train_example_dict["s"])
                zero_list[old_train_example_dict["a"]] = 1
                train_example_dict = {
                    "s": old_train_example_dict["s"],
                    "p": zero_list,
                    "v": old_train_example_dict["v"]
                }
                train_examples.append(train_example_dict)

            train_examples_per_game[game_id] = train_examples

        save_dict = {
            "game_size": old_save_dict["game_size"],
            "n_games": old_save_dict["n_games"],
            "depth": old_save_dict["depth"],
            "data": train_examples_per_game
        }

        with open('data/optimization_data.json', 'w') as f:
            json.dump(save_dict, f)