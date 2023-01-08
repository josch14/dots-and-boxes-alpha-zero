import multiprocessing.pool as mpp


def istarmap(self, func, iterable, chunksize=1):
    """
    starmap-version of imap
    When using multiple threads to perform a loop of the same calculation, tqdm is not able to display actual progress
    correctly when starmap is used. Therefore, implement the patch istarmap() based on the code for imap().
    see: https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError("Chunksize must be 1+, not {0:n}".format(chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put((
        self._guarded_task_generation(result._job, mpp.starmapstar, task_batches), result._set_length
    ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap



def print_parameters(config: dict):
    print("####################### Parameters #######################")
    print("game_size: " + str(config["game_size"]) + " | n_iterations: " + str(config["n_iterations"]))
    print("[MCTS] " + mcts_parameters_to_string(config["mcts_parameters"]))
    print("[Neural Network] " + model_parameters_to_string(config["model_parameters"]))
    print("[Optimizer] " + optimizer_parameters_to_string(config["optimizer_parameters"]))
    print("[Data] " + data_parameters_to_string(config["data_parameters"]))
    print("[Evaluator] " + evaluator_parameters_to_string(config["evaluator_parameters"]))
    print("##########################################################")


def mcts_parameters_to_string(mcts_parameters: dict):
    s = "n_games: " + str(mcts_parameters["n_games"]) + " | " \
        "n_simulations: " + str(mcts_parameters["n_simulations"]) + " | " \
        "temperature_move_threshold: " + str(mcts_parameters["temperature_move_threshold"]) + " | " \
        "c_puct: " + str(mcts_parameters["c_puct"]) + " | " \
        "dirichlet_eps: " + str(mcts_parameters["dirichlet_eps"]) + " | " \
        "dirichlet_alpha: " + str(mcts_parameters["dirichlet_alpha"])
    return s


def model_parameters_to_string(model_parameters: dict):
    s = "hidden_layers: " + str(model_parameters["hidden_layers"]) + " | " \
        "dropout: " + str(model_parameters["dropout"])
    return s


def optimizer_parameters_to_string(optimizer_parameters: dict):
    s = "learning_rate: " + str(optimizer_parameters["learning_rate"]) + " | " \
        "momentum: " + str(optimizer_parameters["momentum"]) + " | " \
        "weight_decay: " + str(optimizer_parameters["weight_decay"])
    return s


def data_parameters_to_string(data_parameters: dict):
    s = "game_buffer: " + str(data_parameters["game_buffer"]) + " | " \
        "n_batches: " + str(data_parameters["n_batches"]) + " | " \
        "batch_size: " + str(data_parameters["batch_size"])
    return s


def evaluator_parameters_to_string(evaluator_parameters: dict):
    s = "n_games: " + str(evaluator_parameters["n_games"]) + " | " \
        "win_fraction: " + str(evaluator_parameters["win_fraction"])
    return s

