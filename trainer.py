import time
from multiprocessing import Pool
from random import shuffle
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
from sys import stdout

# local import
from src import Evaluator, DotsAndBoxesGame, MCTS, AZNeuralNetwork, \
    AlphaBetaPlayer, NeuralNetworkPlayer, RandomPlayer, istarmap


class Trainer:
    """
    Executes the training loop, where each iteration consists of
    1) self-play (using MCTS)
    2) model learning (using generated data)
    3) model comparison (using evaluator)

    Attributes
    ----------
    game_size : int
        board size (width & height) of a Dots-and-Boxes game
    mcts_parameters : dict
        hyperparameters concerning the MCTS
    model_parameters, optimizer_parameters, training_parameters : dict, dict, dict
        hyperparameters concerning the neural network (architecture, training, optimizer)
    evaluator_parameters : dict
        hyperparameters concerning the evaluator
    n_workers : int
        number of threads during self-play. Each thread performs games of self-play
    model : AZNeuralNetwork
        neural network that is to be trained
    model_name : str
        name of the neural network that is to be trained (if existing, the checkpoint may be loaded from local files)
    inference_device : torch.cuda.device
        device with which model interference is performed during MCTS
    training_device : torch.cuda.device
        device with which model training is performed
    train_examples_per_game : List[List[[np.ndarray, [float], float]]]
        list (one element corresponds with single self-play), containing lists of training examples (s, p, v)-tuples (from
        the current player's POV)
    """
    def __init__(self, config: dict, model_name: str, n_workers: int, inference_device: str, training_device: str):

        self.game_size = config["game_size"]
        self.mcts_parameters = config["mcts_parameters"]
        self.model_parameters = config["model_parameters"]
        self.optimizer_parameters = config["optimizer_parameters"]
        self.training_parameters = config["training_parameters"]
        self.evaluator_parameters = config["evaluator_parameters"]

        self.n_workers = n_workers
        # utilize gpu if possible
        if "cuda" in [inference_device, training_device]:
            assert torch.cuda.is_available()
        self.inference_device = torch.device(inference_device)
        self.training_device = torch.device(training_device)
        print(f"Model inference (during MCTS) is performed with device: {self.inference_device}")
        print(f"Model training is performed with device: {self.training_device}")

        # initialize model (potentially from checkpoint)
        self.model = AZNeuralNetwork(
            game_size=self.game_size,
            model_parameters=self.model_parameters,
            inference_device=self.inference_device
        )
        self.model_name = model_name
        if model_name:
            self.model.load_checkpoint(model_name)
        self.model.to(self.inference_device)

        self.train_examples_per_game = []

    def loop(self, n_iterations: int):
        """
        Perform iterations of self-play + model training + model evaluation.

        Parameters
        ----------
        n_iterations : int
            number of iterations to perform
        """

        iteration_of_best_model = 0
        for iteration in range(1, n_iterations + 1):
            print(f"\n#################### Iteration {iteration}/{n_iterations} #################### ")

            # 1) perform games of self-play to obtain training data
            print("------------ Self-play using MCTS ------------")
            self.train_examples_per_game.extend(
                self.perform_self_plays(n_games=self.mcts_parameters["n_games"])
            )
            # cut dataset to desired size
            while len(self.train_examples_per_game) > self.training_parameters["game_buffer_size"]:
                self.train_examples_per_game.pop(0)


            # 2) model learning
            print("\n---------- Neural Network Training -----------")
            prev_model = deepcopy(self.model)
            self.perform_model_training()


            # 3) evaluator: model comparison
            print("\n-------------- Model Comparison --------------")
            # if the trained network wins by a margin of > win_fraction, then it is subsequently used for self-play
            # generation, and also becomes the baseline for subsequent comparisons
            win_fraction = self.evaluator_parameters["win_fraction"]
            n_games = self.evaluator_parameters["n_games"]

            neural_network_player = NeuralNetworkPlayer(self.model, name=f"TrainedModel(Iteration={iteration})")
            opponents = [RandomPlayer(), AlphaBetaPlayer(depth=1), AlphaBetaPlayer(depth=2), AlphaBetaPlayer(depth=3)]
            # 3.1) compare against non-neural network players
            for opponent in opponents:
                evaluator = Evaluator(
                    game_size=self.game_size,
                    player1=neural_network_player,
                    player2=opponent,
                    n_games=n_games
                )
                evaluator.compare()

            # 3.2) compare against previous model
            evaluator = Evaluator(
                game_size=self.game_size,
                player1=NeuralNetworkPlayer(self.model, name=f"TrainedModel(Iteration={iteration})"),
                player2=NeuralNetworkPlayer(prev_model, name=f"PreviousModel(Iteration={iteration_of_best_model})"),
                n_games=n_games
            )
            _, _, _, win_percent = evaluator.compare()

            print(f"Trained model won {round(win_percent * 100, 2)}% of the games vs. previous model ({round(win_fraction * 100, 2)}% needed).")
            if win_percent >= win_fraction:
                iteration_of_best_model = iteration
                print("Continuing with trained model!")
                if self.model_name:
                    self.model.save_checkpoint(model_name=self.model_name + f"_{iteration}")
            else:
                print(f"Continuing with previous best model from iteration {iteration_of_best_model}!")
                self.model = prev_model

            print("###########################################################")


    def perform_self_plays(self, n_games: int):
        """
        Perform games of self-play using MCTS.

        Parameters
        ----------
        n_games : int
            number of games of self-play to perform

        Returns
        -------
        train_examples : List[[np.ndarray, [float], float]]
            list of training examples (s, p, v) (from the current player's POV)
        """
        train_examples_per_game = []
        self.model.eval()

        start_time = time.time()
        with Pool(processes=self.n_workers) as pool:
            for train_examples in pool.istarmap(self.perform_self_play, tqdm([()] * n_games, file=stdout, smoothing=0.0)):
                train_examples_per_game.append(train_examples)


        print("{0:d} games of Self-play resulted in {1:d} new training examples (without augmentations; after {2:.2f}s).".format(
            n_games, len([t for l in train_examples_per_game for t in l]), time.time() - start_time))

        return train_examples_per_game


    def perform_self_play(self):
        """
        Perform a single game of self-play using MCTS. The data for the game is stored as (s, p, v) at each
        time-step (i.e., each turn results in a training example), with s being the position/game state (vector),
        p being the policy vector and v being the value (scalar).

        Returns
        -------
        train_examples : List[[np.ndarray, [float], float]]
            list of training examples (s, p, v) (from the current player's POV)
        """

        game = DotsAndBoxesGame(self.game_size)
        n_moves = 0
        train_examples = []

        # one self-play corresponds with one tree
        mcts = MCTS(
            model=self.model,
            s=deepcopy(game),
            mcts_parameters=self.mcts_parameters
        )

        # when more than temperature_move_threshold moves were performed during self-play, the temperature parameter
        # is set from 1 to 0. This ensures that a diverse set of positions are encountered, as then the first moves
        # during MCTS are selected proportionally to their visit count
        temperature_move_threshold = self.mcts_parameters["temperature_move_threshold"]

        # iteration over time-steps t during the game. At each time-step, a MCTS is executed using the previous iteration
        # of the neural network and a move is played by sampling the search probabilities
        while game.is_running():
            temp = 1 if n_moves < temperature_move_threshold else 0
            n_moves += 1

            # execute MCTS for next move
            probs = mcts.play(temp=temp)

            train_examples.append([
                game.get_canonical_s(),
                probs,
                game.current_player  # correct v is determined later
            ])

            # sample and play move from probability distribution
            move = np.random.choice(
                a=list(range(game.N_LINES)),
                p=probs
            )
            game.execute_move(move)

            # child node corresponding to the played action becomes the new root. The subtree below this child is
            # retained along with all its statistics, while the remainder of the tree is discarded
            mcts.root = mcts.root.get_child_by_move(move)

        # determine correct value v for the activate player in each example
        assert game.result is not None, "Game not yet finished. Unable to determine value v"
        for i, (_, _, current_player) in enumerate(train_examples):
            if current_player == game.result:
                train_examples[i][2] = 1
            elif game.result == 0:
                train_examples[i][2] = 0
            else:
                train_examples[i][2] = -1

        return train_examples


    def perform_model_training(self):
        """
        Train the already existing neural network using the training data which was generated from self-play.

        Loss Function: The neural network is adjusted to minimize the error between the predicted value and the self-play
        winner, and to maximize the similarity of the neural network move probabilities to the search probabilities
        (i.e., the parameters are adjusted by gradient descent on a loss function that sums over the mean-squared error
        and cross-entropy losses. The cross-entropy and MSE losses are weighted equally. L2 weight regularization is
        used to prevent overfitting.

        Optimization: The neural network parameters are optimized by stochastic gradient descent with momentum (as
        opposed to the original paper, without learning rate annealing).
        """

        # run a complete training for a neural network
        epochs = self.training_parameters["epochs"]
        batch_size = self.training_parameters["batch_size"]
        patience = self.training_parameters["patience"]

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.optimizer_parameters["learning_rate"],
            momentum=self.optimizer_parameters["momentum"],
            weight_decay=self.optimizer_parameters["weight_decay"],
        )

        # prepare data
        # augment dataset by including rotations and reflections of each position
        train_examples = []
        for train_examples_list in self.train_examples_per_game:
            for s, p, v in train_examples_list:
                train_examples.extend(
                    [(s_augmented, p, v) for s_augmented in DotsAndBoxesGame.get_rotations_and_reflections(s)]
                )
        game_buffer_size = self.training_parameters["game_buffer_size"]
        print(f"The dataset consist of {len(train_examples)} training examples (including augmentations) from the "
              f"{len(self.train_examples_per_game)}/{game_buffer_size} most recent games.")

        # TODO "Parameters were updated from 700,000 mini-batches of 2,048 positions. ...
        # TODO ... Each mini-batch of data is  sampled uniformly at random from all positions of the most recent 500,000 games
        # TODO of self-play"

        shuffle(train_examples)
        s_train, p_train, v_train = [list(t) for t in zip(*train_examples)]
        s_batched, p_batched, v_batched = [], [], []
        for i in range(0, len(train_examples), batch_size):
            s_batched.append(torch.tensor(np.vstack(s_train[i:i + batch_size]), dtype=torch.float32, device=self.training_device))
            p_batched.append(torch.tensor(np.vstack(p_train[i:i + batch_size]), dtype=torch.float32, device=self.training_device))
            v_batched.append(torch.tensor(v_train[i:i + batch_size], dtype=torch.float32, device=self.training_device))  # scalar v
        n_batches = len(s_batched)

        current_patience = 0
        best_model = None
        best_loss = 1e10

        CrossEntropyLoss = torch.nn.CrossEntropyLoss()
        MSELoss = torch.nn.MSELoss()

        self.model.to(self.training_device)

        for epoch in range(1, epochs + 1):

            if current_patience > patience:
                print(f"Early stopping after {epoch} epochs.")
                break

            start_time = time.time()

            # train model
            self.model.train()
            for i in range(n_batches):
                optimizer.zero_grad()

                p, v = self.model.forward(s_batched[i])

                loss = CrossEntropyLoss(p, p_batched[i]) + MSELoss(v, v_batched[i])
                loss.backward()
                optimizer.step()

            # evaluate model on train set
            self.model.eval()
            with torch.no_grad():
                # calculate loss per training example
                loss = 0
                for i in range(n_batches):
                    optimizer.zero_grad()
                    p, v = self.model.forward(s_batched[i])
                    loss += CrossEntropyLoss(p, p_batched[i]) + MSELoss(v, v_batched[i])
                loss = loss / n_batches

                print("[Epoch {0:d}] Loss: {1:.5f} (after {2:.2f}s). ".format(epoch, loss, time.time() - start_time), end="")

                if loss < best_loss:
                    best_loss = loss
                    best_model = deepcopy(self.model)
                    current_patience = 0
                    print("New best model achieved!")
                else:
                    current_patience += 1
                    print("")

        self.model = best_model
        self.model.to(self.inference_device)