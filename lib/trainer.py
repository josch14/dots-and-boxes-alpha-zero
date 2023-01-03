import time
from multiprocessing import Pool
from random import shuffle
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
from sys import stdout

# local import
from lib.evaluator import Evaluator
from lib.game import DotsAndBoxesGame
from lib.mcts import MCTS
from lib.model import AZNeuralNetwork


class Trainer:
    """
    Executes the training loop, where each iteration consists of
    1) self-play (using MCTS)
    2) model learning (using generated data)
    3) model comparison (using evaluator)

    Attributes
    ----------
    game_size : int
        board size (width & height) of a Dots and Boxes game
    iterations : int
        number of iterations of self-play + model training + model evaluation
    mcts_parameters : dict
        hyperparameters concerning the MCTS
    model_parameters, optimizer_parameters, training_parameters : dict, dict, dict
        hyperparameters concerning the neural network (architecture, training, optimizer)
    evaluator_parameters : dict
        hyperparameters concerning the evaluator
    model : AZNeuralNetwork
        neural network that is to be trained
    model_name : str
        name of the neural network that is to be trained (if existing, the checkpoint may be loaded from local files)
    inference_device : torch.cuda.device
        device with which model interference is performed during MCTS
    training_device : torch.cuda.device
        device with which model training is performed
    train_examples : List[[np.ndarray, [float], float]]
        list of training examples (s, p, v) (from the current player's POV)
    """
    def __init__(self, config: dict, model_name: str, inference_device: str, training_device: str):

        self.game_size = config["game_size"]
        self.iterations = config["iterations"]

        self.mcts_parameters = config["mcts_parameters"]
        self.model_parameters = config["model_parameters"]
        self.optimizer_parameters = config["optimizer_parameters"]
        self.training_parameters = config["training_parameters"]
        self.evaluator_parameters = config["evaluator_parameters"]

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

        self.train_examples = []

    def loop(self):
        """
        Perform a single iteration of self-play + model learning + model comparison.
        """
        # training parameters
        dataset_size = self.training_parameters["dataset_size"]

        # evaluator parameters
        win_fraction = self.evaluator_parameters["win_fraction"]
        n_games = self.evaluator_parameters["n_games"]

        iteration_of_best_model = 0
        for iteration in range(1, self.iterations + 1):
            print(f"\n#################### Iteration {iteration}/{self.iterations} #################### ")

            # 1) perform games of self-play to obtain training data
            print("------------ Self-play using MCTS ------------")
            train_examples = self.perform_self_plays(
                n_games=self.mcts_parameters["n_games"],
                n_simulations=self.mcts_parameters["n_simulations"],
                temperature_move_threshold=self.mcts_parameters["temperature_move_threshold"],
                c_puct=self.mcts_parameters["c_puct"]
            )
            # rules are invariant under rotation and reflection:
            # augment dataset to include rotations and reflections of each position
            augmented_train_examples = []
            for s, p, v in train_examples:
                augmented_train_examples.extend(
                    [(s_augmented, p, v) for s_augmented in DotsAndBoxesGame.get_rotations_and_reflections(s)]
                )
            print(f"Self-play resulted in {len(augmented_train_examples)} new training examples (after augmentation).")

            self.train_examples.extend(augmented_train_examples)
            # cut dataset to desired size
            while len(self.train_examples) > dataset_size:
                self.train_examples.pop(0)


            # 2) model learning
            print("\n---------- Neural Network Training -----------")
            print(f"Dataset currently consist of {len(self.train_examples)}/{dataset_size} training examples.")
            prev_model = deepcopy(self.model)
            self.perform_model_training()


            # 3) evaluator: model comparison
            # if the trained network wins by a margin of > win_fraction, then it is subsequently used for self-play
            # generation, and also becomes the baseline for subsequent comparisons
            print("\n-------------- Model Comparison --------------")
            evaluator = Evaluator(
                game_size=self.game_size,
                model1=self.model,
                model2=prev_model,
                n_games=n_games
            )
            wins_trained_model, wins_prev_model, draws = evaluator.compare()
            win_percent = wins_trained_model / n_games

            print(f"Results: TrainedModel(Iteration={iteration}):Draw:PreviousModel(Iteration={iteration_of_best_model}) - {wins_trained_model}:{draws}:{wins_prev_model}")
            print(f"Trained model won {round(win_percent * 100, 2)}% of the games ({round(win_fraction * 100, 2)}% needed).")

            if win_percent >= win_fraction:
                iteration_of_best_model = iteration
                print("Continuing with trained model!")
                if self.model_name:
                    self.model.save_checkpoint(model_name=self.model_name + f"_{iteration}")
            else:
                print(f"Continuing with previous best model from iteration {iteration_of_best_model}!")
                self.model = prev_model
            print("###########################################################")


    def perform_self_plays(self,
                           n_games: int,
                           n_simulations: int,
                           temperature_move_threshold: int,
                           c_puct: float):
        """
        Perform games of self-play using MCTS.

        Parameters
        ----------
        n_games : int
            number of games of self-play to perform
        n_simulations, temperature_move_threshold, c_puct : int, int, float
            see self.perform_self_play()

        Returns
        -------
        train_examples : List[[np.ndarray, [float], float]]
            list of training examples (s, p, v) (from the current player's POV)
        """
        train_examples = []
        self.model.eval()

        args_repeat = [(n_simulations, temperature_move_threshold, c_puct)] * n_games
        with Pool(processes=4) as pool:
            for train_example in pool.starmap(self.perform_self_play, tqdm(args_repeat, file=stdout)):
                train_examples.extend(train_example)
        return train_examples


    def perform_self_play(self,
                          n_simulations: int,
                          temperature_move_threshold: int,
                          c_puct: float):
        """
        Perform a single game of self-play using MCTS. The data for the game is stored as (s, p, v) at each
        time-step (i.e., each turn results in a training example), with s being the position/game state (vector),
        p being the policy vector and v being the value (scalar).

        Parameters
        ----------
        n_simulations : int
            number of simulations used for each MCTS (i.e., for each move)
        temperature_move_threshold : int
            when more than temperature_move_threshold moves were performed during self-play, the temperature parameter
            is set from 1 to 0. This ensures that a diverse set of positions are encountered, as then the first moves
            during MCTS are selected proportionally to their visit count
        c_puct : float
            constant determining level of exploration (parameter for PUCT algorithm utilized in MCTS: select)

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
            n_simulations=n_simulations,
            c_puct=c_puct
        )

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
        # use copy of training examples to preserve order of data
        train_examples = deepcopy(self.train_examples)
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
                print(f"Early stopping (patience={patience}) after {epoch} epochs.")
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

                if loss < best_loss:
                    best_loss = loss
                    best_model = deepcopy(self.model)
                    current_patience = 0
                    print("New best model achieved after {0:d} epochs (loss: {1:.5f}). "
                          "Execution Time for epoch: {2:.2f}s".format(epoch, best_loss, time.time() - start_time))
                else:
                    current_patience += 1

        self.model = best_model
        self.model.to(self.inference_device)
