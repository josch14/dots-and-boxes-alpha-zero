from random import shuffle
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm

# local import
from lib.arena import Arena
from lib.game import DotsAndBoxesGame
from lib.mcts import MCTS
from lib.model import AZNeuralNetwork

# initialize logging
import logging as logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Trainer")


class Trainer:
    """
    Executes iterations of self-play (using MCTS) + model learning + arena,
    combining logic of all further classes.
    """

    def __init__(self, configuration: dict):
        self.game_size = configuration["game_size"]
        self.iterations = configuration["iterations"]

        self.mcts_parameters = configuration["mcts_parameters"]  # mcts
        self.model_parameters = configuration["model_parameters"]  # neural network
        self.optimizer_parameters = configuration["optimizer_parameters"]  # neural network
        self.training_parameters = configuration["training_parameters"]  # neural network
        self.arena_parameters = configuration["arena_parameters"]  # arena

        # initialize model
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        n_lines = 2 * self.game_size * (self.game_size + 1)
        self.model = AZNeuralNetwork(
            io_units=n_lines,
            model_parameters=self.model_parameters
        )
        self.model.to(self.device)

        self.training_examples = []

    def loop(self):
        """
        Perform iterations of self-play + model learning + arena.
        """
        # training parameters
        dataset_size = self.training_parameters["dataset_size"]

        # arena parameters
        win_fraction = self.arena_parameters["win_fraction"]
        n_games = self.arena_parameters["n_games"]

        for iteration in range(1, self.iterations + 1):
            if (iteration + 1) % 100 == 0:
                log.info(f"Progress: Iteration {iteration}/{self.iterations}")

            # 1) perform games of self-play to obtain training data
            log.info(f"Performing self-play using MCTS ..")
            train_examples = self.perform_self_plays(
                n_games=self.mcts_parameters["n_games"],
                n_simulations=self.mcts_parameters["n_simulations"],
                temperature_move_threshold=self.mcts_parameters["temperature_move_threshold"],
                c_puct=self.mcts_parameters["c_puct"]
            )
            # rules are invariant under rotation and reflection: augment dataset to include rotations and reflections
            # of each position
            augmented_train_examples = []
            for s, p, v in train_examples:
                augmented_train_examples.extend(
                    [(s_augmented, p, v) for s_augmented in DotsAndBoxesGame.get_rotations_and_reflections(s)]
                )

            log.info(f"Self-play resulted in {len(train_examples)} new training examples (after augmentation).")

            self.training_examples.extend(train_examples)
            # cut dataset to desired size
            while len(self.training_examples) > dataset_size:
                self.training_examples.pop(0)

            # 2) model learning
            log.info(f"Training the neural network using dataset of size "
                     f"{len(self.training_examples)}/{dataset_size}")
            prev_model = deepcopy(self.model)
            self.perform_model_training()

            # 3) arena: model comparison
            log.info(f"Trained model plays against previous version ..")
            arena = Arena(
                game_size=self.game_size,
                model1=self.model,
                model2=prev_model,
                n_games=n_games
            )
            wins_trained_model, wins_prev_model, draws = arena.compare()
            win_percent = wins_trained_model / n_games

            log.info(
                f"Compare results: (trained model) {wins_trained_model}:{draws}:{wins_prev_model} (previous model). "
                f"Trained model won {round(win_percent * 100, 2)}% of the games ({round(win_fraction * 100, 2)}% needed).")

            if win_percent >= win_fraction:
                log.info("Continuing with trained model.")
            else:
                log.info("Continuing with previous model.")
                self.model = prev_model

    def perform_self_plays(self,
                           n_games: int,
                           n_simulations: int,
                           temperature_move_threshold: int,
                           c_puct: float):
        """
        Perform iterations of games of self-play.

        Parameters
        ----------
        n_games : int
            number of games of self-play to perform
        n_simulations : int
            see self.perform_self_play()
        temperature_move_threshold : int
            see self.perform_self_play()
        c_puct : float
            see self.perform_self_play()


        Returns
        -------
        train_examples : List[]
            list of training examples (see self.perform_self_play())
        """
        train_examples = []
        self.model.eval()

        for _ in tqdm(range(n_games)):
            # single iteration of self-play
            train_examples.extend(self.perform_self_play(
                n_simulations=n_simulations,
                temperature_move_threshold=temperature_move_threshold,
                c_puct=c_puct
            ))
        return train_examples

    def perform_self_play(self,
                          n_simulations: int,
                          temperature_move_threshold: int,
                          c_puct: float):
        """
        Perform a single iteration of self-play. The data for the game is stored
        as (s, p, v) at each time-step (i.e., each turn results in a training
        example), with s being the position/game state (vector), p the policy
        vector and v the value (scalar).

        Parameters
        ----------
        n_simulations : int
            number of simulations used for each MCTS
        temperature_move_threshold : int
            when more than temperature_move_threshold moves were performed during
            self-play, the temperature parameter is set from 1 to 0. This ensures
            that a diverse set of positions are encountered, as for the first
            moves, moves are selected proportionally to their visit count in MCTS
        c_puct : float
            constant determining level of exploration (PUCT algorithm in MCTS: select)


        Returns
        -------
        train_examples : List[[np.ndarray, [float], float]]
            list of training examples (s, p, v)
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

        # iteration over time-steps t during the game. At each time-step, a
        # MCTS is executed using the previous iteration of the neural network and
        # a move is played by sampling the search probabilities
        while game.is_running():
            temp = 1 if n_moves < temperature_move_threshold else 0
            n_moves += 1

            # execute MCTS
            probs = mcts.play(temp=temp)

            # - Include Symmetries of current Game State:
            #     rules are invariant to colour transposition: represent the board from the perspective of the current player
            train_examples.append([
                game.get_canonical_lines_vector(),
                probs,
                game.current_player  # correct v is determined later
            ])

            # sample and play move from probability distribution
            move = np.random.choice(
                a=list(range(game.N_LINES)),
                p=probs
            )
            game.execute_move(move)

            # child node corresponding to the played action becomes the new root
            # the subtree below this child is retained along with all its statistics,
            # while the remainder of the tree is discarded
            mcts.root = mcts.root.get_child_by_move(move)
            mcts.root.parent, mcts.root.a = None, None

        # determine correct value v for the activate player in each example
        # TODO player perspective
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
        the neural network’s
        parameters are updated to make the move probabilities and value (p,
        v) = fθ(s) more closely match the improved search probabilities and selfplay
        winner (π, z);

        The neural network parameters θ are
        updated to maximize the similarity of the policy vector pt to the search
        probabilities πt, and to minimize the error between the predicted winner vt
        and the game winner z (see equation (1)).


        OPTIMIZATION:
        Neural network parameters are optimized by stochastic gradient
        descent with momentum and learning rate annealing, using the loss in equation
        (1). The learning rate is annealed according to the standard schedule in Extended
        Data Table 3. The momentum parameter is set to 0.9. The cross-entropy and MSE
        losses are weighted equally (this is reasonable because rewards are unit scaled,
        r ∈ {− 1, + 1}) and the L2 regularization parameter is set to c = 10−4

        The
        neural network = (p, v) fθ (s) i is adjusted to minimize the error between
        the predicted value v and the self-play winner z, and to maximize the
        similarity of the neural network move probabilities p to the search
        probabilities π. Specifically, the parameters θ are adjusted by gradient
        descent on a loss function l that sums over the mean-squared error and
        cross-entropy losses, respectively:
        = = − −π + θ (p, v) fθ (s) and l (z v) logp c (1) 2 T 2
        where c is a parameter controlling the level of L2 weight regularization
        (to prevent overfitting).


        Train the neural network with examples obtained from MCTS/self-play. The
        neural network parameters are updated to ..
        1) .. maximize the similarity of the policy vector p to the search
           probabilities π, and to
        2) .. minimize the error between predicted winner v and game winner z
        """

        """
        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value. The examples has
                      board in its canonical form.

        """

        # run a complete training for a neural network
        epochs = self.training_parameters["epochs"]
        batch_size = self.training_parameters["batch_size"]
        patience = self.training_parameters["patience"]

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.optimizer_parameters["learning_rate"],
            momentum=self.optimizer_parameters["momentum"]
        )

        # prepare data
        # use copy of training examples to preserve order of data
        train_examples = deepcopy(self.training_examples)
        shuffle(train_examples)
        s_train, p_train, v_train = [list(t) for t in zip(*train_examples)]

        s_batched, p_batched, v_batched = [], [], []
        for i in range(0, len(train_examples), batch_size):
            s_batched += torch.tensor([s_train[i:i + batch_size]], dtype=torch.float32, device=self.device)
            p_batched += torch.tensor([p_train[i:i + batch_size]], dtype=torch.float32, device=self.device)
            v_batched += torch.tensor([v_train[i:i + batch_size]], dtype=torch.float32, device=self.device)

        current_patience = 0
        best_model = None
        best_loss = 1e10
        CrossEntropyLoss = torch.nn.CrossEntropyLoss()
        MSELoss = torch.nn.MSELoss()

        for epoch in range(1, epochs + 1):
            if current_patience > patience:
                log.info(f"Performance does not improve anymore, model training is stopped after {epoch} epochs.")
                break

            if epoch % 10 == 0:
                log.info(f"Model Training: Epoch {epoch}")

            # train model
            self.model.train()
            for i in range(len(s_batched)):
                optimizer.zero_grad()

                p, v = self.model.forward(s_batched[i])

                loss = CrossEntropyLoss(p, p_batched[i]) + MSELoss(v, v_batched[i])
                loss.backward()
                optimizer.step()

            # evaluate model on train set
            self.model.eval()
            with torch.no_grad():
                total_loss = 0

                for i in range(len(s_batched)):
                    optimizer.zero_grad()

                    p, v = self.model.forward(s_batched[i])

                    total_loss += CrossEntropyLoss(p, p_batched[i]) + MSELoss(v, v_batched[i])

                if total_loss < best_loss:
                    best_loss = total_loss
                    best_model = deepcopy(self.model)
                    current_patience = 0
                    log.info(f"New best model achieved after {epoch} epochs (loss: {best_loss}")
                else:
                    current_patience += 1

        self.model = best_model
