from random import shuffle

import numpy as np
import torch

from .arena import Arena
from .constants import GameState, Value
from .game import DotsAndBoxesGame
from .model import AZNeuralNetwork
from .mcts import MCTS

import copy
from tqdm import tqdm

# initialize logging
import logging as logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Trainer")


class Trainer:
    """
    Execute self-play + learning. 
    """

    def __init__(self, configuration: dict):
        self.game_size = configuration["game_size"]
        self.epochs = configuration["epochs"]

        self.mcts_parameters = configuration["mcts_parameters"]  # mcts
        self.model_parameters = configuration["model_parameters"]  # neural network
        self.optimizer_parameters = configuration["optimizer_parameters"]  # neural network
        self.training_parameters = configuration["training_parameters"]  # neural network
        self.arena_parameters = configuration["arena_parameters"]  # arena

        # model that is to be trained, and opponent
        n_lines = 2 * self.game_size * (self.game_size + 1)
        self.model = AZNeuralNetwork(
            io_units=n_lines,
            model_parameters=self.model_parameters
        )

        # TODO implement GPU support

    def loop(self):
        """
        Perform numEpochs epochs of training. 
        During each epoch, a specific number of games of self-play are performed, 
        resulting in a list training examples.
        The neural network is then trained with the training examples.
        Finally, the updated neural network competes against the previous network
        and is accepted only when it wins a specific fraction of games.
        """

        for epoch in range(1, self.epochs + 1):
            if (epoch + 1) % 100 == 0:
                log.info(f"Progress: Epoch {epoch}/{self.epochs}")

            # 1) perform self-play to obtain training data
            log.info(f"Performing self-play using MCTS ..")
            train_examples = self.perform_self_plays(
                n_games=self.mcts_parameters["n_games"],
                n_simulations=self.mcts_parameters["n_simulations"],
                temperature_move_threshold=self.mcts_parameters["temperature_move_threshold"]
            )

            # train the current model using generated data
            log.info(f"Training the neural network using generated data ..")
            prev_model = copy.deepcopy(self.model)  # save neural network in its current state for comparison
            self.perform_model_training(
                train_examples=train_examples
            )

            # model comparison
            log.info(f"Trained model plays against previous version ..")
            win_fraction = self.arena_parameters["win_fraction"]
            n_games = self.arena_parameters["n_games"]
            arena = Arena(
                game_size=self.game_size,
                model1=self.model,
                model2=prev_model,
                n_games=n_games
            )
            wins_trained_model, wins_prev_model, draws = arena.compare()
            win_percent = wins_trained_model / n_games

            log.info(f"Compare results: (trained model) {wins_trained_model}:{wins_prev_model} (previous model). "
                     f"Trained model won {round(win_percent * 100, 2)}% of the games ({round(win_fraction * 100, 2)}% needed).")

            if win_percent >= win_fraction:
                log.info("Continuing with trained model.")
            else:
                log.info("Continuing with previous model.")
                self.model = prev_model

    def perform_model_training(self, train_examples):
        """
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
        shuffle(train_examples)
        x = [e[0] for e in train_examples]
        p = [e[1] for e in train_examples]
        v = [e[2] for e in train_examples]

        x_batched, p_batched, v_batched = [], [], []
        for i in range(0, len(train_examples), batch_size):
            x_batched += [torch.Tensor(x[i:i + batch_size])]
            p_batched += [torch.Tensor(p[i:i + batch_size])]
            v_batched += [torch.Tensor(v[i:i + batch_size])]

        current_patience = 0
        best_model = None
        best_loss = 1e10
        CrossEntropyLoss = torch.nn.CrossEntropyLoss()
        MSELoss = torch.nn.MSELoss()

        for epoch in range(epochs):
            if current_patience > patience:
                log.info(f"Performance does not improve anymore, model training is stopped after {epoch} epochs.")
                break

            log.info(f"Model Training: Epoch {epoch}")

            # train model
            self.model.train()
            for i in range(len(x_batched)):
                optimizer.zero_grad()

                p, v = self.model.forward(x_batched[i])

                loss = CrossEntropyLoss(p, p_batched[i]) + MSELoss(v, v_batched[i])
                loss.backward()
                optimizer.step()

            # evaluate model on train set
            self.model.eval()
            with torch.no_grad():
                total_loss = 0

                for i in range(len(x_batched)):
                    optimizer.zero_grad()

                    p, v = self.model.forward(x_batched[i])

                    total_loss += CrossEntropyLoss(p, p_batched[i]) + MSELoss(v, v_batched[i])

                if total_loss < best_loss:
                    best_loss = total_loss
                    best_model = copy.deepcopy(self.model)
                    current_patience = 0
                    log.info(f"New best model achieved after {epoch} epochs (loss: {best_loss}")
                else:
                    current_patience += 1

        self.model = best_model

    def perform_self_plays(self, n_games: int, n_simulations: int, temperature_move_threshold: int):
        train_examples = []
        self.model.eval()

        for _ in tqdm(range(n_games)):
            # single iteration of self-play
            train_examples += self.perform_self_play(
                n_simulations=n_simulations,
                temperature_move_threshold=temperature_move_threshold
            )
        return train_examples

    def perform_self_play(self, n_simulations: int, temperature_move_threshold: int):
        """
        Performs a single episode of self-play (until the game end).
        During playing, each turn results in a training example.
        """

        """
        For the first 30 moves of each game, the temperature is set to τ = 1; this
        selects moves proportionally to their visit count in MCTS, and ensures a diverse
        set of positions are encountered. For the remainder of the game, an infinitesimal
        temperature is used, τ→ 0.
        
        After MCTS has completed a round of simulations, and before it has chosen a move to play, 
        it has accumulated a visit count (N above) for each potential next move. 
        MCTS works such that good moves are eventually visited more often than bad moves, 
        and are a good indication of where to play.
        
        Normally, these counts are normalized and used as a distribution to choose the actual move. 
        But a so-called temperature parameter (τ) can be used to first exponentiate these counts: N^(1/τ). 
        They set τ=1 for the first 30 moves of the game (which has no effect) and then set it to an infinitesimal value 
        for the rest of the game (which, post-normalization, suppresses all values except the maximum).
        """

        # At each time-step t, an MCTS is executed using the previous iteration of the neural network
        # and a move is played by sampling the search probabilities
        game = DotsAndBoxesGame(self.game_size)
        n_moves = 0
        train_examples = []  # dataset train, containing lists of [lines_vector, probs, v]

        # iteration over time stop t in game
        while True:
            game = game.copy()  # TODO check when this is necessary
            mcts = MCTS(
                model=self.model,
                game=game,
                n_simulations=n_simulations,
            )

            temp = 1 if n_moves < temperature_move_threshold else 0
            n_moves += 1

            probs = mcts.calculate_probabilities(temp=temp)
            # TODO Include Symmetries of current Game State

            train_examples.append([game.get_lines_vector(), probs, game.get_player_at_turn()])  # v is determined later

            # sample move from using probs, and apply move
            move = np.random.choice(
                a=list(range(game.N_LINES)),
                p=probs
            )
            game.draw_line(move)

            if not game.is_running():
                state = game.get_state()
                if state == GameState.DRAW:
                    winner = 0
                elif state == GameState.WIN_PLAYER_1:
                    winner = Value.PLAYER_1
                elif state == GameState.WIN_PLAYER_2:
                    winner = Value.PLAYER_2

                # v in {-1, 0, 1}, depending on whether the player lost, draw, or won
                for i, (_, _, player_at_turn) in enumerate(train_examples):

                    if player_at_turn == winner:
                        train_examples[i][2] = 1
                    elif winner == 0:
                        train_examples[i][2] = 0
                    else:
                        train_examples[i][2] = -1
                return train_examples
