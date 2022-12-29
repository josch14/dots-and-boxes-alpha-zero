import logging
import os
import torch
from torch import nn
import numpy as np
from typing import Tuple


class AZNeuralNetwork(nn.Module):
    """
    AlphaZero neural network f(s) = (p,v) implementation, combining the roles of a policy network and value network, with
    - p (policy vector): vector of move probabilities p = P(a|s)
    - v (scalar value): probability of the current player winning from position s

    As the position of a Dots and Boxes game is represented as a vector (and is not representable as an image), the use
    of convolutional layers does not make sense (as opposed to the original paper). Therefore, the model makes use of
    simple fully connected layers. Furthermore, the model is initialized to random weights.
    """

    def __init__(self, game_size: int, model_parameters: dict, inference_device: torch.device):
        super(AZNeuralNetwork, self).__init__()

        self.inference_device = inference_device

        self.game_size = game_size
        self.io_units = 2 * self.game_size * (self.game_size + 1)  # = n_lines
        self.hidden_units = model_parameters["hidden_units"]
        self.hidden_layers = model_parameters["hidden_layers"]
        self.dropout = model_parameters["dropout"]

        # input layer
        linear_in = nn.Linear(self.io_units, self.hidden_units)
        nn.init.xavier_normal_(linear_in.weight)  # weight init
        linear_in.bias.data.fill_(0.01)  # bias init
        self.fully_connected_in = nn.Sequential(
            linear_in,
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_units),
            nn.Dropout(self.dropout)
        )

        # hidden layers
        self.fully_connected_layers = []
        for i in range(self.hidden_layers - 1):
            linear = nn.Linear(self.hidden_units, self.hidden_units)
            nn.init.xavier_normal_(linear.weight)  # weight init
            linear.bias.data.fill_(0.01)  # bias init
            self.fully_connected_layers.append(
                nn.Sequential(
                    linear,
                    nn.ReLU(),
                    nn.BatchNorm1d(self.hidden_units),
                    nn.Dropout(self.dropout)
                )
            )
        self.fully_connected_layers = nn.ModuleList(self.fully_connected_layers)

        # output layers
        linear_p_out = nn.Linear(self.hidden_units, self.io_units)
        nn.init.xavier_normal_(linear_p_out.weight)  # weight init
        linear_p_out.bias.data.fill_(0.01)  # bias init
        self.p_out = nn.Sequential(
            linear_p_out,
            nn.Softmax(dim=1),
        )

        linear_v_out = nn.Linear(self.hidden_units, 1)
        nn.init.xavier_normal_(linear_v_out.weight)  # weight init
        linear_v_out.bias.data.fill_(0.01)  # bias init
        self.v_out = nn.Sequential(
            linear_v_out,
            nn.Tanh(),
        )

    def forward(self, s):
        """
        Simple forward though the neural network. In this simple form, this method is only used during training of the
        neural network. During self-play using MCTS and model evaluation, p_v() is used which makes use of this method.

        NOTE: Input to e.g. nn.Linear is expected to be [batch_size, features].
            Therefore, single vectors have to be fed as row vectors.

        Parameters
        ----------
        s : torch.tensor
            position vector s, assumed to be in its canonical form (rules are invariant to colour transposition:
            represent the board from the perspective of the current player (=1))

        Returns
        -------
        p, v : [torch.tensor, torch.tensor]
            policy vector p (potentially containing values > 0 for invalid moves), value v
        """
        s = self.fully_connected_in(s)

        for layer in self.fully_connected_layers:
            s = layer(s)

        p = self.p_out(s)
        v = self.v_out(s).squeeze()  # one-dimensional output

        return p, v

    def p_v(self, s: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Performs a single forward through the neural network. As opposed to forward(), this method ensures that in p
        invalid moves have probability 0 (while still ensuring a probability distribution in p).

        Parameters
        ----------
        s : np.ndarray
            position vector s, assumed to be in its canonical form (rules are invariant to colour transposition:
            represent the board from the perspective of the current player (=1))

        Returns
        -------
        p, v : [np.ndarray, float]
            policy vector p (containing values >= 0 only for valid moves), value v
        """
        valid_moves = np.where(s == 0)[0].tolist()
        assert len(valid_moves) > 0, "No valid move left, model should not be called in this case"

        # model expects ...
        s = torch.from_numpy(s).to(self.inference_device)  # ... tensor
        s = s.unsqueeze(0)  # ... batch due to batch normalization

        # cpu only necessary when gpu is used
        p, v = self.forward(s)
        p = p.squeeze().detach().cpu().numpy()
        v = v.detach().cpu().item()

        # p possibly contains p > 0 for invalid moves -> erase those
        valid = np.zeros(s.squeeze().shape)
        valid[valid_moves] = 1

        p_valid = np.multiply(p, valid)
        if np.sum(p_valid) == 0:
            logging.warning(f"Model did not return a probability larger than zero for any valid move:\n"
                            f"(p,v) = {(p, v)} with valid moves {valid_moves}.")
            # set probability equally for all valid moves
            p_valid = np.multiply([1] * s.shape[0], valid)

        # normalization to sum 1
        p = p_valid / np.sum(p_valid)

        return p, v

    def determine_move(self, s: np.ndarray) -> int:
        """
        For the current position s, determine the next move for the current player. This is done by simply determining
        the move which has the largest probability among all valid moves as put out by the neural network.

        Parameters
        ----------
        s : np.ndarray
            position vector s, assumed to be in its canonical form (rules are invariant to colour transposition:
            represent the board from the perspective of the current player (=1))

        Returns
        -------
        move : int
            the valid move for which the neural network determines the largest probability
        """
        p, _ = self.p_v(s)
        move = p.argmax()

        valid_moves = np.where(s == 0)[0].tolist()
        assert move in valid_moves, f"move {move} is not a valid move in {valid_moves}"

        return move

    def save_checkpoint(self, model_name: str):
        path = "checkpoints/" + model_name + ".pt"
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, model_name: str):
        path = "checkpoints/" + model_name + ".pt"
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            print(f"Loading model from {path} was successful.")
        else:
            print(f"Couldn't load model from {path}: path does not exist (yet)")


