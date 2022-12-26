import torch
from torch import nn
import numpy as np
from typing import Tuple, List

# initialize logging
import logging as logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("AZNeuralNetwork")


class AZNeuralNetwork(nn.Module):
    """
    Implementation of a neural network. The model is initialized to random
    weights. As the position of a Dots and Boxes game is represented as a
    vector (and is not representable as an image), the use of convolutional
    layers does not make sense. Therefore, the model makes use of simple
    fully connected layers.


    Neural network training in
    AlphaGo Zero. The neural network takes the raw board position st as its
    input, passes it through many convolutional layers with parameters θ,
    and outputs both a vector pt, representing a probability distribution over
    moves, and a scalar value vt, representing the probability of the current
    player winning in position st.

    values used for lines_vector, current_player and game_winner:
    x = 1 <-> Current player
    x = 0 <-> None of both
    x = -1 <-> other player

    - Include Symmetries of current Game State:
    rules are invariant to colour transposition: represent the board from the perspective of the current player
    """

    """
    Deep neural network f(s) = (p,v), combining the roles of a policy network
    and value network, with
    - p (policy vector): vector of move probabilities p = P(a|s)
    - v (scalar value): probability of the current player winning from position s

    Input:
        lines_vector: the current board in vector form, where each element
                      corresponds to a line, having a Value (see enum)

    Returns:
        p: numpy array of same length as the input vector
        v: float in [-1,1]

    NOTE: Input to e.g. nn.Linear is expected to be [batch_size, features].
          Therefore, single vectors have to be fed as row vectors.
    """

    def __init__(self, io_units: int, model_parameters: dict):
        super(AZNeuralNetwork, self).__init__()

        self.io_units = io_units
        self.hidden_units = model_parameters["hidden_units"]
        self.hidden_layers = model_parameters["hidden_layers"]
        self.dropout = model_parameters["dropout"]
        self.initialize_gain = model_parameters["initialize_gain"]

        # input layer
        linear_in = nn.Linear(self.io_units, self.hidden_units)
        # nn.init.xavier_uniform_(
        #     linear_in.weight,
        #     gain=nn.init.calculate_gain('relu', self.initialize_gain)
        # )
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
            # nn.init.xavier_uniform_(
            #     linear.weight,
            #     gain=nn.init.calculate_gain('relu', self.initialize_gain)
            # )
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
        # nn.init.uniform_(linear_p_out.weight, a=0.0, b=1.0)
        # nn.init.uniform_(linear_p_out.bias, a=0.0, b=1.0)
        self.p_out = nn.Sequential(
            linear_p_out,
            nn.Softmax(dim=1),
        )

        linear_v_out = nn.Linear(self.hidden_units, 1)
        # nn.init.uniform_(linear_v_out.weight, a=0.0, b=1.0) # TODO discuss: initialization sensible?
        # nn.init.uniform_(linear_v_out.bias, a=0.0, b=1.0)
        self.v_out = nn.Sequential(
            linear_v_out,
            nn.Tanh(),
        )


    """
    s = vector of lines, from current_player points of view.
    """
    def forward(self, s):
        s = self.fully_connected_in(s)

        # TODO check whether model input really contains features in rows (i.e., lines vectors -> transpose necessary?)
        for layer in self.fully_connected_layers:
            s = layer(s)

        p = self.p_out(s)
        v = self.v_out(s).squeeze()  # one-dimensional output

        # TODO remove later (make sure we have probability distribution in each feature vector)
        # INFO: batches make first dimension
        for i in range(p.shape[0]):  # iterate over train examples in batch
            assert 1 - torch.sum(p[i, :]) < 0.00001  # sum of probabilities needs to be 1
        return p, v

    def p_v(self, s: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Simple forward using the model, but ensure that in p (policy vector) 
        invalid moves have probability 0, while sum over p is 1.
        Assumes that the result will never require gradient (.detach()).

        lines_vector is net lines_vector from DotsAndBoxesGame, brought into canonical form (i.e., 1 for current player)!!!
        s = vector of lines, from current_player points of view.
        """
        valid_moves = np.where(s == 0)[0].tolist()

        # model expects ...
        s = torch.from_numpy(s)  # ... tensor
        s = s.unsqueeze(0)  # ... batch due to batch normalization

        p, v = self.forward(s)

        # cpu only necessary when gpu is used
        p = p.squeeze().detach().cpu().numpy()
        v = v.detach().cpu().item()

        # p possibly contains p > 0 for invalid moves -> erase those
        valids = np.zeros(s.squeeze().shape)
        valids[valid_moves] = 1

        p = np.multiply(p, valids)

        # # normalization to 1
        if np.sum(p) > 0:
            p /= np.sum(p)
        else:
            # no move has a probability > 0. This means that either
            # (1) model returned p(a)=0 for all moves (unlikely to occur)
            # (2) there is no valid move left, i.e., the game is actually finished
            # In case of (2), probability of all moves is set equally

            # TODO sensible? During training with MCTS, this situation shouldn't really occur?
            # Does indeed occur as of right now (mostly after model training)
            assert not len(valid_moves) > 0 and np.sum(p) == 0, \
                "Model returned p(a)=0 for all moves."
            log.error("No valid move left. Model output p is set equally for all moves.")
            p = [1] * s.shape[0]
            p /= np.sum(p)

        return p, v

    def determine_move(self, s: np.ndarray) -> int:
        p, _ = self.p_v(s)
        move = p.argmax()
        return move

    # TODO rewrite / implement
    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        pass

    # TODO rewrite / implement
    def load_checkpoint(self, folder, filename):
        """
        Loads parameters of the neural network from folder/filename
        """
        pass
