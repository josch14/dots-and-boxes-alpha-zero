import torch
from torch import nn
import numpy as np
from typing import Tuple, List

# initialize logging
import logging as logging
logging.basicConfig(level = logging.INFO)
log = logging.getLogger("AZNeuralNetwork")

class AZNeuralNetwork(nn.Module):

    def __init__(self, io_dim: int, hidden_dim: int, dropout: float=0.2):
        super(AZNeuralNetwork, self).__init__()

        # input layer
        self.linear = nn.Linear(io_dim, hidden_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(io_dim, hidden_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )

        # last core layer
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )

        # output layers
        self.p = nn.Sequential(
            nn.Linear(hidden_dim, io_dim),
            nn.Softmax(dim=0),  # TODO check whether this is fine for train when batches are used
        )

        self.v = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )


    def forward(self, x):
        """
        Deep neural network f(s) = (p,v) implementation with 
        - p (policy): probability distribution P(a,s) over moves (vector)
        - v (value): estimation of the current player's winning probability (scalar)     
        
        Input:
            lines_vector: the current board in vector form, where each element
                          corresponds to a line, having a Value (see enum)

        Returns: 
            p: numpy array of same length as the input vector
            v: float in [-1,1]

        NOTE: Input to e.g. nn.Linear is expected to be [batch_size, features].
              Thereofre, single vectors have to be fed as row vectors.
        """
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        p = self.p(x) 
        v = self.v(x)
        return p, v


    def p_v(self, lines_vector: np.ndarray, valid_moves: List[int]) -> Tuple[np.ndarray, float]:
        """
        Simple forward using the model, but ensure that in p (policy vector) 
        invalid moves have probability 0, while sum over p is 1.
        Assumes that the result will never require gradient (.detach()).
        """

        # lines_vector is column vector, model/torch expects features in rows
        p, v = self.forward(
            torch.from_numpy(lines_vector) # model expects torch tensor
        )

        # cpu only necessary when gpu is used
        p = p.detach().cpu().numpy()
        v = v.detach().cpu().item()

        # p possibly contains p > 0 for invalid moves -> erase those
        valids = np.zeros(lines_vector.shape)
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
            assert not len(valid_moves) > 0 and np.sum(p) == 0, \
                "Model returned p(a)=0 for all moves."
            log.error("No valid move left. Model output p is set equally for all moves.")
            p = [1] * lines_vector.shape[0]
            p /= np.sum(p)

        return p, v

    def determine_move(self, lines_vector: np.ndarray, valid_moves: List[int]):
        p, _ = self.p_v(lines_vector, valid_moves)
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