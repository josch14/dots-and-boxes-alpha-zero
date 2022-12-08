import torch
from torch import nn
import numpy as np
from typing import Tuple, List


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
            nn.Softmax(),
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
        unvalid moves have probability 0, while sum over p is 1.
        Assumes that the result will never require gradient (.detach()).
        """

        p, v = self.forward(
            torch.from_numpy(lines_vector) # model expects torch tensor
        )

        # cpu only necessary when gpu is used
        p = p.detach().cpu().numpy()
        v = v.detach().cpu().item()

        # p possibly contains p > 0 for invalid moves -> erase those
        valids = np.zeros((lines_vector.shape[1], 1))
        valids[valid_moves] = 1
        
        p = np.multiply(p, valids)

        # normalization to 1
        p /= np.sum(p)

        return p, v
        


    def train(self, examples) -> None:
        """
        Train the neural network with examples obtained from MCTS/self-play. The 
        neural network parameters are updated to .. 
        1) .. maximize the similarity of the policy vector p to the search 
           probabilities π, and to
        2) .. minimize the error between predicted winner v and game winner z
        """

        # TODO rewrite / implement
        """
        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value. The examples has
                      board in its canonical form.

        """
        pass


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