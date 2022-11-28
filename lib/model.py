from torch import nn
import numpy as np
from typing import Tuple, List

class AZNeuralNetwork(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super(AZNeuralNetwork, self).__init__()

        # TODO implement
        # self.linear = nn.Linear(in_dim, out_dim)


    def forward(self, lines_vector: np.ndarray) -> Tuple[np.ndarray, float]:
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

        """
        # TODO implement
        p = np.zeros((lines_vector.shape[0], 1))
        v = 0.0
        return p, v


    def p_v(self, lines_vector: np.ndarray, valid_moves: List[int]):
        """
        Simple forward using the model, but ensure that in p (policy vector) 
        unvalid moves have probability 0, while sum over p is 1
        """
        # get probability distribution, with possible p > 0 for invalid moves
        p, v = self.forward(lines_vector)

        # erase values for non-valid moves
        valids = np.zeros((lines_vector.shape[0], 1))
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