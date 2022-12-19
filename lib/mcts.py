import copy
import numpy as np

# local import
from lib.game import DotsAndBoxesGame
from lib.model import AZNeuralNetwork
from lib.node import AZNode


class MCTS:
    """
    In each position s, an MCTS is executed, guided by the neural network.

    Attributes
    ----------
    model : AZNeuralNetwork
        neural network for evaluating board positions
    root : AZNode
        node from which the MCTS is executed (with input position s)
    n_simulations : int
        # simulations for each MCTS (only to determine the next move)
    """

    def __init__(self,
                 model: AZNeuralNetwork,
                 s: DotsAndBoxesGame,
                 n_simulations: int) -> None:

        self.model = model
        self.root = AZNode(
            parent=None,
            a=None,
            s=s
        )
        self.n_simulations = n_simulations

    def play(self, temp: int) -> [float]:
        """
        (d) Play.
        Provides the core functionality of MCTS: output search probabilities
        recommending moves to play.

        Parameters
        ----------
        temp : int
            temperature controlling parameter

        Returns
        -------
        probs : List[float]
            move probabilities pi(a) ~ N(s,a)^(1/temp)
        """

        # perform MCTS simulations
        for i in range(self.n_simulations):
            self.search(self.root)

        s = self.root.s  # position s of root node

        # only valid moves may have a visit
        assert set(list(self.root.N.keys())).issubset(set(s.get_valid_moves()))

        # probability vector that is returned should contain value for each line
        # when line is already drawn, probability should be 0
        counts = [self.root.N[a] if a in self.root.N else 0 for a in
                  range(s.N_LINES)]

        if temp == 0:
            # select the move with maximum visit count to give the strongest
            # possible play (return value is one-hot vector)
            probs = [0] * len(counts)
            probs[np.array(counts).argmax()] = 1
            return probs

        probs = [n ** (1. / temp) for n in counts]  # pi(a) ~ N(s,a)^(1/temp)
        probs = [p / float(sum(probs)) for p in
                 probs]  # ensure probability distribution
        return probs

    def search(self, node: AZNode) -> float:
        """
        Perform a single simulation within the MCTS.

        Parameters
        ----------
        node : AZNode
            node that corresponds to the MCTS's current position s

        Returns
        -------
        v : float
            probability of the current player winning in position s
        """

        # game is finished before reaching a non-visited node
        if not node.s.is_running():
            # return the actual score v for the current player
            # in case of a winner, player_at_turn contains it (when capturing
            # a box, the player at turn does not switch)
            result = node.s.result

            if node.s.player_at_turn == result:
                v = 1
            elif result == 0:
                v = 0
            else:
                v = -1
            return v

        # reached a leaf node, i.e., a node which is visited for the first time
        if node.P is None:
            v = self.evaluate(node)
            return v

        # node visited before: continue traversing the tree
        a = self.select(node)

        if a not in node.N:
            # applying the selected move means approaching a leaf
            child = self.expand(node, a)
        else:
            child = node.get_child_by_move(a)

        # continue traversing, i.e. call method recursively
        v_child = self.search(child)

        # we now received a score v from the child node, either
        # by reaching a leaf (v in [0,1] as calculated by the neural network) or
        # by finishing the game (v in {-1, 0, 1})
        v = v_child if node.s.player_at_turn == child.s.player_at_turn else -v_child

        # backup before returning v
        self.backup(node, a, v)

        return v

    @staticmethod
    def select(node: AZNode) -> int:
        """
        (a) Select.
        Select the move with maximum action value Q, plus an upper confidence
        bound U that depends on a stored prior probability P and visit count N.

        Parameters
        ----------
        node : AZNode
            (non-leaf) node that corresponds to the MCTS's current position s

        Returns
        -------
        a_max : int
            move a for which Q(s,a) + U(s,a) is maximized
        """

        maximum = float('-inf')
        a_max = -1

        for a in node.s.get_valid_moves():
            # each move corresponds to a child node that may or may not have
            # already been visited

            p = node.P[a]
            if a in node.N:
                q = node.Q[a]
                n = node.N[a]
            else:
                q = 0
                n = 0

            # upper confidence bound U(s, a) ~ P(s, a) / (1 + N(s, a))
            ucb = p / (1 + n)

            # maximize action value Q(s,a) + upper confidence bound U(s,a)
            if q + ucb > maximum:
                maximum = q + ucb
                a_max = a
        assert a_max != -1

        return a_max

    @staticmethod
    def expand(node: AZNode, a: int):
        """
        (b) Expand (and Evaluate).
        For the input node, create the child node (i.e., we are approaching
        a leaf) that is reached when executing move a.

        Parameters
        ----------
        node : AZNode
            node that corresponds to the MCTS's current position s
        a : int
            move with which the leaf is reached from the current node

        Returns
        -------
        leaf : AZNode
            the created child node/leaf
        """
        s = copy.deepcopy(node.s)
        s.execute_move(a)
        leaf = AZNode(
            parent=node,
            a=a,
            s=s
        )
        return leaf

    def evaluate(self, leaf: AZNode) -> float:
        """
        (b) (Expand and) Evaluate.
        Evaluate the associated position of the leaf node by the neural network
        and store the vector of P values.

        Parameters
        ----------
        leaf : AZNode
            (leaf) node that corresponds to the MCTS's current position s

        Returns
        -------
        v : float
            probability of the current player winning in position s
        """
        p, v = self.model.p_v(
            lines_vector=leaf.s.lines_vector,
            valid_moves=leaf.s.get_valid_moves()
        )
        leaf.P = p
        return v

    @staticmethod
    def backup(node: AZNode, a: int, v: float) -> None:
        """
        (c) Backup.
        Update action value Q to track the mean of all evaluations v in the
        subtree below that node, and the visit count N.

        Parameters
        ----------
        node : AZNode
            node that corresponds to the MCTS's current position s
        a : int
            move that was selected and executed within the current search
        v : float
            the resulting score for this node for the current simulation
        """

        if a not in node.N:
            # leaf: node was visited for the first time
            node.Q[a] = v
            node.N[a] = 1

        else:
            n = node.N[a]
            q = node.Q[a]
            node.Q[a] = (n * q + v) / (n + 1)  # Q = mean of v
            node.N[a] += 1
