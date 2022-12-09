from typing import Tuple

from .game import DotsAndBoxesGame
from .model import AZNeuralNetwork
from .constants import GameState, Value
import numpy as np

class AZNode:
    """
    The search tree consists of edges (s,a) (board representation s, move a),
    storing prior probabilities P(s,a), visit counts N(s,a) and 
    action values Q(s,a).

    AZNode implements the search tree of AlphaZero. 
    - each Node corresponds to a board representation s
    - P(s,a), N(s,a), Q(s,a), i.e., values that depend on s and a, are saved in
      the child node s', the result when executing move a on s
    """
    def __init__(
        self,
        parent,               # parent node
        a: int,               # move to get here
        s: DotsAndBoxesGame,  # current game state
        P: np.ndarray): # prior probabilities

        # only root node has no parent
        if parent is not None:
            assert isinstance(parent, AZNode)
            for child in parent.children:
                assert child.s != s and child.a != a, \
                    "identical node should not yet exist for parent (?)"

        # any node except root needs to have a corresponding move
        assert (parent is None and a is None) or \
            (parent is not None and a is not None)
        

        # create link with parent node
        if parent is not None:
            parent.children.append(self)

        # input parameters
        self.parent = parent
        self.a = a  # move to get to this node from parent
        self.s = s  # game state after applying a 
        self.P = P  # policies P(s,a) (vector) as returned by the neural network

        # further parameters
        self.children = [] # child nodes
        self.Q = {} # action values Q(s,a) (dict, with line number as key)
        self.N = {} # visit counts N(s,a) (dict, with line number as key)
        self.score = None

    def is_root(self) -> bool:
        return True if self.parent is None else False

    def is_leaf(self) -> bool:
        return True if len(self.children) == 0 else False

    def get_child_by_move(self, a: int):
        """
        Assumes that the child actually exists.
        """
        for child in self.children:
            if child.a == a:
                return child
        assert False, "Node does not contain a child with given move a."



class MCTS:

    def __init__(self, model: AZNeuralNetwork, game: DotsAndBoxesGame, n_simulations: int) -> None:

        self.model = model
        self.n_simulations = n_simulations

        # root node of the search tree (has no parent) 
        self.root = AZNode(
            parent=None,
            a=None,
            s=game,
            P=self.model.p_v(
                lines_vector=game.get_lines_vector(),
                valid_moves=game.get_valid_moves()
            )[0]
        )




    def calculate_probabilities(self, temp: int) -> [float]:
        """
        Paper: MCTS may be viewed as a self-play algorithm that, given neural network parameters and a root position s,
        computes a vector of search probabilities pi recommending moves to play (refers to (d) Play).
        The move probabilities are proportional to the exponentiated visit count for each move,
        i.e., pi(a) ~ N(s,a)^(1/temp) with
        - N being the visit count of each move from the root state
        - temp being a temperature controlling parameter.
        """

        for i in range(self.n_simulations):
            self.search(self.root)

        s_root = self.root.s

        # only valid moves may have a visit
        assert set(list(self.root.N.keys())).issubset(set(s_root.get_valid_moves())), \
            "Only valid moves may have a visit.\n" \
            f"- Visited moves: {set(list(self.root.N.keys()))}\n" \
            f"- Valid moves: {set(s_root.get_valid_moves())}"

        # probability vector should contain value for each line and should be 0 when line is already drawn
        counts = [self.root.N[a] if a in self.root.N else 0 for a in range(s_root.N_LINES)]

        if temp == 0:
            max_idx = np.array(counts).argmax()  # returns first index which contains the maximum
            probs = [0] * len(counts)
            probs[max_idx] = 1
            return probs

        # pi(a) ~ N(s,a)^(1/temp)
        probs = [n ** (1. / temp) for n in counts]
        probs = [p / float(sum(probs)) for p in probs]
        return probs

    def search(self, node: AZNode) -> float:

        # when the game is finished before reaching a non-visited node, 
        # return the score v (neural network not necessary here for v)
        if not node.s.is_running():
            # current player .. 
            # .. wins: score = 1
            # .. loses: score = -1
            # .. draw: score = 0

            state = node.s.get_state()
            player_at_turn = node.s.get_player_at_turn()

            if player_at_turn == Value.PLAYER_1 and \
                state == GameState.WIN_PLAYER_1:
                v = 1

            elif player_at_turn == Value.PLAYER_2 and \
                state == GameState.WIN_PLAYER_2:
                v = 1

            elif state == GameState.DRAW:
                v = 0

            else:
                v = -1

            return v    

        """
        (a) Select
        Traverse the tree by selecting the move/edge maximum action value Q, 
        plus an upper confidence bound (ucb) U that depends on a stored 
        prior probability P and visit count N for that move. 
        Do (a) UNTIL a leaf is approached.

        maximize Upper Confidence Bound Q(s,a) + U(s,a)
        until leaf is found, i.e., a node which was not visited yet
        U(s, a) ~ P(s,a) / (1 + N(s,a)) (proportional)
        - P(s,a): prior probability, returned by neural network
        - Q(s,a): action value
        - N(s,a): visit count
        idea: Initially prefer moves with low visit count and high prior 
        probability, but asymptotically prefer moves with high action value
        """

        ucb_opt = float('-inf')
        a_opt = -1 # determine the next move
        valid_moves = node.s.get_valid_moves()

        for a in valid_moves:
            # each move corresponds to a child node that may or may not have 
            # already been visited

            p = node.P[a]
            if a in node.N:
                # child visited at least once
                q = node.Q[a]
                n = node.N[a]
            else:
                q = 0
                n = 0

            ucb = q + p / (1 + n)

            if ucb > ucb_opt:
                ucb_opt = ucb
                a_opt = a

        a = a_opt

        if a not in node.N:
            # applying a_opt means approaching a leaf, i.e., a game state
            # that was not visited before
            child, v_child = self.expand(parent=node, a=a)

        else:
            # continue traversing, i.e. call method recursively
            child = node.get_child_by_move(a)
            v_child = self.search(child)


        # we now have a score v for the child node, either 1) by reaching a leaf 
        # while traversing (v from neural network) or 2) by finishing game (v is
        # a game score in {-1, 0, 1})

        # calculate v of current node using the child node's v
        # if-statement necessary since a player may have two turns in a row
        # after captuing a box
        if node.s.get_player_at_turn() == child.s.get_player_at_turn():
            v = v_child
        else:
            v = -v_child
        
        self.backup(node, a, v)

        return v



    def expand(self, parent: AZNode, a: int) -> Tuple[AZNode, float]:
        """
        Applying move a on parent node with state s means approaching a leaf.

        b) Expand and Evaluate
        The leaf node is expanded and the associated position s is evaluated
        by the neural network (P(s, ·),V(s)) = fθ(s); the vector of P values
        are stored in the outgoing edges from s.
        """

        # determine new game state
        s = parent.s.copy()
        s.draw_line(a)

        p, v = self.model.p_v(
            lines_vector=s.get_lines_vector(),
            valid_moves=s.get_valid_moves())

        # create leaf
        leaf = AZNode(parent=parent, a=a, s=s, P=p)

        # NOTE
        # the leaf will not carry any Q or N value (e.g., the visit count for the edge from parent to
        # the leaf will be incremented with the parent's visit count N)

        # we expanded the tree, and return v as calculated by the neural network
        # -> game will not be played until the end
        return leaf, v

    def backup(self, node: AZNode, a: int, v: float):
        """
        (c) Backup
        Action value Q is updated to track the mean of all evaluations V
        in the subtree below that action.
        """

        # update Q and N values of node
        if a in node.N:
            # child was visited before
            n = node.N[a]
            q = node.Q[a]
            node.Q[a] = (n * q + v) / (n + 1) # Q is average v value
            node.N[a] += 1

        else:
            # child is leaf that was jsut created
            node.Q[a] = v
            node.N[a] = 1