from lib.game import DotsAndBoxesGame


class AZNode:
    """
    Implements the search tree of AlphaZero. Each node corresponds to a
    position s.

    Attributes
    ----------
    parent : AZNode
        parent node
    s : DotsAndBoxesGame
        position of the node
    a : int
        move that was executed at parent's position, resulting in this node's
        position s
    children : List[AZNode]
        child nodes
    Q : dict
        action values Q[a] = Q(s,a)
    N : dict
        visit counts N[a] = N(s,a)
    P : np.ndarray
        action values P(s,a) as returned by the neural network
    """

    def __init__(self,
                 parent,
                 s: DotsAndBoxesGame,
                 a):

        if parent is not None:  # only root node has no parent
            assert isinstance(parent, AZNode)
            for child in parent.children:  # node shouldn't already exist for parent
                assert child.s != s and child.a != a

        # any node except root needs to have a corresponding move
        assert (parent is None and a is None) or \
               (parent is not None and isinstance(a, int))

        # create link with parent node
        if parent is not None:
            parent.children.append(self)

        # use constructor parameters
        self.parent = parent
        self.a = a
        self.s = s

        self.children = []
        self.Q = {}
        self.N = {}
        self.P = None

    def is_root(self) -> bool:
        return self.parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_child_by_move(self, a: int):
        for child in self.children:
            if child.a == a:
                return child
