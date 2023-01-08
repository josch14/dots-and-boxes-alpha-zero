from typing import Tuple, List
from random import randint
import math
import numpy as np


class DotsAndBoxesGame:
    """
    Implementation of the Dots-and-Boxes game, including relevant parameters for representing the game state
    and the logic for playing the game.

    Attributes
    ----------
    SIZE : int
        board size (in number of boxes per row and column)
    current_player : int
        player which is playing the next move. It can be determined manually or randomly which player should have the
        first turn of the game
        values: {-1, 1} = {player 2, player 1}
    result : int
        game result
        values: {None, -1, 0, 1} = {game is running, win player 2, draw, win player 1}
    N_LINES : int
        total number of lines that can be drawn as a result of the board size
    s : np.ndarray
        position vector s (i.e., the board representation) of length N_LINES. Each element corresponds to one line on
        the board
        element values: {-1, 0, 1} = {line drawn by player 2, line is free, line drawn by player 1}
        the line indices correspond with the lines of a Dots-and-Boxes game in the following manner (i.e., first the
        horizontal lines are numbered, then the vertical lines):
        +  0 +  1 +
        6    8   10
        +  2 +  3 +
        7    9   11
        +  4 +  5 +
    N_BOXES : int
        total number of boxes that can be captured as a result of the board size
    boxes : np.ndarray
        tracks which boxes were captured by which player
        element values: {-1, 0, 1} = {box captured by player 2, box not captured yet, box captured by player 1}
    """
    def __init__(self, size: int, starting_player: int = None):

        self.SIZE = size
        self.current_player = (1 if randint(0, 1) == 1 else -1) if starting_player is None else starting_player
        self.result = None

        # board representation in position vector s
        self.N_LINES = 2 * size * (size + 1)
        self.s = np.zeros((self.N_LINES,), dtype=np.float32)

        # boxes which can be captured by drawing lines
        self.N_BOXES = size * size
        self.boxes = np.zeros((size, size))

    def __eq__(self, obj) -> bool:
        if obj is None:
            return False

        if not isinstance(obj, DotsAndBoxesGame):
            return False

        if not self.current_player == obj.current_player or \
                not self.result == obj.result or \
                not self.SIZE == obj.SIZE or \
                not self.N_LINES == obj.N_LINES or \
                not np.array_equal(self.s, obj.s) or \
                not self.N_BOXES == obj.N_BOXES or \
                not np.array_equal(self.boxes, obj.boxes):
            return False
        return True


    """
    Class setters.
    """
    def draw_line(self, line: int):
        assert self.s[line] == 0, "line is already drawn"
        self.s[line] = self.current_player

    def switch_current_player(self):
        self.current_player *= -1

    def capture_box(self, row: int, col: int):
        assert self.boxes[row][col] == 0, "box is already captured"
        self.boxes[row][col] = self.current_player


    """
    Game Logic.
    """
    def execute_move(self, line: int):

        # execute move means drawing the line
        self.draw_line(line)

        # check whether a new box was captured
        # this is the case when the line belongs to a box (maximum of two boxes) which now has 4 drawn lines
        box_captured = False
        for box in self.get_boxes_of_line(line):
            lines = self.get_lines_of_box(box)

            if len([self.s[line] for line in lines if self.s[line] != 0]) == 4:
                self.capture_box(
                    row=box[0],
                    col=box[1]
                )
                box_captured = True

        # switch current player when the player did not capture a box by drawing the line
        if not box_captured:
            self.switch_current_player()
        else:
            self.check_finished()

    def check_finished(self):
        assert self.result is None, "result is already set"

        # player reached necessary number of captured boxes to win the game
        boxes_to_win = math.floor(self.N_BOXES / 2) + 1

        if ((self.boxes == 1).sum()) >= boxes_to_win:
            self.result = 1

        elif ((self.boxes == -1).sum()) >= boxes_to_win:
            self.result = -1

        elif np.count_nonzero(self.s == 0) == 0:  # no free lines left
            self.result = 0  # draw

    def is_running(self) -> bool:
        return self.result is None

    def get_valid_moves(self) -> List[int]:
        return np.where(self.s == 0)[0].tolist()


    """
    Important methods to get line and box information.
    """
    def get_boxes_of_line(self, line: int) -> List[Tuple[int, int]]:

        if line < int(self.N_LINES / 2):
            # horizontal line
            i = line // self.SIZE
            j = line % self.SIZE  # column for both boxes

            if i == 0:
                return [(i, j)]
            elif i == self.SIZE:
                return [(i - 1, j)]
            else:
                return [(i - 1, j), (i, j)]

        else:
            # vertical line
            line = line - int(self.N_LINES / 2)
            j = line // self.SIZE
            i = line % self.SIZE  # row for both boxes

            if j == 0:
                return [(i, j)]
            elif j == self.SIZE:
                return [(i, j - 1)]
            else:
                return [(i, j - 1), (i, j)]  # [left box, right box]

    def get_lines_of_box(self, box: Tuple[int, int]) -> List[int]:
        i = box[0]
        j = box[1]

        # horizontal lines
        line_top = i * self.SIZE + j  # top line
        line_bottom = (i + 1) * self.SIZE + j  # bottom line

        # vertical lines
        line_left = int(self.N_LINES / 2) + j * self.SIZE + i  # left line
        line_right = int(self.N_LINES / 2) + (j + 1) * self.SIZE + i  # right line

        return [line_top, line_bottom, line_left, line_right]


    """
    Import methods for self-play using MCTS and the neural network that is to be trained.
    """
    def get_canonical_s(self) -> np.ndarray:
        """
        The neural network expects the position vector s from the POV of the current player (=1).
        """
        canonical_s = self.current_player * self.s
        canonical_s[canonical_s == 0.] = 0.
        return canonical_s

    @staticmethod
    def get_rotations_and_reflections(s: np.ndarray) -> List[np.ndarray]:
        """
        For the position s, determine the equivalent position vectors, i.e., rotations and reflections.

        Parameters
        -------
        s : np.ndarray
            position vector for which the equivalent position vectors should be determined

        Returns
        -------
        equivalents : [np.ndarray]
            all rotations and reflections of current position s (8 positions, including s itself)
        """

        # rotations
        h, v = DotsAndBoxesGame.s_to_h_v(s)
        rotations = [np.copy(s)]
        for i in range(3):
            v, h = np.rot90(h), np.rot90(v)
            rotations.append(DotsAndBoxesGame.h_v_to_s(h=h, v=v))

        # reflections
        reflections = []
        for s in rotations:
            h, v = DotsAndBoxesGame.s_to_h_v(s)
            reflections.append(DotsAndBoxesGame.h_v_to_s(h=np.fliplr(h), v=np.fliplr(v)))

        return rotations + reflections

    @staticmethod
    def n_lines_to_size(n_lines: int) -> int:
        return int(-0.5 + math.sqrt(4 + 8 * n_lines) / 4)

    @staticmethod
    def s_to_h_v(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert position vector s to (h, v)-matrices representation (containing the horizontal and vertical lines).
        Example (numbers are indices of the position vector s):
        +  0 +  1 +
        6    8    10            [0  1]
        +  2 +  3 +    -->  h = [2  3]  and   v = [6  8  10]
        7    9    11            [4  5]            [7  9  11]
        +  4 +  5 +
        """

        n_lines = s.size
        size = DotsAndBoxesGame.n_lines_to_size(n_lines)

        h = np.zeros((size + 1, size), dtype=np.float32)
        v = np.zeros((size, size + 1), dtype=np.float32)

        for line in range(n_lines):
            if line < n_lines / 2:
                # horizontal line
                i = int(line // size)
                j = int(line % size)
                h[i][j] = s[line]

            else:
                # vertical line
                j = int((line - n_lines / 2) // size)
                i = int((line - n_lines / 2) % size)
                v[i][j] = s[line]

        return h, v

    @staticmethod
    def h_v_to_s(h: np.ndarray, v: np.ndarray) -> np.ndarray:

        s = np.concatenate((
            np.matrix.flatten(h, order='C'),  # row-major
            np.matrix.flatten(v, order='F')   # column-major
        ))

        return s
