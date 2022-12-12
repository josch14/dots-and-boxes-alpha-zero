from typing import Tuple, List
from random import randint
import math
import numpy as np


class DotsAndBoxesGame:

    def __init__(self, size: int):
        """
        size: int
            - size of the board; size of n means that there are a total of 
            n*n boxes for the players to capture by drawing the lines
            - example: n=2 -> 4 boxes -> 2*n*(n+1) = 2*2*3 = 12 possible lines
            - the board with the corresponding line numbers then look as follows
            (i.e. first horizontal lines are numbered, then the vertical lines)
                +  0 + 1  +
                6    8   10
                +  2 + 3  +
                7    9   11
                +  4 + 5  +   
        """

        # x values used lines_vector, player_at_turn and game_winner:
        # x = 1 <-> Player 1
        # x = 0 <-> None of both
        # x = -1 <-> Player 2
        self.result = None
        self.player_at_turn = 1 if randint(0, 1) == 1 else -1  # player with first turn
        self.SIZE = size
        self.N_LINES = 2 * size * (size + 1)

        # lines (board values)
        self.n_lines_drawn = 0
        self.lines_vector = np.zeros((self.N_LINES,), dtype=np.float32)  # torch uses float32

        # boxes which can be captured by drawing lines
        self.N_BOXES = size * size
        self.boxes = np.zeros((size, size))

    def __eq__(self, obj) -> bool:
        if obj is None:
            return False

        if not isinstance(obj, DotsAndBoxesGame):
            return False

        if not self.player_at_turn == obj.player_at_turn or \
                not self.result == obj.result or \
                not self.SIZE == obj.SIZE or \
                not self.N_LINES == obj.N_LINES or \
                not self.n_lines_drawn == obj.n_lines_drawn or \
                not np.array_equal(self.lines_vector, obj.lines_vector) or \
                not self.N_BOXES == obj.N_BOXES or \
                not np.array_equal(self.boxes, obj.boxes):
            return False

        return True

    """
    Game Logic.
    """

    def execute_move(self, line: int) -> None:

        # execute move means drawing the line
        self.draw_line(line, self.player_at_turn)
        self.n_lines_drawn += 1

        # check whether a new box was captured
        # this is the case when the line belongs to a box (maximum of two boxes) which now has 4 drawn lines

        box_captured = False
        # step 1: get the box or boxes (i.e., the indices) to which the line belongs
        for box in self.get_boxes_of_line(line):
            lines = self.get_lines_of_box(box)

            # step 2: check whether such box (now) has 4 lines -> box captures
            if len([self.lines_vector[l] for l in lines if self.lines_vector[l] != 0]) == 4:
                self.capture_box(
                    row=box[0],
                    col=box[1],
                    value=self.player_at_turn
                )
                box_captured = True

        # when the player captured a box by drawing a line, it's the player's turn again
        if not box_captured:
            self.switch_player_at_turn()
        else:
            # check whether the game is finished now
            self.check_finished()

    def check_finished(self) -> None:
        assert self.result is None, "when check_finished() is called, self.result should be None"

        # player reached necessary number of captured boxes to win the game
        boxes_to_win = math.floor(self.N_BOXES / 2) + 1

        if ((self.boxes == 1).sum()) >= boxes_to_win:
            self.result = 1  # win: player 1

        elif ((self.boxes == 2).sum()) >= boxes_to_win:
            self.result = -1  # win: player 2

        elif self.n_lines_drawn == self.N_LINES:
            self.result = 0  # draw (no lines left to draw)

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
    Class getters.
    """

    def is_running(self) -> bool:
        return self.result is None

    def get_valid_moves(self) -> List[int]:
        return np.where(self.lines_vector == 0)[0].tolist()

    """
    Class setters.
    """

    def draw_line(self, line: int, value: int) -> None:
        assert 0 <= line < self.N_LINES, "Invalid line number."
        assert value in [-1, 1], f"Invalid line value ({line} not in {[-1, 1]})."
        assert self.lines_vector[line] == 0, "Invalid line value (line should not be drawn yet)."
        self.lines_vector[line] = value

    def switch_player_at_turn(self) -> None:
        self.player_at_turn *= -1

    def capture_box(self, row: int, col: int, value: int) -> None:
        assert value in [-1, 1], \
            "Box needs to be captured by a Player."
        self.boxes[row][col] = value
