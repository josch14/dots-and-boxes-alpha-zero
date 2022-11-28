# local import
from .constants import GameState, Value

# system import
from typing import Tuple, List
import math
import numpy as np
import copy

class DotsAndBoxesGame:

    def __init__(self, size: int=3):
        """
        size: int
            - size of the board; size of n means that there are a total of 
            n*n boxes for the players to capture by drawing the lines
            - example: n=2 -> 4 boxes -> 2*n*(n+1) = 2*2*3 = 12 possible lines
            - the board with the corresponding line numbers then looks as follow
            (i.e. first horizontal lines are numbered, then the vertical lines)
                +  0 + 1  +
                6    8   10
                +  2 + 3  +
                7    9   11
                +  4 + 5  +   
        """
        self.__player_at_turn = Value.PLAYER_1
        self.__state = GameState.RUNNING
        self.SIZE = size
        self.N_LINES = 2 * size * (size + 1)

        # lines
        self.__n_lines_drawn = 0 
        self.__lines_vector = np.zeros((self.N_LINES, 1)) # Values.FREE

        # boxes which can be captured by drawing lines
        self.N_BOXES = size * size
        self.__boxes = np.zeros((size, size)) # Values.FREE

    def __eq__(self, obj):
        if obj is None:
            return False
            
        if not isinstance(obj, DotsAndBoxesGame):
            return False
        
        if not self.get_player_at_turn() == obj.get_player_at_turn() or \
            not self.get_state() == obj.get_state() or \
            not self.get_n_lines_drawn() == obj.get_n_lines_drawn() or \
            not np.array_equal(self.get_lines_vector(), obj.get_lines_vector()) or \
            not np.array_equal(self.get_boxes(), obj.get_boxes()):

            return False

        return True


    """
    Game Logic.
    """
    def draw_line(self, line: int) -> None:
        
        # update lines_vector
        self.set_line_value(line, self.get_player_at_turn())

        # increment number of drawn lines
        self.incr_n_lines_drawn()

        # heck whether a new box was captured. This is the case when the line 
        # belongs to a box (maximum of two boxes) which now has 4 drawn lines
        # 1) get the box or boxes (i.e., the indices) to which the line belongs
        # 2) check whether a box now has 4 lines
        box_captured = False
        for box in self.get_boxes_of_line(line):
            
            lines = self.get_lines_of_box(box)
            line_values = [self.get_line_value(l) for l in lines]
            drawn_lines = [l for l in line_values if l != Value.FREE]

            if len(drawn_lines) == 4:
                # drawing the line resulted in capturing the box
                self.capture_box(
                    i=box[0], 
                    j=box[1],
                    player=self.get_player_at_turn()
                )
                box_captured = True

        # when a box was captured by a player, the player it is the player's 
        # turn again
        if not box_captured:
            self.switch_player_at_turn()
        else:
            # box was captured
            # check if match is finished now
            self.check_finished()


    def check_finished(self) -> None:
        # player reached necessary number of boxes to capture to win the game
        boxes_to_win = math.floor(self.get_boxes().size / 2) + 1

        if ((self.get_boxes() == Value.PLAYER_1).sum()) >= boxes_to_win:
            self.__state = GameState.WIN_PLAYER_1

        elif ((self.get_boxes() == Value.PLAYER_2).sum()) >= boxes_to_win:
            self.__state = GameState.WIN_PLAYER_2

        # no lines left to draw: draw
        elif self.get_n_lines_drawn() == self.N_LINES:
            self.__state = GameState.DRAW


    """
    Important methods to get line and box information.
    """
    def get_boxes_of_line(self, line: int) -> List[Tuple[int, int]]:

        if line < int(self.N_LINES/2):
            # horizontal line
            i = line // self.SIZE
            j = line % self.SIZE # column for both boxes

            if i == 0:
                return [(i, j)]
            elif i == self.SIZE:
                return [(i-1, j)]
            else:
                return [(i-1, j), (i, j)]

        else:
            # vertical line
            line = line - int(self.N_LINES/2)
            j = line // self.SIZE
            i = line % self.SIZE # row for both boxes 

            if j == 0:
                return [(i, j)]
            elif j == self.SIZE:
                return [(i, j-1)]
            else:
                return [(i, j-1), (i, j)] # [left box, right box]


    def get_lines_of_box(self, box: Tuple[int, int]) -> List[int]:
        i = box[0]
        j = box[1]

        # horizontal lines
        line_top = i * self.SIZE + j # top line
        line_bottom = (i+1) * self.SIZE + j # bottom line

        # vertical lines
        line_left = int(self.N_LINES/2) + j * self.SIZE + i # left line
        line_right = int(self.N_LINES/2) + (j+1) * self.SIZE + i # right line 

        return [line_top, line_bottom, line_left, line_right]


    """
    Class getters.
    """
    def get_lines_vector(self) -> np.ndarray:
        return self.__lines_vector

    def get_line_value(self, line: int) -> Value:
        assert 0 <= line and line < self.N_LINES, \
            f"Invalid line number (received {line}, limits: [0, {self.N_LINES}))."
        return self.__lines_vector[line]

    def get_box_value(self, box: Tuple[int, int]) -> Value:
        return self.__boxes[box[0], box[1]]

    def get_n_lines_drawn(self) -> int:
        return self.__n_lines_drawn

    def get_player_at_turn(self) -> Value:
        return self.__player_at_turn

    def get_boxes(self) -> np.ndarray:
        return self.__boxes

    def is_running(self) -> bool:
        return self.__state == GameState.RUNNING

    def get_state(self) -> GameState:
        return self.__state

    def get_valid_moves(self) -> List[int]:
        return np.where(self.__lines_vector == Value.FREE)[0].tolist()

    def copy(self):
        return copy.deepcopy(self)


    """
    Class setters.
    """
    # def set_player_at_turn(self, player_at_turn: int):
    #     self
    def set_line_value(self, line: int, value: int) -> None:
        assert 0 <= line and line < self.N_LINES, \
            "Invalid line number."
        assert value in [Value.PLAYER_1, Value.PLAYER_2], \
            f"Invalid line value ({line} not in {[Value.PLAYER_1, Value.PLAYER_2]})."
        assert self.get_line_value(line) == Value.FREE, \
            "Invalid line value (line should not be drawn yet)."
        self.__lines_vector[line] = value

    def incr_n_lines_drawn(self) -> None:
        self.__n_lines_drawn += 1

    def switch_player_at_turn(self) -> None:
        if self.__player_at_turn == Value.PLAYER_1:
            self.__player_at_turn = Value.PLAYER_2
        else:
            self.__player_at_turn = Value.PLAYER_1

    def capture_box(self, i: int, j: int, player: int) -> None:
        assert player in [Value.PLAYER_1, Value.PLAYER_2], \
            "Box needs to be captured by a Player."
        self.__boxes[i][j] = player


game = DotsAndBoxesGame(3)
game2 = DotsAndBoxesGame(3)
game.draw_line(1)
game.draw_line(2)
game2.draw_line(1)
game2.draw_line(2)

a = [game]

print(game == game2)