# local import
from constants import GameState, Color, Value

# system import
from typing import Tuple, List
import math
import sys
import numpy as np

# print grid with colored lines and boxes
from termcolor import colored
# to make the ANSI colors used in termcolor work with the windows terminal
import colorama
colorama.init()

class DotsAndBoxes:

    def __init__(self, size: int=3):
        """
        size: int
            - size of the grid. Size of n means that there are a total of 
            n*n boxes for the players to capture by drawing the lines
            - example: n=2 -> 4 boxes -> 2*n*(n+1) = 2*2*3 = 12 possible lines
            - the grid with the corresponding line numbers then looks as follow
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
    Methods to print the game state to console.
    """
    def state_string(self) -> str:
        boxes_player_1 = (self.get_boxes() == Value.PLAYER_1).sum()
        boxes_player_2 = (self.get_boxes() == Value.PLAYER_2).sum()
        str = f"Game State: {self.__state.value}, " \
            "Score: (" + \
            colored("Player 1", Color.PLAYER_1) + \
            f") {boxes_player_1}:{boxes_player_2} (" + \
            colored("Player 2", Color.PLAYER_2) + ")"
        return str

    def grid_string(self) -> str:
        if self.SIZE > 6:
            sys.exit("ERROR: To ensure the output quality in the console, " + \
                "the grid size of games that are prointed is limited to 6.\n")

        str = ""

        def value_to_color(value: int):
            return Color.PLAYER_1 if value == Value.PLAYER_1 else Color.PLAYER_2


        def str_horizontal_line(line: int, last_column: bool) -> str:
            
            value = self.get_line_value(line)
            color = value_to_color(value)

            str = "+" + colored("------", color) if value > 0 else \
                    "+  {: >2d}  ".format(line)
            return (str + "+") if last_column else str


        def str_vertical_line(left_line: int, print_line_number: bool) -> str:

            value = self.get_line_value(left_line)
            color = value_to_color(value)

            if value > 0:
                str = colored("|", color) # line with line color

                # color the box if the box right to the line is already captured
                box_value = self.get_box_value(
                    box=self.get_boxes_of_line(left_line)[-1]
                )
                if box_value == Value.FREE:
                    return str + "      "
                else:
                    color = value_to_color(box_value)
                    return str + colored("======", color)

            else:
                if print_line_number:
                    return "{: >2d}     ".format(left_line)
                else:
                    return "       "


        # iterate through boxes from top to bottom, left to right
        for i in range(self.SIZE):

            # 1) use top line
            for j in range(self.SIZE):
                str += str_horizontal_line(
                    line=self.get_lines_of_box((i, j))[0], 
                    last_column=(j == self.SIZE - 1)
                )
            str += "\n"

            # 2) use left and right lines
            for repeat in range(3):
                for j in range(self.SIZE):
                    str += str_vertical_line(
                        left_line=self.get_lines_of_box((i, j))[2], 
                        print_line_number=(repeat == 1)
                    )

                # last vertical line in a row
                right_line=self.get_lines_of_box((i, self.SIZE-1))[3]
                value = self.get_line_value(right_line)
                if value > 0:
                    str += colored("|", value_to_color(value))
                else:
                    if repeat == 1:
                        str += f"{right_line}"
                str += "\n"

            # 3) print bottom lines for the last row of boxes
            if i == self.SIZE - 1:
                for j in range(self.SIZE):
                    str += str_horizontal_line(
                        line=self.get_lines_of_box((i, j))[1], 
                        last_column=(j == self.SIZE - 1)
                    )
                str += "\n"
        return str
        

    """
    Class getters.
    """
    def get_line_value(self, line: int):
        assert 0 <= line and line < self.N_LINES, \
            f"Invalid line number (received {line}, limits: [0, {self.N_LINES}))."
        return self.__lines_vector[line]

    def get_box_value(self, box: Tuple[int, int]):
        return self.__boxes[box[0], box[1]]

    def get_n_lines_drawn(self):
        return self.__n_lines_drawn

    def get_player_at_turn(self):
        return self.__player_at_turn

    def get_boxes(self):
        return self.__boxes

    def is_running(self):
        return self.__state == GameState.RUNNING

    """
    Class setters.
    """
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
            "Box needs by captured by a Player."
        self.__boxes[i][j] = player