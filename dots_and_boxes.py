from typing import Tuple, List
import numpy as np
from enum import Enum
import math

PLAYER_1 = 1
PLAYER_2 = 2
FREE = 0

class GameState(Enum):
    RUNNING = "Running"
    WIN_PLAYER_1 = "Finished: Player 1 won"
    WIN_PLAYER_2 = "Finished: Player 2 won"
    DRAW = "Finished: Draw"

class DotsAndBoxes:

    def __init__(self, size: int=3):
        """
        size: int
            - size of the grid. Size of n means that there are a total of 
            n*n boxes for the players to mark/win by drawing the lines
            - example: n=2 -> 4 boxes -> 2*n*(n+1) = 2*2*3 = 12 possible lines
        lines_horizontal: np.ndarray
            - drawed lines in horizontal direction
            - code: 0 (no line), 1 (Player 1), 2 (Player 2)
            - example: lines_horizontal[1][0]=1 means that the first player 
            drawed the line in the second row (i=1) from the first to the 
            second dot (j=0)
        lines_vertical: np.ndarray
            - drawed lines in vertical direction
            - example: lines_vertical[1][0]=1 means that the first player 
            drawed the line in the second column (i=1) from the first to the 
            second dot (j=0)
        """
        self.__player_at_turn = PLAYER_1
        self.__state = GameState.RUNNING
        self.SIZE = size
        self.N_LINES = 2 * size * (size + 1)

        # line representation
        """
        .  0 . 1  .
        6    8   10
        .  2 . 3  .
        7    9   11
        .  4 . 5  .        
        """
        self.__n_lines_drawn = 0 
        self.__lines_vector = np.zeros((self.N_LINES, 1))

        # box representation
        self.N_BOXES = size * size
        self.__boxes = np.zeros((size, size))


    def get_line_value(self, line: int):

        assert 0 <= line and line < self.N_LINES, \
            f"Invalid line number (received {line}, limits: [0, {self.N_LINES}))."
        return self.__lines_vector[line]


    def set_line_value(self, line: int, value: int):
        assert 0 <= line and line < self.N_LINES, \
            "Invalid line number."
        assert value in [PLAYER_1, PLAYER_2], \
            "Invalid line value."
        assert self.get_line_value(line) == FREE, \
            "Invalid line value (line should not be drawn yet)."

        self.__lines_vector[line] = value


    def incr_n_lines_drawn(self):
        self.__n_lines_drawn += 1

    def get_n_lines_drawn(self):
        return self.__n_lines_drawn

    def switch_player_at_turn(self):
        self.__player_at_turn = not self.__player_at_turn

    def get_player_at_turn(self):
        return self.__player_at_turn

    def get_state(self):
        return self.__state

    def get_boxes(self):
        return self.__boxes

    def get_state(self):
        return self.__state

    def capture_box(self, i: int, j: int, player: int):
        assert player in [PLAYER_1, PLAYER_2], \
            "Box needs by captured by a Player."

        self.__boxes[i][j] = player


    def draw_line(self, line: int, line_value: int):
        """
        Draw a line. 
        Returns true when drawing the line resulted in winning the game.
        """
        # update lines_vector
        self.set_line_value(line, line_value)

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
            drawn_lines = [l for l in line_values if l != FREE]

            if len(drawn_lines) == 4:
                # drawing the line resulted in capturing the box
                self.capture_box(
                    i=box[0], 
                    j=box[1],
                    player=line_value
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


    def check_finished(self):
        # player reached necessary number of boxes to capture to win the game
        boxes_to_win = math.floor(self.get_boxes().size / 2) + 1

        if ((self.get_boxes() == PLAYER_1).sum()) >= boxes_to_win:
            self.__state = GameState.WIN_PLAYER_1

        elif ((self.get_boxes() == PLAYER_2).sum()) >= boxes_to_win:
            self.__state = GameState.WIN_PLAYER_2

        # no lines left to draw: draw
        elif self.get_n_lines_drawn() == self.N_LINES:
            self.__state = GameState.DRAW

        # game not yet finished
        else:
            return False


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
                return [(i, j-1), (i, j)]


    def get_lines_of_box(self, box: Tuple[int, int]) -> List[int]:
        """
        Get indices of the lines that belong to the input box.
        """
        i = box[0]
        j = box[1]

        # horizontal lines
        line_top = i * self.SIZE + j # top line
        line_bottom = (i+1) * self.SIZE + j # bottom line

        # vertical lines
        line_left = int(self.N_LINES/2) + j * self.SIZE + i # left line
        line_right = int(self.N_LINES/2) + (j+1) * self.SIZE + i # right line 

        return [line_top, line_bottom, line_left, line_right]

        

    def state_string(self) -> str:
        boxes_player_1 = (self.get_boxes() == PLAYER_1).sum()
        boxes_player_2 = (self.get_boxes() == PLAYER_2).sum()
        str = f"Game State: {self.__state.value}, " \
            f"Score: (Player 1) {boxes_player_1}:{boxes_player_2} (Player 2)"
        return str

    def grid_string(self) -> str:
        # TODO make sure that board is small enough that only lines with numbers
        # < 100 are allowed for sensible print of the board
        
        str = ""

        def str_horizontal_line(line: int, last_column: bool) -> str:
            str = "+------" if self.get_line_value(line) > 0 else \
                    "+  {: >2d}  ".format(line)

            return (str + "+") if last_column else str


        def str_vertical_line(left_line: int, print_line_number: bool) -> str:

            if self.get_line_value(left_line) > 0:
                return "|      "
            else:
                if print_line_number:
                    return "{: >2d}     ".format(left_line)
                else:
                    return "       "


        # iterate through boxes from top to bottom, left to right
        for i in range(self.SIZE):

            # use top line
            for j in range(self.SIZE):
                str += str_horizontal_line(
                    line=self.get_lines_of_box((i, j))[0], 
                    last_column=(j == self.SIZE - 1)
                )
            str += "\n"

            # use left and right lines
            for repeat in range(3):
                for j in range(self.SIZE):
                    str += str_vertical_line(
                        left_line=self.get_lines_of_box((i, j))[2], 
                        print_line_number=(repeat == 1)
                    )

                # last vertical line in a row
                right_line=self.get_lines_of_box((i, self.SIZE-1))[3]
                if self.get_line_value(right_line) > 0:
                    str += "|"
                else:
                    if repeat == 1:
                        str += f"{right_line}"
                str += "\n"


            # print bottom lines for the last row of boxes
            if i == self.SIZE - 1:
                for j in range(self.SIZE):
                    str += str_horizontal_line(
                        line=self.get_lines_of_box((i, j))[1], 
                        last_column=(j == self.SIZE - 1)
                    )
                str += "\n"

        return str
        

def main():
    game = DotsAndBoxes(size=3)

    while (True):
        game_state = game.get_state()
        if game_state != GameState.RUNNING:
            break
        
        print()
        print(game.state_string())
        print(game.grid_string())

        player_str = "Player 1" if game.get_player_at_turn() == PLAYER_1 else "Player 2"
        line = int(input(player_str + ": Please enter a free line number: "))
        game.draw_line(line, 1)

    print(game.state_string())


if __name__ == '__main__':
    main()





"""
TRASH
"""

        # # 2) update lines_horizontal OR lines_vertical
        # if line < self.N_LINES/2:
        #     # horizontal line
        #     i = line // self.SIZE
        #     j = line % self.SIZE
        #     self.lines_horizontal[i][j] = line_value
        # else:
        #     # vertical line
        #     line = line - self.N_LINES/2
        #     i = line // self.SIZE
        #     j = line % self.SIZE
        #     self.lines_vertical[i][j] = line_value



# def get_line_value(self, line: int):
#     """
#     - the lines are numbered (line = 0...N_LINES-1), first the horizontal
#     lines (0..2*size-1), then the vertical lines
#     - example for size=2 (every number is a line):
#         .  0 . 1  .
#         6    8   10
#         .  2 . 3  .
#         7    9   11
#         .  4 . 5  .
#     """
#     assert 0 <= line and line < self.N_LINES, \
#         "Invalid line number."
    
#     if line < self.N_LINES/2:
#         # horizontal line
#         i = line // self.SIZE
#         j = line % self.SIZE
#         return self.lines_horizontal[i][j]
#     else:
#         # vertical line
#         line = line - self.N_LINES/2
#         i = line // self.SIZE
#         j = line % self.SIZE
#         return self.lines_vertical[i][j]

