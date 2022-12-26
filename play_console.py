import copy
import sys

import numpy as np

# local import
from lib.game import DotsAndBoxesGame

# enable colored prints
from termcolor import colored
import colorama
colorama.init()


def main():
    game = DotsAndBoxesPrinter(size=2)
    print()
    print(game.state_string())
    print(game.board_string())

    while game.is_running():

        # print draw request
        player_str = colored(
            f"Player {1 if game.current_player == 1 else 2}",
            Color.PLAYER_1 if game.current_player == 1 else Color.PLAYER_2
        )
        print(player_str + ": Please enter a free line number: ", end="")

        # process draw request
        while True:
            line = int(input())
            if line in game.get_valid_moves():
                break
            print(f"Line {line} is not a valid move. Please select a move in {game.get_valid_moves()}.")
        game.execute_move(line)

        # print new game state
        print()
        print(game.state_string())
        print(game.board_string())
        print()

        print("TESTESTEST")
        save = np.copy(game.lines_vector)
        equivalents = game.get_equivalent_positions()
        for i, s in enumerate(equivalents):
            print(f"\n\n\nNUM: {i}")
            game.lines_vector = s
            print(game.board_string())

        game.lines_vector = save





    print(game.state_string())


class Color:
    PLAYER_1 = "red"
    PLAYER_2 = "green"


class DotsAndBoxesPrinter(DotsAndBoxesGame):

    def __init__(self, size):
        super().__init__(size=size)

    """
    Add methods to the DotsAndBoxes game class the enable you to play the
    game in a console.
    """

    def state_string(self) -> str:
        boxes_player_1 = (self.boxes == 1).sum()
        boxes_player_2 = (self.boxes == -1).sum()
        s = "Score: (" + \
            colored("Player 1", Color.PLAYER_1) + \
            f") {boxes_player_1}:{boxes_player_2} (" + \
            colored("Player 2", Color.PLAYER_2) + ")"
        return s

    def str_horizontal_line(self, line: int, last_column: bool) -> str:

        value = self.lines_vector[line]
        color = value_to_color(value)

        s = "+" + colored("------", color) if value != 0 else \
            "+  {: >2d}  ".format(line)
        return (s + "+") if last_column else s

    def str_vertical_line(self, left_line: int, print_line_number: bool) -> str:

        value = self.lines_vector[left_line]
        color = value_to_color(value)

        if value != 0:
            s = colored("|", color)

            # color the box when the box right to the line is already captured
            box = self.get_boxes_of_line(left_line)[-1]
            box_value = self.boxes[box[0], box[1]]
            if box_value == 0:
                return s + "      "
            else:
                color = value_to_color(box_value)
                return s + colored("======", color)

        else:
            if print_line_number:
                return "{: >2d}     ".format(left_line)
            else:
                return "       "

    def board_string(self) -> str:
        if self.SIZE > 6:
            sys.exit("ERROR: To ensure the output quality in the console, " + \
                     "the board size of games that are printed is limited to 6.\n")

        # iterate through boxes from top to bottom, left to right
        s = ""
        for i in range(self.SIZE):

            # 1) use top line
            for j in range(self.SIZE):
                s += self.str_horizontal_line(
                    line=self.get_lines_of_box((i, j))[0],
                    last_column=(j == self.SIZE - 1)
                )
            s += "\n"

            # 2) use left and right lines
            for repeat in range(3):
                for j in range(self.SIZE):
                    s += self.str_vertical_line(
                        left_line=self.get_lines_of_box((i, j))[2],
                        print_line_number=(repeat == 1)
                    )

                # last vertical line in a row
                right_line = self.get_lines_of_box((i, self.SIZE - 1))[3]
                value = self.lines_vector[right_line]
                if value != 0:
                    s += colored("|", value_to_color(value))
                else:
                    if repeat == 1:
                        s += f"{right_line}"
                s += "\n"

            # 3) print bottom lines for the last row of boxes
            if i == self.SIZE - 1:
                for j in range(self.SIZE):
                    s += self.str_horizontal_line(
                        line=self.get_lines_of_box((i, j))[1],
                        last_column=(j == self.SIZE - 1)
                    )
                s += "\n"
        return s


def value_to_color(value: int):
    return Color.PLAYER_1 if value == 1 else Color.PLAYER_2


if __name__ == '__main__':
    main()
