import sys
from lib.game import DotsAndBoxesGame
from lib.constants import Value

# print board with colored lines and boxes
from termcolor import colored
# to make the ANSI colors used in termcolor work with the windows terminal
import colorama
colorama.init()


def main():

    game = DotsAndBoxesPrinter(size=2)
    print()
    print(game.state_string())
    print(game.board_string())

    while (game.is_running()):
        
        # print draw request
        color = Color.PLAYER_1 if game.get_player_at_turn() == Value.PLAYER_1 \
            else Color.PLAYER_2
        player_str = colored(f"Player {game.get_player_at_turn()}", color)
        player_str += ": Please enter a free line number: "
        print(player_str, end="")

        # process draw request
        line = int(input())
        game.draw_line(line)
        # TODO Parse input 1) check: int 2) check: move can be executed

        # print new game statey
        print()
        print(game.state_string())
        print(game.board_string())
        print()

    print(game.state_string())




class Color:
    PLAYER_1 = "red" # player 1
    PLAYER_2 = "green" # player 2

class DotsAndBoxesPrinter(DotsAndBoxesGame):

    def __init__(self, size: int=3):
        super().__init__(size=size)

    """
    Add methods to the DotsAndBoxes game class the enable you to play the
    game in a console.
    """
    def state_string(self) -> str:
        boxes_player_1 = (self.get_boxes() == Value.PLAYER_1).sum()
        boxes_player_2 = (self.get_boxes() == Value.PLAYER_2).sum()
        str = f"Game State: {self.get_state().value}, " \
            "Score: (" + \
            colored("Player 1", Color.PLAYER_1) + \
            f") {boxes_player_1}:{boxes_player_2} (" + \
            colored("Player 2", Color.PLAYER_2) + ")"
        return str

    def board_string(self) -> str:
        if self.SIZE > 6:
            sys.exit("ERROR: To ensure the output quality in the console, " + \
                "the board size of games that are printed is limited to 6.\n")

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

if __name__ == '__main__':
    main()