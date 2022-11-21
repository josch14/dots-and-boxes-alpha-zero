from dots_and_boxes import DotsAndBoxes, Color, Value

# print player name in color
from termcolor import colored
import colorama
colorama.init()

def main():

    game = DotsAndBoxes(size=7)
    print()
    print(game.state_string())
    print(game.grid_string())

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

        # print new game statey
        print()
        print(game.state_string())
        print(game.grid_string())
        print()

    print(game.state_string())


if __name__ == '__main__':
    main()