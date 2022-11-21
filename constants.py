from enum import Enum

class Color:
    PLAYER_1 = "red" # player 1
    PLAYER_2 = "green" # player 2

class Value:
    """
    Valid values for lines and boxes.
    """
    PLAYER_1 = 1 # red
    PLAYER_2 = 2 # blue
    FREE = 0


class GameState(Enum):
    RUNNING = "Running"
    WIN_PLAYER_1 = "Finished: Player 1 won"
    WIN_PLAYER_2 = "Finished: Player 2 won"
    DRAW = "Finished: Draw"