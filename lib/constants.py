from enum import Enum


class GameResult:
    WIN_PLAYER_1 = 1
    WIN_PLAYER_2 = 2
    DRAW = 0


class Value:
    """
    Valid values for lines and boxes.
    """
    PLAYER_1 = 1  # red
    PLAYER_2 = 2  # blue
    FREE = 0


class GameState(Enum):
    RUNNING = "Running"
    WIN_PLAYER_1 = "Finished: Player 1 won"
    WIN_PLAYER_2 = "Finished: Player 2 won"
    DRAW = "Finished: Draw"
