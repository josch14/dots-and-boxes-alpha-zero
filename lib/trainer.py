from .game import DotsAndBoxesGame



class Trainer():

    def __init__(self, size: int = 3):
        self.game = DotsAndBoxesGame(size)
