import math

from src.Player import Player
from src.colors import *


class MrX(Player):
    def __init__(self, number: int, name: str = "", color: () = WHITE):
        super().__init__(number, name, color)
        self.last_known_position = None
        self.position = None
        self.start_position = None

    def mr_x_reveal_position(self):
        if self.number == 0:
            self.last_known_position = self.position
        return self
    
    def is_mr_x(self):
        return True
