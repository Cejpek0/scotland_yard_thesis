import math

from src.Player import Player
from src.colors import *


class Cop(Player):
    def __init__(self, number: int, name: str = "", color: () = WHITE):
        super().__init__(number, name, color)
        self.position = None
        self.start_position = None

    def is_cop(self):
        return True
