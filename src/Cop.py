"""
File description: Cop class represents a cop player in the game. It is a subclass of the Player class.

Author: Michal Cejpek (xcejpe05@stud.fit.vutbr.cz)
"""

from src.Player import Player
from src.colors import *


class Cop(Player):
    def __init__(self, number: int, name: str = "", color: () = GREEN):
        super().__init__(number, name, color)
        self.position = None
        self.start_position = None

    def is_cop(self):
        return True
