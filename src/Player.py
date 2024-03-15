import math

from src.colors import *


class Player:
    def __init__(self, number: int, name: str = "", color: () = WHITE):
        self.position = None
        self.number = number
        self.color = color
        self.start_position = None
        self.name = name
        if self.name == "":
            print(f"WARN: Player {self.number} has no name")

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return f"Player {self.number} ({self.name})"

    def get_position(self) -> (int, int):
        return self.position

    def set_start_position(self, position: ()):
        self.start_position = position
        self.position = position
        return self

    def get_distance_to(self, position: (int, int)) -> float:
        return math.sqrt((self.position[0] - position[0]) ** 2 + (self.position[1] - position[1]) ** 2)

    def is_mr_x(self):
        return False

    def is_cop(self):
        return False
