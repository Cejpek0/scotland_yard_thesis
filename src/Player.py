from src.colors import *


class Player:
    def __init__(self, number: int, name: str = "", color: () = WHITE):
        self.last_known_position = None
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

    def set_start_position(self, position: ()):
        self.start_position = position
        self.position = position
        return self

    def get_distance_to(self, *args) -> int:
        if len(args) == 1 and isinstance(args[0], Player):
            return self.get_distance_to(args[0].position)
        else:
            return abs(self.position[0] - args[0][0]) + abs(self.position[1] - args[0][1])

    def mr_x_reveal_position(self):
        if self.number == 0:
            self.last_known_position = self.position
        return self