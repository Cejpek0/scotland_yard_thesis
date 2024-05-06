"""
File description: This file is the main entry point of the application.
It creates an instance of the GameController and runs the game loop until the game is running.

Author: Michal Cejpek (xcejpe05@stud.fit.vutbr.cz)
"""

from src.GameController import GameController

if __name__ == '__main__':
    g = GameController(verbose=True)
    while g.running:
        g.game_loop()
