from src.GameController import GameController

if __name__ == '__main__':
    g = GameController()
    while g.running:
        g.game_loop()
