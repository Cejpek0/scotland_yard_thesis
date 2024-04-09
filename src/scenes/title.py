from src.Button import Button
from src.colors import *
from src.scenes.scene import Scene
from src.game.scotland_yard_game_logic import DefinedAlgorithms


class Title(Scene):
    def __init__(self, game_controller, gui_controller):
        Scene.__init__(self, game_controller, gui_controller)
        self.cop_selected_algo = DefinedAlgorithms.PPO
        self.mr_x_selected_algo = DefinedAlgorithms.PPO

    def update(self, delta_time, actions):
        from src.GameController import UserActions
        if actions[UserActions.enter.name]:
            from src.scenes.game_scene import ScotlandYardScene
            new_state = ScotlandYardScene(self.game_controller, self.gui_controller)
            new_state.enter_scene()
            self.game_controller.scene_stack.append(new_state)

    def render(self, display):
        display.fill(MIDNIGHT_BLUE)
        btn_start = Button("AI VS AI", 100, 100, 200, 50, display, BLACK, WHITE, WHITE, BLACK, WHITE, BLACK, 3, 5)
        btn_cop_ppo, btn_cop_dqn, btn_cop_random, mr_x_ppo, mr_x_dqn, mr_x_random = self.to_draw_ai_change_buttons()

        btn_start.draw()

        from src.GameController import UserActions
        if self.game_controller.user_actions[UserActions.mouse_left_up.name]:
            if btn_start.is_hovered():
                from src.scenes.game_scene import ScotlandYardScene
                new_state = ScotlandYardScene(self.game_controller, self.gui_controller)
                new_state.enter_scene()
                
    def to_draw_ai_change_buttons(self):
        
        btn_cop_ppo = Button("PPO", 100, 200, 200, 50, self.gui_controller.game_canvas, BLACK, WHITE, WHITE, BLACK, WHITE, BLACK, 3, 5)
        
