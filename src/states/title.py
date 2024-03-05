from src.Button import Button
from src.colors import *
from src.states.state import State


class Title(State):
    def __init__(self, game_controller, gui_controller):
        State.__init__(self, game_controller, gui_controller)

    def update(self, delta_time, actions):
        from src.GameController import UserActions
        if actions[UserActions.enter.name]:
            from src.states.scotland_yard import ScotlandYard
            new_state = ScotlandYard(self.game_controller, self.gui_controller)
            new_state.enter_state()
            self.game_controller.state_stack.append(new_state)

    def render(self, display):
        display.fill(MIDNIGHT_BLUE)
        btn_ai = Button("AI VS AI", 100, 100, 200, 50, display, BLACK, WHITE, WHITE, BLACK, WHITE, BLACK, 3, 5)
        btn_cop = Button("PLAY AS COPS", 100, 200, 200, 50, display, BLACK, WHITE, WHITE, BLACK, WHITE, BLACK, 3, 5)
        btn_mr_x = Button("PLAY AS MR X", 100, 300, 200, 50, display, BLACK, WHITE, WHITE, BLACK, WHITE, BLACK, 3, 5)

        btn_ai.draw()
        btn_cop.draw()
        btn_mr_x.draw()
        
        from src.GameController import UserActions
        if self.game_controller.user_actions[UserActions.mouse_left_up.name]:
            if btn_ai.is_hovered():
                from src.states.scotland_yard import ScotlandYard
                new_state = ScotlandYard(self.game_controller, self.gui_controller)
                new_state.enter_state()
                self.game_controller.state_stack.append(new_state)
