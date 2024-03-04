from src.Button import Button
from src.GameController import UserActions
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
        display.fill(BLACK)
        btn_ai = Button("AI", 100, 100, 100, 50, display, BLUE, LIGHT_BLUE, WHITE, BLACK, BLACK, WHITE, 3, 5)

        if btn_ai.is_hovered():
            btn_ai.color = btn_ai.hover_color
            btn_ai.text_color = btn_ai.text_hover_color
        btn_ai.draw()

        self.gui_controller.to_draw_text("Game States Demo", WHITE,
                                         (self.gui_controller.width / 2, self.gui_controller.height / 2))
