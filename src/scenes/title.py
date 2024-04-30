import pygame

from src.Button import Button
from src.colors import *
from src.scenes.scene import Scene
from src.game.scotland_yard_game_logic import DefinedAlgorithms
from src.GameController import UserActions


class Title(Scene):
    def __init__(self, game_controller, gui_controller):
        Scene.__init__(self, game_controller, gui_controller)
        self.cop_selected_algo = DefinedAlgorithms.PPO
        self.mr_x_selected_algo = DefinedAlgorithms.PPO
        self.text_font = pygame.font.Font(None, 30)

        self.screen_width = self.game_controller.gui_controller.game_canvas.get_width()
        self.screen_height = self.game_controller.gui_controller.game_canvas.get_height()

        self.button_width = 100
        self.button_height = 40
        self.button_margin = 20

        self.cop_button_center_x = self.screen_width // 3
        self.mrx_button_center_x = self.cop_button_center_x * 2

        self.text_surface_cops = self.text_font.render("Cops algorithm", True, WHITE)
        self.text_width_cops = self.text_surface_cops.get_width()
        self.text_x_cops = self.cop_button_center_x - self.text_width_cops // 2

        self.text_surface_mrx = self.text_font.render("Mr X algorithm", True, WHITE)
        self.text_width_mrx = self.text_surface_mrx.get_width()
        self.text_x_mrx = self.mrx_button_center_x - self.text_width_mrx // 2

        self.mr_x_button_pos_x = self.mrx_button_center_x - self.button_width // 2
        self.cop_button_pos_x = self.cop_button_center_x - self.button_width // 2

        self.btn_start, self.btn_cop_ppo, self.btn_cop_dqn, self.btn_cop_random, self.btn_mr_x_ppo, self.btn_mr_x_dqn, self.btn_mr_x_random = self.to_draw_buttons(
            self.game_controller.gui_controller.game_canvas)
        self.list_cop_btns = [self.btn_cop_ppo, self.btn_cop_dqn, self.btn_cop_random]
        self.list_mrx_btns = [self.btn_mr_x_ppo, self.btn_mr_x_dqn, self.btn_mr_x_random]

    def update(self, delta_time, actions):
        return

    def render(self, display):
        display.fill(MIDNIGHT_BLUE)

        for btn in self.list_cop_btns:
            self.set_cop_colors(btn)
            btn.draw()

        for btn in self.list_mrx_btns:
            self.set_mrx_colors(btn)
            btn.draw()
        self.btn_start.draw()

        display.blit(self.text_surface_cops, (self.text_x_cops, 50))
        display.blit(self.text_surface_mrx, (self.text_x_mrx, 50))

        if self.game_controller.user_actions[UserActions.mouse_left_up.name]:
            if self.btn_start.is_hovered():
                from src.scenes.game_scene import ScotlandYardScene

                new_state = ScotlandYardScene(self.game_controller, self.gui_controller,
                                              self.cop_selected_algo, self.mr_x_selected_algo)
                new_state.enter_scene()
                return
            elif self.btn_cop_ppo.is_hovered():
                self.cop_selected_algo = DefinedAlgorithms.PPO
            elif self.btn_cop_dqn.is_hovered():
                self.cop_selected_algo = DefinedAlgorithms.DQN
            elif self.btn_cop_random.is_hovered():
                self.cop_selected_algo = DefinedAlgorithms.RANDOM
            elif self.btn_mr_x_ppo.is_hovered():
                self.mr_x_selected_algo = DefinedAlgorithms.PPO
            elif self.btn_mr_x_dqn.is_hovered():
                self.mr_x_selected_algo = DefinedAlgorithms.DQN
            elif self.btn_mr_x_random.is_hovered():
                self.mr_x_selected_algo = DefinedAlgorithms.RANDOM

    def reset_colors(self, button: Button):
        button.color = GREEN
        button.hover_color = WHITE
        button.text_color = WHITE
        button.text_hover_color = BLACK
        button.border_color = WHITE
        button.border_color_hovered = BLACK
        return button

    def set_selected_colors(self, button: Button):
        button.color = RED
        button.hover_color = RED
        button.text_color = WHITE
        button.text_hover_color = WHITE
        button.border_color = WHITE
        button.border_color_hovered = WHITE
        return button

    def set_cop_colors(self, button: Button):
        if self.cop_selected_algo == button.value:
            return self.set_selected_colors(button)
        return self.reset_colors(button)

    def set_mrx_colors(self, button: Button):
        if self.mr_x_selected_algo == button.value:
            return self.set_selected_colors(button)
        return self.reset_colors(button)

    def to_draw_buttons(self, display):
        btn_color, btn_hover_color, text_color, hover_text_color, border_color, border_color_hovered = GREEN, WHITE, WHITE, BLACK, WHITE, BLACK
        btn_cop_ppo = Button("PPO", DefinedAlgorithms.PPO, self.cop_button_pos_x, 200, self.button_width,
                             self.button_height, display, btn_color, btn_hover_color, text_color, hover_text_color,
                             border_color, border_color_hovered, 3, 5)
        btn_cop_dqn = Button("DQN", DefinedAlgorithms.DQN, self.cop_button_pos_x, 250, self.button_width,
                             self.button_height, display, btn_color, btn_hover_color, text_color, hover_text_color,
                             border_color, border_color_hovered, 3, 5)
        btn_cop_random = Button("Random", DefinedAlgorithms.RANDOM, self.cop_button_pos_x, 300, self.button_width,
                                self.button_height, display, btn_color, btn_hover_color, text_color, hover_text_color,
                                border_color, border_color_hovered, 3, 5)

        btn_mr_x_ppo = Button("PPO", DefinedAlgorithms.PPO, self.mr_x_button_pos_x, 200, self.button_width,
                              self.button_height, display, btn_color, btn_hover_color, text_color, hover_text_color,
                              border_color, border_color_hovered, 3, 5)
        btn_mr_x_dqn = Button("DQN", DefinedAlgorithms.DQN, self.mr_x_button_pos_x, 250, self.button_width,
                              self.button_height, display, btn_color, btn_hover_color, text_color, hover_text_color,
                              border_color, border_color_hovered, 3, 5)
        btn_mr_x_random = Button("Random", DefinedAlgorithms.RANDOM, self.mr_x_button_pos_x, 300, self.button_width,
                                 self.button_height, display, btn_color, btn_hover_color, text_color, hover_text_color,
                                 border_color, border_color_hovered, 3, 5)

        start_button_center_x = self.screen_width // 2
        start_button_y = self.screen_height - self.button_height - self.button_margin - 20

        btn_start = Button("Start", None, start_button_center_x - self.button_width, start_button_y,
                           self.button_width * 2, self.button_height, display, BLACK, WHITE, WHITE, BLACK, WHITE, BLACK,
                           3, 5)

        return btn_start, btn_cop_ppo, btn_cop_dqn, btn_cop_random, btn_mr_x_ppo, btn_mr_x_dqn, btn_mr_x_random
