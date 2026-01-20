import argparse
import os
import copy
import math
import random
from threading import Thread

import numpy as np
import pygame
import torch

from connect4 import Connect4
from mcts import AlphaZeroMCTS
from neuralnetwork import AlphaZeroResNet


def parse_args():
    parser = argparse.ArgumentParser(description="Play Connect4 against agents.")
    parser.add_argument("--agent1", choices=["human", "mcts", "random"], default="human")
    parser.add_argument("--agent2", choices=["human", "mcts", "random"], default="mcts")
    parser.add_argument("--mcts-iters", type=int, default=200)
    parser.add_argument("--net", type=str, default="connect4_resnet.net")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


class StarBurst:
    def __init__(self, width, height, count=120):
        self.width = width
        self.height = height
        self.stars = []
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1.0, 4.0)
            self.stars.append({
                "x": width / 2,
                "y": height / 2,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "r": random.randint(2, 5),
                "life": random.randint(60, 140),
            })

    def update(self):
        for star in self.stars:
            star["x"] += star["vx"]
            star["y"] += star["vy"]
            star["life"] -= 1

    def draw(self, surface):
        for star in self.stars:
            if star["life"] <= 0:
                continue
            pygame.draw.circle(surface, (255, 215, 0), (int(star["x"]), int(star["y"])), star["r"])


class Connect4Pygame:
    def __init__(self, args):
        self.args = args
        self.cell = 90
        self.cols = 7
        self.rows = 6
        self.margin = 10
        self.width = self.cols * self.cell
        self.height = self.rows * self.cell + 80
        self.board_height = self.rows * self.cell

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Connect4: Red (P1) vs Black (P2)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 18)

        self.game = Connect4()
        self.net = None
        if self.args.agent1 == "mcts" or self.args.agent2 == "mcts":
            device = self.args.device or ("cuda" if torch.cuda.is_available() else "cpu")
            if os.path.isfile(self.args.net):
                self.net = torch.load(self.args.net, map_location=device)
            else:
                self.net = AlphaZeroResNet(self.args.channels, num_blocks=self.args.blocks, device=device)
            self.net.eval()

        self.drop_anim = None
        self.ai_thinking = False
        self.win_animation = None
        self.pending_action = None
        self.expected_player = None

    def current_agent(self):
        return self.args.agent1 if self.game.player == 1 else self.args.agent2

    def reset(self):
        self.game.reset()
        self.drop_anim = None
        self.ai_thinking = False
        self.win_animation = None
        self.pending_action = None
        self.expected_player = None

    def handle_click(self, x):
        if self.game.game_over or self.ai_thinking or self.drop_anim:
            return
        if self.current_agent() != "human":
            return
        col = int(x // self.cell)
        if col in self.game.get_valid_actions():
            self.start_drop(col)

    def start_drop(self, col):
        row = self.game.get_open_row(col)
        if row < 0:
            return
        start_y = -self.cell
        end_y = row * self.cell
        self.drop_anim = {
            "col": col,
            "row": row,
            "y": start_y,
            "end_y": end_y,
            "speed": 18,
            "player": self.game.player,
        }

    def finish_drop(self):
        if not self.drop_anim:
            return
        col = self.drop_anim["col"]
        self.game.take_action(col)
        self.drop_anim = None
        if self.game.game_over:
            if not self.game.draw:
                self.win_animation = StarBurst(self.width, self.board_height)
        else:
            self.maybe_ai_move()

    def maybe_ai_move(self):
        if self.game.game_over or self.drop_anim:
            return
        agent = self.current_agent()
        if agent == "human":
            return
        if self.ai_thinking:
            return
        if agent == "random":
            action = int(np.random.choice(self.game.get_valid_actions()))
            self.start_drop(action)
        elif agent == "mcts":
            self.ai_thinking = True
            snapshot = copy.deepcopy(self.game)
            self.expected_player = self.game.player
            Thread(
                target=self.compute_mcts_action_async,
                args=(snapshot, self.expected_player),
                daemon=True,
            ).start()

    def compute_mcts_action_async(self, snapshot, expected_player):
        mcts = AlphaZeroMCTS(
            snapshot,
            self.net,
            cpuct=1.5,
            dirichlet_alpha=0.0,
            dirichlet_frac=0.0,
        )
        policy = mcts.search(self.args.mcts_iters)
        action = int(np.argmax(policy))
        self.pending_action = (expected_player, action)

    def draw_board(self):
        self.screen.fill((20, 34, 74))
        for r in range(self.rows):
            for c in range(self.cols):
                x0 = c * self.cell + self.margin
                y0 = r * self.cell + self.margin
                x1 = (c + 1) * self.cell - self.margin
                y1 = (r + 1) * self.cell - self.margin
                value = self.game.board[r, c]
                if value == 1:
                    color = (239, 68, 68)
                elif value == 2:
                    color = (17, 24, 39)
                else:
                    color = (248, 250, 252)
                pygame.draw.ellipse(self.screen, color, (x0, y0, x1 - x0, y1 - y0))
                pygame.draw.ellipse(self.screen, (15, 23, 42), (x0, y0, x1 - x0, y1 - y0), 2)

        if self.drop_anim:
            c = self.drop_anim["col"]
            x0 = c * self.cell + self.margin
            x1 = (c + 1) * self.cell - self.margin
            y0 = self.drop_anim["y"] + self.margin
            y1 = self.drop_anim["y"] + self.cell - self.margin
            color = (239, 68, 68) if self.drop_anim["player"] == 1 else (17, 24, 39)
            pygame.draw.ellipse(self.screen, color, (x0, y0, x1 - x0, y1 - y0))
            pygame.draw.ellipse(self.screen, (15, 23, 42), (x0, y0, x1 - x0, y1 - y0), 2)

    def draw_status(self):
        p1 = self.args.agent1
        p2 = self.args.agent2
        legend = "P1 (Red): %s | P2 (Black): %s" % (p1, p2)
        legend_text = self.small_font.render(legend, True, (226, 232, 240))
        self.screen.blit(legend_text, (12, self.board_height + 6))

        if self.game.game_over:
            if self.game.draw:
                status = "Draw."
            else:
                winner = "Red (P1)" if self.game.winner == 1 else "Black (P2)"
                status = "Winner: %s" % winner
        else:
            current = "Red (P1)" if self.game.player == 1 else "Black (P2)"
            status = "Turn: %s" % current
            if self.ai_thinking:
                status += " (thinking...)"

        status_text = self.font.render(status, True, (250, 250, 210))
        self.screen.blit(status_text, (12, self.board_height + 34))

        if self.game.game_over and not self.game.draw:
            banner = pygame.Surface((self.width, self.board_height), pygame.SRCALPHA)
            banner.fill((255, 215, 0, 40))
            self.screen.blit(banner, (0, 0))

        if self.win_animation:
            self.win_animation.draw(self.screen)

    def update(self):
        if self.drop_anim:
            self.drop_anim["y"] += self.drop_anim["speed"]
            if self.drop_anim["y"] >= self.drop_anim["end_y"]:
                self.drop_anim["y"] = self.drop_anim["end_y"]
                self.finish_drop()
        if self.win_animation:
            self.win_animation.update()
        if self.pending_action and not self.drop_anim and not self.game.game_over:
            expected_player, action = self.pending_action
            if self.game.player == expected_player:
                self.start_drop(action)
            self.pending_action = None
            self.ai_thinking = False

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.handle_click(event.pos[0])

            if not self.game.game_over and not self.drop_anim:
                self.maybe_ai_move()

            self.update()
            self.draw_board()
            self.draw_status()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


def main():
    args = parse_args()
    gui = Connect4Pygame(args)
    gui.run()


if __name__ == "__main__":
    main()
