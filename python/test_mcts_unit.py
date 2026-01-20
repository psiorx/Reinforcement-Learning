import os
import sys
import unittest
import numpy as np
import torch

sys.path.append(os.path.dirname(__file__))

from connect4 import Connect4
from mcts import AlphaZeroMCTS, Node


class DummyNet:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)

    def predict(self, board, player=None):
        policy = torch.zeros((1, 7), device=self.device)
        policy = torch.log_softmax(policy, dim=1)
        value = torch.zeros((1, 1), device=self.device)
        return policy, value


class OneStepWinGame:
    def __init__(self):
        self.board = np.zeros((1, 1), dtype=int)
        self.player = 1
        self.game_over = False
        self.draw = False
        self.winner = 0

    def get_valid_actions(self):
        return np.array([0]) if not self.game_over else np.array([], dtype=int)

    def take_action(self, action):
        if self.game_over:
            return
        self.board[0, 0] = self.player
        self.winner = self.player
        self.game_over = True
        self.draw = False
        self.player = 1 if self.player == 2 else 2


class TestMCTS(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.net = DummyNet()
        self._verbose = True

    def _print_board(self, board):
        print(np.flip(board, axis=0))

    def _print_policy(self, policy, top_k=3):
        top_idx = np.argsort(policy)[-top_k:][::-1]
        top_vals = [(int(i), float(policy[i])) for i in top_idx]
        print("policy_top:", top_vals)

    def _mcts_action(self, game, iters=200):
        mcts = AlphaZeroMCTS(game, self.net, cpuct=1.5, dirichlet_alpha=0.0, dirichlet_frac=0.0)
        policy = mcts.search(iters)
        if self._verbose:
            self._print_policy(policy)
        return int(np.argmax(policy))

    def test_selects_immediate_win(self):
        game = Connect4()
        for a in [0, 0, 1, 1, 2, 2]:
            game.take_action(a)
        if self._verbose:
            print("test_selects_immediate_win")
            self._print_board(game.board)
        action = self._mcts_action(game, iters=200)
        if self._verbose:
            print("chosen_action:", action)
        self.assertEqual(action, 3)

    def test_blocks_immediate_loss(self):
        game = Connect4()
        for a in [0, 0, 1, 0, 1, 0]:
            game.take_action(a)
        if self._verbose:
            print("test_blocks_immediate_loss")
            self._print_board(game.board)
        action = self._mcts_action(game, iters=300)
        if self._verbose:
            print("chosen_action:", action)
        self.assertEqual(action, 0)

    def test_avoids_full_column(self):
        game = Connect4()
        for _ in range(6):
            game.take_action(0)
        if self._verbose:
            print("test_avoids_full_column")
            self._print_board(game.board)
        mcts = AlphaZeroMCTS(game, self.net, cpuct=1.5, dirichlet_alpha=0.0, dirichlet_frac=0.0)
        policy = mcts.search(50)
        if self._verbose:
            self._print_policy(policy)
        self.assertEqual(policy[0], 0.0)

    def test_wins_from_forced_position_vs_random(self):
        for seed in range(5):
            np.random.seed(seed)
            game = Connect4()
            for a in [0, 0, 1, 1, 2, 2]:
                game.take_action(a)
            if self._verbose:
                print("test_wins_from_forced_position_vs_random seed=%d" % seed)
                self._print_board(game.board)
            action = self._mcts_action(game, iters=200)
            game.take_action(action)
            if self._verbose:
                print("chosen_action:", action, "winner:", game.winner, "game_over:", game.game_over)
            self.assertTrue(game.game_over)
            self.assertEqual(game.winner, 1)

    def test_backup_sign_flip_one_step_game(self):
        game = OneStepWinGame()
        mcts = AlphaZeroMCTS(game, self.net, cpuct=1.5, dirichlet_alpha=0.0, dirichlet_frac=0.0)
        board_key = mcts.game.board.tobytes()
        mcts.nodes[board_key] = Node(mcts.game.get_valid_actions(), np.array([1.0]))
        v = mcts.search_internal(0)
        node = mcts.nodes[board_key]
        parent_q = node.Q[0]
        child_terminal = -1
        if self._verbose:
            print("test_backup_sign_flip_one_step_game")
            print("returned_v:", float(v), "parent_q:", float(parent_q), "child_terminal:", child_terminal)
        self.assertEqual(child_terminal, -1)
        self.assertAlmostEqual(v, -child_terminal, delta=1e-6)
        self.assertAlmostEqual(parent_q, v, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
