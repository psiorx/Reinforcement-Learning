import numpy as np
import copy

class Node:
    def __init__(self, valid_actions, policy):  
        num_actions = len(policy)
        self.valid_actions = np.zeros(num_actions)
        self.valid_actions[valid_actions] = 1
        self.P = policy
        self.P[np.where(self.valid_actions != 1)] = 0
        policy_sum = sum(self.P)
        if policy_sum == 0:
            self.P = self.valid_actions / max(1, sum(self.valid_actions))
        else:
            self.P /= policy_sum
        self.Q = np.zeros(num_actions)
        self.W = np.zeros(num_actions)
        self.N = np.zeros(num_actions)

class AlphaZeroMCTS:
    def __init__(self, game, net, cpuct=1.0, dirichlet_alpha=0.0, dirichlet_frac=0.0):
        self.net = net
        self.orig_game = game
        self.game = copy.deepcopy(game)
        self.nodes = dict()
        self.c = cpuct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac
        self.max_depth = 0
        self.player = self.orig_game.player
        self.functional = hasattr(game, "apply_action_to_board") and hasattr(game, "get_valid_actions_from")
        self.root_board = None
        self.root_player = None

    def _key(self, board, player=None):
        if self.functional:
            return (board.tobytes(), player)
        return board.tobytes()

    def get_policy(self, board=None, player=None):
        if board is None:
            board = self.game.board
            player = self.game.player
        board_key = self._key(board, player)
        visit_sum = sum(self.nodes[board_key].N)
        if visit_sum == 0:
            return self.nodes[board_key].P
        new_policy = self.nodes[board_key].N / visit_sum
        return new_policy

    def search(self, num_iterations):
        if self.functional:
            self.root_board = self.orig_game.board.copy()
            self.root_player = self.orig_game.player
            root_state = (self.root_board, self.root_player, self.orig_game.game_over, self.orig_game.draw, self.orig_game.winner)
            for _ in range(num_iterations):
                self.search_internal_functional(*root_state, depth=0)
            return self.get_policy(self.root_board, self.root_player)
        for _ in range(num_iterations):
            self.search_internal_stateful(0)
            self.game = copy.deepcopy(self.orig_game)
        return self.get_policy()

    def search_internal(self, depth=0):
        if self.functional:
            return self.search_internal_functional(
                self.game.board.copy(),
                self.game.player,
                self.game.game_over,
                self.game.draw,
                self.game.winner,
                depth=depth,
            )
        return self.search_internal_stateful(depth)

    def search_internal_functional(self, board, player, game_over, draw, winner, depth=0):
        if depth > self.max_depth:
            self.max_depth = depth

        if game_over:
            if draw:
                return 0
            return 1 if winner == player else -1

        board_key = self._key(board, player)
        if board_key not in self.nodes:
            output = self.net.predict(board, player)
            policy = np.exp(output[0].detach().cpu().numpy().squeeze())
            value = float(output[1].detach().cpu().numpy().squeeze())
            self.nodes[board_key] = Node(self.orig_game.get_valid_actions_from(board), policy)
            if depth == 0 and self.dirichlet_alpha > 0:
                node = self.nodes[board_key]
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(node.P))
                node.P = (1 - self.dirichlet_frac) * node.P + self.dirichlet_frac * noise
            return value

        node = self.nodes[board_key]
        U = node.Q + self.c * node.P * np.sqrt(sum(node.N)) / (1 + node.N)
        U[np.where(node.valid_actions == 0)] = -1e9
        max_indexes = np.where(U == max(U))[0]
        action_index = np.random.choice(max_indexes)

        next_board, next_player, next_game_over, next_draw, next_winner = self.orig_game.apply_action_to_board(
            board, player, action_index
        )
        child_value = self.search_internal_functional(
            next_board, next_player, next_game_over, next_draw, next_winner, depth + 1
        )
        v = -child_value

        node.N[action_index] += 1
        node.W[action_index] += v
        node.Q[action_index] = node.W[action_index] / node.N[action_index]
        return v

    def search_internal_stateful(self, depth=0):
        if depth > self.max_depth:
            self.max_depth = depth

        if self.game.game_over:
            if self.game.draw:
                return 0
            return 1 if self.game.winner == self.game.player else -1

        board_key = self.game.board.tobytes()
        if board_key not in self.nodes:
            output = self.net.predict(self.game.board, self.game.player)
            policy = np.exp(output[0].detach().cpu().numpy().squeeze())
            value = float(output[1].detach().cpu().numpy().squeeze())
            self.nodes[board_key] = Node(self.game.get_valid_actions(), policy)
            if depth == 0 and self.dirichlet_alpha > 0:
                node = self.nodes[board_key]
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(node.P))
                node.P = (1 - self.dirichlet_frac) * node.P + self.dirichlet_frac * noise
            return value

        node = self.nodes[board_key]
        U = node.Q + self.c * node.P * np.sqrt(sum(node.N)) / (1 + node.N)
        U[np.where(node.valid_actions == 0)] = -1e9
        max_indexes = np.where(U == max(U))[0]
        action_index = np.random.choice(max_indexes)

        self.game.take_action(action_index)
        child_value = self.search_internal_stateful(depth + 1)
        v = -child_value

        node.N[action_index] += 1
        node.W[action_index] += v
        node.Q[action_index] = node.W[action_index] / node.N[action_index]
        return v
