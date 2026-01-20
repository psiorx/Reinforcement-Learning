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
    def __init__(self, game, net, cpuct=1.5, dirichlet_alpha=0.3, dirichlet_frac=0.25):
        self.net = net
        self.game = copy.deepcopy(game)
        self.orig_game = game
        self.nodes = dict()
        self.c = cpuct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac
        self.max_depth = 0
        self.player = self.orig_game.player

    def get_policy(self):
        board_key = self.game.board.tobytes()
        visit_sum = sum(self.nodes[board_key].N)
        if visit_sum == 0:
            return self.nodes[board_key].P
        new_policy = self.nodes[board_key].N / visit_sum
        return new_policy

    def search(self, num_iterations):
        for i in range(num_iterations):
            reward = self.search_internal(0)
            self.game = copy.deepcopy(self.orig_game)            
        return self.get_policy()        

    def search_internal(self, depth = 0):
        if depth > self.max_depth:
            self.max_depth = depth
        
        ## BASE CASES
        #terminal state
        if self.game.game_over:
            if self.game.draw:
                return 0
            return 1 if self.game.winner == self.game.player else -1

        #expansion
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

        ## RECURSIVE CASE
        #selection
        node = self.nodes[board_key] 
        #compute confidence bounds
        U = node.Q + self.c * node.P * np.sqrt(sum(node.N)) / (1 + node.N)

        U[np.where(node.valid_actions == 0)] = -1e9
        max_indexes = np.where(U == max(U))[0]
        action_index = np.random.choice(max_indexes)

        self.game.take_action(action_index)
        child_value = self.search_internal(depth + 1)
        v = -child_value

        #backpropagation
        node.N[action_index] += 1
        node.W[action_index] += v
        node.Q[action_index] = node.W[action_index] / node.N[action_index]

        return v
