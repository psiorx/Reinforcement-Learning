import numpy as np
import copy

class Node:
    def __init__(self, valid_actions, policy):  
        num_actions = len(policy)
        self.valid_actions = np.zeros(num_actions)
        self.valid_actions[valid_actions] = 1
        self.P = policy
        self.P[np.where(self.valid_actions != 1)] = 0
        self.Q = np.zeros(num_actions)
        self.N = np.zeros(num_actions)

class AlphaZeroMCTS:
    def __init__(self, game, net):
        self.net = net
        self.game = copy.deepcopy(game)
        self.orig_game = game
        self.nodes = dict()
        self.c = 0.5
        self.max_depth = 0

    def get_policy(self):
        board_key = self.game.board.tobytes()
        new_policy = self.nodes[board_key].N/sum(self.nodes[board_key].N)
        return new_policy

    def search(self, num_iterations):
        for i in range(num_iterations):
            self.search_internal(0)
            self.game = copy.deepcopy(self.orig_game)   

    def search_internal(self, depth = 0):
        if depth > self.max_depth:
            self.max_depth = depth
        
        ## BASE CASES            
        #terminal state
        if self.game.game_over:
            return -self.game.reward

        #expansion
        board_key = self.game.board.tobytes()
        if board_key not in self.nodes:
            output = self.net.predict(self.game.board)
            policy = output[0].detach().cpu().numpy().squeeze()
            value = output[1].detach().cpu().numpy()
            self.nodes[board_key] = Node(self.game.get_valid_actions(), policy)
            return -value

        ## RECURSIVE CASE
        
        #selection
        node = self.nodes[board_key] 
        #compute confidence bounds
        U = node.Q + self.c * node.P * np.sqrt(sum(node.N)) / (1 + node.N)
        U[np.where(node.valid_actions == 0)] = -1
        max_indexes = np.where(U == max(U))[0]
        action_index = np.random.choice(max_indexes)
        self.game.take_action(action_index)
        v = self.search_internal(depth + 1)

        #backpropagation
        node.Q[action_index] = (node.N[action_index] * node.Q[action_index] + v) / (node.N[action_index] + 1)
        node.N[action_index] += 1
        return -v
