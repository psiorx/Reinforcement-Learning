import numpy as np
import copy

class Node:
    def __init__(self, valid_actions, policy):  
        self.valid_actions = valid_actions
        self.P = policy[valid_actions]
        self.Q = np.zeros(len(valid_actions))
        self.N = np.zeros(len(valid_actions))

class AlphaZeroMCTS:
    def __init__(self, game, net):
        self.net = net
        self.game = copy.deepcopy(game)
        self.orig_game = game
        self.nodes = dict()
        self.c = 0.5

    def search(self):
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

        max_indexes = np.where(U == max(U))[0]
        action_index = np.random.choice(max_indexes)
        self.game.take_action(node.valid_actions[action_index])
        v = self.search()

        #backpropagation
        node.Q[action_index] = (node.N[action_index] * node.Q[action_index] + v) / (node.N[action_index] + 1)
        node.N[action_index] += 1
        self.game = copy.deepcopy(self.orig_game)
        return -v
