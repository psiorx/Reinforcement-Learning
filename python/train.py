import torch
import os
import copy
import numpy as np
from connect4 import Connect4
from neuralnetwork import NNet
from mcts import AlphaZeroMCTS
from collections import deque
from threading import Thread
from queue import Queue

# hyperparams
batch_size = 256
mcts_iterations = 256
exp_pool_size = 4098
duel_interval = 20
duel_acceptance = 60.0
channel_width = 128
training_iterations = 10
network_file = "connectfo_128.net"

np.set_printoptions(precision=4)

if os.path.isfile(network_file):
    print("loading previous network")
    net = torch.load(network_file)
else:
    print("starting new agent")
    net = NNet(channel_width, "cuda")
net.eval()

training_net = copy.deepcopy(net)
training_net.train()

def get_experience(current_net):
    current_net.eval()
    game = Connect4()            
    experiences = []
    while not game.game_over:
        mcts = AlphaZeroMCTS(game, current_net)
        policy = mcts.search(mcts_iterations)
        action = np.random.choice(range(len(policy)), p=policy)
        e = [copy.deepcopy(game.board), copy.deepcopy(policy), game.player, None]
        experiences.append(e)
        game.take_action(action)
    
    for e in experiences:
        if game.draw:
            e[3] = 0
        else:
            e[3] = 1 if game.winner == e[2] else -1

    return experiences


def compare_agents(new_net, orig_net, num_games):
    orig_net.eval()
    new_net.eval()
    scores = {0: 0, 1:0, 2:0}
    win_percentage = 0
    for i in range(num_games):
        game = Connect4()           
        print("Playing game %d:" % i) 
        while not game.game_over:
            action = []
            policy = []
            if game.player == 1:
                mcts = AlphaZeroMCTS(game, new_net)
                policy = mcts.search(mcts_iterations)
                action = np.argmax(policy)
            else:
                mcts = AlphaZeroMCTS(game, orig_net)
                policy = mcts.search(mcts_iterations)
                action = np.argmax(policy)
            game.take_action(action)

        if game.draw:
            scores[0] += 1
        else:
            scores[game.winner] += 1
        win_percentage = 100 * scores[1] / float(num_games)
        print(scores)
    print("Win Percentage: %f" % (win_percentage))
    return win_percentage

q_out = Queue()

def generate_experience(net, q_out):
    while True:
        exp = get_experience(net)   
        q_out.put(exp)

threads = []
num_workers = 4
for i in range(num_workers):
    t = Thread(target=generate_experience, args=[net, q_out])
    t.start()
    threads.append(t)

exp_pool = deque()

training_episodes = 0

while True:
    for e in q_out.get():
        exp_pool.append(e)
        if len(exp_pool) > exp_pool_size:
            exp_pool.popleft()
        
    if len(exp_pool) > batch_size:
        training_indexes = np.random.choice(range(len(exp_pool)), batch_size, replace=False)
        batch = []
        for idx in training_indexes:
            batch.append(exp_pool[idx])
        print("training episode %d with batch: %d" % (training_episodes, len(batch)))
        training_net.process_data(batch, training_iterations)
        training_episodes += 1
        if training_episodes % duel_interval == 0:
            training_net.eval()
            print("Duel Commencing - THERE CAN BE ONLY ONE")
            win_rate = compare_agents(training_net, net, 10)
            if win_rate >= duel_acceptance:
                net.load_state_dict(training_net.state_dict())
                torch.save(training_net, network_file)
                print("UPGRADE COMPLETE")

    print("total experience: %d" % len(exp_pool))
    