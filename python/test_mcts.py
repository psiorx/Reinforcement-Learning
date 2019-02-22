import numpy as np
from connect4 import Connect4
from neuralnetwork import NNet
from mcts import AlphaZeroMCTS

net = NNet(256, "cuda")
net.eval()
np.set_printoptions(precision=4)

#
#
#
#
# 2 2 2
# 1 1 1  <-- 1 should win this
def testGame1(iters):
    game = Connect4()
    for a in [0, 0, 1, 1, 2, 2]:
        game.take_action(a)

    mcts = AlphaZeroMCTS(game, net)
    action = np.argmax(mcts.search(iters))
    if action != 3:
        return 0
    else:
        return 1

#
#
# 2
# 2
# 2 1 
# 1 1     <-- 1 should block this

def testGame2(iters):
    game = Connect4()
    for a in [0, 0, 1, 0, 1, 0]:
        game.take_action(a)

    mcts = AlphaZeroMCTS(game, net)
    policy = mcts.search(iters)
    action = np.argmax(policy)
    if action != 0:
        return 0
    else:
        return 1

#
#
# 2
# 2 1
# 2 1 
# 1 1     <-- 2 should win this

def testGame3(iters):
    game = Connect4()
    for a in [0, 0, 1, 0, 1, 0, 1]:
        game.take_action(a)
    mcts = AlphaZeroMCTS(game, net)
    policy = mcts.search(iters)
    action = np.argmax(policy)
    if action != 0:
        return 0
    else:
        return 1

#
#
#
#
# 2 2
# 1 1 1  <-- 2 should block this

def testGame4(iters):
    game = Connect4()
    for a in [0, 0, 1, 1, 2]:
        game.take_action(a)

    mcts = AlphaZeroMCTS(game, net)
    action = np.argmax(mcts.search(iters))
    
    if action != 3:
        return 0
    else:
        return 1

iters = 100
samples = 10

score1 = sum([testGame1(iters) for i in range(samples)])
print("Score 1: %d/%d" % (score1, samples))
score2 = sum([testGame2(iters) for i in range(samples)])
print("Score 2: %d/%d" % (score2, samples))
score3 = sum([testGame3(iters) for i in range(samples)])
print("Score 3: %d/%d" % (score3, samples))
score4 = sum([testGame4(iters) for i in range(samples)])
print("Score 4: %d/%d" % (score4, samples))
