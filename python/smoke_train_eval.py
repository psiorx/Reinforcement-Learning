import argparse
import copy
import numpy as np
import torch

from connect4 import Connect4
from neuralnetwork import AlphaZeroResNet
from mcts import AlphaZeroMCTS


def select_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def self_play_episode(net, mcts_iters, temperature_moves):
    net.eval()
    game = Connect4()
    experiences = []
    move_count = 0
    while not game.game_over:
        mcts = AlphaZeroMCTS(game, net, cpuct=1.5, dirichlet_alpha=0.3, dirichlet_frac=0.25)
        policy = mcts.search(mcts_iters)
        if move_count < temperature_moves:
            action = np.random.choice(range(len(policy)), p=policy)
        else:
            action = int(np.argmax(policy))
        experiences.append([copy.deepcopy(game.board), copy.deepcopy(policy), game.player, None])
        game.take_action(action)
        move_count += 1

    for e in experiences:
        if game.draw:
            e[3] = 0
        else:
            e[3] = 1 if game.winner == e[2] else -1
    return experiences


def compare_agents(new_net, orig_net, num_games, mcts_iters):
    new_net.eval()
    orig_net.eval()
    scores = {0: 0, 1: 0, 2: 0}
    for _ in range(num_games):
        game = Connect4()
        while not game.game_over:
            if game.player == 1:
                mcts = AlphaZeroMCTS(game, new_net, cpuct=1.5, dirichlet_alpha=0.0, dirichlet_frac=0.0)
            else:
                mcts = AlphaZeroMCTS(game, orig_net, cpuct=1.5, dirichlet_alpha=0.0, dirichlet_frac=0.0)
            policy = mcts.search(mcts_iters)
            action = np.argmax(policy)
            game.take_action(action)
        if game.draw:
            scores[0] += 1
        else:
            scores[game.winner] += 1
    win_percentage = 100.0 * scores[1] / float(num_games)
    return win_percentage, scores


def parse_args():
    parser = argparse.ArgumentParser(description="Quick AlphaZero smoke train/eval.")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--mcts-iters", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--episodes-per-epoch", type=int, default=4)
    parser.add_argument("--training-iters", type=int, default=2)
    parser.add_argument("--eval-games", type=int, default=6)
    parser.add_argument("--temperature-moves", type=int, default=6)
    return parser.parse_args()


def run_smoke():
    device = select_device()
    np.set_printoptions(precision=4)

    args = parse_args()
    net = AlphaZeroResNet(args.channels, num_blocks=args.blocks, device=device)
    net.eval()
    baseline = copy.deepcopy(net)

    exp_pool = []
    for _ in range(args.epochs):
        for _ in range(args.episodes_per_epoch):
            exp_pool.extend(self_play_episode(net, args.mcts_iters, args.temperature_moves))
        batch = np.random.choice(len(exp_pool), min(64, len(exp_pool)), replace=False)
        data = [exp_pool[i] for i in batch]
        net.process_data(data, args.training_iters)

    win_rate, scores = compare_agents(net, baseline, num_games=args.eval_games, mcts_iters=args.mcts_iters)
    print("Smoke eval: win_rate=%.1f%% scores=%s" % (win_rate, scores))


if __name__ == "__main__":
    run_smoke()
