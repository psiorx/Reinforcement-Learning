import argparse
import copy
import numpy as np
import torch

from connect4 import Connect4
from neuralnetwork import AlphaZeroNet, AlphaZeroResNet
from mcts import AlphaZeroMCTS


def select_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def apply_temperature(policy, temperature):
    if temperature <= 0:
        one_hot = np.zeros_like(policy)
        one_hot[int(np.argmax(policy))] = 1.0
        return one_hot
    if temperature == 1:
        return policy
    tempered = np.power(policy, 1.0 / temperature)
    tempered_sum = np.sum(tempered)
    if tempered_sum <= 0:
        return policy
    return tempered / tempered_sum

def self_play_episode(net, mcts_iters, temperature_moves):
    net.eval()
    game = Connect4()
    experiences = []
    move_count = 0
    while not game.game_over:
        mcts = AlphaZeroMCTS(game, net, cpuct=1.0, dirichlet_alpha=0.0, dirichlet_frac=0.0)
        policy = mcts.search(mcts_iters)
        temp = 1 if move_count < temperature_moves else 0
        policy = apply_temperature(policy, temp)
        if temp > 0:
            action = np.random.choice(range(len(policy)), p=policy)
        else:
            action = int(np.argmax(policy))
        experiences.append([copy.deepcopy(game.board), copy.deepcopy(policy), game.player, None])
        game.take_action(action)
        move_count += 1

    for e in experiences:
        if game.draw:
            e[3] = 1e-4
        else:
            e[3] = 1 if game.winner == e[2] else -1
    augmented = []
    for board, policy, player, value in experiences:
        augmented.append([board, policy, player, value])
        augmented.append([board[:, ::-1], policy[::-1], player, value])
    return augmented


def compare_agents(new_net, orig_net, num_games, mcts_iters):
    new_net.eval()
    orig_net.eval()
    draws = 0
    new_net_wins = 0
    new_net_losses = 0
    for i in range(num_games):
        game = Connect4()
        new_net_is_player1 = (i % 2 == 0)
        while not game.game_over:
            use_new_net = (game.player == 1 and new_net_is_player1) or (game.player == 2 and not new_net_is_player1)
            net_to_use = new_net if use_new_net else orig_net
            mcts = AlphaZeroMCTS(game, net_to_use, cpuct=1.0, dirichlet_alpha=0.0, dirichlet_frac=0.0)
            policy = mcts.search(mcts_iters)
            action = int(np.argmax(policy))
            game.take_action(action)
        if game.draw:
            draws += 1
        else:
            if (game.winner == 1 and new_net_is_player1) or (game.winner == 2 and not new_net_is_player1):
                new_net_wins += 1
            else:
                new_net_losses += 1
    decisive = max(1, new_net_wins + new_net_losses)
    win_percentage = 100.0 * new_net_wins / float(decisive)
    score_percentage = 100.0 * (new_net_wins + 0.5 * draws) / float(num_games)
    return win_percentage, score_percentage, {"wins": new_net_wins, "losses": new_net_losses, "draws": draws}


def parse_args():
    parser = argparse.ArgumentParser(description="Quick AlphaZero smoke train/eval.")
    parser.add_argument("--net", choices=["light", "resnet"], default="light")
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--mcts-iters", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--episodes-per-epoch", type=int, default=4)
    parser.add_argument("--training-iters", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--temperature-moves", type=int, default=15)
    return parser.parse_args()


def run_smoke():
    device = select_device()
    np.set_printoptions(precision=4)

    args = parse_args()
    if args.net == "light":
        net = AlphaZeroNet(args.channels, device=device)
    else:
        net = AlphaZeroResNet(args.channels, num_blocks=args.blocks, device=device)
    net.eval()
    baseline = copy.deepcopy(net)

    exp_pool = []
    for _ in range(args.epochs):
        for _ in range(args.episodes_per_epoch):
            exp_pool.extend(self_play_episode(net, args.mcts_iters, args.temperature_moves))
        net.process_data(
            exp_pool,
            epochs=args.training_iters,
            batch_size=64,
        )

    win_rate, score_rate, scores = compare_agents(net, baseline, num_games=args.eval_games, mcts_iters=args.mcts_iters)
    print(
        "Smoke eval: win_rate=%.1f%% score=%.1f%% wins=%d losses=%d draws=%d"
        % (win_rate, score_rate, scores["wins"], scores["losses"], scores["draws"])
    )


if __name__ == "__main__":
    run_smoke()
