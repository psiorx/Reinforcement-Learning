import argparse
import time
from datetime import datetime
import torch
import os
import copy
import numpy as np
from threading import Event
from queue import Queue, Empty
from torch.utils.tensorboard import SummaryWriter
from connect4 import Connect4
from neuralnetwork import AlphaZeroResNet
from mcts import AlphaZeroMCTS
from collections import deque
from threading import Thread

def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero training loop for Connect4.")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--mcts-iters", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--exp-pool-size", type=int, default=8192)
    parser.add_argument("--training-iters", type=int, default=20)
    parser.add_argument("--duel-interval", type=int, default=20)
    parser.add_argument("--duel-games", type=int, default=10)
    parser.add_argument("--temperature-moves", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()

args = parse_args()

# hyperparams
batch_size = args.batch_size
mcts_iterations = args.mcts_iters
exp_pool_size = args.exp_pool_size
duel_interval = args.duel_interval
duel_acceptance = 60.0
channel_width = args.channels
num_blocks = args.blocks
training_iterations = args.training_iters
temperature_moves = args.temperature_moves
network_file = "connect4_resnet.net"
device = "cuda" if torch.cuda.is_available() else "cpu"
elo_k = 32
max_episodes = args.max_episodes if args.max_episodes > 0 else None

np.set_printoptions(precision=4)

if os.path.isfile(network_file):
    print("loading previous network")
    net = torch.load(network_file, map_location=device)
else:
    print("starting new agent")
    net = AlphaZeroResNet(channel_width, num_blocks=num_blocks, device=device)
net.eval()

training_net = copy.deepcopy(net)
training_net.train()

def update_elo(new_elo, old_elo, score, k_factor):
    expected = 1.0 / (1.0 + 10 ** ((old_elo - new_elo) / 400.0))
    delta = k_factor * (score - expected)
    return new_elo + delta, old_elo - delta

def get_experience(current_net):
    current_net.eval()
    game = Connect4()            
    experiences = []
    move_count = 0
    while not game.game_over:
        mcts = AlphaZeroMCTS(game, current_net, cpuct=1.5, dirichlet_alpha=0.3, dirichlet_frac=0.25)
        policy = mcts.search(mcts_iterations)
        if move_count < temperature_moves:
            action = np.random.choice(range(len(policy)), p=policy)
        else:
            action = int(np.argmax(policy))
        e = [copy.deepcopy(game.board), copy.deepcopy(policy), game.player, None]
        experiences.append(e)
        game.take_action(action)
        move_count += 1
    
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
                mcts = AlphaZeroMCTS(game, new_net, cpuct=1.5, dirichlet_alpha=0.0, dirichlet_frac=0.0)
                policy = mcts.search(mcts_iterations)
                action = np.argmax(policy)
            else:
                mcts = AlphaZeroMCTS(game, orig_net, cpuct=1.5, dirichlet_alpha=0.0, dirichlet_frac=0.0)
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
    return win_percentage, scores

q_out = Queue()
stop_event = Event()

def generate_experience(net, q_out, stop_event):
    while not stop_event.is_set():
        exp = get_experience(net)
        if stop_event.is_set():
            break
        q_out.put(exp)

threads = []
num_workers = args.num_workers
for i in range(num_workers):
    t = Thread(target=generate_experience, args=[net, q_out, stop_event])
    t.daemon = True
    t.start()
    threads.append(t)

exp_pool = deque()
recent_game_lengths = deque(maxlen=100)
total_self_play_games = 0
total_self_play_moves = 0
start_time = time.time()

training_episodes = 0
elo_new, elo_old = 1500.0, 1500.0
run_name = args.run_name or "resnet_c%d_b%d_mcts%d_bs%d" % (
    channel_width,
    num_blocks,
    mcts_iterations,
    batch_size,
)
run_name = run_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(log_dir=os.path.join("runs", "connect4", run_name))

try:
    while True:
        try:
            game_experience = q_out.get(timeout=1.0)
        except Empty:
            if stop_event.is_set():
                break
            continue
        total_self_play_games += 1
        total_self_play_moves += len(game_experience)
        recent_game_lengths.append(len(game_experience))
        writer.add_scalar("self_play/games_total", total_self_play_games, training_episodes)
        writer.add_scalar("self_play/moves_total", total_self_play_moves, training_episodes)
        writer.add_scalar("self_play/last_game_length", len(game_experience), training_episodes)
        writer.add_scalar(
            "self_play/avg_game_length",
            float(np.mean(recent_game_lengths)),
            training_episodes,
        )
        writer.add_scalar("self_play/exp_pool_size", len(exp_pool), training_episodes)
        elapsed = time.time() - start_time
        writer.add_scalar("self_play/seconds_elapsed", elapsed, training_episodes)
        if elapsed > 0:
            writer.add_scalar("self_play/games_per_min", 60.0 * total_self_play_games / elapsed, training_episodes)
            writer.add_scalar("self_play/moves_per_sec", total_self_play_moves / elapsed, training_episodes)
        writer.flush()

        for e in game_experience:
            exp_pool.append(e)
            if len(exp_pool) > exp_pool_size:
                exp_pool.popleft()
            
        if len(exp_pool) > batch_size:
            training_indexes = np.random.choice(range(len(exp_pool)), batch_size, replace=False)
            batch = []
            for idx in training_indexes:
                batch.append(exp_pool[idx])
            print("training episode %d with batch: %d" % (training_episodes, len(batch)))
            losses = training_net.process_data(batch, training_iterations)
            if losses:
                writer.add_scalar("loss/value", losses["value_loss"], training_episodes)
                writer.add_scalar("loss/policy", losses["policy_loss"], training_episodes)
                writer.flush()
            training_episodes += 1
            if max_episodes is not None and training_episodes >= max_episodes:
                print("Reached max episodes (%d). Exiting." % max_episodes)
                break
            if training_episodes % duel_interval == 0:
                training_net.eval()
                print("Duel Commencing - THERE CAN BE ONLY ONE")
                win_rate, scores = compare_agents(training_net, net, args.duel_games)
                score = (scores[1] + 0.5 * scores[0]) / float(args.duel_games)
                elo_new, elo_old = update_elo(elo_new, elo_old, score, elo_k)
                writer.add_scalar("duel/win_rate", win_rate, training_episodes)
                writer.add_scalar("duel/wins", scores[1], training_episodes)
                writer.add_scalar("duel/losses", scores[2], training_episodes)
                writer.add_scalar("duel/draws", scores[0], training_episodes)
                writer.add_scalar("elo/new", elo_new, training_episodes)
                writer.add_scalar("elo/old", elo_old, training_episodes)
                writer.flush()
                if win_rate >= duel_acceptance:
                    net.load_state_dict(training_net.state_dict())
                    torch.save(training_net, network_file)
                    print("UPGRADE COMPLETE")

        print("total experience: %d" % len(exp_pool))
        if max_episodes is not None and training_episodes >= max_episodes:
            break
except KeyboardInterrupt:
    print("Interrupted. Shutting down workers...")
finally:
    stop_event.set()
    writer.close()
    
