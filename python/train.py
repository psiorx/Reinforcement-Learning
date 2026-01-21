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
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
from connect4 import Connect4
from neuralnetwork import AlphaZeroNet, AlphaZeroResNet

def build_net(net_type, channels, blocks, device):
    if net_type == "light":
        return AlphaZeroNet(channels, device=device)
    return AlphaZeroResNet(channels, num_blocks=blocks, device=device)

def load_checkpoint(path, device, fallback_net, quiet=False):
    if not os.path.isfile(path):
        return None, fallback_net
    if not quiet:
        print("loading previous network")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    except Exception:
        try:
            from torch.serialization import safe_globals
            with safe_globals([AlphaZeroNet, AlphaZeroResNet]):
                checkpoint = torch.load(path, map_location=device, weights_only=False)
        except Exception:
            raise

    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint, fallback_net
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint, checkpoint.get("net", fallback_net)
    if isinstance(checkpoint, dict):
        # assume raw state_dict
        return {"state_dict": checkpoint}, fallback_net
    return None, fallback_net
from mcts import AlphaZeroMCTS
from collections import deque
from threading import Thread

def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero training loop for Connect4.")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--net", choices=["light", "resnet"], default="light")
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--mcts-iters", type=int, default=25)
    parser.add_argument("--duel-mcts-iters", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--exp-pool-size", type=int, default=15000)
    parser.add_argument("--training-iters", type=int, default=10)
    parser.add_argument("--duel-interval", type=int, default=1)
    parser.add_argument("--duel-games", type=int, default=40)
    parser.add_argument("--duel-acceptance", type=float, default=60.0)
    parser.add_argument("--temperature-moves", type=int, default=15)
    parser.add_argument("--min-exp-size", type=int, default=15000)
    parser.add_argument("--num-eps", type=int, default=100, help="Number of self-play games per iteration")
    parser.add_argument("--history-iters", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-frac", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--stop-after-upgrades", type=int, default=0)
    parser.add_argument("--stop-after-stalemates", type=int, default=0, help="Stop after N consecutive 100%% draw duels")
    return parser.parse_args()

args = parse_args()
minimal_logging = True

# hyperparams
batch_size = args.batch_size
mcts_iterations = args.mcts_iters
duel_mcts_iterations = args.duel_mcts_iters or mcts_iterations
exp_pool_size = args.exp_pool_size
min_exp_size = min(args.min_exp_size, exp_pool_size)
history_iters = args.history_iters
duel_interval = args.duel_interval
duel_acceptance = args.duel_acceptance
channel_width = args.channels
num_blocks = args.blocks
training_iterations = args.training_iters
temperature_moves = args.temperature_moves
dirichlet_alpha = args.dirichlet_alpha
dirichlet_frac = args.dirichlet_frac
learning_rate = args.lr
weight_decay = args.weight_decay
stop_after_upgrades = args.stop_after_upgrades
stop_after_stalemates = args.stop_after_stalemates
draw_value = 1e-4
network_file = "connect4_%s.net" % args.net
if args.device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = args.device
elo_k = 32
max_episodes = args.max_episodes if args.max_episodes > 0 else None

np.set_printoptions(precision=4)

checkpoint, net_type = load_checkpoint(network_file, device, args.net, quiet=minimal_logging)
if isinstance(checkpoint, torch.nn.Module):
    net = checkpoint
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    net = build_net(net_type, channel_width, num_blocks, device)
    net.load_state_dict(checkpoint["state_dict"])
else:
    if not minimal_logging:
        print("starting new agent")
    net = build_net(args.net, channel_width, num_blocks, device)
net.eval()

training_net = copy.deepcopy(net)
training_net.train()

def update_elo(new_elo, old_elo, score, k_factor):
    expected = 1.0 / (1.0 + 10 ** ((old_elo - new_elo) / 400.0))
    delta = k_factor * (score - expected)
    return new_elo + delta, old_elo - delta

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

def get_experience(current_net):
    current_net.eval()
    game = Connect4()            
    experiences = []
    move_count = 0
    while not game.game_over:
        mcts = AlphaZeroMCTS(
            game,
            current_net,
            cpuct=1.0,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_frac=dirichlet_frac,
            draw_value=draw_value,
        )
        policy = mcts.search(mcts_iterations)
        temp = 1 if move_count < temperature_moves else 0
        policy = apply_temperature(policy, temp)
        if temp > 0:
            action = np.random.choice(range(len(policy)), p=policy)
        else:
            action = int(np.argmax(policy))
        e = [copy.deepcopy(game.board), copy.deepcopy(policy), game.player, None]
        experiences.append(e)
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


def compare_agents(new_net, orig_net, num_games, mcts_iters, use_tqdm=True):
    orig_net.eval()
    new_net.eval()
    draws = 0
    new_net_wins = 0
    new_net_losses = 0
    use_tqdm = use_tqdm and tqdm is not None
    duel_bar = tqdm(total=num_games, desc="duel", ascii=True, leave=False) if use_tqdm else None
    for i in range(num_games):
        game = Connect4()
        new_net_is_player1 = (i % 2 == 0)
        while not game.game_over:
            use_new_net = (game.player == 1 and new_net_is_player1) or (game.player == 2 and not new_net_is_player1)
            net_to_use = new_net if use_new_net else orig_net
            mcts = AlphaZeroMCTS(
                game,
                net_to_use,
                cpuct=1.0,
                dirichlet_alpha=0.0,
                dirichlet_frac=0.0,
                draw_value=draw_value,
            )
            policy = mcts.search(mcts_iters)
            # Random tie-breaking among best actions
            max_val = np.max(policy)
            best_actions = np.where(policy == max_val)[0]
            action = int(np.random.choice(best_actions))
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
        draw_percentage = 100.0 * draws / float(i + 1)
        if duel_bar is not None:
            duel_bar.set_postfix(
                wins=new_net_wins,
                losses=new_net_losses,
                draws=draws,
                win_rate="%.1f%%" % win_percentage,
                draw_rate="%.1f%%" % draw_percentage,
            )
            duel_bar.update(1)
    if duel_bar is not None:
        duel_bar.close()
    decisive = max(1, new_net_wins + new_net_losses)
    win_percentage = 100.0 * new_net_wins / float(decisive)
    draw_percentage = 100.0 * draws / float(num_games)
    score_percentage = 100.0 * (new_net_wins + 0.5 * draws) / float(num_games)
    return win_percentage, score_percentage, {"wins": new_net_wins, "losses": new_net_losses, "draws": draws}

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
    t.daemon = False
    t.start()
    threads.append(t)

exp_pool = deque()
history_examples = deque(maxlen=history_iters)
recent_game_lengths = deque(maxlen=100)
total_self_play_games = 0
total_self_play_moves = 0
iter_games = 0  # Games played in current iteration
num_eps = args.num_eps  # Target games per iteration
start_time = time.time()
pool_bar = None
last_pool_count = 0
pending_self_play_notice = True

training_episodes = 0
elo_new, elo_old = 1500.0, 1500.0
consecutive_upgrades = 0
consecutive_stalemates = 0
if args.net == "light":
    run_name = args.run_name or "light_c%d_mcts%d_bs%d" % (
        channel_width,
        mcts_iterations,
        batch_size,
    )
else:
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
        if minimal_logging and pending_self_play_notice:
            iter_id = training_episodes + 1
            print("Iter %d self-play start" % iter_id)
            pending_self_play_notice = False
        total_self_play_games += 1
        iter_games += 1
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
        # Progress bar showing games in current iteration
        if iter_games < num_eps:
            if pool_bar is None and not minimal_logging:
                pool_bar = tqdm(total=num_eps, desc="self_play", ascii=True, leave=False)
                last_pool_count = 0
            delta = max(0, iter_games - last_pool_count)
            if delta and pool_bar is not None:
                pool_bar.update(delta)
                last_pool_count = iter_games
        elif pool_bar is not None:
            pool_bar.close()
            pool_bar = None
            last_pool_count = 0

        # Train after completing num_eps games, but require minimum examples
        if iter_games >= num_eps and len(exp_pool) >= min(min_exp_size, batch_size * 10):
            training_indexes = np.random.choice(range(len(exp_pool)), batch_size, replace=False)
            batch = []
            for idx in training_indexes:
                batch.append(exp_pool[idx])
            train_examples = []
            for past in history_examples:
                train_examples.extend(past)
            train_examples.extend(exp_pool)
            batches_per_epoch = max(1, len(train_examples) // batch_size)
            iter_id = training_episodes + 1
            if minimal_logging:
                print("Iter %d self-play end: examples=%d" % (iter_id, len(exp_pool)))
            losses = training_net.process_data(
                train_examples,
                epochs=training_iterations,
                batch_size=batch_size,
                lr=learning_rate,
                weight_decay=weight_decay,
                minimal_logging=minimal_logging,
            )
            if minimal_logging:
                if losses and losses.get("start") and losses.get("end"):
                    start_pi, start_v = losses["start"]
                    end_pi, end_v = losses["end"]
                    print("Iter %d train start: loss_pi=%.6f loss_v=%.6f" % (iter_id, start_pi, start_v))
                    print("Iter %d train end: loss_pi=%.6f loss_v=%.6f" % (iter_id, end_pi, end_v))
                else:
                    print("Iter %d train start: loss_pi=nan loss_v=nan" % iter_id)
                    print("Iter %d train end: loss_pi=nan loss_v=nan" % iter_id)
            if losses:
                writer.add_scalar("loss/value", losses["value_loss"], training_episodes)
                writer.add_scalar("loss/policy", losses["policy_loss"], training_episodes)
                writer.flush()
            training_episodes += 1
            if training_episodes % duel_interval == 0:
                training_net.eval()
                win_rate, score_rate, scores = compare_agents(
                    training_net,
                    net,
                    args.duel_games,
                    duel_mcts_iterations,
                    use_tqdm=not minimal_logging,
                )
                score = (scores["wins"] + 0.5 * scores["draws"]) / float(args.duel_games)
                elo_new, elo_old = update_elo(elo_new, elo_old, score, elo_k)
                writer.add_scalar("duel/win_rate", win_rate, training_episodes)
                writer.add_scalar("duel/score_rate", score_rate, training_episodes)
                writer.add_scalar("duel/draw_rate", 100.0 * scores["draws"] / args.duel_games, training_episodes)
                writer.add_scalar("duel/wins", scores["wins"], training_episodes)
                writer.add_scalar("duel/losses", scores["losses"], training_episodes)
                writer.add_scalar("duel/draws", scores["draws"], training_episodes)
                writer.add_scalar("elo/new", elo_new, training_episodes)
                writer.add_scalar("elo/old", elo_old, training_episodes)
                writer.flush()
                if minimal_logging:
                    print("Iter %d duel: new=%d prev=%d draws=%d win_rate=%.1f%%" % (
                        iter_id, scores["wins"], scores["losses"], scores["draws"], win_rate
                    ))
                # Check for stalemate (100% draws)
                is_stalemate = scores["draws"] == args.duel_games
                if is_stalemate:
                    consecutive_stalemates += 1
                    if minimal_logging:
                        print("Iter %d stalemate detected (%d consecutive)" % (iter_id, consecutive_stalemates))
                else:
                    consecutive_stalemates = 0
                # Check for stalemate termination
                if stop_after_stalemates > 0 and consecutive_stalemates >= stop_after_stalemates:
                    if minimal_logging:
                        print("Reached %d consecutive stalemates. Training converged!" % stop_after_stalemates)
                    break
                if win_rate >= duel_acceptance:
                    net.load_state_dict(training_net.state_dict())
                    torch.save(
                        {
                            "state_dict": training_net.state_dict(),
                            "net": args.net,
                            "channels": channel_width,
                            "blocks": num_blocks,
                        },
                        network_file,
                    )
                    if minimal_logging:
                        print("Iter %d result: accepted" % iter_id)
                    consecutive_upgrades += 1
                else:
                    training_net.load_state_dict(net.state_dict())
                    training_net.train()
                    if minimal_logging:
                        print("Iter %d result: rejected" % iter_id)
                    consecutive_upgrades = 0
                history_examples.append(deque(exp_pool))
                exp_pool = deque()
                iter_games = 0  # Reset game counter for next iteration
                pending_self_play_notice = True
                if stop_after_upgrades > 0 and consecutive_upgrades >= stop_after_upgrades:
                    if not minimal_logging:
                        print("Reached %d consecutive upgrades. Exiting." % stop_after_upgrades)
                    break
            elif minimal_logging:
                print("Iter %d duel: new=0 prev=0 draws=0" % iter_id)
                print("Iter %d result: skipped" % iter_id)
                history_examples.append(deque(exp_pool))
                exp_pool = deque()
                iter_games = 0  # Reset game counter for next iteration
                pending_self_play_notice = True
            if max_episodes is not None and training_episodes >= max_episodes:
                if not minimal_logging:
                    print("Reached max episodes (%d). Exiting." % max_episodes)
                break
        if max_episodes is not None and training_episodes >= max_episodes:
            break
except KeyboardInterrupt:
    print("Interrupted. Shutting down workers...")
finally:
    stop_event.set()
    for t in threads:
        t.join(timeout=5.0)
    writer.close()
    
