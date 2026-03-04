"""Train a lightweight ValueNet using short MCTS evaluations as targets.

Why this script exists (your "方案2"):
- You already have a decent MCTS player.
- We can use MCTS to label positions with an approximate value.
- Then AlphaBeta can blend that learned value into its leaf evaluation.

This is CPU-friendly and doesn't require long RL self-play.

Usage (from this folder):

  python train_value_from_mcts.py --samples 6000 --mcts_sims 220 --epochs 8

It will write: value_model.pt
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from board3d import Board3D
from config import P1, P2, MAX_ROLLOUT_STEPS, VALUE_MODEL_PATH
from value_net import ValueNet, encode_board


def other(p: int) -> int:
    return P1 if p == P2 else P2


def center_score(move: Tuple[int, int], X: int = 5, Y: int = 5) -> float:
    cx = (X - 1) * 0.5
    cy = (Y - 1) * 0.5
    x, y = move
    return -((x - cx) ** 2 + (y - cy) ** 2)


def immediate_winning_moves(board: Board3D, player: int) -> List[Tuple[int, int]]:
    wins = []
    for m in board.valid_moves():
        board.drop(m[0], m[1], player)
        if board.check_winner() == player:
            wins.append(m)
        board.undo()
    return wins


def immediate_block_moves(board: Board3D, player: int) -> List[Tuple[int, int]]:
    opp = other(player)
    opp_wins = set(immediate_winning_moves(board, opp))
    return list(opp_wins)


def rollout_policy_move(board: Board3D, player: int) -> Tuple[int, int]:
    wins = immediate_winning_moves(board, player)
    if wins:
        return random.choice(wins)
    blocks = immediate_block_moves(board, player)
    if blocks:
        return random.choice(blocks)

    moves = board.valid_moves()
    scores = [center_score(m) for m in moves]
    mx = max(scores)
    weights = [math.exp((s - mx) * 0.35) for s in scores]
    total = sum(weights)
    r = random.random() * total
    acc = 0.0
    for m, w in zip(moves, weights):
        acc += w
        if acc >= r:
            return m
    return moves[-1]


@dataclass
class Node:
    parent: Optional["Node"]
    move: Optional[Tuple[int, int]]
    visits: int = 0
    value: float = 0.0
    children: Dict[Tuple[int, int], "Node"] = None
    untried: List[Tuple[int, int]] = None

    def __post_init__(self):
        self.children = {} if self.children is None else self.children
        self.untried = [] if self.untried is None else self.untried


def uct_select(node: Node, c: float = 1.35) -> Node:
    logN = math.log(max(1, node.visits))
    best_child = None
    best_score = -1e18
    for ch in node.children.values():
        if ch.visits == 0:
            score = 1e9
        else:
            exploit = ch.value / ch.visits
            explore = c * math.sqrt(logN / ch.visits)
            score = exploit + explore
        if score > best_score:
            best_score = score
            best_child = ch
    return best_child


def mcts_value_estimate(board: Board3D, player: int, simulations: int = 200) -> float:
    """Return an estimated value v in [-1, 1] from *player* perspective."""

    # Tactical short-circuit
    if immediate_winning_moves(board, player):
        return 1.0
    if immediate_winning_moves(board, other(player)):
        # If opponent has an immediate win, current player is in trouble.
        return -1.0

    root = Node(parent=None, move=None)
    root.untried = board.valid_moves()
    root_player = player

    for _ in range(simulations):
        b = board.clone()
        node = root
        p = player

        # Selection
        while (not node.untried) and node.children:
            node = uct_select(node)
            b.drop(node.move[0], node.move[1], p)
            p = other(p)

        # Expansion
        if node.untried:
            moves = node.untried
            scored = sorted(moves, key=lambda m: center_score(m), reverse=True)
            k = min(6, len(scored))
            m = random.choice(scored[:k])
            node.untried.remove(m)

            b.drop(m[0], m[1], p)
            p = other(p)
            child = Node(parent=node, move=m)
            child.untried = b.valid_moves()
            node.children[m] = child
            node = child

        # Rollout
        winner = b.check_winner()
        steps = 0
        while winner == 0 and (not b.is_full()) and steps < MAX_ROLLOUT_STEPS:
            mv = rollout_policy_move(b, p)
            b.drop(mv[0], mv[1], p)
            winner = b.check_winner()
            p = other(p)
            steps += 1

        # Backprop (root perspective)
        if winner == root_player:
            reward = 1.0
        elif winner == 0:
            reward = 0.5
        else:
            reward = 0.0

        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    if not root.children:
        return 0.0

    # Best move value in [0,1]
    best = max((ch.value / max(1, ch.visits)) for ch in root.children.values())
    # map to [-1,1]
    return float(2.0 * best - 1.0)


def sample_positions(samples: int, max_random_plies: int, seed: int) -> List[Tuple[Board3D, int]]:
    rng = random.Random(seed)
    out: List[Tuple[Board3D, int]] = []
    for _ in range(samples):
        b = Board3D()
        p = P1
        plies = rng.randint(0, max_random_plies)
        for _k in range(plies):
            if b.check_winner() != 0 or b.is_full():
                break
            mv = rng.choice(b.valid_moves())
            b.drop(mv[0], mv[1], p)
            p = other(p)
        if b.check_winner() == 0 and (not b.is_full()):
            out.append((b, p))  # state + side to move
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--max_random_plies", type=int, default=18)
    ap.add_argument("--mcts_sims", type=int, default=220)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=str, default=VALUE_MODEL_PATH)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Sampling {args.samples} positions ...")
    positions = sample_positions(args.samples, args.max_random_plies, args.seed)
    print(f"Got {len(positions)} positions. Labeling with MCTS ({args.mcts_sims} sims each) ...")

    xs = []
    ys = []
    for i, (b, p) in enumerate(positions, 1):
        v = mcts_value_estimate(b, p, simulations=args.mcts_sims)
        xs.append(encode_board(b, p))
        ys.append(v)
        if i % 400 == 0:
            print(f"  labeled {i}/{len(positions)}")

    X = torch.stack(xs)  # (N, 250)
    y = torch.tensor(ys, dtype=torch.float32)  # (N,)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=False)

    model = ValueNet()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(1, args.epochs + 1):
        total = 0.0
        n = 0
        for xb, yb in dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        print(f"epoch {ep:02d} | mse {total / max(1, n):.5f}")

    ckpt = {"state_dict": model.state_dict()}
    torch.save(ckpt, args.out)
    print(f"Saved model -> {args.out}")


if __name__ == "__main__":
    main()
