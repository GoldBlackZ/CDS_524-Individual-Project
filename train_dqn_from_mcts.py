from __future__ import annotations
import argparse
import random
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from board3d import Board3D
from config import P1, P2, X_SIZE, Y_SIZE
from mcts_ai import MCTSAI
from dqn_net import DQNNet, encode_board, N_ACTIONS, legal_action_mask

# -------------------------------
# Teacher: short MCTS to estimate value of a position
# -------------------------------
def mcts_value(board: Board3D, player: int, sims: int = 200) -> float:
    """Estimate V(s) in [-1,1] for `player` by sampling rollouts via MCTS AI.
    This uses the existing MCTSAI.choose() logic as a driver, but returns a value.
    Implementation: run `sims` random playouts with light rollout policy.
    """
    # We'll reuse MCTSAI internals only lightly by doing our own rollouts here
    wins = 0
    draws = 0
    for _ in range(sims):
        b = board.clone()
        cur = player
        # Play until terminal or max steps
        for _step in range(200):
            w = b.check_winner()
            if w != 0:
                if w == player: wins += 1
                break
            if b.is_full():
                draws += 1
                break
            mv = random.choice(b.valid_moves())
            b.drop(mv[0], mv[1], cur)
            cur = P1 if cur == P2 else P2
    total = sims
    # map win/draw/loss to [-1,1]
    # loss count inferred
    losses = total - wins - draws
    return (wins - losses) / max(1, total)

def one_step_q_targets(board: Board3D, player: int, gamma: float, sims: int) -> List[Tuple[torch.Tensor, int, float, torch.Tensor, bool]]:
    """Generate transitions (s, a, target_q_for_a, mask) for all legal actions.
    target uses a teacher value estimate:
        Q(s,a) ≈ r + gamma * V_teacher(s')
    where r is terminal reward in {-1,0,1} from player's perspective.
    """
    out = []
    for (x,y) in board.valid_moves():
        b2 = board.clone()
        b2.drop(x,y,player)
        winner = b2.check_winner()
        done = (winner != 0) or b2.is_full()
        if done:
            if winner == player: r = 1.0
            elif winner == 0: r = 0.0
            else: r = -1.0
            target = r
        else:
            opp = P1 if player == P2 else P2
            # after player moves, it's opponent's turn, so value for player is -V(opp perspective)
            v_opp = mcts_value(b2, opp, sims=sims)
            v_player = -v_opp
            target = gamma * v_player

        s = encode_board(board, player)
        a = x * Y_SIZE + y
        out.append((s, a, float(target)))
    return out

class QDataset(Dataset):
    def __init__(self, rows: List[Tuple[torch.Tensor,int,float]]):
        self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        s,a,t = self.rows[i]
        return s, torch.tensor(a, dtype=torch.long), torch.tensor(t, dtype=torch.float32)

def random_position(max_plies: int = 16) -> Tuple[Board3D,int]:
    b = Board3D()
    cur = P1
    plies = random.randint(0, max_plies)
    for _ in range(plies):
        if b.check_winner() != 0 or b.is_full():
            break
        x,y = random.choice(b.valid_moves())
        b.drop(x,y,cur)
        cur = P1 if cur == P2 else P2
    # return position and player-to-move
    return b, cur

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=1200, help="how many random positions to sample")
    ap.add_argument("--max_plies", type=int, default=18, help="max random plies to reach a position")
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--teacher_sims", type=int, default=120, help="rollouts per value estimate")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="dqn_model.pt")
    args = ap.parse_args()

    rows: List[Tuple[torch.Tensor,int,float]] = []
    for _ in range(args.positions):
        b, player = random_position(max_plies=args.max_plies)
        if b.check_winner() != 0 or b.is_full():
            continue
        rows.extend(one_step_q_targets(b, player, gamma=args.gamma, sims=args.teacher_sims))

    ds = QDataset(rows)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    device = torch.device("cpu")
    model = DQNNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for s, a, t in dl:
            s = s.to(device)
            a = a.to(device)
            t = t.to(device)
            q_all = model(s)  # (B, N_ACTIONS)
            q_sa = q_all.gather(1, a.view(-1,1)).squeeze(1)
            loss = loss_fn(q_sa, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * s.size(0)
            n += s.size(0)
        print(f"epoch {ep}: loss={total_loss/max(1,n):.6f}  samples={n}")

    torch.save(model.state_dict(), args.out)
    print(f"Saved DQN model -> {args.out}")

if __name__ == "__main__":
    main()
