from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from board3d import Board3D
from config import X_SIZE, Y_SIZE, Z_SIZE, EMPTY, P1, P2

N_ACTIONS = X_SIZE * Y_SIZE  # 25 for 5x5

def encode_board(board: Board3D, player: int) -> torch.Tensor:
    """Encode board from `player` perspective into shape (2*X*Y*Z,).
    Channel 0: player's stones, Channel 1: opponent's stones.
    """
    opp = P1 if player == P2 else P2
    feat = torch.zeros((2, X_SIZE, Y_SIZE, Z_SIZE), dtype=torch.float32)
    for x in range(X_SIZE):
        for y in range(Y_SIZE):
            for z in range(Z_SIZE):
                v = board.get(x,y,z)
                if v == player:
                    feat[0, x, y, z] = 1.0
                elif v == opp:
                    feat[1, x, y, z] = 1.0
    return feat.flatten()

def legal_action_mask(board: Board3D) -> torch.Tensor:
    """Return float mask shape (N_ACTIONS,) with 1.0 for legal (x,y) columns."""
    m = torch.zeros((N_ACTIONS,), dtype=torch.float32)
    for (x,y) in board.valid_moves():
        m[x * Y_SIZE + y] = 1.0
    return m

class DQNNet(nn.Module):
    """A small MLP that outputs Q(s,a) for all (x,y) columns."""
    def __init__(self, hidden: int = 512):
        super().__init__()
        in_dim = 2 * X_SIZE * Y_SIZE * Z_SIZE
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
