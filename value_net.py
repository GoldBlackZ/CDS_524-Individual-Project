"""Learned value evaluation for 5x5x5 Connect-4-3D.

This file is intentionally lightweight:
- A small MLP ValueNet (CPU-friendly)
- A board encoder that matches the project's (x,y,z) indexing

Value convention used here:
- Network predicts v in [-1, 1] from the *current player's* perspective.
  +1 = current player is winning, -1 = losing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from board3d import Board3D
from config import X_SIZE, Y_SIZE, Z_SIZE, P1, P2


def other(p: int) -> int:
    return P1 if p == P2 else P2


def encode_board(board: Board3D, player: int) -> torch.Tensor:
    """Encode board as a (250,) float tensor: 2 channels (me, opp) over 5x5x5."""
    opp = other(player)
    # Shape: (2, X, Y, Z) then flatten.
    t = torch.zeros((2, X_SIZE, Y_SIZE, Z_SIZE), dtype=torch.float32)
    for x in range(X_SIZE):
        for y in range(Y_SIZE):
            top = board.top_z(x, y)
            for z in range(top + 1):
                v = board.get(x, y, z)
                if v == player:
                    t[0, x, y, z] = 1.0
                elif v == opp:
                    t[1, x, y, z] = 1.0
    return t.reshape(-1)


class ValueNet(nn.Module):
    """Small MLP value network (CPU friendly)."""

    def __init__(self, in_dim: int = 2 * X_SIZE * Y_SIZE * Z_SIZE, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        return self.net(x).squeeze(-1)


@dataclass
class ValueEvaluator:
    """Loads a ValueNet checkpoint (optional) and evaluates boards."""

    ckpt_path: str
    device: str = "cpu"
    model: Optional[ValueNet] = None
    ok: bool = False

    def __post_init__(self):
        try:
            m = ValueNet()
            state = torch.load(self.ckpt_path, map_location=self.device)
            # Accept either raw state_dict or a dict with 'state_dict'
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            m.load_state_dict(state)
            m.eval()
            self.model = m
            self.ok = True
        except Exception:
            self.model = None
            self.ok = False

    @torch.no_grad()
    def value(self, board: Board3D, player: int) -> float:
        """Return v in [-1, 1] from *player* perspective."""
        if not self.ok or self.model is None:
            return 0.0
        x = encode_board(board, player).unsqueeze(0)  # (1, 250)
        v = self.model(x).item()
        # Safety clamp
        if v > 1.0:
            v = 1.0
        if v < -1.0:
            v = -1.0
        return float(v)
