from __future__ import annotations
from typing import Tuple, Optional
import random
import torch

from board3d import Board3D
from dqn_net import DQNNet, encode_board, legal_action_mask, N_ACTIONS
from config import X_SIZE, Y_SIZE

class DQNAI:
    """Use a trained DQNNet to choose a move (x,y).

    - If epsilon>0, will do epsilon-greedy exploration (handy for self-play testing).
    - Masks illegal columns (full columns).
    """
    def __init__(self, model_path: str = "dqn_model.pt", device: str = "cpu", epsilon: float = 0.0):
        self.device = torch.device(device)
        self.model = DQNNet().to(self.device)
        self.epsilon = float(epsilon)
        self.loaded = False
        try:
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            self.loaded = True
        except Exception:
            # Model not found or incompatible -> stay uninitialized, will fall back to random legal move
            self.loaded = False

    def choose(self, board: Board3D, player: int) -> Tuple[int,int]:
        legal = board.valid_moves()
        if not legal:
            return (0,0)

        # exploration / fallback
        if (not self.loaded) or (self.epsilon > 0.0 and random.random() < self.epsilon):
            return random.choice(legal)

        with torch.no_grad():
            s = encode_board(board, player).to(self.device).unsqueeze(0)  # (1, in_dim)
            q = self.model(s).squeeze(0).cpu()  # (N_ACTIONS,)
            mask = legal_action_mask(board)  # 1 for legal
            q = q.masked_fill(mask == 0, float("-inf"))
            a = int(torch.argmax(q).item())

        x = a // Y_SIZE
        y = a % Y_SIZE
        return (x,y)
