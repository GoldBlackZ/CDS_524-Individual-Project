from __future__ import annotations
from typing import Optional, Tuple
from config import P1, P2, DEFAULT_EPISODES
from board3d import Board3D
from mcts_ai import MCTSAI
from alpha_beta_ai import AlphaBetaAI
from dqn_ai import DQNAI

class Game:
    def __init__(self):
        self.board = Board3D()
        self.current = P1

        # MCTS control (episodes)
        self.episodes = DEFAULT_EPISODES

        # AlphaBeta control
        self.ab_depth = 4

        # P2 mode: 'human' | 'mcts' | 'ab' | 'dqn'
        self.p2_mode = "ab"
        self.p2_ai_mcts: Optional[MCTSAI] = None
        self.p2_ai_ab: Optional[AlphaBetaAI] = AlphaBetaAI(depth=self.ab_depth)
        self.p2_ai_dqn: Optional[DQNAI] = None

    def reset(self):
        self.board.reset()
        self.current = P1

    def set_p2_mode(self, mode: str):
        self.p2_mode = mode
        if mode == "human":
            self.p2_ai_mcts = None
            self.p2_ai_ab = None
        elif mode == "mcts":
            self.p2_ai_ab = None
            self.p2_ai_mcts = MCTSAI(simulations=self.episodes)
        elif mode == "ab":
            self.p2_ai_mcts = None
            self.p2_ai_dqn = None
            self.p2_ai_ab = AlphaBetaAI(depth=self.ab_depth)
        elif mode == "dqn":
            self.p2_ai_mcts = None
            self.p2_ai_ab = None
            self.p2_ai_dqn = DQNAI()
        else:
            raise ValueError("mode must be 'human' or 'mcts' or 'ab'")

    def set_episodes(self, episodes:int):
        self.episodes = max(50, min(8000, episodes))
        if self.p2_ai_mcts is not None:
            self.p2_ai_mcts.simulations = self.episodes

    def set_ab_depth(self, depth:int):
        self.ab_depth = max(2, min(6, depth))
        if self.p2_ai_ab is not None:
            self.p2_ai_ab.depth = self.ab_depth

    def mode_text(self) -> str:
        if self.p2_mode == "human":
            return "P2: Human"
        if self.p2_mode == "mcts":
            return f"P2: AI(MCTS)"
        return f"P2: AI(AlphaBeta d={self.ab_depth})" if self.p2_mode == "ab" else "P2: AI(DQN)"

    def handle_drop(self, col: Tuple[int,int]) -> bool:
        if self.board.check_winner() != 0 or self.board.is_full():
            return False
        x,y = col
        mv = self.board.drop(x,y,self.current)
        if mv is None:
            return False
        self.current = P1 if self.current == P2 else P2
        return True

    def maybe_ai_move(self):
        if self.board.check_winner() != 0 or self.board.is_full():
            return
        if self.current == P2 and self.p2_mode != "human":
            if self.p2_mode == "mcts" and self.p2_ai_mcts is not None:
                x,y = self.p2_ai_mcts.choose(self.board, P2)
                self.handle_drop((x,y))
            elif self.p2_mode == "ab" and self.p2_ai_ab is not None:
                x,y = self.p2_ai_ab.choose(self.board, P2)
                self.handle_drop((x,y))
            elif self.p2_mode == "dqn" and self.p2_ai_dqn is not None:
                x,y = self.p2_ai_dqn.choose(self.board, P2)
                self.handle_drop((x,y))
