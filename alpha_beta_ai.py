from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math, random
from board3d import Board3D
from config import P1, P2, CONNECT_N, X_SIZE, Y_SIZE, Z_SIZE, VALUE_MODEL_PATH, VALUE_BLEND, DQN_MODEL_PATH, USE_DQN_MOVE_ORDERING

# Optional learned value evaluator (blended into leaf evaluation)
try:
    from value_net import ValueEvaluator
except Exception:  # pragma: no cover
    ValueEvaluator = None  # type: ignore

# Optional DQN for move ordering (Q(s,a) guidance)
try:
    import torch
    from dqn_net import DQNNet, encode_board as _encode_dqn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    DQNNet = None  # type: ignore
    _encode_dqn = None  # type: ignore

_DQN_NET = None
_DQN_DEVICE = None

def _get_dqn_net():
    global _DQN_NET, _DQN_DEVICE
    if not USE_DQN_MOVE_ORDERING:
        return None
    if _DQN_NET is not None:
        return _DQN_NET
    if torch is None or DQNNet is None or _encode_dqn is None:
        _DQN_NET = None
        return None
    _DQN_DEVICE = torch.device("cpu")
    net = DQNNet().to(_DQN_DEVICE)
    try:
        state = torch.load(DQN_MODEL_PATH, map_location=_DQN_DEVICE)
        net.load_state_dict(state)
        net.eval()
        _DQN_NET = net
    except Exception:
        _DQN_NET = None
    return _DQN_NET

def _dqn_q_for_moves(board: Board3D, player: int, moves: List[Tuple[int,int]]) -> dict[Tuple[int,int], float]:
    """Return a dict of Q estimates for the given moves. Missing if no model."""
    net = _get_dqn_net()
    if net is None:
        return {}
    with torch.no_grad():
        s = _encode_dqn(board, player).to(_DQN_DEVICE).unsqueeze(0)
        q = net(s).squeeze(0).cpu()
    out = {}
    for (x,y) in moves:
        out[(x,y)] = float(q[x * Y_SIZE + y].item())
    return out

_VALUE_EVAL = None
_VALUE_SCALE = 4500.0  # maps value [-1,1] into heuristic-ish score range

def _get_value_eval():
    global _VALUE_EVAL
    if _VALUE_EVAL is not None:
        return _VALUE_EVAL
    if ValueEvaluator is None:
        _VALUE_EVAL = None
        return None
    _VALUE_EVAL = ValueEvaluator(VALUE_MODEL_PATH)
    if not getattr(_VALUE_EVAL, "ok", False):
        _VALUE_EVAL = None
    return _VALUE_EVAL

def other(p:int) -> int:
    return P1 if p == P2 else P2

def in_bounds(x:int,y:int,z:int) -> bool:
    return 0 <= x < X_SIZE and 0 <= y < Y_SIZE and 0 <= z < Z_SIZE

DIRS = [
    (1,0,0),(0,1,0),(0,0,1),
    (1,1,0),(1,-1,0),
    (1,0,1),(1,0,-1),
    (0,1,1),(0,1,-1),
    (1,1,1),(1,1,-1),(1,-1,1),(1,-1,-1),
]

def precompute_lines() -> List[List[Tuple[int,int,int]]]:
    lines=[]
    for x in range(X_SIZE):
        for y in range(Y_SIZE):
            for z in range(Z_SIZE):
                for dx,dy,dz in DIRS:
                    ex = x + (CONNECT_N-1)*dx
                    ey = y + (CONNECT_N-1)*dy
                    ez = z + (CONNECT_N-1)*dz
                    if in_bounds(ex,ey,ez):
                        lines.append([(x+k*dx, y+k*dy, z+k*dz) for k in range(CONNECT_N)])
    return lines

LINES = precompute_lines()

# You can tune W[3] here
W = {0:0, 1:1, 2:14, 3:220, 4:100000}

def center_bonus(x:int,y:int) -> float:
    cx = (X_SIZE-1)*0.5
    cy = (Y_SIZE-1)*0.5
    return -((x-cx)**2 + (y-cy)**2)

def z_for_move(board: Board3D, m: Tuple[int,int]) -> int:
    z = board.next_free_z(m[0], m[1])
    return z if z is not None else 999

def immediate_wins(board: Board3D, player:int) -> List[Tuple[int,int]]:
    wins=[]
    for (x,y) in board.valid_moves():
        board.drop(x,y,player)
        if board.check_winner() == player:
            wins.append((x,y))
        board.undo()
    return wins

def count_immediate_wins_after(board: Board3D, player:int) -> int:
    c = 0
    for (x,y) in board.valid_moves():
        board.drop(x,y,player)
        if board.check_winner() == player:
            c += 1
        board.undo()
    return c

def fork_moves(board: Board3D, player:int) -> List[Tuple[int,int]]:
    forks=[]
    for (x,y) in board.valid_moves():
        board.drop(x,y,player)
        if board.check_winner() != player:
            if count_immediate_wins_after(board, player) >= 2:
                forks.append((x,y))
        board.undo()
    return forks

def evaluate(board: Board3D, root:int) -> float:
    opp = other(root)
    score = 0.0

    for seg in LINES:
        rc = 0
        oc = 0
        for x,y,z in seg:
            v = board.get(x,y,z)
            if v == root:
                rc += 1
            elif v == opp:
                oc += 1
        if rc and oc:
            continue
        if rc:
            score += W[rc]
        elif oc:
            # defense coefficient is here: 1.15 -> 1.25 to defend harder
            score -= W[oc] * 1.15

    for x in range(X_SIZE):
        for y in range(Y_SIZE):
            tz = board.top_z(x,y)
            if tz >= 0:
                v = board.get(x,y,tz)
                if v == root:
                    score += 0.35 * center_bonus(x,y)
                elif v == opp:
                    score -= 0.35 * center_bonus(x,y)

    # z-height regularizer: discourage wasting moves on vertical stacking early
    for x in range(X_SIZE):
        for y in range(Y_SIZE):
            top = board.top_z(x,y)
            for z in range(top+1):
                v = board.get(x,y,z)
                if v == root:
                    score -= 0.35 * z
                elif v == opp:
                    score += 0.10 * z

    return score

def evaluate_blended(board: Board3D, root:int) -> float:
    """Blend hand-crafted heuristic with optional learned value.

    VALUE_BLEND in config controls the mix:
      0.0 => pure heuristic
      1.0 => pure learned value (scaled)
    """
    base = evaluate(board, root)
    b = float(VALUE_BLEND)
    if b <= 0.0:
        return base
    ev = _get_value_eval()
    if ev is None:
        return base
    v = ev.value(board, root)  # [-1, 1]
    learned = v * _VALUE_SCALE
    return (1.0 - b) * base + b * learned

_rng = random.Random(1337)
_Z = [[[[_rng.getrandbits(64) for _ in range(3)] for _ in range(Z_SIZE)] for _ in range(Y_SIZE)] for _ in range(X_SIZE)]

def board_hash(board: Board3D) -> int:
    h = 0
    for x in range(X_SIZE):
        for y in range(Y_SIZE):
            top = board.top_z(x,y)
            for z in range(top+1):
                v = board.get(x,y,z)
                h ^= _Z[x][y][z][v]
    return h

EXACT = 0
LOWER = 1
UPPER = 2

@dataclass
class TTEntry:
    depth: int
    value: float
    flag: int
    best_move: Optional[Tuple[int,int]]

@dataclass
class AlphaBetaAI:
    depth: int = 4
    tt_size: int = 200000
    use_forks: bool = True

    def __post_init__(self):
        self.tt: Dict[int, TTEntry] = {}

    def choose(self, board: Board3D, player:int) -> Tuple[int,int]:
        self.tt.clear()
        
        
        wins = immediate_wins(board, player)
        if wins:
            return max(wins, key=lambda m: center_bonus(m[0],m[1]))

        blocks = immediate_wins(board, other(player))
        if blocks:
            return max(blocks, key=lambda m: center_bonus(m[0],m[1]))

        if self.use_forks:
            forks = fork_moves(board, player)
            if forks:
                return max(forks, key=lambda m: center_bonus(m[0],m[1]))
            opp_forks = set(fork_moves(board, other(player)))
            if opp_forks:
                direct = [m for m in board.valid_moves() if m in opp_forks]
                if direct:
                    return max(direct, key=lambda m: center_bonus(m[0],m[1]))

        moves = self._ordered_moves(board, board.valid_moves(), player, pv_move=None)

        best_move = moves[0]
        best_val = -math.inf
        alpha = -math.inf
        beta = math.inf

        for m in moves:
            board.drop(m[0], m[1], player)
            val = -self._negamax(board, other(player), self.depth-1, -beta, -alpha, root=player)
            board.undo()
            if val > best_val:
                best_val = val
                best_move = m
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        return best_move

    def _ordered_moves(self, board: Board3D, moves: List[Tuple[int,int]], player:int, pv_move: Optional[Tuple[int,int]]) -> List[Tuple[int,int]]:
        opp = other(player)
        scored=[]
        qmap = _dqn_q_for_moves(board, player, moves)
        # DQN guidance is only for ordering; keep weight modest to avoid overriding tactics
        DQN_W = 55.0
        for m in moves:
            s = 0.0
            if pv_move is not None and m == pv_move:
                s += 1e6

            # Prefer lower z (spread on x-y plane) unless tactics override
            s -= 250.0 * z_for_move(board, m)

            board.drop(m[0], m[1], player)
            if board.check_winner() == player:
                s += 9e5
            board.undo()

            board.drop(m[0], m[1], opp)
            if board.check_winner() == opp:
                s += 6e5
            board.undo()

            s += 220.0 * center_bonus(m[0],m[1])
            if qmap:
                s += DQN_W * qmap.get(m, 0.0)
            scored.append((s,m))
        scored.sort(reverse=True, key=lambda t:t[0])
        return [m for _,m in scored]

    def _negamax(self, board: Board3D, to_move:int, depth:int, alpha:float, beta:float, root:int) -> float:
        winner = board.check_winner()
        if winner != 0:
            if winner == root:
                return 1e6 + depth*30
            else:
                return -1e6 - depth*30
        if board.is_full():
            return 0.0
        if depth <= 0:
            return evaluate_blended(board, root)

        h = board_hash(board) ^ (to_move * 0x9e3779b97f4a7c15)
        ent = self.tt.get(h)
        if ent is not None and ent.depth >= depth:
            if ent.flag == EXACT:
                return ent.value
            if ent.flag == LOWER:
                alpha = max(alpha, ent.value)
            elif ent.flag == UPPER:
                beta = min(beta, ent.value)
            if alpha >= beta:
                return ent.value

        iw = immediate_wins(board, to_move)
        if iw:
            if to_move == root:
                return 1e6 + depth*30
            else:
                return -1e6 - depth*30

        pv = ent.best_move if ent is not None else None
        moves = self._ordered_moves(board, board.valid_moves(), to_move, pv_move=pv)

        best = -math.inf
        best_move = moves[0]
        orig_alpha = alpha

        for m in moves:
            board.drop(m[0], m[1], to_move)
            val = -self._negamax(board, other(to_move), depth-1, -beta, -alpha, root)
            board.undo()

            if val > best:
                best = val
                best_move = m
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break

        flag = EXACT
        if best <= orig_alpha:
            flag = UPPER
        elif best >= beta:
            flag = LOWER

        if len(self.tt) >= self.tt_size:
            self.tt.clear()
        self.tt[h] = TTEntry(depth=depth, value=best, flag=flag, best_move=best_move)
        return best
