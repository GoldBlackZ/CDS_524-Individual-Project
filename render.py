from __future__ import annotations
from typing import Dict, Tuple, Optional
import pygame
from config import X_SIZE, Y_SIZE, GRID_SPACING, DISC_RADIUS, HOVER_RADIUS, UI_FONT_SIZE, EMPTY, P1, P2
from camera import Camera

def cell_to_world(x:int, y:int, z:int, X:int, Y:int, Z:int) -> Tuple[float,float,float]:
    cx = (X - 1) * 0.5
    cy = (Y - 1) * 0.5
    cz = (Z - 1) * 0.5
    wx = (x - cx) * GRID_SPACING
    wy = (y - cy) * GRID_SPACING
    wz = (z - cz) * GRID_SPACING
    return wx, wy, wz

class Renderer:
    def __init__(self, screen: pygame.Surface, camera: Camera, X:int, Y:int, Z:int):
        self.screen = screen
        self.cam = camera
        self.X, self.Y, self.Z = X, Y, Z
        self.font = pygame.font.SysFont(None, UI_FONT_SIZE)
        self.small = pygame.font.SysFont(None, 18)

    def draw(self, board, current_player:int, hover_col: Optional[Tuple[int,int]], mode_text:str, episodes:int):
        w,h = self.screen.get_size()
        self.screen.fill((14, 14, 20))

        # project column centers at z=0
        col_centers: Dict[Tuple[int,int], Tuple[int,int,float]] = {}
        for x in range(self.X):
            for y in range(self.Y):
                wx,wy,wz = cell_to_world(x,y,0, self.X,self.Y,self.Z)
                sx,sy,depth = self.cam.project(wx,wy,wz,w,h)
                col_centers[(x,y)] = (sx,sy,depth)

        self._draw_grid(col_centers)

        # discs far->near
        discs = []
        for x in range(self.X):
            for y in range(self.Y):
                top = board.top_z(x,y)
                for z in range(top+1):
                    p = board.get(x,y,z)
                    if p != EMPTY:
                        wx,wy,wz = cell_to_world(x,y,z, self.X,self.Y,self.Z)
                        sx,sy,depth = self.cam.project(wx,wy,wz,w,h)
                        discs.append((depth, z, p, sx, sy))
        discs.sort(reverse=True, key=lambda t: t[0])

        for depth, z, p, sx, sy in discs:
            self._draw_disc(p, sx, sy, z)

        if hover_col is not None:
            x,y = hover_col
            if board.next_free_z(x,y) is not None:
                sx,sy,_ = col_centers[(x,y)]
                pygame.draw.circle(self.screen, (230,230,255), (sx,sy), HOVER_RADIUS, 2)

        self._draw_ui(board, current_player, mode_text, episodes)

    def _draw_grid(self, col_centers: Dict[Tuple[int,int], Tuple[int,int,float]]):
        corners = [(0,0), (self.X-1,0), (self.X-1, self.Y-1), (0, self.Y-1)]
        pts = [col_centers[c][:2] for c in corners]
        pygame.draw.polygon(self.screen, (38,38,54), pts, width=0)
        pygame.draw.polygon(self.screen, (110,110,145), pts, width=3)

        for y in range(self.Y):
            for x in range(self.X-1):
                p1 = col_centers[(x,y)][:2]
                p2 = col_centers[(x+1,y)][:2]
                pygame.draw.line(self.screen, (85,85,115), p1, p2, 2)
        for x in range(self.X):
            for y in range(self.Y-1):
                p1 = col_centers[(x,y)][:2]
                p2 = col_centers[(x,y+1)][:2]
                pygame.draw.line(self.screen, (85,85,115), p1, p2, 2)

        for (x,y),(sx,sy,_) in col_centers.items():
            pygame.draw.circle(self.screen, (155,155,185), (sx,sy), 5)
            txt = self.small.render(f"{x},{y}", True, (215,215,235))
            self.screen.blit(txt, (sx+6, sy+6))

    def _draw_disc(self, player:int, sx:int, sy:int, z:int):
        if player == P1:
            base = (235, 70, 80)
            rim  = (255, 150, 160)
        else:
            base = (70, 175, 245)
            rim  = (155, 220, 255)

        shade = max(0.65, 1.0 - 0.07*z)
        color = (int(base[0]*shade), int(base[1]*shade), int(base[2]*shade))
        rimc  = (int(rim[0]*shade),  int(rim[1]*shade),  int(rim[2]*shade))

        pygame.draw.circle(self.screen, color, (sx,sy), DISC_RADIUS)
        pygame.draw.circle(self.screen, rimc, (sx,sy), DISC_RADIUS, 3)
        pygame.draw.circle(self.screen, (255,255,255), (sx-6, sy-8), 5, 2)

    def _draw_ui(self, board, current_player:int, mode_text:str, episodes:int):
        winner = board.check_winner()
        if winner != 0:
            msg = f"Winner: {'X' if winner==P1 else 'O'}   (R 重开, Q/Esc 退出)"
        elif board.is_full():
            msg = "Draw! (R 重开, Q/Esc 退出)"
        else:
            msg = f"Turn: {'X' if current_player==P1 else 'O'} | {mode_text} | episodes={episodes} | 鼠标点击落子"

        surf = self.font.render(msg, True, (240,240,250))
        self.screen.blit(surf, (18, 16))
        tip = self.small.render("视角: A/D 旋转 W/S 抬头 | 1人类 2MCTS 3AlphaBeta | +/-调episodes | [ ]调AB深度 | R重开 Q退出", True, (210,210,230))
        self.screen.blit(tip, (18, 44))

    def pick_column_from_mouse(self, board, mx:int, my:int) -> Optional[Tuple[int,int]]:
        w,h = self.screen.get_size()
        best = None
        best_d2 = 10**18
        for x in range(self.X):
            for y in range(self.Y):
                if board.next_free_z(x,y) is None:
                    continue
                wx,wy,wz = cell_to_world(x,y,0, self.X,self.Y,self.Z)
                sx,sy,_ = self.cam.project(wx,wy,wz,w,h)
                dx = mx - sx
                dy = my - sy
                d2 = dx*dx + dy*dy
                if d2 < best_d2:
                    best_d2 = d2
                    best = (x,y)
        if best is None:
            return None
        return best if best_d2 <= (HOVER_RADIUS*HOVER_RADIUS) else None
