# 立体四子棋 5×5×5（pygame 可旋转视角）+ episodes=500 更聪明 AI

## 规则
- 棋盘：5×5×5 三维空间
- 落子：你在 **x-y 平面**选择一个 (x,y) “柱子”，棋子沿 **z 轴从下往上堆叠**
- 胜利：任意三维直线方向（横/竖/斜/空间对角线等）连成 4 子即胜

## 依赖
- Python 3.9+
- pygame

安装：
```bash
python3 -m pip install pygame
```

运行：
```bash
python3 main.py
```

## 操作
- 鼠标左键：点击某个 (x,y) 柱子落子
- A / D：左右旋转（yaw）
- W / S：抬头/低头（pitch）
- R：重开
- Q / ESC：退出
- 1：P2 = 人类
- 2：P2 = AI（MCTS）
- +/-：调整 AI 每步模拟次数（episodes，默认 500）

## episodes=500 是什么？
这里把 episodes 解释为 **MCTS 每一步的模拟次数（simulations）**：
- episodes 越大，AI 越会算，更聪明，但每一步更慢
- 默认 500，兼顾速度与强度（M1 可跑）


## 强化AI说明（不增加太多算力）
- AlphaBeta 默认深度 d=4，但加入：两步 fork 检测 + 置换表（transposition table）+ 更强走法排序。
- 这会显著提升“先防后攻”的表现，不需要把深度硬拉到 d=5。


## 针对“AI老往z轴堆叠导致落后”的修正
- AlphaBeta 的走法排序与评估加入了 **轻微的z高度惩罚**，鼓励前期更多占据 x-y 平面来争夺空间。
- 惩罚很小：如果堆叠能形成强威胁/必胜，仍然会优先选择。
