# expert_dataset.py
# 封装专家示例数据，并提供生成专家示例所需的辅助函数
import torch
from torch.utils.data import Dataset
import numpy as np
import random
from typing import Tuple, List

class ExpertDataset(Dataset):
    """
    PyTorch Dataset，用于加载专家示例数据
    每个样本是 (state_tensor, move_idx)
    """
    def __init__(self, path: str):
        # 从磁盘加载由 generate_expert_data.py 生成的列表
        self.examples = torch.load(path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, int]:
        state, move = self.examples[idx]
        return state, move


def board_to_tensor(board: np.ndarray) -> torch.FloatTensor:
    """
    将棋盘状态转换为网络输入张量（3 通道：黑子、白子、空位）
    """
    black_plane = (board == 1).astype(np.float32)
    white_plane = (board == -1).astype(np.float32)
    empty_plane = (board == 0).astype(np.float32)
    tensor = np.stack([black_plane, white_plane, empty_plane], axis=0)
    return torch.from_numpy(tensor)


def find_open_four_moves(board: np.ndarray, player: int) -> List[Tuple[int, int]]:
    """
    检测并返回所有能让 player 直接连成五或堵住对手四连的空位坐标。
    支持水平、垂直和两条对角线方向。
    """
    H, W = board.shape
    moves: List[Tuple[int,int]] = []
    # 扫描四个方向：右 (0,1)，下 (1,0)，右下 (1,1)，右上 (-1,1)
    directions = [(0,1), (1,0), (1,1), (-1,1)]
    for dr, dc in directions:
        for r in range(H):
            for c in range(W):
                # 收集 window 五个格子的坐标
                coords = [(r + i*dr, c + i*dc) for i in range(5)]
                # 判断是否都在棋盘范围内
                if any(rr < 0 or rr >= H or cc < 0 or cc >= W for rr, cc in coords):
                    continue
                window = np.array([board[rr, cc] for rr, cc in coords])
                # 四颗 player 和 一颗空
                if np.count_nonzero(window == player) == 4 and np.count_nonzero(window == 0) == 1:
                    idx0 = int(np.where(window == 0)[0][0])
                    rr, cc = coords[idx0]
                    # 确保空位两端至少一端开放
                    prev_r, prev_c = rr - dr, cc - dc
                    next_r, next_c = rr + dr, cc + dc
                    open_prev = (0 <= prev_r < H and 0 <= prev_c < W and board[prev_r, prev_c] == 0)
                    open_next = (0 <= next_r < H and 0 <= next_c < W and board[next_r, next_c] == 0)
                    if open_prev or open_next:
                        moves.append((rr, cc))
    return moves


def expert_policy(board: np.ndarray, me: int = 1) -> Tuple[int, int]:
    """
    简单专家策略：
      1) 我方能连五就连五；
      2) 否则堵住对手的 open-four；
      3) 否则随机落子。
    """
    opp = 2 if me == 1 else 1
    # 1) 自己的必赢手
    my_moves = find_open_four_moves(board, me)
    if my_moves:
        return random.choice(my_moves)
    # 2) 堵对手的 open-four
    opp_moves = find_open_four_moves(board, opp)
    if opp_moves:
        return random.choice(opp_moves)
    # 3) 随机走
    empties = list(zip(*np.where(board == 0)))
    return random.choice(empties)
