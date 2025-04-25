import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

BOARD_SIZE = 15  # 棋盘大小


def encode_pov_tensor(board: np.ndarray, me: int) -> torch.FloatTensor:
    """
    将棋盘转换为网络的 POV 输入张量：
      - board: np.ndarray, shape (H,W)，1 表示黑，-1 表示白，0 表示空
      - me: 1 表示黑 to move，-1 表示白 to move
    输出 torch.FloatTensor, shape (3, H, W)，第 0 通道始终是“to-move”一方的棋子
    """
    # 生成三个平面
    black = (board == 1).astype(np.float32)
    white = (board == -1).astype(np.float32)
    empty = (board == 0).astype(np.float32)
    # 按照当前执子方排列通道
    if me == 1:
        arr = np.stack([black, white, empty], axis=0)
    else:
        arr = np.stack([white, black, empty], axis=0)
    return torch.from_numpy(arr)


class ResidualBlock(nn.Module):
    """简单的 ResNet 残差块"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class YourModelClass(nn.Module):
    def __init__(self):
        super(YourModelClass, self).__init__()
        # 输入部分：3 通道
        self.conv_input = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # 3 个残差块
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )

        # policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU()
        )
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

        # value head
        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv_input(x)
        x = self.residual_blocks(x)

        # policy
        p = self.policy_head(x)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # value
        v = self.value_head(x)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v