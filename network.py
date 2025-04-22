import torch
import torch.nn as nn

BOARD_SIZE = 15  # 棋盘大小15x15

class YourModelClass(nn.Module):
    def __init__(self):
        super(YourModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 输入3通道（黑/白/空）
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Policy Head
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1)  # 输出2通道
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)  # 每个位置打分

        # Value Head
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1)  # 输出1通道
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, 64)
        self.value_fc2 = nn.Linear(64, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Policy Head
        p = self.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value Head
        v = self.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = self.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v