# test_block_four.py
import argparse
import random
import numpy as np
import torch
from network import YourModelClass
from network import encode_pov_tensor
from mcts import MCTS

BOARD_SIZE = 15

def print_board(board):
    """用 ASCII 打印 0=.  1=X  -1=O"""
    symbols = {0:'.', 1:'X', -1:'O'}
    for row in board:
        print(' '.join(symbols[v] for v in row))
    print()


def make_block_four_board(H, W):
    """
    随机在 H×W 棋盘上构造一个对手的“活四”（4 子连成一线、两端各空一个点）：
    返回 board（二值矩阵，-1 表示对手棋子）、以及两个可封堵位置的一维索引列表
    """
    directions = [(0,1), (1,0), (1,1), (-1,1)]
    dr, dc = random.choice(directions)

    # 找到一个起点，使得从起点 dr,dc 方向连 4 子，且两端各留一个空
    while True:
        r0 = random.randint(1, H-2)
        c0 = random.randint(1, W-2)
        cells = [(r0 + k*dr, c0 + k*dc) for k in range(4)]
        end1 = (cells[0][0] - dr, cells[0][1] - dc)
        end2 = (cells[-1][0] + dr, cells[-1][1] + dc)
        all_pts = cells + [end1, end2]
        if all(0 <= x < H and 0 <= y < W for x, y in all_pts):
            break

    board = np.zeros((H, W), dtype=int)
    # 把活四放上去，用 -1 表示对手棋子
    for x, y in cells:
        board[x, y] = -1
    
    # 随机撒一些黑棋干扰
    num_blockers = 4
    empties = list(zip(*np.where(board == 0)))
    for _ in range(num_blockers):
        bx, by = random.choice(empties)
        board[bx, by] = 1
        empties.remove((bx, by))

    # 计算两端的平面索引
    block_positions = [end1[0] * W + end1[1], end2[0] * W + end2[1]]
    return board, block_positions


def test_block_four(net, device, H, W, n_tests):
    net.eval()
    correct = 0
    example = None  # 保存首个成功封堵样例

    for i in range(n_tests):
        board, block_positions = make_block_four_board(H, W)
        # 对局面进行 POV 编码，自己 me=1（黑棋）
        tensor = encode_pov_tensor(board, me=1).unsqueeze(0).to(device)
        #不用蒙特卡洛
        """
        with torch.no_grad():
            policy, _ = net(tensor)
        policy = policy.cpu().numpy().flatten()
        """
        #用蒙特卡洛
        #"""
        mcts = MCTS(net, board_size=BOARD_SIZE, n_simulations=100)  # 或者你想试的次数
        pi = mcts.search(board)       # 直接传 numpy board
        policy = pi.flatten()
        #"""

        legal = np.where(board.flatten() == 0)[0]
        policy_masked = policy[legal]
        move = legal[np.argmax(policy_masked)]

        if move in block_positions:
            correct += 1
            if example is None:
                example = (board.copy(), block_positions, move)

    # 打印总体准确率
    print(f"Block-four accuracy: {correct}/{n_tests} = {correct/n_tests*100:.1f}%")

    # 如果有例子，打印第一个
    if example is not None:
        board, blocks, move = example
        print("\n=== 示例：第一个正确封堵样例 ===")
        print_board(board)
        coords = [(pos // W, pos % W) for pos in blocks]
        print("合法的封堵位置（row, col）：", coords)
        print("网络实际选的落点（row, col）：", (move // W, move % W))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",   type=str,   required=True,
                   help="模型文件路径")
    p.add_argument("--n_tests", type=int,   default=500,
                   help="测试局数")
    p.add_argument("--board_h", type=int,   default=15,
                   help="棋盘高度")
    p.add_argument("--board_w", type=int,   default=15,
                   help="棋盘宽度")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = YourModelClass().to(device)
    ckpt = torch.load(args.model, map_location=device)
    state_dict = ckpt.get("model_state", ckpt)
    net.load_state_dict(state_dict)
    net.eval()

    test_block_four(net, device, args.board_h, args.board_w, args.n_tests)
