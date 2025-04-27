import os
import pickle
import random
import numpy as np

from mcts import MCTS
from gomoku_game import GomokuGame
from replay_buffer import ReplayBuffer
from network import encode_pov_tensor

# —— 动态给 GomokuGame 添加无头自对弈所需接口 ——

def get_board(self):
    return self.board
GomokuGame.get_board = get_board

def get_state_tensor(self, player):
    return encode_pov_tensor(self.board, player)
GomokuGame.get_state_tensor = get_state_tensor

def get_legal_moves(self, board):
    return list(np.where(board.flatten() == 0)[0])
GomokuGame.get_legal_moves = get_legal_moves

def play(self, action, player):
    size = self.board.shape[0]
    y, x = divmod(action, size)
    self.board[y][x] = player
GomokuGame.play = play

def get_winner(self):
    H, W = self.board.shape
    for y in range(H):
        for x in range(W):
            p = self.board[y][x]
            if p != 0 and self.check_winner(x, y, p):
                return p
    return 0
GomokuGame.get_winner = get_winner

def is_over(self):
    return (self.get_winner() != 0) or np.all(self.board.flatten() != 0)
GomokuGame.is_over = is_over


def self_play_games(
    model_black=None,
    model_white=None,
    num_games=100,
    save_path="self_play_data.pkl",
    cycle_index=None,
    num_simulations=200,
    buffer_size=10000
):
    """
    进行自对弈，支持两模型对抗或随机走子。

    Args:
        model_black:      黑方模型，None 时随机走子
        model_white:      白方模型，None 时退到 model_black 或随机
        num_games:        对局数
        save_path:        保存路径（.pkl）
        cycle_index:      当前 Cycle 索引，仅用于打印
        num_simulations:  每步 MCTS 模拟次数
        buffer_size:      ReplayBuffer 容量
    """
    if model_white is None:
        model_white = model_black

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    buffer = ReplayBuffer(capacity=buffer_size)

    for idx in range(1, num_games + 1):
        game = GomokuGame()
        board = game.get_board()
        size  = board.shape[0]

        mcts_black = MCTS(model_black, board_size=size, n_simulations=num_simulations) if model_black else None
        mcts_white = MCTS(model_white, board_size=size, n_simulations=num_simulations) if model_white else None

        state_hist  = []
        pi_hist     = []
        player_hist = []
        current     = 1  # 黑方 +1，白方 -1

        while not game.is_over():
            # 获取当前策略分布
            if current == 1:
                if mcts_black:
                    policy = mcts_black.search(game.get_board())
                else:
                    legal = game.get_legal_moves(game.get_board())
                    policy = np.zeros(size*size, dtype=np.float32)
                    for a in legal:
                        policy[a] = 1/len(legal)
            else:
                if mcts_white:
                    policy = mcts_white.search(game.get_board())
                else:
                    legal = game.get_legal_moves(game.get_board())
                    policy = np.zeros(size*size, dtype=np.float32)
                    for a in legal:
                        policy[a] = 1/len(legal)

            # 对完整分布归一化，并屏蔽非法走子
            policy = policy / (policy.sum() + 1e-8)
            legal = game.get_legal_moves(game.get_board())
            mask = np.ones_like(policy, dtype=bool)
            mask[legal] = False
            policy[mask] = 0
            policy = policy / (policy.sum() + 1e-8)

            # 按概率采样动作
            if policy.sum() <= 0:
                action = random.choice(legal)
            else:
                action = np.random.choice(len(policy), p=policy)

            # 保存完整向量
            pi = policy.tolist()

            # 记录状态与策略
            state = game.get_state_tensor(current)
            state_hist.append(state)
            pi_hist.append(pi)
            player_hist.append(current)

            game.play(action, current)
            current *= -1

        winner = game.get_winner()
        # 将每一步 (state, pi, value) 插入 buffer
        for st, pi_vec, pl in zip(state_hist, pi_hist, player_hist):
            value = winner * pl
            buffer.add(st, pi_vec, value)

        if idx % 1 == 0 or idx == num_games:
            print(f"[SelfPlay] Cycle {cycle_index} game {idx}/{num_games} finished. Winner: {winner}")

    # 保存 ReplayBuffer 对象
    with open(save_path, 'wb') as f:
        pickle.dump(buffer, f)
    print(f"[SelfPlay] Saved {num_games} games to {save_path}")


if __name__ == "__main__":
    # 示例调用：python self_play.py 100 models/pretrained.pth self_play_data.pkl
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="Run self-play games to generate training data.")
    parser.add_argument('num_games', type=int, help='Number of games to play')
    parser.add_argument('model_path', type=str, nargs='?', default=None, help='Path to pretrained model')
    parser.add_argument('save_path', type=str, nargs='?', default='self_play_data.pkl', help='Output pickle file')
    parser.add_argument('--simulations', type=int, default=200, help='MCTS simulations per move')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu','cuda'], help='Device for model')
    args = parser.parse_args()

    # 加载模型
    model = None
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location=args.device)
        from network import YourModelClass
        model = YourModelClass()
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
        model.to(args.device)

    self_play_games(
        model_black=model,
        model_white=None,
        num_games=args.num_games,
        save_path=args.save_path,
        cycle_index=None,
        num_simulations=args.simulations,
        buffer_size=10000
    )
