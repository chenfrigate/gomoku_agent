import os
import pickle
import random
import numpy as np

from mcts import MCTS
from gomoku_game_headless import GomokuGame
from replay_buffer import ReplayBuffer
from network import encode_pov_tensor

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
        board = game.get_board_state()  # 使用 get_board_state 替代 get_board
        size = board.shape[0]

        mcts_black = MCTS(model_black, board_size=size, n_simulations=num_simulations) if model_black else None
        mcts_white = MCTS(model_white, board_size=size, n_simulations=num_simulations) if model_white else None

        state_hist = []
        pi_hist = []
        player_hist = []
        current = 1  # 黑方 1，白方 -1

        while not game.is_game_over():  # 使用 is_game_over 替代 is_over
            # 获取当前策略分布
            if current == 1:
                policy = mcts_black.search(board) if mcts_black else _uniform_policy(game)
            else:
                policy = mcts_white.search(board) if mcts_white else _uniform_policy(game)

            # 掩码非法动作并归一化
            policy = _mask_and_normalize(policy, game)

            # 按概率采样动作
            action = _sample_action(policy, game)

            # 记录数据
            state = game.get_state_tensor(current)
            state_hist.append(state)
            pi_hist.append(policy.tolist())
            player_hist.append(current)

            # 执行动作并切换玩家
            game.play(action, current)
            board = game.get_board_state()  # 更新棋盘状态
            current *= -1

        # 保存对局数据
        winner = game.get_winner()
        for st, pi_vec, pl in zip(state_hist, pi_hist, player_hist):
            value = winner * pl  # 胜者视角的价值（1 赢，-1 输，0 平）
            buffer.add(st, pi_vec, value)

        print(f"[SelfPlay] Cycle {cycle_index} game {idx}/{num_games} finished. Winner: {winner}")

    # 保存数据
    with open(save_path, 'wb') as f:
        pickle.dump(buffer, f)
    print(f"[SelfPlay] Saved {num_games} games to {save_path}")

def _uniform_policy(game):
    """生成均匀分布的合法动作概率"""
    legal = game.get_legal_moves(game.get_board_state())
    policy = np.zeros(game.board.size, dtype=np.float32)
    policy[legal] = 1.0 / len(legal)
    return policy

def _mask_and_normalize(policy, game):
    """掩码非法动作并归一化概率"""
    legal = game.get_legal_moves(game.get_board_state())
    mask = np.zeros_like(policy, dtype=bool)
    mask[legal] = True
    policy[~mask] = 0  # 非法位置置零
    if policy.sum() > 0:
        policy /= policy.sum()  # 归一化
    else:
        policy[legal] = 1.0 / len(legal)  # 避免除零（理论上不会触发）
    return policy

def _sample_action(policy, game):
    """根据概率分布采样动作"""
    legal = game.get_legal_moves(game.get_board_state())
    if policy.sum() <= 0:
        return random.choice(legal)  # 概率全零时随机选择
    return np.random.choice(len(policy), p=policy)

if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="Run self-play games to generate training data.")
    parser.add_argument('num_games', type=int, help='Number of games to play')
    parser.add_argument('model_path', type=str, nargs='?', default=None, help='Path to pretrained model')
    parser.add_argument('save_path', type=str, nargs='?', default='self_play_data.pkl', help='Output pickle file')
    parser.add_argument('--simulations', type=int, default=200, help='MCTS simulations per move')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = None
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location=device)
        from network import YourModelClass
        model = YourModelClass()
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)

    self_play_games(
        model_black=model,
        model_white=None,
        num_games=args.num_games,
        save_path=args.save_path,
        cycle_index=None,
        num_simulations=args.simulations,
        buffer_size=10000
    )