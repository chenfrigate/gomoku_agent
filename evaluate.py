# evaluate.py

import argparse
import torch
import numpy as np
from mcts import MCTS
from gomoku_game import GomokuGame
from network import YourNet

def play_one_game(model_black, model_white, game, num_simulations):
    """
    用两路模型和 MCTS 模拟一局对弈，返回赢家（1 表示黑胜，-1 表示白胜）。
    """
    # 为黑白各自构造 MCTS
    mcts_black = MCTS(game, model_black, num_simulations)
    mcts_white = MCTS(game, model_white, num_simulations)

    board = game.get_init_board()
    current = 1  # 1 黑，-1 白

    while not game.is_over(board):
        if current == 1:
            pi, action = mcts_black.run(board)
        else:
            pi, action = mcts_white.run(board)
        board = game.apply_move(board, action, current)
        current *= -1

    # 游戏结束，判断赢家
    winner = game.get_winner(board)
    return winner

def evaluate_models(model_a, model_b, num_games=20, num_simulations=200, device=None):
    """
    对 model_a（先手） vs model_b（后手）进行若干局对弈，返回 (wins_a, wins_b)。
    """
    game = GomokuGame()
    wins = [0, 0]

    # 如果有 GPU，先把模型移动过去
    if device:
        model_a.to(device)
        model_b.to(device)

    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        for i in range(num_games):
            winner = play_one_game(model_a, model_b, game, num_simulations)
            if winner == 1:
                wins[0] += 1
            elif winner == -1:
                wins[1] += 1
            else:
                # 平局按各半记
                wins[0] += 0.5
                wins[1] += 0.5
    return wins

def load_model(path, device):
    model = YourNet().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", type=str, required=True, help="先手模型 checkpoint")
    parser.add_argument("--model_b", type=str, required=True, help="后手模型 checkpoint")
    parser.add_argument("--games",   type=int,   default=20,   help="对局数")
    parser.add_argument("--sims",    type=int,   default=200,  help="每步 MCTS 模拟次数")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_a = load_model(args.model_a, device)
    model_b = load_model(args.model_b, device)

    wins = evaluate_models(
        model_a, model_b,
        num_games=args.games,
        num_simulations=args.sims,
        device=device
    )
    print(f"Model A(先手) wins: {wins[0]}, Model B(后手) wins: {wins[1]}")
