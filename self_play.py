# self_play.py
import numpy as np
import torch
from replay_buffer import ReplayBuffer
from mcts import MCTS
from network import YourModelClass

BOARD_SIZE = 15

def self_play_games():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YourModelClass().to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()

    mcts = MCTS(model, board_size=BOARD_SIZE, n_simulations=50)
    buffer = ReplayBuffer()

    num_games = 5  # 玩5局
    for _ in range(num_games):
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        history = []
        player = 1

        for move in range(BOARD_SIZE * BOARD_SIZE):
            policy = mcts.search(board)
            action = np.random.choice(len(policy), p=policy)
            y, x = divmod(action, BOARD_SIZE)

            if board[y][x] != 0:
                continue  # 确保合法

            board[y][x] = player
            state = encode_board(board, player)
            history.append((state, policy, player))

            if check_winner(board, x, y, player):
                winner = player
                break
            player = -player
        else:
            winner = 0  # 平局

        # 把历史数据放入buffer
        for state, policy, p in history:
            value = 1 if p == winner else -1 if winner != 0 else 0
            buffer.add(state, policy, value)

    buffer.save('self_play_data.pkl')
    print(f"自我对弈完成，保存数据，大小: {len(buffer.buffer)}")

def encode_board(board, current_player):
    black = (board == 1).astype(np.float32)
    white = (board == -1).astype(np.float32)
    empty = (board == 0).astype(np.float32)
    if current_player == 1:
        return np.stack([black, white, empty], axis=0)
    else:
        return np.stack([white, black, empty], axis=0)

def check_winner(board, x, y, player):
    directions = [(1,0), (0,1), (1,1), (1,-1)]
    for dx, dy in directions:
        count = 1
        for d in [1, -1]:
            nx, ny = x, y
            while True:
                nx += d * dx
                ny += d * dy
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[ny][nx] == player:
                    count += 1
                else:
                    break
        if count >= 5:
            return True
    return False
