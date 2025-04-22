import numpy as np
from replay_buffer import ReplayBuffer

BOARD_SIZE = 15

def self_play_games():
    buffer = ReplayBuffer()

    num_games = 5  # 玩5局
    for _ in range(num_games):
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        player = 1
        history = []

        for move in range(BOARD_SIZE * BOARD_SIZE):
            # 随机选择一个合法动作
            empty = np.argwhere(board == 0)
            if len(empty) == 0:
                break
            action = empty[np.random.choice(len(empty))]
            y, x = action

            board[y][x] = player

            # 记录当前状态（复制），动作（one-hot），当前玩家
            state = encode_board(board, player)
            policy = np.zeros(BOARD_SIZE * BOARD_SIZE)
            policy[y * BOARD_SIZE + x] = 1.0  # 当前选的动作

            history.append((state, policy, player))

            if check_winner(board, x, y, player):
                winner = player
                break
            player = -player  # 切换玩家
        else:
            winner = 0  # 平局

        # 保存到buffer
        for state, policy, player in history:
            value = 1 if player == winner else -1 if winner != 0 else 0
            buffer.add(state, policy, value)

    # 保存数据
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
