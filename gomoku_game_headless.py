import numpy as np

BOARD_SIZE = 15  # 可根据你的项目调整
WIN_CONDITION = 5

class GomokuGame:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = 1
        self.game_over = False
        self.winner = None

    def reset(self):
        self.board.fill(0)
        self.current_player = 1
        self.game_over = False
        self.winner = None

    def get_legal_actions(self):
        return [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if self.board[i][j] == 0]

    def play_move(self, move):
        if self.game_over:
            return False


        row, col = move
        if self.board[row][col] != 0:
            return False

        self.board[row][col] = self.current_player

        if self.check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
        elif len(self.get_legal_actions()) == 0:
            self.game_over = True  # 平局
            self.winner = 0
        else:
            self.current_player = -self.current_player  # 1 -> -1, -1 -> 1


        return True

    def check_win(self, row, col):
        player = self.board[row][col]
        directions = [(1,0), (0,1), (1,1), (1,-1)]  # (dr, dc)
        for dr, dc in directions:
            count = 1  # 当前落子点已计数
            #print(f"检查位置: ({row}, {col}), 玩家: {player}")
            for step in [1, -1]:  # 正反两个方向
                r, c = row, col
                while True:
                    r += step * dr
                    c += step * dc
                    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == player:
                        count += 1
                    else:
                        break
            #print(f"方向 {dr}, {dc}: 计数={count}")
            if count >= WIN_CONDITION:
                return True
        return False

    def get_current_player(self):
        return self.current_player

    def is_game_over(self):
        return self.game_over

    def get_winner(self):
        return self.winner

    def get_board_state(self):
        return self.board.copy()

    def clone(self):
        clone = GomokuGame()
        clone.board = self.board.copy()
        clone.current_player = self.current_player
        clone.game_over = self.game_over
        clone.winner = self.winner
        return clone

    def get_state_tensor(self, player):
        from network import encode_pov_tensor
        return encode_pov_tensor(self.board, player)

    def get_legal_moves(self, board):
        return list(np.where(board.flatten() == 0)[0])

    # 在 gomoku_game_headless.py 中修改 play 方法
    def play(self, action, player):
        size = self.board.shape[0]
        row, col = divmod(action, size)
        if self.board[row][col] != 0:
            return False  # 非法移动
        
        self.board[row][col] = player
        
        # 检查游戏是否结束
        if self.check_win(row, col):
            self.game_over = True
            self.winner = player
        elif len(self.get_legal_actions()) == 0:
            self.game_over = True
            self.winner = 0
        
        return True
