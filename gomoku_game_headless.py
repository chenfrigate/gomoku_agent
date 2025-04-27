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

        x, y = move
        if self.board[x][y] != 0:
            return False

        self.board[x][y] = self.current_player

        if self.check_win(x, y):
            self.game_over = True
            self.winner = self.current_player
        elif len(self.get_legal_actions()) == 0:
            self.game_over = True  # 平局
            self.winner = 0
        else:
            self.current_player = 3 - self.current_player  # 1 -> 2, 2 -> 1

        return True

    def check_win(self, x, y, player):
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        for dx, dy in directions:
            count = 1
            for d in [1, -1]:
                nx, ny = x, y
                while True:
                    nx += d * dx
                    ny += d * dy
                    if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[ny][nx] == player:
                        count += 1
                    else:
                        break
            if count >= 5:
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


    # 下面这几行就直接复制粘贴进类里，替换掉打补丁
    def get_board(self):
        return self.board

    def get_state_tensor(self, player):
        from network import encode_pov_tensor
        return encode_pov_tensor(self.board, player)

    def get_legal_moves(self, board):
        return list(np.where(board.flatten() == 0)[0])

    def play(self, action, player):
        size = self.board.shape[0]
        y, x = divmod(action, size)
        self.board[y][x] = player

    def get_winner(self):
        H, W = self.board.shape
        for y in range(H):
            for x in range(W):
                p = self.board[y][x]
                if p != 0 and self.check_win(x, y, p):
                    return p
        return 0

    def is_over(self):
        return (self.get_winner() != 0) or np.all(self.board.flatten() != 0)
