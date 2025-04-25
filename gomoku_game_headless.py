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

    def check_win(self, x, y):
        def count(dx, dy):
            cnt = 0
            i, j = x + dx, y + dy
            while 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE and self.board[i][j] == self.current_player:
                cnt += 1
                i += dx
                j += dy
            return cnt

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            if count(dx, dy) + count(-dx, -dy) + 1 >= WIN_CONDITION:
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
