# gomoku_game.py
import tkinter as tk
from tkinter import messagebox
import numpy as np
import torch
from mcts import MCTS

BOARD_SIZE = 15
GRID_SIZE = 40
PADDING = 20

class GomokuGame:
    def __init__(self, model):
        self.window = tk.Tk()
        self.window.title("五子棋 - 智能体对弈")

        canvas_size = PADDING * 2 + GRID_SIZE * (BOARD_SIZE - 1)
        self.canvas = tk.Canvas(self.window, width=canvas_size, height=canvas_size, bg='burlywood')
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.human_move)
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.current_player = 1  # 1是玩家，-1是Agent

        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mcts = MCTS(self.model, board_size=BOARD_SIZE, n_simulations=50)

        self.draw_board()

    def draw_board(self):
        for i in range(BOARD_SIZE):
            self.canvas.create_line(PADDING, PADDING + i * GRID_SIZE,
                                    PADDING + (BOARD_SIZE - 1) * GRID_SIZE, PADDING + i * GRID_SIZE)
            self.canvas.create_line(PADDING + i * GRID_SIZE, PADDING,
                                    PADDING + i * GRID_SIZE, PADDING + (BOARD_SIZE - 1) * GRID_SIZE)

    def draw_piece(self, x, y, player):
        color = "black" if player == 1 else "white"
        self.canvas.create_oval(PADDING + x * GRID_SIZE - 15, PADDING + y * GRID_SIZE - 15,
                                PADDING + x * GRID_SIZE + 15, PADDING + y * GRID_SIZE + 15, fill=color)

    def human_move(self, event):
        x = round((event.x - PADDING) / GRID_SIZE)
        y = round((event.y - PADDING) / GRID_SIZE)
        if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and self.board[y][x] == 0:
            self.board[y][x] = 1
            self.draw_piece(x, y, 1)
            if self.check_winner(x, y, 1):
                messagebox.showinfo("游戏结束", "你赢了！")
                self.window.quit()
                return
            self.current_player = -1
            self.window.after(500, self.agent_move)

    def agent_move(self):
        x, y = self.get_agent_action()
        if x is not None:
            self.board[y][x] = -1
            self.draw_piece(x, y, -1)
            if self.check_winner(x, y, -1):
                messagebox.showinfo("游戏结束", "智能体赢了！")
                self.window.quit()
            self.current_player = 1

    def get_agent_action(self):
        policy = self.mcts.search(self.board)
        legal_moves = np.where(self.board.flatten() == 0)[0]
        policy_masked = policy[legal_moves]
        if policy_masked.sum() == 0:
            return None, None
        action = legal_moves[np.argmax(policy_masked)]
        y, x = divmod(action, BOARD_SIZE)
        return x, y

    def check_winner(self, x, y, player):
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

    def run(self):
        self.window.mainloop()