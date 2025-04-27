import tkinter as tk
from tkinter import messagebox
import numpy as np
import random
import torch
from mcts import MCTS
from gomoku_game_headless import GomokuGame as HeadlessGame  # 引入无头逻辑

BOARD_SIZE = 15
GRID_SIZE = 40
PADDING = 20

class GomokuGameGUI:
    def __init__(self, model=None,n_simulations=1000):
        # 初始化无头游戏实例
        self.game = HeadlessGame()
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model is not None:
            self.mcts = MCTS(self.model, board_size=BOARD_SIZE, n_simulations=n_simulations)
        else:
            self.mcts = None

        # GUI 界面
        self.window = tk.Tk()
        self.window.title("五子棋 - 智能体对弈")
        canvas_size = PADDING*2 + GRID_SIZE*(BOARD_SIZE - 1)
        self.canvas = tk.Canvas(self.window, width=canvas_size, height=canvas_size, bg='burlywood')
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.human_move)

        self.draw_board()
        # Agent 先手
        self.window.after(500, self.agent_move)

    def draw_board(self):
        for i in range(BOARD_SIZE):
            x0, y0 = PADDING, PADDING + i * GRID_SIZE
            x1, y1 = PADDING + (BOARD_SIZE-1)*GRID_SIZE, y0
            self.canvas.create_line(x0, y0, x1, y1)
            x0, y0 = PADDING + i * GRID_SIZE, PADDING
            x1, y1 = x0, PADDING + (BOARD_SIZE-1)*GRID_SIZE
            self.canvas.create_line(x0, y0, x1, y1)

    def draw_piece(self, x, y, player):
        color = 'black' if player == 1 else 'white'
        cx = PADDING + x*GRID_SIZE
        cy = PADDING + y*GRID_SIZE
        r = GRID_SIZE//2 - 5
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill=color)

    def human_move(self, event):
        col = int((event.x - PADDING + GRID_SIZE/2) // GRID_SIZE)
        row = int((event.y - PADDING + GRID_SIZE/2) // GRID_SIZE)
        # 调用无头逻辑验证并执行
        if 0 <= col < BOARD_SIZE and 0 <= row < BOARD_SIZE:
            ok = self.game.play_move((row, col))  # headless 接口使用 (row,col)
            if not ok:
                return
            self.draw_piece(col, row, -1)
            #self.draw_piece(row,col)
            # if self.game.check_win(x, y,-1):
            # 延迟100ms后检查输赢，确保棋子已渲染
            self.window.after(100, lambda: self.check_winner(row, col, -1))
            self.window.after(500, self.agent_move)

    def agent_move(self):
        # 1) 先拿到所有合法动作 和 扁平索引
        legal_moves = self.game.get_legal_actions()            # [(r,c), ...]
        legal = [r * BOARD_SIZE + c for r, c in legal_moves]   # [flat_idx, ...]
        board = self.game.get_board_state()

        # 2) 如果有模型，用 MCTS 搜索并对输出做掩码；否则给均匀分布
        if self.mcts:
            raw_policy = self.mcts.search(board)                # 网络输出的原始概率向量
            policy = np.zeros_like(raw_policy)
            policy[legal] = raw_policy[legal]                   # 只保留合法落子概率
        else:
            policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
            policy[legal] = 1.0 / len(legal)


        # 掩码并采样
        # legal 现在是扁平索引列表
        mask = np.zeros_like(policy, dtype=bool)
        mask[legal] = True
        probs = policy * mask
        if probs.sum() == 0:
            choice = random.choice(legal)
        else:
            probs = probs / probs.sum()
            choice = np.random.choice(len(probs), p=probs)

        # 将 flat index 转为行列
        row, col = divmod(choice, BOARD_SIZE)
        # 执行落子
        self.game.play_move((row, col))
        self.draw_piece(col, row, 1)
        print(f"智能体落子: ({row}, {col})")
        # 检查胜负
        # 延迟100ms后检查输赢
        self.window.after(100, lambda: self.check_winner(row, col, 1))


    def check_winner(self, row, col, player):
        if self.game.check_win(row, col):
            winner = "你" if player == -1 else "智能体"
            messagebox.showinfo("游戏结束", f"{winner}赢了！")
            self.window.quit()

    def run(self):
        self.window.mainloop()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--simulations', type=int, default=1000)
    args = parser.parse_args()

    # 自动检测 CUDA/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    gui = GomokuGameGUI(model, n_simulations=args.simulations)
    gui.run()
