import tkinter as tk
from tkinter import messagebox
import numpy as np
import torch
from mcts import MCTS
from network import encode_pov_tensor

BOARD_SIZE = 15
GRID_SIZE = 40
PADDING = 20

class GomokuGame:
    #def __init__(self, model):
    def __init__(self, model=None):
        self.debug_net_policy = False
        self.window = tk.Tk()
        self.window.title("五子棋 - 智能体对弈")

        canvas_size = PADDING * 2 + GRID_SIZE * (BOARD_SIZE - 1)
        self.canvas = tk.Canvas(self.window, width=canvas_size, height=canvas_size, bg='burlywood')
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.human_move)
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        # Agent 先手，扮演黑棋（1），玩家扮白棋（-1）
        self.current_player = -1

        # self.model = model
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.mcts = MCTS(self.model, board_size=BOARD_SIZE, n_simulations=1000)
        # 接收可选的 model；只有传进来才会初始化 MCTS
        self.model  = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model is not None:
            self.mcts = MCTS(self.model, board_size=BOARD_SIZE, n_simulations=1000)
        else:
            self.mcts = None


        self.draw_board()
        # 初始化后暂停 500ms，再让 Agent（黑）先下一子
        self.window.after(500, self.agent_move)

    def draw_board(self):
        for i in range(BOARD_SIZE):
            self.canvas.create_line(PADDING, PADDING + i * GRID_SIZE,
                                    PADDING + (BOARD_SIZE - 1) * GRID_SIZE, PADDING + i * GRID_SIZE)
            self.canvas.create_line(PADDING + i * GRID_SIZE, PADDING,
                                    PADDING + i * GRID_SIZE, PADDING + (BOARD_SIZE - 1) * GRID_SIZE)

    def draw_piece(self, x, y, player):
        color = "black" if player == 1 else "white"
        self.canvas.create_oval(
            PADDING + x * GRID_SIZE - 15, PADDING + y * GRID_SIZE - 15,
            PADDING + x * GRID_SIZE + 15, PADDING + y * GRID_SIZE + 15,
            fill=color
        )

    def human_move(self, event):
        x = round((event.x - PADDING) / GRID_SIZE)
        y = round((event.y - PADDING) / GRID_SIZE)
        if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and self.board[y][x] == 0:
            # 玩家扮白棋，用 -1 表示
            self.board[y][x] = -1
            self.draw_piece(x, y, -1)
            if self.check_winner(x, y, -1):
                messagebox.showinfo("游戏结束", "你赢了！")
                self.window.quit()
                return
            self.current_player = -1
            # 白棋落完，换黑棋（Agent）
            self.window.after(500, self.agent_move)

    def agent_move(self):
        x, y = self.get_agent_action()
        #print(f"i am trying put {x} and {y}")
        if x is not None:
            # Agent 扮演黑棋，用 1 表示
            self.board[y][x] = 1
            self.draw_piece(x, y, 1)
            if self.check_winner(x, y, 1):
                messagebox.showinfo("游戏结束", "智能体赢了！")
                self.window.quit()
            # 下完黑，换白棋玩家
            self.current_player = -1

    def get_agent_action(self):
        if getattr(self, 'debug_net_policy', False):
            # 1. 先把当前棋盘编码为模型输入 tensor
            #    假设 self.board 是 shape=(BOARD_SIZE, BOARD_SIZE) 的 numpy 数组，
            #    self.device + self.net 在 __init__ 里已经设置好了。
            tensor = encode_pov_tensor(self.board, me=1)      # (C, H, W)
            tensor = tensor.unsqueeze(0).to(self.device)      # (1, C, H, W)

            # 2. 纯前向，不计算梯度
            with torch.no_grad():
                policy, _ = self.model(tensor)                  # policy.shape == (1, BOARD_SIZE*BOARD_SIZE)
            policy = policy.cpu().numpy().flatten()           # (BOARD_SIZE*BOARD_SIZE,)

            # 3. 掩掉已经落子的位置，只在空位里选
            legal_moves = np.where(self.board.flatten() == 0)[0]  # 空点的 flat 索引列表
            policy_masked = policy[legal_moves]                  

            # 4. 如果网络在所有空位上都打了 0（极端情况），就随便下一个防止卡死
            if policy_masked.sum() == 0:
                choice = np.random.choice(legal_moves)
                y, x = divmod(choice, BOARD_SIZE)
                return x, y

            # 5. 否则选概率最高的位置
            action = legal_moves[np.argmax(policy_masked)]
            y, x = divmod(action, BOARD_SIZE)
            return x, y

        else:# —— 原本 MCTS 走子流程 —— 
            policy = self.mcts.search(self.board)
            #print(f"i get a policy,{policy} based on the board of {self.board}")
            legal_moves = np.where(self.board.flatten() == 0)[0]
            policy_masked = policy[legal_moves]
            if policy_masked.sum() == 0:
                # 随机挑一个合法点，下个棋以免卡死
                choice = np.random.choice(legal_moves)
                y, x = divmod(choice, BOARD_SIZE)
                #return None, None
                return x,y
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
