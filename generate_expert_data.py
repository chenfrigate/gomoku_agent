## 4. generate_expert_data.py 负责随机中局采样 + expert_policy，保存 `(state_tensor, move_idx)` 的列表。

# generate_expert_data.py
import random
import numpy as np
import torch
from config    import N_EXPERT_EXAMPLES, BOARD_H, BOARD_W, EXPERT_DATA_PATH
from expert_dataset import board_to_tensor, expert_policy

# —— 在这里插入我们的新函数 —— 
def gen_block_four_examples(n_block):
    examples = []
    H, W = BOARD_H, BOARD_W
    directions = [(0,1), (1,0), (1,1), (-1,1)]
    for _ in range(n_block):
        board = np.zeros((H, W), dtype=int)
        dr, dc = random.choice(directions)
        # 随机起点
        if dr == 1:    r0 = random.randint(0, H-5)
        elif dr == -1: r0 = random.randint(4, H-1)
        else:          r0 = random.randint(0, H-1)
        if dc == 1:    c0 = random.randint(0, W-5)
        elif dc == -1: c0 = random.randint(4, W-1)
        else:          c0 = random.randint(0, W-1)

        coords = [(r0 + i*dr, c0 + i*dc) for i in range(5)]
        empty_idx = random.randrange(5)
        for i,(r,c) in enumerate(coords):
            if i != empty_idx:
                board[r,c] = -1     # 对手棋子

        # 4) 额外撒 4 颗己方黑棋干扰
        empties = [(r, c) for r in range(H) for c in range(W) if board[r, c] == 0]
        random.shuffle(empties)
        for r, c in empties[:4]:
            board[r, c] = 1

        state = board_to_tensor(board)
        rr, cc = coords[empty_idx]
        move_idx = rr * W + cc
        examples.append((state, move_idx))
    return examples
# —— 新函数结束 ——



def sample_expert_examples(n):
    examples = []
    for _ in range(n):
        board = np.zeros((BOARD_H, BOARD_W), dtype=int)
        # 随机下 20–50 步
        for m in range(random.randint(20,50)):
            empties = list(zip(*np.where(board==0)))
            r,c = random.choice(empties)
            board[r,c] = 1 if m%2==0 else -1
        # expert move
        r,c = expert_policy(board, me=1)
        state = board_to_tensor(board)        
        move_idx = r*BOARD_W + c
        examples.append((state, move_idx))
        if (_+1) % 1000 == 0 or _ == n-1:
            print(f"[{_+1}/{n}] random examples generated")
    return examples        
    #torch.save(examples, EXPERT_DATA_PATH)
    #print(f"[+] Saved {len(examples)} examples → {EXPERT_DATA_PATH}")

if __name__ == "__main__":
    #sample_expert_examples(N_EXPERT_EXAMPLES)
    #1 先生成原始的 50k 随机中局样本
    base_examples = sample_expert_examples(N_EXPERT_EXAMPLES)

    # 1. 加载原始的 50k 样本
    #base_examples = torch.load(EXPERT_DATA_PATH)
    #print(f"[+] Loaded {len(base_examples)} base examples")

    # 2) 再生成专门的堵四样本，比如再加 10000 条
    block_examples = gen_block_four_examples(10000)
    print(f"[+] Generated {len(block_examples)} block-four examples")

    # 3) 合并
    base_examples.extend(block_examples)
    all_examples = base_examples
    random.shuffle(all_examples) #打乱合并后的列表

    # 4) 保存
    torch.save(all_examples, EXPERT_DATA_PATH)
    print(f"[+] Saved total {len(all_examples)} examples → {EXPERT_DATA_PATH}")
