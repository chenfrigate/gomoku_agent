# config.py

# 棋盘尺寸
BOARD_H, BOARD_W = 15, 15

# 专家示例数量
N_EXPERT_EXAMPLES = 50000
#N_EXPERT_EXAMPLES = 500

# 训练超参
PRETRAIN_EPOCHS    = 120
PRETRAIN_LR        = 1e-3
PRETRAIN_BATCH     = 128

# 强化学习循环
GAMES_PER_CYCLE    = 500
EPOCHS_PER_CYCLE   = 2
MCTS_PLAYOUTS      = 800

# 路径
EXPERT_DATA_PATH   = "expert_data.pkl"
PRETRAINED_MODEL   = "models/pretrained_expert.pth"
