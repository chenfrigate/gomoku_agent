# cycle_train.py
import os
import argparse
import torch
from config import (
    PRETRAINED_MODEL, GAMES_PER_CYCLE, EPOCHS_PER_CYCLE,
    MCTS_PLAYOUTS
)
from network import YourModelClass
from self_play import self_play_games
from train import train_agent



def load_model(path, device):
    model = YourModelClass().to(device)
    if path and os.path.exists(path):
        # 直接当成完整 checkpoint 处理，取出 'model_state' 字典
        ckpt = torch.load(path, map_location=device)
        state_dict = ckpt['model_state']
        model.load_state_dict(state_dict)  # 如有需要可加 strict=False
        print(f"[*] Loaded model weights from {path}")
    else:
        print(f"[!] No pretrained model found at {path}, initializing randomly.")
    return model

def cycle_training_loop(
    num_cycles,
    games_per_cycle,
    epochs_per_cycle,
    device,
    white_strategy='expert',  # 'expert', 'latest', 'none'
    winrate_threshold=0.6
):
    """
    主训练循环：
      - 黑方模型每轮自训练并更新；
      - 白方模型可基于策略 ('expert','latest','none') 初始，并在胜率阈值后动态切换。
    """
    expert_model = load_model(PRETRAINED_MODEL, device)

    import glob, re

    # 先尝试去 models/ 目录里找所有 RL 循环保存的 .pth
    all_rl = glob.glob("models/model_cycle*.pth")
    if all_rl:
        # 从文件名里提取 cycle 数字，按数字大小排序，取最新那一个
        def cycle_num(path):
            m = re.search(r"cycle(\d+)", path)
            return int(m.group(1)) if m else -1

        all_rl.sort(key=cycle_num)
        cur_model_path = all_rl[-1]
        print(f"[*] Found latest RL model: {cur_model_path}")
    else:
        # 如果找不到，就用专家预训练
        cur_model_path = PRETRAINED_MODEL if os.path.exists(PRETRAINED_MODEL) else None
        print(f"[!] No RL model found, fallback to pretrained: {cur_model_path}")

    # 初始化白方
    white_model = expert_model if white_strategy=='expert' else None

    for cycle in range(1, num_cycles+1):
        print(f"\n=== 🚀 Cycle {cycle}/{num_cycles} 开始 ===")
        black_model = load_model(cur_model_path, device)
        # 白方策略
        if white_strategy=='latest':
            white_model = black_model
        elif white_strategy=='none':
            white_model = None

        # 自对弈
        save_path = f"pkls/self_play_cycle{cycle}.pkl"
        self_play_games(
            model_black=black_model,
            model_white=white_model,
            num_games=games_per_cycle,
            save_path=save_path,
            cycle_index=cycle,
            num_simulations=MCTS_PLAYOUTS
        )
        # 训练
        print("🎯 开始训练黑方模型...")
        p_loss, v_loss = train_agent(
            model=black_model,
            epochs=epochs_per_cycle,
            data_path=save_path
        )
        # 保存
        next_model_path = f"models/model_cycle{cycle}.pth"
        os.makedirs(os.path.dirname(next_model_path), exist_ok=True)
        # torch.save(black_model.state_dict(), next_model_path)
        torch.save({'model_state': black_model.state_dict()}, next_model_path)
        print(f"💾 模型保存到：{next_model_path}")
        # 动态切换
        if white_strategy in ('expert','latest'):
            try:
                from evaluate import evaluate_models
                wins = evaluate_models(
                    model_a=black_model,
                    model_b=white_model,
                    num_games=10,
                    num_simulations=MCTS_PLAYOUTS
                )
                winrate = wins[0]/sum(wins) if sum(wins)>0 else 0
                print(f"[Eval] Black vs White winrate: {winrate:.2f}")
                if winrate > winrate_threshold:
                    print(f"[Swap] 胜率 {winrate:.2f} > {winrate_threshold}，切换白方模型")
                    white_model = black_model
            except ImportError:
                print("[Warning] 无 evaluate_models，跳过切换")
        cur_model_path = next_model_path
    print("=== 所有 Cycle 完成 ===")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycles', type=int, default=4, help='训练 Cycle 数')
    parser.add_argument('--games', type=int, default=GAMES_PER_CYCLE, help='每 Cycle 对局数')
    parser.add_argument('--epochs', type=int, default=EPOCHS_PER_CYCLE, help='每 Cycle 训练轮数')
    parser.add_argument('--strategy', choices=['expert','latest','none'], default='expert', help='白方模型策略')
    parser.add_argument('--threshold', type=float, default=0.6, help='胜率阈值，超过则切换白方模型')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cycle_training_loop(
        num_cycles=args.cycles,
        games_per_cycle=args.games,
        epochs_per_cycle=args.epochs,
        device=device,
        white_strategy=args.strategy,
        winrate_threshold=args.threshold
    )
