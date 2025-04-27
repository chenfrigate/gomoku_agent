import argparse
import os
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="五子棋智能体项目")
    parser.add_argument('--mode', type=str, choices=['train', 'self_play', 'play'], required=True,
                        help="选择运行模式：train（训练智能体）/ self_play（自我对弈收集数据）/ play（与智能体对弈）")
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                        help="智能体模型路径 (play 模式需要)")
    parser.add_argument('--pretrained_path', type=str, default='models/pretrained_expert.pth',
                        help="（可选）基础专家模型权重路径，用于 train 模式初始化")
    parser.add_argument('--epochs', type=int, default=10,
                        help="train 模式下的训练轮数")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="train 模式下的小批量大小")
    parser.add_argument('--data_path', type=str, default='self_play_data.pkl',
                        help="train 模式下的自博弈数据路径")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.mode == 'train':
        print("📚 Train 模式：开始训练智能体…")
        from network import YourModelClass
        from train import train_agent

        # 1. 设备 & 模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = YourModelClass().to(device)

        # 2. 加载预训练专家模型（如果文件存在）
        if os.path.exists(args.pretrained_path):
            ckpt = torch.load(args.pretrained_path, map_location=device)
            # 如果保存的是完整 checkpoint
            state_dict = ckpt.get('model_state', ckpt)
            model.load_state_dict(state_dict)
            print(f"✅ Loaded pretrained expert model from {args.pretrained_path}")
        else:
            print(f"⚠️ Pretrained path not found: {args.pretrained_path}, using random init")

        # 3. 调用 train_agent
        train_agent(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            data_path=args.data_path,
            device=device
        )

    elif args.mode == 'self_play':
        print("🤖 Self-play 模式：开始自我对弈，生成训练数据…")
        from self_play import self_play_games
        self_play_games()

    elif args.mode == 'play':
        print("🎮 Play 模式：加载模型，与智能体对弈…")
        from gomoku_game import GomokuGameGUI
        from network import YourModelClass

        if not os.path.exists(args.model_path):
            print(f"❌ 模型文件 {args.model_path} 不存在，请先训练！")
            return

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = YourModelClass().to(device)

        # 加载 checkpoint
        ckpt = torch.load(args.model_path, map_location=device)
        state_dict = ckpt.get('model_state', ckpt)
        model.load_state_dict(state_dict)
        model.eval()

        game = GomokuGameGUI(model)
        game.run()

if __name__ == "__main__":
    main()