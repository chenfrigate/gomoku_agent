import argparse
import os
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="五子棋智能体项目")
    parser.add_argument('--mode', type=str, choices=['train', 'self_play', 'play'], required=True,
                        help="选择运行模式：train（训练智能体）/ self_play（自我对弈收集数据）/ play（与智能体对弈）")
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                        help="智能体模型路径 (play模式需要)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == 'train':
        print("开始训练智能体...")
        from train import train_agent
        train_agent()

    elif args.mode == 'self_play':
        print("开始自我对弈，生成训练数据...")
        from self_play import self_play_games
        self_play_games()

    elif args.mode == 'play':
        print("加载模型，与智能体对弈...")
        from gomoku_game import GomokuGame
        from network import YourModelClass

        if not os.path.exists(args.model_path):
            print(f"模型文件 {args.model_path} 不存在，请先训练！")
            return

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = YourModelClass().to(device)

        # 加载 checkpoint
        ckpt = torch.load(args.model_path, map_location=device)
        # 如果是包含完整训练状态的 dict，则提取 'model_state'
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            state_dict = ckpt['model_state']
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict)
        model.eval()

        game = GomokuGame(model)
        game.run()


if __name__ == "__main__":
    main()
