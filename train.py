import torch
import torch.nn as nn
import torch.optim as optim
from replay_buffer import ReplayBuffer
from torch.nn.utils import clip_grad_norm_



def train_agent(
    model,
    epochs: int = 10,
    batch_size: int = 64,
    data_path: str = 'self_play_data.pkl',
):
    """
    训练智能体网络：
      - model:       YourModelClass() 实例
      - epochs:      训练轮数
      - batch_size:  每轮小批量大小
      - data_path:   replay buffer 保存的 pkl 路径
      - device:      torch.device('cuda') or torch.device('cpu')
    """
    # 1. 准备设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # 2. 加载数据
    buffer = ReplayBuffer()
    buffer.load(data_path)

    # 🔥 调试打印 buffer 内容
    # buffer_list = buffer.buffer
    # print(f"✅ Buffer 加载成功，长度：{len(buffer)}, buffer 类型：{type(buffer_list)}, 前3元素类型：{[type(buffer_list[i]) for i in range(min(3, len(buffer_list)))]}")

    dataset_size = len(buffer)
    if dataset_size == 0:
        raise RuntimeError(f"ReplayBuffer 为空：{data_path}")

    # 3. 优化器 & 损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    policy_loss_fn = nn.KLDivLoss(reduction='batchmean')
    value_loss_fn = nn.MSELoss()

    # 4. 开始训练
    for epoch in range(1, epochs + 1):
        model.train()
        total_p_loss = 0.0
        total_v_loss = 0.0

        # 每个 epoch 随机分 batch 训练
        # num_batches = max(1, dataset_size // batch_size)
        num_batches = (dataset_size + batch_size - 1) // batch_size
        for _ in range(num_batches):
            # 从 buffer 中随机采样一个小批次
            states, policies, values = buffer.sample(batch_size)
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)

            # 前向
            pred_p, pred_v = model(states)
            log_p = nn.LogSoftmax(dim=1)(pred_p)


            # 计算策略 loss 和价值 loss
            p_loss = policy_loss_fn(log_p, policies)
            v_loss = value_loss_fn(pred_v, values)
            loss = p_loss + v_loss

            # 反向 + 更新
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪：将所有参数的梯度范数剪到 1.0
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_p_loss += p_loss.item()
            total_v_loss += v_loss.item()

        avg_p = total_p_loss / num_batches
        avg_v = total_v_loss / num_batches
        # 更新学习率
        scheduler.step()
        print(f"[Epoch {epoch:02d}/{epochs}] Policy Loss: {avg_p:.4f} | Value Loss: {avg_v:.4f}")

    return avg_p, avg_v


if __name__ == "__main__":
    import argparse
    from network import YourModelClass  # 替换为实际模型类

    parser = argparse.ArgumentParser(description="Train the Gomoku agent.")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_path', type=str, default='self_play_data.pkl')
    parser.add_argument('--model_path', type=str, default=None)
    args = parser.parse_args()

    model = YourModelClass()

    if args.model_path:
        checkpoint = torch.load(args.model_path)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            print(f"✅ 从 checkpoint 加载模型参数: {args.model_path}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ 从权重文件加载模型: {args.model_path}")

    # 调用训练
    train_agent(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_path=args.data_path
    )