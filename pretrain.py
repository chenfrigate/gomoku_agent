# pretrain.py  在专家示例上做纯监督交叉熵预训练并支持从最终模型继续训练。
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from config import (
    PRETRAIN_EPOCHS, PRETRAIN_LR, PRETRAIN_BATCH,
    EXPERT_DATA_PATH, PRETRAINED_MODEL
)
from network import YourModelClass
from expert_dataset import ExpertDataset

"""
此脚本在 ExpertDataset 上做交叉熵预训练，并将最终模型及优化器状态保存在 PRETRAINED_MODEL。
再次运行时，如果 PRETRAINED_MODEL 存在，会从其 epoch 继续训练，继续优化最终模型权重。
"""

def pretrain():
    # Ensure save directory exists
    save_dir = os.path.dirname(PRETRAINED_MODEL)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Load dataset
    ds = ExpertDataset(EXPERT_DATA_PATH)
    loader = DataLoader(ds, batch_size=PRETRAIN_BATCH, shuffle=True, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and optimizer
    model = YourModelClass().to(device)
    optimizer = optim.Adam(model.parameters(), lr=PRETRAIN_LR)
    criterion = nn.CrossEntropyLoss()

    # Resume from pretrained model if exists
    start_epoch = 0
    if os.path.exists(PRETRAINED_MODEL):
        ckpt = torch.load(PRETRAINED_MODEL, map_location=device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt.get("optim_state", {}))
            start_epoch = ckpt.get("epoch", 0)
            print(f"[*] Resuming training from epoch {start_epoch}")
        else:
            # Legacy file: only state_dict
            model.load_state_dict(ckpt)
            print("[*] Loaded pretrained model weights, starting new training session.")

    model.train()
    for ep in range(start_epoch, PRETRAIN_EPOCHS):
        total_loss, total_samples = 0.0, 0
        for states, moves in loader:
            states, moves = states.to(device), moves.to(device)
            logits, _ = model(states)
            loss = criterion(logits, moves)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * states.size(0)
            total_samples += states.size(0)

        avg_loss = total_loss / total_samples
        print(f"Epoch {ep+1}/{PRETRAIN_EPOCHS}, loss={avg_loss:.4f}")

        # Save checkpoint to PRETRAINED_MODEL (contains optimizer state)
        torch.save({
            "epoch": ep+1,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict()
        }, PRETRAINED_MODEL)

    print(f"[+] Training complete. Final model and optimizer saved → {PRETRAINED_MODEL}")


if __name__ == "__main__":
    pretrain()
