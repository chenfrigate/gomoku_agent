# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from network import YourModelClass
from replay_buffer import ReplayBuffer


def train_agent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = YourModelClass().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn_policy = nn.CrossEntropyLoss()
    loss_fn_value = nn.MSELoss()

    buffer = ReplayBuffer()
    buffer.load('self_play_data.pkl')

    batch_size = 64
    epochs = 10

    for epoch in range(epochs):
        model.train()

        states, target_policies, target_values = buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(device)
        target_policies = torch.FloatTensor(target_policies).to(device)
        target_values = torch.FloatTensor(target_values).unsqueeze(1).to(device)

        optimizer.zero_grad()

        pred_policies, pred_values = model(states)

        # 保障policy是logits，使用CrossEntropyLoss需要long label
        pred_policies_softmax = torch.softmax(pred_policies, dim=1)
        loss_p = -(target_policies * torch.log(pred_policies_softmax + 1e-8)).sum(dim=1).mean()
        loss_v = loss_fn_value(pred_values, target_values)
        loss = loss_p + loss_v

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} - Policy Loss: {loss_p.item():.4f} - Value Loss: {loss_v.item():.4f}")

    torch.save(model.state_dict(), "best_model.pth")
    print("\n训练完成，保存模型为 best_model.pth")