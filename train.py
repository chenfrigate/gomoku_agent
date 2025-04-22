import torch
import torch.nn as nn
import torch.optim as optim
from replay_buffer import ReplayBuffer

def train_agent(model, epochs=10, batch_size=64, data_path='self_play_data.pkl'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    buffer = ReplayBuffer()
    buffer.load(data_path)
    states, policies, values = buffer.sample(len(buffer.buffer))  # 全量采样

    states = states.to(device)
    policies = policies.to(device)
    values = values.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()

        pred_policy, pred_value = model(states)
        policy_loss = policy_loss_fn(pred_policy, torch.argmax(policies, dim=1))
        value_loss = value_loss_fn(pred_value, values)

        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"[Epoch {epoch}] Policy Loss: {policy_loss.item():.4f} | Value Loss: {value_loss.item():.4f}")
    return policy_loss.item(), value_loss.item()
