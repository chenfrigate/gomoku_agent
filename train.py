import torch
import torch.nn as nn
import torch.optim as optim
from network import YourModelClass
from replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_agent():
    model = YourModelClass().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn_policy = nn.CrossEntropyLoss()
    loss_fn_value = nn.MSELoss()

    buffer = ReplayBuffer()
    buffer.load('self_play_data.pkl')

    batch_size = 32
    epochs = 10

    for epoch in range(epochs):
        states, policies, values = buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(device)
        policies = torch.FloatTensor(policies).to(device)
        values = torch.FloatTensor(values).unsqueeze(1).to(device)

        optimizer.zero_grad()
        pred_policy, pred_value = model(states)

        loss_p = loss_fn_policy(pred_policy, policies)
        loss_v = loss_fn_value(pred_value, values)
        loss = loss_p + loss_v

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "best_model.pth")
    print("训练完成，保存模型为 best_model.pth")