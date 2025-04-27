import random
import pickle
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []

    def add(self, state, policy, value):
        self.buffer.append((state, policy, value))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        samples = random.sample(list(self.buffer), batch_size)
        states, policies, values = zip(*samples)

        states = torch.tensor(np.stack(states), dtype=torch.float32)  # (batch_size, 1, 15, 15)
        # policies = torch.tensor(np.array(policies), dtype=torch.float32)  # (batch_size, action_size)
        # ===== 改这里 =====
        try:
            policies_arr = np.stack(policies)
        except ValueError:
            # 打印所有 policy 的 shape，帮助你找出哪条数据格式不对
            for idx, p in enumerate(policies):
                print(f"[Bad policy #{idx}] type={type(p)}, shape={np.shape(p)}")
            raise
        policies = torch.tensor(policies_arr, dtype=torch.float32)
        # ==================


        values = torch.tensor(np.array(values), dtype=torch.float32).unsqueeze(1)  # (batch_size, 1)

        return states, policies, values

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            # 如果它本身就是一个 ReplayBuffer 对象，那就把它的 buffer 取出来
            if isinstance(data, ReplayBuffer):
                self.buffer = data.buffer
            # 否则，假设它就是个列表，直接赋值
            else:
                self.buffer = data
    def __len__(self):
        return len(self.buffer)