import pickle
import random

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []

    def add(self, state, policy, value):
        self.buffer.append((state, policy, value))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, policies, values = zip(*samples)
        return states, policies, values

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.buffer = pickle.load(f)
