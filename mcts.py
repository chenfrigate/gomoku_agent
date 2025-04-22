# mcts.py
import math
import numpy as np
import copy
import torch

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        


class MCTS:
    def __init__(self, model, board_size=15, c_puct=1.0, n_simulations=100):
        self.model = model
        self.board_size = board_size
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.device = next(model.parameters()).device

    def search(self, root_state):
        root = MCTSNode(root_state)

        for _ in range(self.n_simulations):
            node = root
            state = copy.deepcopy(root_state)

            # Selection
            while node.children:
                action, node = self.select(node)
                y, x = divmod(action, self.board_size)
                state[y][x] = self.current_player(state)

            # Expansion
            if not self.is_terminal(state):
                self.expand(node, state)

            # Simulation (use value network)
            value = self.evaluate(state)

            # Backpropagation
            self.backpropagate(node, value)

        return self.get_policy(root)

    def select(self, node):
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in node.children.items():
            q = child.value / (child.visits + 1e-8)
            u = self.c_puct * math.sqrt(node.visits + 1) / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def expand(self, node, state):
        legal_moves = self.get_legal_moves(state)
        for move in legal_moves:
            node.children[move] = MCTSNode(copy.deepcopy(state), parent=node)

    def backpropagate(self, node, value):
        while node:
            node.visits += 1
            node.value += value
            node = node.parent
            value = -value

    def evaluate(self, state):
        input_state = self.encode_state(state)
        input_tensor = torch.FloatTensor(input_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, value = self.model(input_tensor)
        return value.item()

    def get_policy(self, root):
        visits = np.zeros(self.board_size * self.board_size)
        for action, child in root.children.items():
            visits[action] = child.visits
        policy = visits / (np.sum(visits) + 1e-8)
        return policy

    def encode_state(self, board):
        black = (board == 1).astype(np.float32)
        white = (board == -1).astype(np.float32)
        empty = (board == 0).astype(np.float32)
        return np.stack([black, white, empty], axis=0)

    def get_legal_moves(self, board):
        return [i for i, v in enumerate(board.flatten()) if v == 0]

    def current_player(self, board):
        blacks = np.sum(board == 1)
        whites = np.sum(board == -1)
        return 1 if blacks <= whites else -1

    def is_terminal(self, board):
        return np.all(board != 0)
