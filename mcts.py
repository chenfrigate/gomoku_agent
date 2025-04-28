import math
import numpy as np
import copy
import torch

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state   = state
        self.parent  = parent
        self.children= {}
        self.visits  = 0
        self.value   = 0.0
        self.prior   = 0.0  # network policy prior

class MCTS:
    def __init__(self, model, board_size=15, c_puct=1.0, n_simulations=1000):
        self.model         = model
        self.board_size    = board_size
        self.c_puct        = c_puct
        self.n_simulations = n_simulations
        # assume model.parameters() is non-empty
        self.device        = next(model.parameters()).device

    # def search(self, root_state):
    #     root = MCTSNode(root_state)

    #     for _ in range(self.n_simulations):
    #         node  = root
    #         state = copy.deepcopy(root_state)

    #         # Selection
    #         while node.children:
    #             action, node = self.select(node)
    #             y, x = divmod(action, self.board_size)
    #             state[y][x] = self.current_player(state)

    #         # Expansion
    #         if not self.is_terminal(state):
    #             self.expand(node, state)

    #         # Simulation (use value network)
    #         value = self.evaluate(state)

    #         # Backpropagation
    #         self.backpropagate(node, value)

    #     return self.get_policy(root)


    def search(self, root_state, batch_size=16):
        root = MCTSNode(root_state)
        sims_done = 0

        while sims_done < self.n_simulations:
            # 本批次要做多少模拟
            this_batch = min(batch_size, self.n_simulations - sims_done)
            leaves = []   # 本批次的 (node, state, path)

            # 1) Selection 阶段：本批次多次 descent
            for _ in range(this_batch):
                node  = root
                state = copy.deepcopy(root_state)
                path  = []

                # 沿树走到一个未 expand 的叶子
                while node.children:
                    action, node = self.select(node)
                    path.append(node)
                    y, x = divmod(action, self.board_size)
                    state[y][x] = self.current_player(state)

                leaves.append((node, state, path))

            # 2) Batch Inference
            batch_states = torch.stack([
                torch.from_numpy(self.encode_state(s)).to(self.device)
                for (_, s, _) in leaves
            ], dim=0)  # [B, C, H, W]
            with torch.no_grad():
                logits, values = self.model(batch_states)
                pis   = torch.softmax(logits, dim=-1).cpu().numpy()
                vals  = values.cpu().numpy()

            # 3) Expand + Backpropagate
            for (node, state, path), pi, v in zip(leaves, pis, vals):
                # 3.1 Expand
                if not self.is_terminal(state):
                    for move in self.get_legal_moves(state):
                        child = MCTSNode(copy.deepcopy(state), parent=node)
                        child.prior = max(pi[move], 1e-6)
                        node.children[move] = child

                # 3.2 Backpropagate：把 v 沿 path 向上累加
                #    注意要根据信号翻转（-v, +v, -v…）
                self.backpropagate(node, v)

            sims_done += this_batch

        # 最后收集根节点的访问次数分布
        return self.get_policy(root)


    def select(self, node):
        """
        PUCT selection: Q + U, where U = c_puct * P * sqrt(N_parent) / (1 + N_child)
        """
        best_score  = -float('inf')
        best_action = None
        best_child  = None

        for action, child in node.children.items():
            q = child.value / (child.visits + 1e-8)
            u = self.c_puct * child.prior * math.sqrt(node.visits) / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score  = score
                best_action = action
                best_child  = child
        return best_action, best_child

    # def expand(self, node, state):
    #     """
    #     Expand node with network policy priors.
    #     """
    #     # 1) network forward to get policy over moves
    #     input_state  = self.encode_state(state)
    #     input_tensor = torch.FloatTensor(input_state).unsqueeze(0).to(self.device)
    #     with torch.no_grad():
    #         policy, _ = self.model(input_tensor)
    #     policy = policy.cpu().numpy().flatten()

    #     # 2) create children with priors
    #     legal_moves = self.get_legal_moves(state)
    #     for move in legal_moves:
    #         child = MCTSNode(copy.deepcopy(state), parent=node)
    #         # assign a minimal prior if network gives zero
    #         child.prior = max(policy[move], 1e-6)
    #         node.children[move] = child



    def backpropagate(self, node, value):
        """
        Propagate evaluation up the tree, alternating sign.
        """
        while node:
            node.visits += 1
            node.value  += value
            node       = node.parent
            value      = -value

    def evaluate(self, state):
        """
        Use value head of network to evaluate state.
        """
        input_state  = self.encode_state(state)
        input_tensor = torch.FloatTensor(input_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, value = self.model(input_tensor)
        return value.item()

    def get_policy(self, root):
        """
        Return visit-count normalized policy from root children.
        """
        visits = np.zeros(self.board_size * self.board_size)
        for action, child in root.children.items():
            visits[action] = child.visits
        policy = visits / (np.sum(visits) + 1e-8)
        return policy

    def encode_state(self, board):
        """
        Convert board to 3 x H x W tensor: black, white, empty.
        """
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