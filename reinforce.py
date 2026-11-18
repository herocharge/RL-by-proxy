"""Reward-only autoregressive constructor trained with REINFORCE."""
import torch
from torch import nn
from torch.distributions import Categorical
from dsl import Program, SIZES


class Constructor(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.embedding = nn.Embedding(11, 32)  # 0=start; 1..6 predicates; 7..9 actions
        self.gru = nn.GRUCell(32, hidden)
        self.predicate_head = nn.Linear(hidden, 6)
        self.action_head = nn.Linear(hidden, 3)

    def sample(self):
        device = next(self.parameters()).device
        h = torch.zeros(1, self.gru.hidden_size, device=device)
        previous = torch.tensor([0], device=device)
        choices, logps, entropies = [], [], []
        for i, size in enumerate(SIZES):
            h = self.gru(self.embedding(previous), h)
            distribution = Categorical(logits=(self.predicate_head(h) if size == 6 else self.action_head(h)).squeeze(0))
            choice = distribution.sample()
            choices.append(int(choice.item()))
            logps.append(distribution.log_prob(choice))
            entropies.append(distribution.entropy())
            previous = torch.tensor([1 + choices[-1] if size == 6 else 7 + choices[-1]], device=device)
        return Program(tuple(choices)), torch.stack(logps).sum(), torch.stack(entropies).sum()


class Reinforce:
    def __init__(self, learning_rate=0.001, baseline_decay=0.95, entropy_weight=0.001):
        self.network = Constructor()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.baseline = None
        self.baseline_decay = baseline_decay
        self.entropy_weight = entropy_weight

    def sample_and_update(self, evaluator):
        program, logp, entropy = self.network.sample()
        result = evaluator(program)
        prior_baseline = 0.0 if self.baseline is None else self.baseline
        loss = -(result.reward - prior_baseline) * logp - self.entropy_weight * entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.baseline = (result.reward if self.baseline is None else
                         self.baseline_decay * self.baseline + (1 - self.baseline_decay) * result.reward)
        return program, result
