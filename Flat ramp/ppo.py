import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_std, device):
        super(ActorCritic, self).__init__()
        self.device = device
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(self.device)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)
        self.log_std = nn.Parameter(torch.log(torch.tensor([action_std]))).to(self.device)

    def forward(self, state):
        action_mean = self.actor(state)
        action_std = self.log_std.exp()
        state_value = self.critic(state)
        return action_mean, action_std, state_value

    def evaluate(self, states, actions):
        action_mean, action_std, state_values = self.forward(states)
        dist = Normal(action_mean, action_std)
        action_logprobs = dist.log_prob(actions).sum(axis=-1)
        dist_entropy = dist.entropy().sum(axis=-1)
        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self, state_dim, lr, betas, gamma, K_epochs, eps_clip, device, action_std=0.1):
        self.device = device
        self.policy = ActorCritic(state_dim, action_std, device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.policy_old = ActorCritic(state_dim, action_std, device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, memory, state):
        state = state.to(self.device)
        with torch.no_grad():
            action_mean, action_std, _ = self.policy_old(state)
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum()

        memory.states.append(state.cpu())
        memory.actions.append(action.cpu())
        memory.logprobs.append(action_logprob.cpu())

        return action.cpu().numpy()

    def update(self, memory):
        rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(self.device).unsqueeze(1)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.stack(memory.states).to(self.device)
        old_actions = torch.stack(memory.actions).to(self.device)
        old_logprobs = torch.stack(memory.logprobs).to(self.device)

        _, old_state_values, _ = self.policy.evaluate(old_states, old_actions)
        advantages = rewards - old_state_values.detach()

        cumulative_loss = 0
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.03 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            cumulative_loss += loss.mean().item()

        self.policy_old.load_state_dict(self.policy.state_dict())
        memory.clear_memory()

        return cumulative_loss / self.K_epochs


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()




