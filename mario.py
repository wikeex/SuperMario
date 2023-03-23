import random
from collections import deque

import numpy as np
import torch

import environment
import master_buffer
from net import ActorNet, CriticNet
from utils import device


class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = device()

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.critic = CriticNet(self.state_dim, self.action_dim).float()
        self.actor = ActorNet(self.state_dim, self.action_dim).float()
        self.critic = self.critic.to(device=self.device)
        self.actor = self.actor.to(device=self.device)
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.05
        self.curr_step = 0

        self.save_every = 5e5  # no. of experiences between saving Mario Net
        self.memory = deque(maxlen=40000)
        self.batch_size = 32

        # load master buffer memory
        # TODO: 文件路径改成配置
        self.master_memory = master_buffer.load('/Users/admin/PycharmProjects/SuperMario/master_buffer_files')

        self.gamma = 0.9

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.00025)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

        self.tau = 0.02

        self.loss_sum = 0

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_values = 1 - np.random.random((1, self.action_dim))
            action_values = torch.tensor(action_values, dtype=torch.float32).to(device=self.device)

        # EXPLOIT
        else:
            state = state.__array__()
            state = torch.tensor(state).to(device=self.device)
            state = state.unsqueeze(0)
            action_values = self.actor(state, model="online").detach()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_values

    def cache(self, state, next_state, action, reward, done, loss):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = state.__array__()
        next_state = next_state.__array__()

        state = torch.tensor(state).to(device=self.device)
        next_state = torch.tensor(next_state).to(device=self.device)
        reward = torch.tensor([reward]).to(device=self.device)
        done = torch.tensor([done]).to(device=self.device)

        self.memory.append((state, next_state, action, reward, done,))
        # if self.curr_step < self.burnin:
        #     self.memory.append((state, next_state, action, reward, done,))
        # else:
        #     self.loss_sum += loss
        #     if loss > self.loss_sum / self.curr_step:
        #         self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        if random.randint(0, 9) in range(0, 10):
            batch = random.sample(self.memory, self.batch_size)
        else:
            batch = random.sample(self.master_memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    @torch.no_grad()
    def td_target(self, reward, next_state):
        next_action = self.actor(next_state, model="target")
        next_q = self.critic(next_state, next_action, model="target")

        return (reward.unsqueeze(-1) + self.gamma * next_q).float()

    @staticmethod
    def sync_target(net, tau):
        for target_param, online_param in zip(net.target.parameters(), net.online.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + online_param.data * tau)

    def save(self):
        save_path = (
                self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.critic.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def critic_learn(self, state, next_state, action, reward):

        # Get TD Estimate
        td_est = self.critic(state, action, model="online")

        # Get TD Target
        td_tgt = self.td_target(reward, next_state)

        # Back propagate loss through Q_online
        critic_loss = self.loss_fn(td_tgt, td_est)
        self.critic_optimizer.zero_grad()

        critic_loss.backward()
        self.critic_optimizer.step()

        return td_est.mean().item(), critic_loss.item()

    def actor_learn(self, state):
        actor_loss = -torch.mean(self.critic(state, self.actor(state, model="online"), model="online"))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_target(self.critic, self.tau)
            self.sync_target(self.actor, self.tau)

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        self.actor_learn(state)
        q, loss = self.critic_learn(state, next_state, action, reward)

        return q, loss

