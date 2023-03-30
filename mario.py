import random
from collections import deque

import numpy as np
import torch

import master_buffer
from master_buffer import mario_params_to_tensors
from net import MarioNet


class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device=torch.device('cuda'))

        self.curr_step = 0

        self.save_every = 1e5  # no. of experiences between saving Mario Net
        self.memory = deque(maxlen=80000)
        self.batch_size = 32

        # load master buffer memory
        self.master_memory = master_buffer.load('./master_buffer_files')

        self.gamma = 0.9

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 5e3  # min. experiences before training
        self.learn_every = 1  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

        self.loss_sum = 0

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """

        state = state.__array__()
        if self.use_cuda:
            state = torch.tensor(state).cuda()
        else:
            state = torch.tensor(state)
        state = state.unsqueeze(0)
        action_values = self.net(state, model="online")
        action_idx = torch.argmax(action_values, dim=1).item()

        # increment step
        self.curr_step += 1
        return action_idx

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
        state, next_state, action, reward, done = mario_params_to_tensors(state, next_state, action, reward, done)

        if self.curr_step < self.burnin:
            self.memory.append((state, next_state, action, reward, done,))
        else:
            self.loss_sum += loss
            if loss > self.loss_sum / self.curr_step:
                self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        if random.randint(0, 9) in range(0, 5):
            batch = random.sample(self.memory, self.batch_size)
        else:
            batch = random.sample(self.master_memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        if self.use_cuda:
            return (state.cuda(), next_state.cuda(), action.squeeze().cuda(),
                    reward.squeeze().cuda(), done.squeeze().cuda())
        else:
            return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_q, dim=1)
        next_q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_q).float()

    def update_q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
                self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict()),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Back propagate loss through Q_online
        loss = self.update_q_online(td_est, td_tgt)

        return td_est.mean().item(), loss
