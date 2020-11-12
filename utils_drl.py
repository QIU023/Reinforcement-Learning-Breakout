from typing import (
    Optional,
)

import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils_types import (
    TensorStack4,
    TorchDevice,
)

from utils_memory import ReplayMemory
from utils_model import DQN,DuelingDQN


class Agent(object):

    def __init__(
            self,
            action_dim: int,
            device: TorchDevice,
            gamma: float,
            seed: int,

            eps_start: float,
            eps_final: float,
            eps_decay: float,

            dueling: bool,
            restore: Optional[str] = None,
            stable_arg = 0.1,
    ) -> None:
        self.__action_dim = action_dim
        self.__device = device
        self.__gamma = gamma

        self.__eps_start = eps_start
        self.__eps_final = eps_final
        self.__eps_decay = eps_decay

        self.__eps = eps_start
        self.__r = random.Random()
        self.__r.seed(seed)
        
        self.__stable_arg = stable_arg

        if dueling:
            self.__policy = DuelingDQN(action_dim, device).to(device)
            self.__target = DuelingDQN(action_dim, device).to(device)
        else:
            self.__policy = DQN(action_dim, device).to(device)
            self.__target = DQN(action_dim, device).to(device)
        if restore is None:
            self.__policy.apply(DQN.init_weights)
        else:
            #if dueling:
            #self.__policy.Convs_load(restore)
            #else:
            self.__policy.load_state_dict(torch.load(restore))
        self.__target.load_state_dict(self.__policy.state_dict())
        self.__optimizer = optim.Adam(
            self.__policy.parameters(),
            lr=0.0000625,
            eps=1.5e-4,
        )
        self.__target.eval()

    def run(self, state: TensorStack4, training: bool = False) -> int:              #epsilon greedy policy
        """run suggests an action for the given state."""
        if training:
            self.__eps -= \
                (self.__eps_start - self.__eps_final) / self.__eps_decay
            self.__eps = max(self.__eps, self.__eps_final)

        if self.__r.random() > self.__eps:
            with torch.no_grad():
                action = self.__policy(state).max(1).indices.item()
        else: 
            action =  self.__r.randint(0, self.__action_dim - 1)
        value_this = self.__policy(state)[0][action]
        return action, value_this

    def get_target_value(self, state):
        value_next = self.__target(state).max(1).indices.item()
        return value_next

    def learn(self, memory: ReplayMemory, batch_size: int) -> float:
        """learn trains the value network via TD-learning.应该是Q network """
        state_batch, action_batch, reward_batch, \
            next_batch, done_batch, idx_batch = memory.sample(batch_size)

        values = self.__policy(state_batch.float()).gather(1, action_batch)
        values_next = self.__target(next_batch.float()).max(1).values.detach()
        expected = (self.__gamma * values_next.unsqueeze(1)) * \
            (1. - done_batch) + reward_batch                                        #TD target
        loss_batch = F.smooth_l1_loss(values, expected, reduce=False)               #TD error
        loss = torch.mean(loss_batch, dim=0)
        loss.requires_grad = True
        memory.update(loss_batch.detach(), idx_batch)
        
        self.__optimizer.zero_grad()
        loss.backward()                                                             #backward
        for param in self.__policy.parameters():
            if param.grad is not None:#grad clamp to (-1,1)
                param.grad.data.clamp_(-1, 1)
        self.__optimizer.step()                                                     #update
        
        return loss.item()

    def stable_learn(self, folded_state, action, reward, done):
        '''learn stable and q_value'''
        state = folded_state[:4]
        state_next = folded_state[1:]
        value_next = agent.get_target_value(state_next)
        value_now = self.__policy(state.float()).gather(1, action)
        td_target = (self.__gamma * values_next.unsqueeze(1)) * \
            (1. - done) + reward 
        td_error = F.smooth_l1_loss(value_now, td_target)
        stable_loss = torch.zeros(1).float()
        for i in [1, 2]:
            for j in [i-1, i+1]:
                stable_loss += 1*(action[j]*action[i]==2)
        stable_loss -= 1*(action[1]*action[2]==2)
        stable_loss.requires_grad = True
        loss = td_error + self.stable_arg*stable_loss
        self.__optimizer.zero_grad()
        loss.backward()                                                             #backward
        for param in self.__policy.parameters():
            if param.grad is not None:#grad clamp to (-1,1)
                param.grad.data.clamp_(-1, 1)
        self.__optimizer.step()  
        return loss.item()
    
    def sync(self) -> None:
        """sync synchronizes the weights from the policy network to the target
        network."""
        self.__target.load_state_dict(self.__policy.state_dict())

    def save(self, path: str) -> None:
        """save saves the state dict of the policy network."""
        torch.save(self.__policy.state_dict(), path)

    def load(self, path: str) -> None:
        """save saves the state dict of the policy network."""
        self.__policy.load_state_dict(path)
        self.__target.load_state_dict(path)