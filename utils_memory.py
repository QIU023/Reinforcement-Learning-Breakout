from typing import (
    Tuple,
)

from queue import PriorityQueue

import torch
import numpy as np

from utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    TensorStack5,
    TorchDevice,
)


class ReplayMemory(object):

    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice,
    ) -> None:
        self.device = device
        self.__capacity = capacity
        self.size = 0
        self.pos = 0

        self.m_states = torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8)
        self.m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.m_dones = torch.zeros((capacity, 1), dtype=torch.bool)

    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool
    ) -> None:
        self.m_states[self.pos] = folded_state
        self.m_actions[self.pos, 0] = action
        self.m_rewards[self.pos, 0] = reward
        self.m_dones[self.pos, 0] = done
        self.after_push()
       
    def get_pos(self):
        return self.pos
        
    def after_push(self):
        self.pos = (self.pos + 1) % self.__capacity
        self.size = max(self.size, self.pos)

    def sample(self, batch_size: int) -> Tuple[
            BatchState,
            BatchAction,
            BatchReward,
            BatchNext,
            BatchDone,
    ]:
        indices = torch.randint(0, high=self.size, size=(batch_size,))
        b_state = self.m_states[indices, :4].to(self.device).float()
        b_next = self.m_states[indices, 1:].to(self.device).float()
        b_action = self.m_actions[indices].to(self.device)
        b_reward = self.m_rewards[indices].to(self.device).float()
        b_done = self.m_dones[indices].to(self.device).float()
        return b_state, b_action, b_reward, b_next, b_done

    def __len__(self) -> int:
        return self.size

class CompareAble(object):
    def __init__(self, sample, priority):
        self.sample = sample
        self.priority = priority
    def __cmp__(self, other):
        if self.priority < other.priority:
            return -1
        elif self.priority == other.priority:
            return 0
        else:
            return 1

import numpy

# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class Memory_Buffer_PER(object):
    # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, memory_size=1000, a = 0.6, e = 0.01):
        self.tree = SumTree(memory_size)
        self.memory_size = memory_size
        self.prio_max = 0.1
        self.a = a
        self.e = e
        
    def push(self, fold_state, action, reward, done, td_error):
        data = (fold_state[:4], action, reward, fold_state[1:], done)
        p = (np.abs(self.prio_max) + self.e) ** self.a #  proportional priority
        self.tree.add(td_error, data)

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            
            state, action, reward, next_state, done = data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            priorities.append(p)
            idxs.append(idx)
        return torch.from_numpy(np.concatenate(states)),\
            torch.from_numpy(actions),\
            torch.from_numpy(rewards),\
            torch.from_numpy(np.concatenate(next_states)),\
            torch.from_numpy(dones)
    
    def update(self, idxs, errors):
        self.prio_max = max(self.prio_max, max(np.abs(errors)))
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.e) ** self.a
            self.tree.update(idx, p) 
        
    def size(self):
        return self.tree.n_entries

class PERMemory(ReplayMemory):
    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice):
        super().__init__(channels, capacity, device)
        self.m_td_errors = torch.zeros((capacity, 1), dtype=torch.float)
        #self.memory = PriorityQueue(maxsize=capacity)
        
    #@overloaded
    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
            td_error: float,
    ) -> None:
        super().push(folded_state=folded_state, action=action, reward=reward, done=done)
        self.m_td_errors[self.pos, 0] = td_error
        self.after_push()

    def update(self, td_errors, idx_batch):
        for i in range(len(idx_batch)):
            idx = idx_batch[i]
            error = td_errors[i]           
            self.m_td_errors[idx, 0] = error
        
    #@overloaded
    def sample(self, batch_size: int) -> Tuple[
            BatchState,
            BatchAction,
            BatchReward,
            BatchNext,
            BatchDone,
    ]:
        
        pro = torch.softmax(self.m_td_errors[:self.size], dim=0).squeeze(1).detach().numpy()
        indices = np.random.choice(range(self.size), size=(batch_size,), p=pro)
        #indices = torch.randint(0, high=self.size, size=(batch_size,))
        b_state = self.m_states[indices, :4].to(self.device).float()
        b_next = self.m_states[indices, 1:].to(self.device).float()
        b_action = self.m_actions[indices].to(self.device)
        b_reward = self.m_rewards[indices].to(self.device).float()
        b_done = self.m_dones[indices].to(self.device).float()
        return b_state, b_action, b_reward, b_next, b_done, indices
