import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        
        self.model_state = np.zeros((max_size, state_dim))
        self.model_action = np.zeros((max_size, action_dim))
        self.model_next_state = np.zeros((max_size, state_dim))
        self.model_reward = np.zeros((max_size, 1))
        self.model_not_done = np.zeros((max_size, 1))
        

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def add_to_model_pool(self, state, action, next_state, reward, done, size):
        if self.model_pool_ptr + size <= self.max_size:
            self.model_state[self.model_pool_ptr:self.model_pool_ptr + size] = state
            self.model_action[self.model_pool_ptr:self.model_pool_ptr + size] = action
            self.model_next_state[self.model_pool_ptr:self.model_pool_ptr + size] = next_state
            self.model_reward[self.model_pool_ptr:self.model_pool_ptr + size] = reward
            self.model_not_done[self.model_pool_ptr:self.model_pool_ptr + size] = 1. - done

            self.model_pool_ptr = (self.model_pool_ptr + size) % self.max_size
            self.model_pool_size = min(self.model_pool_size + size, self.max_size)
        
        else:
            self.max_size = self.model_pool_size
            self.model_pool_ptr = 0
            self.model_state[self.model_pool_ptr:self.model_pool_ptr + size] = state
            self.model_action[self.model_pool_ptr:self.model_pool_ptr + size] = action
            self.model_next_state[self.model_pool_ptr:self.model_pool_ptr + size] = next_state
            self.model_reward[self.model_pool_ptr:self.model_pool_ptr + size] = reward
            self.model_not_done[self.model_pool_ptr:self.model_pool_ptr + size] = 1. - done
            
            self.model_pool_ptr = (self.model_pool_ptr + size) % self.max_size
            
    def sample_mixed(self, batch_size, ratio=1):
        size = int(batch_size * ratio)
        ind = np.random.randint(0, self.size, size=size)
        ind_model = np.random.randint(0, self.model_pool_size, size=batch_size-size)

        return (
            torch.FloatTensor(np.concatenate([self.state[ind], self.model_state[ind_model]], 0)).to(self.device),
            torch.FloatTensor(np.concatenate([self.action[ind], self.model_action[ind_model]], 0)).to(self.device),
            torch.FloatTensor(np.concatenate([self.next_state[ind], self.model_next_state[ind_model]], 0)).to(self.device),
            torch.FloatTensor(np.concatenate([self.reward[ind], self.model_reward[ind_model]], 0)).to(self.device),
            torch.FloatTensor(np.concatenate([self.not_done[ind], self.model_not_done[ind_model]], 0)).to(self.device),
        )

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
    
    def convert_D4RL(self, dataset):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards'].reshape(-1,1)
        self.not_done = 1. - dataset['terminals'].reshape(-1,1)
        self.size = self.state.shape[0]
        
    def normalize_states(self, eps = 1e-3):
        mean = self.state.mean(0,keepdims=True)
        std = self.state.std(0,keepdims=True) + eps
        self.state = (self.state - mean)/std
        self.next_state = (self.next_state - mean)/std
        return mean, std