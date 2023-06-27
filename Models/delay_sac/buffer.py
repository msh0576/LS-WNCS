from collections import deque

import numpy as np
import torch


class LazyFrames:
    """
    Stacked frames which never allocate memory to the same frame.
    """

    def __init__(self, frames):
        self._frames = list(frames)

    def __array__(self, dtype):
        return np.array(self._frames, dtype=dtype)

    def __len__(self):
        return len(self._frames)


class SequenceBuffer:
    """
    Buffer for storing sequence data.
    """

    def __init__(self, num_sequences=8):
        self.num_sequences = num_sequences
        self._reset_episode = False
        self.state_ = deque(maxlen=self.num_sequences + 1)
        self.action_ = deque(maxlen=self.num_sequences)
        self.reward_ = deque(maxlen=self.num_sequences)
        self.done_ = deque(maxlen=self.num_sequences)

    def reset(self):
        self._reset_episode = False
        self.state_.clear()
        self.action_.clear()
        self.reward_.clear()
        self.done_.clear()

    def reset_episode(self, state):
        assert not self._reset_episode
        self._reset_episode = True
        self.state_.append(state)
        # print(f"after reset_episode | ")

    def append(self, action, reward, done, next_state):
        assert self._reset_episode
        self.action_.append(action)
        self.reward_.append([reward])
        self.done_.append([done])
        self.state_.append(next_state)

    def get(self):
        state_ = LazyFrames(self.state_)
        action_ = np.array(self.action_, dtype=np.float32)
        reward_ = np.array(self.reward_, dtype=np.float32)
        done_ = np.array(self.done_, dtype=np.float32)
        return state_, action_, reward_, done_

    def is_empty(self):
        return len(self.reward_) == 0

    def is_full(self):
        return len(self.reward_) == self.num_sequences

    def __len__(self):
        return len(self.reward_)


class ReplayBuffer:
    """
    Replay Buffer.
    """

    def __init__(self, buffer_size, num_sequences, state_shape, action_shape, device, use_image, rms=None):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.num_sequences = num_sequences
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.use_image = use_image
        self.rms = rms

        # Store the sequence of images as a list of LazyFrames on CPU. It can store images with 9 times less memory.
        self.state_ = [None] * buffer_size
        # Store other data on GPU to reduce workloads.
        self.action_ = torch.empty(buffer_size, num_sequences, *action_shape, device=device)
        self.reward_ = torch.empty(buffer_size, num_sequences, 1, device=device)
        self.done_ = torch.empty(buffer_size, num_sequences, 1, device=device)
        # Buffer to store a sequence of trajectories.
        self.buff = SequenceBuffer(num_sequences=num_sequences)

    def reset_episode(self, state):
        """
        Reset the buffer and set the initial observation. This has to be done before every episode starts.
        """
        self.buff.reset_episode(state)

    def append(self, action, reward, done, next_state, episode_done):
        """
        Store trajectory in the buffer. If the buffer is full, the sequence of trajectories is stored in replay buffer.
        Please pass 'masked' and 'true' done so that we can assert if the start/end of an episode is handled properly.
        """
        self.buff.append(action, reward, done, next_state)

        if self.buff.is_full():
            state_, action_, reward_, done_ = self.buff.get()
            self._append(state_, action_, reward_, done_)

        if episode_done:
            self.buff.reset()

    def _append(self, state_, action_, reward_, done_):
        self.state_[self._p] = state_
        self.action_[self._p].copy_(torch.from_numpy(action_))
        self.reward_[self._p].copy_(torch.from_numpy(reward_))
        self.done_[self._p].copy_(torch.from_numpy(done_))

        self._n = min(self._n + 1, self.buffer_size)
        self._p = (self._p + 1) % self.buffer_size

    def sample_latent(self, batch_size):
        """
        Sample trajectories for updating latent variable model.
        """
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        if self.use_image:
            state_ = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
        else:
            state_ = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.float)

        for i, idx in enumerate(idxes):
            state_[i, ...] = self.state_[idx]
        if self.use_image:
            state_ = torch.tensor(state_, dtype=torch.uint8, device=self.device).float().div_(255.0)
        else:
            if self.rms is not None:
                state_ = self.rms.unnormalize(state_)
            else:
                state_ = torch.tensor(state_, dtype=torch.float, device=self.device)

        return state_, self.action_[idxes], self.reward_[idxes], self.done_[idxes]

    def sample_sac(self, batch_size):
        """
        Sample trajectories for updating SAC.
        """
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        if self.use_image:
            state_ = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
        else:
            state_ = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.float)

        for i, idx in enumerate(idxes):
            state_[i, ...] = self.state_[idx]
        if self.use_image:
            state_ = torch.tensor(state_, dtype=torch.uint8, device=self.device).float().div_(255.0)
        else:
            if self.rms is not None:
                state_ = self.rms.unnormalize(state_)
            else:
                state_ = torch.tensor(state_, dtype=torch.float, device=self.device)

        return state_, self.action_[idxes], self.reward_[idxes, -1], self.done_[idxes, -1]

    def __len__(self):
        return self._n
