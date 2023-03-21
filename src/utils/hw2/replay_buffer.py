from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, action, reward, next_state, done = zip(*[self.buffer[i] for i in indices])
        return np.stack(state), np.stack(action), np.stack(reward), np.stack(next_state), np.stack(done)

    def __len__(self):
        return len(self.buffer)
