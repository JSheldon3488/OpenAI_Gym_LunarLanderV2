import numpy as np
import random
from collections import namedtuple, deque
from model import QNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters for the Agents
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_LOCAL = 4        # how often to update the local network
UPDATE_TARGET = 50      # how often to update the target network

# Set up to run on GPU or CPU
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def __str__(self):
        return "DQN_Agent"

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step += 1
        if self.t_step%UPDATE_LOCAL == 0:
            self.t_step = 0
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
            # ------------------- soft update target network ------------------- #
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def act(self, state, eps=0.):
        """Returns action for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # Turn off training and gradient calculations and just get the action_values
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()  # Turning back on training for the future

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max Q values for each of the next_states from target model
        next_Qvalues = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        td_target_values = rewards + (gamma * next_Qvalues * (1 - dones))
        # Get expected Q values from local model for all actions taken
        prev_qvalues = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(td_target_values, prev_qvalues)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        target network moves in the direction of local network by TAU amount
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class Agent_Double(Agent):
    """Double DQN Agent that interacts with and learns from the environment"""

    def __init__(self, state_size, action_size, seed):
        """Initialize a Double DQN Agent object using inheritance from Agent.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        super().__init__(state_size, action_size, seed)

    def __str__(self):
        return "Double_DQN_Agent"

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples. (Double DQN)

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        #           Double DQN Update                      #
        # Find max actions for next states based on the local_network
        local_argmax_actions = self.qnetwork_local(next_states).detach().argmax(dim=1).unsqueeze(1)
        # Use local_best_actions and target_network to get predicted value for next states
        next_qvalues = self.qnetwork_target(next_states).gather(1,local_argmax_actions).detach()

        #          Everything else same as DQN             #
        # Compute Q target values for current states (1-dones computes to 0 if next state is terminal)
        td_target_values = rewards + (gamma * next_qvalues * (1 - dones))
        # Get expected Q values from local model for all actions taken
        prev_qvalues = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(td_target_values, prev_qvalues)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Prioritized_Agent(Agent):
    """ DQN Agent that interacts with and learns from the environment and uses Prioritized Experience Replay """

    def __init__(self, state_size, action_size, seed):
        """Initialize a DQN Agent object using inheritance from Agent but take into account PER memory.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        super().__init__(state_size, action_size, seed)
        self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.alpha = 0.7
        self.alpha_decay = 0.9995

    def __str__(self):
        return "PER_DQN_Agent"

    def step(self, state, action, reward, next_state, done):
        """ Save experience in replay memory taking into account priority """
        # Initial priority from new experience is large to incentivize every experience to be sampled at least once
        priority = 1000000
        self.memory.add(state, action, reward, next_state, done, priority)

        # Learn and update memory every UPDATE_EVERY time steps.
        self.t_step += 1
        if self.t_step%UPDATE_LOCAL == 0:
            self.t_step = 0
            # If enough samples are available in memory, get subset based on priority and learn
            if len(self.memory) > BATCH_SIZE:
                self.memory.sort()
                sample_indexes, experiences = self.memory.sample(self.alpha)
                self.learn(experiences, GAMMA, sample_indexes)    # Need sample_indexes to update memory buffer with new priority
                self.alpha *= self.alpha_decay    # Favor priority selections early and uniform selections later
            # ------------------- soft update target network ------------------- #
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def learn(self, experiences, gamma, sample_indexes):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            sample_indexes: indexes of the memories in the sample (useful for updating memory with new priority)
        """
        states, actions, rewards, next_states, dones, priorities = experiences

        # Get max Q values for each of the next_states from target model
        next_Qvalues = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        td_target_values = rewards + (gamma * next_Qvalues * (1 - dones))
        # Get expected Q values from local model for all actions taken
        prev_qvalues = self.qnetwork_local(states).gather(1, actions)

        # Calculate new priorities and update memory
        with torch.no_grad():
            new_priority_values = torch.abs(td_target_values - prev_qvalues).cpu()
            self.memory.update_replay_buffer(sample_indexes, (states, actions, rewards, next_states, dones, new_priority_values))

        # Compute loss
        loss = F.mse_loss(td_target_values, prev_qvalues)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Prioritized_Double_Agent(Prioritized_Agent):
    """ Double DQN Agent that interacts with and learns from the environment and uses Prioritized Experience Replay.
        Everything is the same as the Priority_Agent except the learn method now uses the Double DQN formula
    """

    def __init__(self, state_size, action_size, seed):
        """Initialize a Double DQN Agent object using inheritance from Prioritized_Agent.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        super().__init__(state_size, action_size, seed)

    def __str__(self):
        return "PER_Double_DQN_Agent"

    def learn(self, experiences, gamma, sample_indexes):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            sample_indexes: indexes of the memories in the sample (useful for updating memory with new priority)
        """
        states, actions, rewards, next_states, dones, priorities = experiences

        #           Double DQN Update                      #
        # Find max actions for next states based on the local_network
        local_argmax_actions = self.qnetwork_local(next_states).detach().argmax(dim=1).unsqueeze(1)
        # Use local_best_actions and target_network to get predicted value for next states
        next_qvalues = self.qnetwork_target(next_states).gather(1,local_argmax_actions).detach()

        #          Everything else same as DQN             #
        # Compute Q target values for current states (1-dones computes to 0 if next state is terminal)
        td_target_values = rewards + (gamma * next_qvalues * (1 - dones))
        # Get expected Q values from local model for all actions taken
        prev_qvalues = self.qnetwork_local(states).gather(1, actions)

        # Calculate new priorities and update memory
        with torch.no_grad():
            new_priority_values = torch.abs(td_target_values - prev_qvalues).cpu().numpy()
            self.memory.update_replay_buffer(sample_indexes, (states, actions, rewards, next_states, dones, new_priority_values))

        # Compute loss
        loss = F.mse_loss(td_target_values, prev_qvalues)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        #Note: np.vstack creates a column vector (vertical stack items in array)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedReplayBuffer(ReplayBuffer):
    """ Fixed-size buffer to store experience tuples that also allows for storing a priority score. """
    def __init__(self, action_size, buffer_size, batch_size, seed):
        super().__init__(action_size, buffer_size, batch_size, seed)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority"])

    def add(self, state, action, reward, next_state, done, priority):
        """ Add a new experience to memory with a priority score """
        e = self.experience(state, action, reward, next_state, done, priority)
        self.memory.append(e)

    def sample(self, alpha):
        """ Sample a batch of experiences from memory based on ranked-based priority """
        # Calculate probabilities using ranked-based method and generate a sample from memory
        # P(i) = priority(i)^alpha / sum_all_priority^alpha where priority value is based on its order in a sorted container
        sum_all_priority = sum((1/i)**alpha for i in range(1, len(self.memory)+1))
        probs = [((1/i)**alpha)/sum_all_priority for i in range(1, len(self.memory)+1)]
        # Get a sample based on probabilities
        sample_indexes = np.random.choice(len(self.memory), self.batch_size, p=probs)
        experiences = [self.memory[i] for i in sample_indexes]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        priorities = torch.from_numpy(np.vstack([e.priority for e in experiences if e is not None])).float().to(device)

        return sample_indexes, (states, actions, rewards, next_states, dones, priorities)

    def update_replay_buffer(self, sample_indexes, experiences):
        """ Update existing experiences in memory to account for new priority value """
        states, actions, rewards, next_states, dones, new_priorities = experiences
        for i in range(self.batch_size):
            e = self.experience(states[i], int(actions[i]), float(rewards[i]), next_states[i], bool(dones[i]), float(new_priorities[i]))
            self.memory[sample_indexes[i]] = e

    def sort(self):
        """ Sort memory based on priority (TD error) """
        items = [self.memory.pop() for i in range(len(self.memory))]
        items.sort(key=lambda x: x[5], reverse=True)
        self.memory.extend(items)