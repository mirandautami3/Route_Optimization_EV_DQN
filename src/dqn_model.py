import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import os

# Model DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Agent DQN
class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma  
        self.epsilon = epsilon  
        self.epsilon_min = epsilon_min  
        self.epsilon_decay = epsilon_decay  

        # Inisialisasi model
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            action_values = self.model(state)
        return torch.argmax(action_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Prediksi nilai Q saat ini dan masa depan
        q_values = self.model(states)
        next_q_values = self.model(next_states).detach()

        # Hitung target Q-values
        targets = rewards + (1 - dones) * self.gamma * next_q_values.max(1)[0]

        # Loss function
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.criterion(current_q_values, targets)

        # Optimasi model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon untuk eksplorasi
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path="dqn_model.pth"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="dqn_model.pth"):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device), strict=False)
            self.model.to(self.device)
            print(f"✅ Model DQN berhasil dimuat dari {path}")
        else:
            print(f"⚠️ Model tidak ditemukan di {path}, mulai dari awal.")

if __name__ == "__main__":
    agent = DQNAgent(state_size=10, action_size=5)  # Pastikan sesuai dengan input yang digunakan
    print("🧠 DQN Model siap digunakan!")
