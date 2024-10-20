import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Conv_QNet(nn.Module):
    def __init__(self, input_channels, hidden_sizes, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_sizes[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_sizes[0], hidden_sizes[1], kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(hidden_sizes[1], hidden_sizes[2], kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(hidden_sizes[2] * 256 * 256,
                            output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 3:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        pred = self.model(state)

        # ------------------------------------------------------------------------------------------------
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
        #-------------------------------------------------------------------------------------------------

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
