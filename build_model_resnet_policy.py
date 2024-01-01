import chess
import chess.engine
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as opt

#setting
batch_size = 32
epoch_total = 300
lr = 1e-3


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ChessResNet(nn.Module):
    def __init__(self):
        super(ChessResNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 64, 2)
        self.layer2 = self.make_layer(64, 128, 2)
        self.layer3 = self.make_layer(128, 256, 2)
        self.layer4 = self.make_layer(256, 512, 2)
        self.fc = nn.Linear(512, 1)
        self.output_activation = nn.Sigmoid()

    def make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)
        x = self.output_activation(self.fc(x))
        return x

# Example usage:
input_size = 128
hidden_neurons = [128, 100, 64, 64, 64]  # Customize the number of neurons in each hidden layer
output_size = 1  # Assuming a regression task, adjust for classification tasks

# Instantiate the neural network
model = ChessResNet().double()
# Define Binary Cross Entropy Loss
criterion = nn.BCELoss()
#criterion = nn.BCEWithLogitsLoss()
print(model)

all_input = np.load('feature_policy2d.npy')
all_label = np.load('label_policy2d.npy')

print(np.shape(all_input), np.shape(all_label))
#print(list(set(all_label)))
#exit()

min_input = np.min(all_input)
max_input = np.max(all_input)

all_input = (all_input - min_input) / (max_input - min_input)

#print(all_input)
#exit()

all_input_torch = torch.from_numpy(all_input)
all_label_torch = torch.from_numpy(all_label)

dataset_train = TensorDataset(all_input_torch, all_label_torch)
dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

optimizer = opt.Adam(model.parameters(), lr=lr)
for epoch in range(epoch_total):
    epoch_loss = 0
    for i, (input, label) in enumerate(dataloader):
        label = label.double()
        predict = model(input).squeeze()
        #print('predict', predict)
        #print('label', label)

        optimizer.zero_grad()
        #print(input)
        #print('cek label', label)
        loss = criterion(predict, label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        #print(loss)
        #break
    print('Epoch:{}, Loss:{}'.format(epoch+1, epoch_loss/(i+1)))
    if (epoch+1) % 10 == 0:
        checkpoint = {'model': model.state_dict()}
        torch.save(checkpoint, "model_policy2d_epoch-%s.pt"%(epoch+1))
    #break

checkpoint = {'model': model.state_dict()}
torch.save(checkpoint, "model_policy2d_final.pt")