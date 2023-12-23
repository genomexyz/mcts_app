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
batch_size = 64
epoch_total = 100
lr = 1e-3

# Define your neural network architecture
class ChessNet(nn.Module):
    def __init__(self, input_size, hidden_neurons, output_size):
        super(ChessNet, self).__init__()

        # Input layer
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_neurons[0])])

        # Hidden layers
        for i in range(1, len(hidden_neurons)):
            self.layers.append(nn.Linear(hidden_neurons[i-1], hidden_neurons[i]))

        # Output layer
        self.output_layer = nn.Linear(hidden_neurons[-1], output_size)

        self.output_activation = nn.Sigmoid()

        # Activation function (you can customize this)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.output_activation(self.output_layer(x))
        return x

# Example usage:
input_size = 128
hidden_neurons = [128, 100, 64, 64]  # Customize the number of neurons in each hidden layer
output_size = 1  # Assuming a regression task, adjust for classification tasks

# Instantiate the neural network
model = ChessNet(input_size, hidden_neurons, output_size).double()
# Define Binary Cross Entropy Loss
criterion = nn.BCELoss()
print(model)

all_input = np.load('feature_policy_new.npy')
all_label = np.load('label_policy_new.npy')

print('cek data')
print('%s/%s'%(np.sum(all_label), len(all_label)))
exit()

min_input = np.min(all_input)
max_input = np.max(all_input)

all_input = (all_input - min_input) / (max_input - min_input)

all_input_torch = torch.from_numpy(all_input)
all_label_torch = torch.from_numpy(all_label)

dataset_train = TensorDataset(all_input_torch, all_label_torch)
dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

acc_arr = []
for i, (input, label) in enumerate(dataloader):
    print('iter %s'%(i))
    label = label.double()
    predict = model(input).detach().squeeze()
    predict[predict < 0.5] = 0
    predict[predict > 0] = 1
    print('predict', predict)
    print('label', label)
    batch_acc = abs((predict - label).numpy())
    if len(acc_arr) == 0:
        acc_arr = batch_acc
    else:
        acc_arr = np.concatenate([acc_arr, batch_acc])

print(acc_arr)

total_data = len(acc_arr)
acc = np.sum(acc_arr) / total_data
print('acc %s'%(acc))