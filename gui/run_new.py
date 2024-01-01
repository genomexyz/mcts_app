# Importing the Flask module
from flask import Flask, render_template, request, jsonify
import numpy as np
from io import BytesIO
import pandas as pd
from flask import Flask, send_file
import pickle

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from base64 import b64decode
from cryptography.hazmat.primitives import padding
import base64

import random

import chess
import chess.pgn
import chess.engine
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as opt

#setting
encryption_key = b'0123456789abcdef'  # Use a valid key size: 16 bytes (128 bits)

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

knowledge_new = {}
knowledge_map_new = {}

# Creating a Flask web application
app = Flask(__name__)

# Defining a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_move', methods=['POST'])
def get_move():
    #begin chess
    board = chess.Board()
    # Get the JSON data from the request
    json_data = request.get_json()
    hist_move = json_data['history']

    res = {}
    res['legal'] = False
    for iter_hist in range(len(hist_move)):
        single_hist = hist_move[iter_hist]
        legal_move = list(board.legal_moves)
        legal_move_str = []
        for iter_legal in range(len(legal_move)):
            legal_move_str.append(str(legal_move[iter_legal]))
        
        if single_hist not in legal_move_str:
            return res
        
        move_uci = chess.Move.from_uci(single_hist)
        board.push(move_uci)
    
    code_hist_move = '|'.join(hist_move)
    try:
        state = knowledge[code_hist_move]
    except KeyError:
        knowledge[code_hist_move] = {}
        knowledge[code_hist_move]['visit'] = 1
        #knowledge[code_move_parent_encode]['value'] = 0
        knowledge[code_hist_move]['children'] = []
        knowledge_map[code_hist_move] = 0
        state = knowledge[code_hist_move]
    
    if len(state['children']) == 0:
        print('enter random choice')
        all_move = list(board.legal_moves)
        all_move_str = []
        for iter_move in range(len(all_move)):
            all_move_str.append(str(all_move[iter_move]))
        move_choice = random.choice(all_move_str)
    else:
        print('enter determined choice')
        all_move_val = []
        avail_choice = state['children']
        for iter_child in range(len(avail_choice)):
            single_key = code_hist_move+'|'+avail_choice[iter_child]
            single_val = knowledge_map[single_key]
            all_move_val.append(single_val)
        all_move_val = np.array(all_move_val)
        max_val = np.max(all_move_val)
        if max_val < 1:
            all_move = list(board.legal_moves)
            all_move_str = []
            for iter_move in range(len(all_move)):
                all_move_str.append(str(all_move[iter_move]))
            move_choice = random.choice(all_move_str)
        else:
            idx_choice = np.argmax(all_move_val)
            move_choice = avail_choice[idx_choice]
        print('choice', max_val, move_choice)
    res['move'] = move_choice


    res['legal'] = True
    return res

# Running the application on the local development server
if __name__ == '__main__':
    app.run(debug=True)
