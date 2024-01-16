# Importing the Flask module
from flask import Flask, render_template, request, jsonify
import numpy as np
from io import BytesIO
import pandas as pd
from flask import Flask, send_file
import pickle
from copy import copy

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

def encode_board(board):
    encoding = np.zeros((8, 8), dtype=np.int8)

    piece_mapping = {'p': -1, 'r': -2, 'n': -3, 'b': -4, 'q': -5, 'k': -6,
                     'P': 1, 'R': 2, 'N': 3, 'B': 4, 'Q': 5, 'K': 6}

    for square, piece in board.piece_map().items():
        row, col = chess.square_rank(square), chess.square_file(square)
        encoding[row, col] = piece_mapping[piece.symbol()]

    #encoding = encoding.flatten()

    return encoding

def check_game_status(board):
    if board.is_checkmate():
        winner = "white" if board.turn == chess.BLACK else "black"
        print('cek winner', winner)
    elif board.is_stalemate():
        winner = 'draw'
    elif board.is_insufficient_material():
        winner = 'draw'
    else:
        winner = None
    return winner

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

# Instantiate the neural network
checkpoint = torch.load("model_policy2d_new_epoch-61.pt", map_location=torch.device('cpu'))
model = ChessResNet().double()
model.load_state_dict(checkpoint['model'])

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
    res['winner'] = None
    for iter_hist in range(len(hist_move)):
        single_hist = hist_move[iter_hist]
        legal_move = list(board.legal_moves)
        legal_move_str = []
        for iter_legal in range(len(legal_move)):
            legal_move_str.append(str(legal_move[iter_legal]))
        
        if single_hist not in legal_move_str:
            print('cek single hist', single_hist)
            print('cek legal move', legal_move_str)
            return res
        
        move_uci = chess.Move.from_uci(single_hist)
        board.push(move_uci)
        status_game = check_game_status(board)
        if status_game is not None:
            #print('terpicu, harusnya selesai')
            res['legal'] = True
            res['winner'] = status_game
            res['move'] = ''
            return res
            #break
        
    print('cek game status', status_game, board.is_checkmate(), board.turn, chess.BLACK)
    #evaluate best move by bot
    legal_move = list(board.legal_moves)
    legal_move_bot = []
    for iter_legal in range(len(legal_move)):
        legal_move_bot.append(str(legal_move[iter_legal]))
    old_board_state = copy(board)
    all_input = []
    for iter_op in range(len(legal_move_bot)):
        new_board = copy(board)
        move_uci = chess.Move.from_uci(legal_move_bot[iter_op])
        new_board.push(move_uci)
        
        new_state_board = encode_board(new_board)
        old_state_board = encode_board(old_board_state)
        
        input_feature = [old_state_board, new_state_board]
        input_feature = np.array(input_feature)

        if len(all_input) == 0:
            all_input = np.reshape(input_feature, (1, 2, len(old_state_board), len(new_state_board[0])))
        else:
            all_input = np.concatenate([all_input, np.reshape(input_feature, (1, 2, len(old_state_board), len(new_state_board[0])))])
    min_input = np.min(all_input)
    max_input = np.max(all_input)

    all_input = (all_input - min_input) / (max_input - min_input)

    all_input_torch = torch.from_numpy(all_input)
    predict = model(all_input_torch)
    predict = predict.squeeze()
    idx_bot = torch.argmax(predict)
    bot_move_choosen = legal_move_bot[idx_bot]
    res['move'] = bot_move_choosen
    #print('cek predict', predict)

    res['legal'] = True
    return res

# Running the application on the local development server
if __name__ == '__main__':
    app.run(debug=True)
