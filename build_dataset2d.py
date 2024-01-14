import pandas as pd
import re
import numpy as np
import chess
import chess.pgn
import io
import pickle
from copy import copy

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import base64

#setting
#file_path = 'CHESS_DATA.csv'
encryption_key = b'0123456789abcdef'  # Use a valid key size: 16 bytes (128 bits)
file_path = 'lichess_db_standard_rated_2014-07.pgn'
elo_threshold = 1800
cnt_limit = 10000
cnt = 0
limit_sample = 10000
limit_per_case = 10000
limit_cnt = 80
#limit_sample = 100
knowledge = {}
knowledge_map_val = {}
win_score = 1
lose_score = -1

def get_state_board(hist_move):
    board = chess.Board()
    for iter_move in range(len(hist_move)):
        movement = chess.Move.from_uci(hist_move[iter_move])
        board.push(movement)
    return encode_board(board)

def encode_board(board):
    encoding = np.zeros((8, 8), dtype=np.int8)

    piece_mapping = {'p': -1, 'r': -2, 'n': -3, 'b': -4, 'q': -5, 'k': -6,
                     'P': 1, 'R': 2, 'N': 3, 'B': 4, 'Q': 5, 'K': 6}

    for square, piece in board.piece_map().items():
        row, col = chess.square_rank(square), chess.square_file(square)
        encoding[row, col] = piece_mapping[piece.symbol()]

    #encoding = encoding.flatten()

    return encoding

def get_game_data(data):
    data_io = io.StringIO(data)
    game = chess.pgn.read_game(data_io)
    #game_board = game.board()
    all_move = []
    for move in game.mainline_moves():
        all_move.append(str(move))
    return all_move, game.headers

#mode_absorb = True
#mode_absorb_str = '[Event '
#len_absorb = len(mode_absorb_str)
#content = ''
#all_move_white_win = []
#all_move_black_win = []
#with open(file_path, 'r') as file:
#    for line in file:
#        # Process each line
#        #print(line.strip())  # Example: Print each line (without leading/trailing whitespaces)
#        single_line = line.strip()
#        if line == '':
#            continue
#        if single_line[:len_absorb] == mode_absorb_str:
#            content = content.strip()
#            mode_absorb = True
#            if content != '':
#                #extract_data
#                game_move, header = get_game_data(content)
#                try:
#                    rating_black = float(header.get('BlackElo'))
#                except ValueError:
#                    rating_black = 0
#                try:
#                    rating_white = float(header.get('WhiteElo'))
#                except ValueError:
#                    rating_white = 0
#                if header.get('Termination') == 'Normal':
#                    res_match = header.get('Result')
#                    if rating_white > elo_threshold or rating_black > elo_threshold:
#                        if res_match == '0-1':
#                            #print(game_move)
#                            all_move_black_win.append(game_move)
#                            #print('cek black', all_move_black_win)
#                        elif res_match == '1-0':
#                            all_move_white_win.append(game_move)
#                total_sample = len(all_move_white_win)+len(all_move_black_win)
#                if total_sample > limit_sample:
#                    break
#                print('collect %s samples'%(total_sample))
#            content = ''
#        content += line
#
#with open('good_white.pkl', 'wb') as f:
#    pickle.dump(all_move_white_win, f)
#
#with open('good_black.pkl', 'wb') as f:
#    pickle.dump(all_move_black_win, f)

with open('good_white.pkl', 'rb') as f:
    white_win = pickle.load(f)

with open('good_black.pkl', 'rb') as f:
    black_win = pickle.load(f)

#print(black_win)

#board = chess.Board()
cnt = 1
all_input = []
all_label = []
while True:
    dict_move = {}
    begin_total = len(all_input)
    case_closed = False

    if cnt > limit_cnt:
        break

    for iter_hist in range(len(black_win)):
        single_hist = black_win[iter_hist][0:cnt]
        single_hist_str = '|'.join(single_hist)
        if single_hist_str == '':
            continue
        try:
            single_choice = black_win[iter_hist][cnt]
        except IndexError:
            continue
        try:
            dict_move[single_hist_str].append(single_choice)
        except KeyError:
            dict_move[single_hist_str] = [single_choice]
    all_hist = list(dict_move.keys())
    if len(all_hist) == 0: #there is no more training material, then end
        break
    for iter_hist in range(len(all_hist)):
        single_case = all_hist[iter_hist]
        single_case_arr = single_case.split('|')
        next_move = list(set(dict_move[single_case]))
        print('case', single_case_arr, next_move, len(all_input))

        board = chess.Board()
        for iter_move in range(len(single_case_arr)):
            #print('cek movement', single_case_arr[iter_move])
            movement = chess.Move.from_uci(single_case_arr[iter_move])
            board.push(movement)
        all_opsi = list(board.legal_moves)
        all_opsi_str = []
        for iter_op in range(len(all_opsi)):
            all_opsi_str.append(str(all_opsi[iter_op]))
        
        #get train data for valid human move
        #get label 1 as valid human expert move
        for iter_next in range(len(next_move)):
            only_hist = get_state_board(single_case_arr)
            pos_move = single_case_arr.copy()
            pos_move.append(next_move[iter_next])
            future_move = get_state_board(pos_move)
            #input_feature = np.concatenate([only_hist, future_move])
            input_feature = [only_hist, future_move]
            input_feature = np.array(input_feature)
            #print('cek shape input feature', np.shape(input_feature))
            #print(only_hist)
            #print('--------------------------------')
            #print(future_move)
            #exit()
            label = np.array([1])
            if len(all_input) == 0:
                all_input = np.reshape(input_feature, (1, 2, len(only_hist), len(only_hist[0])))
                all_label = label
            else:
                all_input = np.concatenate([all_input, np.reshape(input_feature, (1, 2, len(only_hist), len(only_hist[0])))])
                all_label = np.concatenate([all_label, label])
            
            if len(all_input)-begin_total > limit_per_case:
                case_closed = True
                break

        if case_closed:
            break

            #print('data input', input_feature, np.shape(input_feature))
        #get train data for invalid human move
        #get label 0 as invalid human expert move
        for iter_next in range(len(all_opsi_str)):
            single_opsi = all_opsi_str[iter_next]
            if single_opsi in next_move:
                continue
            only_hist = get_state_board(single_case_arr)
            pos_move = single_case_arr.copy()
            pos_move.append(single_opsi)
            future_move = get_state_board(pos_move)
            input_feature = np.concatenate([only_hist, future_move])
            label = np.array([0])
            if len(all_input) == 0:
                all_input = np.reshape(input_feature, (1, 2, len(only_hist), len(only_hist[0])))
                all_label = label
            else:
                all_input = np.concatenate([all_input, np.reshape(input_feature, (1, 2, len(only_hist), len(only_hist[0])))])
                all_label = np.concatenate([all_label, label])
            
            if len(all_input)-begin_total > limit_per_case:
                case_closed = True
                break
        
        if case_closed:
            break

        #print('movement input')
        #print(all_input, np.shape(all_input))
        #print('movement output')
        #print(all_label, np.shape(all_label))
        #print('all opsi', all_opsi_str)
    #print(dict_move)
    #break
    cnt += 2
#    cnt += 6

np.save('feature_policy2d.npy', all_input)
np.save('label_policy2d.npy', all_label)