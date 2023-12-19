import pandas as pd
import re
import numpy as np
import chess
import chess.pgn
import io
import pickle

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import base64

#setting
#file_path = 'CHESS_DATA.csv'
encryption_key = b'0123456789abcdef'  # Use a valid key size: 16 bytes (128 bits)
file_path = 'lichess_db_standard_rated_2014-07.pgn'
cnt_limit = 10000
cnt = 0
limit_sample = 10000
knowledge = {}
knowledge_map_val = {}
win_score = 1
lose_score = -1

def get_game_data(data):
    data_io = io.StringIO(data)
    game = chess.pgn.read_game(data_io)
    #game_board = game.board()
    all_move = []
    for move in game.mainline_moves():
        all_move.append(str(move))
    return all_move, game.headers

def pad(data):
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    return padder.update(data) + padder.finalize()

def encode_name(name, key):
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    encryptor = cipher.encryptor()
    padded_data = pad(name.encode('utf-8'))
    #print(padded_data)
    encrypted_name = encryptor.update(padded_data) + encryptor.finalize()
    return base64.b64encode(encrypted_name).decode('utf-8')  # Adjust the length as needed


mode_absorb = True
mode_absorb_str = '[Event '
len_absorb = len(mode_absorb_str)
content = ''
all_move_white_win = []
all_move_black_win = []
with open(file_path, 'r') as file:
    for line in file:
        # Process each line
        #print(line.strip())  # Example: Print each line (without leading/trailing whitespaces)
        single_line = line.strip()
        if line == '':
            continue
        if single_line[:len_absorb] == mode_absorb_str:
            content = content.strip()
            mode_absorb = True
            if content != '':
                #extract_data
                game_move, header = get_game_data(content)
                if header.get('Termination') == 'Normal':
                    res_match = header.get('Result')
                    if res_match == '0-1':
                        all_move_black_win.append(game_move)
                        #print('cek black', all_move_black_win)
                    elif res_match == '1-0':
                        all_move_white_win.append(game_move)
                total_sample = len(all_move_white_win)+len(all_move_black_win)
                if total_sample > limit_sample:
                    break
                print('collect %s samples'%(total_sample))
            content = ''
        content += line

#print('cek')
#print(all_move_white_win)

#################
#learn white win#
#################
for iter_move in range(len(all_move_white_win)):
    single_move_hist = all_move_white_win[iter_move]
    #print('cek len', len(single_move_hist))
    #print(single_move_hist)
    history_move = []
    for iter_hist in range(len(single_move_hist)):
        code_move_parent = '|'.join(single_move_hist[0:iter_hist])
        code_move = '|'.join(single_move_hist[0:iter_hist+1])
        code_move_tip = single_move_hist[iter_hist]

        #code_move_parent_encode = encode_name(code_move_parent, encryption_key)
        #code_move_encode = encode_name(code_move, encryption_key)

        code_move_parent_encode = code_move_parent
        code_move_encode = code_move
        code_move_tip_encode = code_move_tip

        try:
            knowledge[code_move_parent_encode]['visit'] += 1
        except KeyError:
            knowledge[code_move_parent_encode] = {}
            knowledge[code_move_parent_encode]['visit'] = 1
            #knowledge[code_move_parent_encode]['value'] = 0
            knowledge[code_move_parent_encode]['children'] = []
            knowledge_map_val[code_move_parent_encode] = 0
        
        if code_move_encode not in knowledge[code_move_parent_encode]['children']:
            knowledge[code_move_parent_encode]['children'].append(code_move_tip_encode)
        
        try:
            knowledge[code_move_encode]['visit'] += 1
        except KeyError:
            knowledge[code_move_encode] = {}
            knowledge[code_move_encode]['visit'] = 1
            knowledge[code_move_encode]['children'] = []
            knowledge_map_val[code_move_encode] = 0
        
        if iter_hist % 2 == 0:
            #knowledge[code_move_parent]['value'] += win_score
            knowledge_map_val[code_move_parent_encode] += win_score
            knowledge_map_val[code_move_encode] += win_score
        else:
            knowledge_map_val[code_move_parent_encode] += lose_score
            knowledge_map_val[code_move_encode] += lose_score

#################
#learn black win#
#################
for iter_move in range(len(all_move_black_win)):
    single_move_hist = all_move_black_win[iter_move]
    #print('cek len', len(single_move_hist))
    #print(single_move_hist)
    history_move = []
    for iter_hist in range(len(single_move_hist)):
        code_move_parent = '|'.join(single_move_hist[0:iter_hist])
        code_move = '|'.join(single_move_hist[0:iter_hist+1])
        code_move_tip = single_move_hist[iter_hist]

        #code_move_parent_encode = encode_name(code_move_parent, encryption_key)
        #code_move_encode = encode_name(code_move, encryption_key)

        code_move_parent_encode = code_move_parent
        code_move_encode = code_move
        code_move_tip_encode = code_move_tip

        try:
            knowledge[code_move_parent_encode]['visit'] += 1
        except KeyError:
            knowledge[code_move_parent_encode] = {}
            knowledge[code_move_parent_encode]['visit'] = 1
            #knowledge[code_move_parent_encode]['value'] = 0
            knowledge[code_move_parent_encode]['children'] = []
            knowledge_map_val[code_move_parent_encode] = 0
        
        if code_move_encode not in knowledge[code_move_parent_encode]['children']:
            knowledge[code_move_parent_encode]['children'].append(code_move_tip_encode)
        
        try:
            knowledge[code_move_encode]['visit'] += 1
        except KeyError:
            knowledge[code_move_encode] = {}
            knowledge[code_move_encode]['visit'] = 1
            knowledge[code_move_encode]['children'] = []
            knowledge_map_val[code_move_encode] = 0
        
        if iter_hist % 2 == 0:
            #knowledge[code_move_parent]['value'] += win_score
            knowledge_map_val[code_move_parent_encode] += lose_score
            knowledge_map_val[code_move_encode] += lose_score
        else:
            knowledge_map_val[code_move_parent_encode] += win_score
            knowledge_map_val[code_move_encode] += win_score

print(knowledge_map_val)
print(knowledge)

with open('knowledge.pkl', 'wb') as f:
    pickle.dump(knowledge, f)

with open('knowledge_map_val.pkl', 'wb') as f:
    pickle.dump(knowledge_map_val, f)
