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

#setting
encryption_key = b'0123456789abcdef'  # Use a valid key size: 16 bytes (128 bits)

def save_knowledge():
    with open('knowledge.pkl', 'wb') as f:
        pickle.dump(knowledge, f)
    with open('knowledge_map_val.pkl', 'wb') as f:
        pickle.dump(knowledge_map, f)

def pad(data):
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    return padder.update(data) + padder.finalize()

def shorten_name(name, key):
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    encryptor = cipher.encryptor()
    padded_data = pad(name.encode('utf-8'))
    print(padded_data)
    encrypted_name = encryptor.update(padded_data) + encryptor.finalize()
    return base64.b64encode(encrypted_name)[:8].decode('utf-8')  # Adjust the length as needed

def decrypt_aes(ciphertext, key):
    key = key.encode('utf-8')
    ciphertext = b64decode(ciphertext)  # Assuming the ciphertext is Base64 encoded

    # Use AES in CBC mode with PKCS7 padding
    cipher = Cipher(algorithms.AES(key), modes.CBC(b'\0' * 16), backend=default_backend())
    decryptor = cipher.decryptor()

    # Decrypt the ciphertext
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()

    return plaintext.decode('utf-8')

with open('knowledge.pkl', 'rb') as f:
    knowledge = pickle.load(f)
with open('knowledge_map_val.pkl', 'rb') as f:
    knowledge_map = pickle.load(f)

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
    res['legal'] = True
    return res

# Running the application on the local development server
if __name__ == '__main__':
    app.run(debug=True)