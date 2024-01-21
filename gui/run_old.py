# Importing the Flask module
from flask import Flask, render_template, request, jsonify
import numpy as np
from io import BytesIO
import pandas as pd
from flask import Flask, send_file

import chess
import chess.pgn

def check_game_status(board):
    if board.is_checkmate():
        winner = "white" if board.turn == chess.BLACK else "black"
    elif board.is_stalemate():
        winner = 'draw'
    elif board.is_insufficient_material():
        winner = 'draw'
    else:
        winner = None
    return winner

# Creating a Flask web application
app = Flask(__name__)

# Defining a route for the home page
@app.route('/')
def home():
    return render_template('index_multiplayer.html')

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
            print('cek non legal move', single_hist)
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

    res['legal'] = True
    return res

# Running the application on the local development server
if __name__ == '__main__':
    app.run(debug=True)