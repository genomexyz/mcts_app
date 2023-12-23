import chess
import numpy as np

def encode_board(board):
    encoding = np.zeros((8, 8), dtype=np.int8)

    piece_mapping = {'p': -1, 'r': -2, 'n': -3, 'b': -4, 'q': -5, 'k': -6,
                     'P': 1, 'R': 2, 'N': 3, 'B': 4, 'Q': 5, 'K': 6}

    for square, piece in board.piece_map().items():
        row, col = chess.square_rank(square), chess.square_file(square)
        encoding[row, col] = piece_mapping[piece.symbol()]

    # Additional features
    #encoding = np.concatenate([encoding.flatten(), [board.has_kingside_castling_rights(chess.WHITE),
    #                                                 board.has_queenside_castling_rights(chess.WHITE),
    #                                                 board.has_kingside_castling_rights(chess.BLACK),
    #                                                 board.has_queenside_castling_rights(chess.BLACK)]])
    encoding = encoding.flatten()

    return encoding

board = chess.Board()
pos = encode_board(board)

print(pos)
print(np.shape(pos))
print(board.turn)