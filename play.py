import chess

board = chess.Board()
print(board.legal_moves)
print(board)

Nf3 = chess.Move.from_uci("g1f3")
board.push(Nf3)  # Make the move

Nf3 = chess.Move.from_uci("e7e5")
board.push(Nf3)  # Make the move
print(board)
print(board.peek())
print(board.move_stack)