import chess
import math
import pandas as pd

#setting
history_move = []
knowledge = {}

#class Node:
#    def __init__(self, state, parent=None):
#        self.state = state
#        self.parent = parent
#        self.children = []
#        self.visits = 0
#        self.value = 0
#    
#    def uct(self, node):
#        if node.visits == 0:
#            return float('inf')  # Exploring unvisited nodes
#        return (node.value / node.visits) + 1.41 * math.sqrt(math.log(node.parent.visits) / node.visits)
#    
#    def select(self, node):
#        while node.children:
#            node = max(node.children, key=self.uct)
#        return node
    
    #def expand(node):
    #    # Simple rule-based expansion for illustration
    #    possible_responses = get_possible_responses(node.state)
    #    for response in possible_responses:
    #        new_state = generate_new_state(node.state, response)
    #        new_node = Node(new_state, parent=node)
    #        node.children.append(new_node)
    #    return random.choice(node.children)

def get_possible_responses(board):
    return list(board.legal_moves)

board = chess.Board()
cek = board.is_checkmate()
print('cek checkmate', cek, board.turn, chess.WHITE)
all_move = get_possible_responses(board)
print(str(all_move[0]))
psn = board.piece_map