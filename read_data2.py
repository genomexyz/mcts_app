import pandas as pd
import re
import numpy as np
import chess
import chess.pgn

#setting
#file_path = 'CHESS_DATA.csv'
file_path = 'lichess_db_standard_rated_2014-07.pgn'
cnt_limit = 10000
cnt = 0

mode_absorb = True
mode_absorb_str = '[Event '
len_absorb = len(mode_absorb_str)
content = ''
with open(file_path, 'r') as file:
    for line in file:
        # Process each line
        print(line.strip())  # Example: Print each line (without leading/trailing whitespaces)
        single_line = line.strip()
        if line == '':
            continue
        if single_line[:len_absorb] == mode_absorb_str:
            content = content.strip()
            mode_absorb = True
            if content != '':
                #extract_data
                print('===================')
                print(content)
                print('===================')
                pass
            content = ''
        content += line
        
        

