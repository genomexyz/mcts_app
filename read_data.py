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

def read_data(filename):
    for segment in pd.read_csv(filename, chunksize=100000):
        yield segment

# Function to extract values using regex
def extract_value(pattern, data):
    match = pattern.search(data)
    return match.group(1) if match else None


pgn_data = """
[Event "Rated Classical game"]
[Site "https://lichess.org/hys9pdz0"]
[White "vippp82"]
[Black "aadajeniuc"]
[Result "1-0"]
[UTCDate "2014.06.30"]
[UTCTime "22:21:32"]
[WhiteElo "1453"]
[BlackElo "1441"]
[WhiteRatingDiff "+12"]
[BlackRatingDiff "-14"]
[ECO "C44"]
[Opening "Scotch Game: Benima Defense"]
[TimeControl "300+8"]
[Termination "Normal"]
"""

# Define regex patterns for each value
event_pattern = re.compile(r'\[Event "(.*?)"\]')
site_pattern = re.compile(r'\[Site "(.*?)"\]')
white_pattern = re.compile(r'\[White "(.*?)"\]')
black_pattern = re.compile(r'\[Black "(.*?)"\]')
result_pattern = re.compile(r'\[Result "(.*?)"\]')
utc_date_pattern = re.compile(r'\[UTCDate "(.*?)"\]')
utc_time_pattern = re.compile(r'\[UTCTime "(.*?)"\]')
white_elo_pattern = re.compile(r'\[WhiteElo "(.*?)"\]')
black_elo_pattern = re.compile(r'\[BlackElo "(.*?)"\]')
white_rating_diff_pattern = re.compile(r'\[WhiteRatingDiff "(.*?)"\]')
black_rating_diff_pattern = re.compile(r'\[BlackRatingDiff "(.*?)"\]')
eco_pattern = re.compile(r'\[ECO "(.*?)"\]')
opening_pattern = re.compile(r'\[Opening "(.*?)"\]')
time_control_pattern = re.compile(r'\[TimeControl "(.*?)"\]')
termination_pattern = re.compile(r'\[Termination "(.*?)"\]')

all_pattern = [event_pattern, site_pattern, white_pattern, black_pattern, result_pattern, utc_date_pattern, utc_time_pattern, white_elo_pattern, black_elo_pattern,
               white_rating_diff_pattern, black_rating_diff_pattern, eco_pattern, opening_pattern, time_control_pattern, termination_pattern]

all_pattern_str = ['event', 'site', 'white', 'black', 'result', 'utc_date', 'utc_time', 'white_elo', 'black_elo', 'white_ratting_diff', 'black_rating_diff', 'eco', 'opening',
                   'time_control', 'termination']

def create_stat():
    move_stat = {}
    for iter_pat in range(len(all_pattern_str)):
        move_stat[all_pattern_str[iter_pat]] = None
    move_stat['white_move'] = []
    move_stat['black_move'] = []
    return move_stat

# Extract values
#event = extract_value(event_pattern, pgn_data)
#site = extract_value(site_pattern, pgn_data)
#white = extract_value(white_pattern, pgn_data)
#black = extract_value(black_pattern, pgn_data)
#result = extract_value(result_pattern, pgn_data)
#utc_date = extract_value(utc_date_pattern, pgn_data)
#utc_time = extract_value(utc_time_pattern, pgn_data)
#white_elo = extract_value(white_elo_pattern, pgn_data)
#black_elo = extract_value(black_elo_pattern, pgn_data)
#white_rating_diff = extract_value(white_rating_diff_pattern, pgn_data)
#black_rating_diff = extract_value(black_rating_diff_pattern, pgn_data)
#eco = extract_value(eco_pattern, pgn_data)
#opening = extract_value(opening_pattern, pgn_data)
#time_control = extract_value(time_control_pattern, pgn_data)
#termination = extract_value(termination_pattern, pgn_data)
#
## Print extracted values
#print(f"Event: {event}")
#print(f"Site: {site}")
#print(f"White: {white}")
#print(f"Black: {black}")
#print(f"Result: {result}")
#print(f"UTC Date: {utc_date}")
#print(f"UTC Time: {utc_time}")
#print(f"White Elo: {white_elo}")
#print(f"Black Elo: {black_elo}")
#print(f"White Rating Diff: {white_rating_diff}")
#print(f"Black Rating Diff: {black_rating_diff}")
#print(f"ECO: {eco}")
#print(f"Opening: {opening}")
#print(f"Time Control: {time_control}")
#print(f"Termination: {termination}")

all_move_stat = []
move_stat = create_stat()
#check_stat = [0] * len(all_pattern_str)
#check_stat = np.array(check_stat)
with open(file_path, 'r') as file:
    for line in file:
        # Process each line
        print(line.strip())  # Example: Print each line (without leading/trailing whitespaces)
        single_line = line.strip()
        if line == '':
            continue
        stat_detected = False
        for iter_pattern in range(len(all_pattern)):
            val = extract_value(all_pattern[iter_pattern], single_line)
            if val is None:
                continue
            move_stat[all_pattern_str[iter_pattern]] = val
            stat_detected = True
        if not stat_detected:
            # extract move
            pass
        #break