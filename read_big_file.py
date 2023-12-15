import pandas as pd

#setting
file_path = 'CHESS_DATA.csv'
cnt_limit = 1000
cnt = 0

def read_data(filename):
    for segment in pd.read_csv(filename, chunksize=100000):
        yield segment

for df in read_data(file_path):
    print(df.keys())
    break
exit()


#with open(file_path, 'r') as file:
#    for line in file:
#        # Process each line
#        print(line.strip())  # Example: Print each line (without leading/trailing whitespaces)
#        break
        #cnt += 1
        #if cnt > cnt_limit:
        #    break

with open(file_path, 'r') as file:
    all_data = file.read().split('\n')
    for line in all_data:
        print(line.strip())
        break
        #cnt += 1
        #if cnt > cnt_limit:
        #    break