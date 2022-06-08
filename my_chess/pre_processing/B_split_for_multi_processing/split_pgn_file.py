import argparse
import os
import multiprocessing

from pre_processing.A_Data_Statistics.numer_of_games_in_file import number_of_games

"""
Splits a single pgn file into multiple pgn files so they can each be processed separately.
Assumes that there are 2 empty lines between each game and the next.
"""

def split_file_into_multiple_files(path, number_of_files):
    games = number_of_games(path)
    number_of_games_per_file = games//number_of_files
    with open(path, encoding="ISO-8859-1") as fp:
        file = fp.readlines()
    c=0
    game_counter = 0
    buffer = []
    for line in file:
        buffer.append(line)
        if line == '\n':
            c += 1
        if (c//2) >= number_of_games_per_file:
            game_path = os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0] + '_'+ str(game_counter)+'.pgn')
            with open(game_path, 'w') as fp:
                fp.writelines(buffer)
            game_counter+=1
            c = 0
            buffer = []

    return number_of_files



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('-number_of_files', help='number of files to split the pgn into',
                        default=multiprocessing.cpu_count()-1)
    args = parser.parse_args()
    print("number of games:", split_file_into_multiple_files(args.path, args.number_of_files))
