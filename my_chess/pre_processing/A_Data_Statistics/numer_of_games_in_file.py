import argparse

"""
Assumes that there are 2 empty lines between each game and the next.
"""

def number_of_games(path):
    with open(path, encoding="ISO-8859-1") as fp:
        file = fp.readlines()
    c=0
    for line in file:
        if line=='\n':
            c+=1
    return c//2



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    print("number of games:", number_of_games(args.path))
