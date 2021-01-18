import argparse
import chess.pgn
import os
import pickle

from my_chess.shared import StatValues


def create_new_player_stats(stats, player):
    if player in stats:
        return
    stats[player]= {}
    stats[player]['wins']=0
    stats[player]['losses']=0
    stats[player]['draws']=0

def winning_stats(path, file_name, stats):
        c=0
        full_path = os.path.join(path, file_name)
        player_name = file_name[:-4].lower()
        create_new_player_stats(stats, player_name)
        fails = 0
        with open(full_path) as pgn:
            while True:

                try:
                    game = chess.pgn.read_game(pgn)
                except:
                    continue
                if game is None:
                    break

                c += 1
                results = game.headers['Result']
                white_player = game.headers['White'].lower()
                black_player  = game.headers['Black'].lower()
                if (player_name not in white_player.lower()) and (player_name not in black_player.lower()):
                    fails+=1
                    if fails >= 0.5 * c and c >= 100:  # skip competitions
                        break

                    continue
                if results=='1-0':
                    if (player_name in white_player): # player name can appear differently in file, e.g: 'Kasparov, Gary' and 'Kasparov,G'
                        stats[player_name]['wins'] = stats[player_name]['wins']+1
                    if (player_name in black_player):
                        stats[player_name]['losses'] = stats[player_name]['losses'] +1
                elif results=='0-1':
                    if (player_name in white_player):
                        stats[player_name]['losses'] = stats[player_name]['losses']+1
                    if (player_name in black_player):
                        stats[player_name]['wins'] = stats[player_name]['wins'] +1
                elif results=='1/2-1/2':
                    if (player_name in white_player):
                        stats[player_name]['draws'] = stats[player_name]['draws']+1
                    if (player_name in black_player):
                        stats[player_name]['draws'] = stats[player_name]['draws'] +1
                else:
                    print('results:',results, game.headers)

        if fails>=0.1*c:
            print('Failed', player_name, fails,'/', c)
        print('total:', c, file_name)

def move_stats(path, file_name, stats, mod_index=0):
    # stats format:
    # {board : {move: {'wins':int, 'draws':int, 'losses':int}}} where board is the fen representation of the move and move is the str of move
    c = 0
    full_path = os.path.join(path, file_name)
    player_name = file_name[:-4].lower()
    fails = 0
    print('starting parsing of game:', player_name)
    with open(full_path) as pgn:
        while True:

            try:
                game = chess.pgn.read_game(pgn)
            except:
                continue
            if game is None:
                break

            c += 1
            if c%500==0:
                print(c)
            results = game.headers['Result']
            white_player = game.headers['White'].lower()
            black_player = game.headers['Black'].lower()
            if (player_name not in white_player.lower()) and (player_name not in black_player.lower()):
                fails += 1
                if fails >= 0.5 * c and c >= 100:  # skip competitions
                    break
                continue

            if c%10!=mod_index: # record 1 out of every 50 games
                continue

            board = chess.Board()
            if results == '1-0':
                for m in game.mainline_moves():
                    move_dict = stats.get(board.fen(),{})
                    stats[board.fen()] = move_dict
                    stat_values = move_dict.get(str(m),{'wins':0, 'draws':0, 'losses':0})
                    if board.turn:
                        stat_values['wins']+=1
                    else:
                        stat_values['losses'] += 1
                    move_dict[str(m)] = stat_values
                    board.push(m)
            elif results == '0-1':
                for m in game.mainline_moves():
                    move_dict = stats.get(board.fen(),{})
                    stats[board.fen()] = move_dict
                    stat_values = move_dict.get(str(m),{'wins':0, 'draws':0, 'losses':0})
                    if not board.turn:
                        stat_values['wins']+=1
                    else:
                        stat_values['losses'] += 1
                    move_dict[str(m)] = stat_values
                    board.push(m)
            elif results == '1/2-1/2':
                for m in game.mainline_moves():
                    move_dict = stats.get(board.fen(),{})
                    stats[board.fen()] = move_dict
                    stat_values = move_dict.get(str(m),{'wins':0, 'draws':0, 'losses':0})
                    stat_values['draws']+=1
                    move_dict[str(m)] = stat_values
                    board.push(m)

    if fails >= 0.1 * c:
        print('Failed', player_name, fails, '/', c)
    print('total:', c, file_name)


def get_statistics(path, mod_index):
    files = os.listdir(path)
    # filter only pgn files in director
    files = [f for f in files if f[-4:]=='.pgn']

    stats = {}
    for f in files:
        full_path = os.path.join(path, f)
        # winning_stats(path, f, stats)
        move_stats(path, f, stats, mod_index)
    return stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path')
    parser.add_argument('-save-path', default=None)
    args = parser.parse_args()
    for i in range(10):
        statistics = get_statistics(args.file_path, i)
        if args.save_path is not None:
            with open(args.save_path[:-4] + str(i)+args.save_path[-4:], 'wb') as f:
                pickle.dump(statistics, f)

    print(statistics)

