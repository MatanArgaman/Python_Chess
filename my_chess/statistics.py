import argparse
import chess.pgn
import os
import pickle

from my_chess.shared.shared_functionality import StatValues


def create_new_player_stats(stats, player):
    if player in stats:
        return
    stats[player]= {}
    stats[player]['wins']=0
    stats[player]['losses']=0
    stats[player]['draws']=0

def winning_stats(path, file_name, stats):
        c=0
        failed = 0
        full_path = os.path.join(path, file_name)
        fails = 0
        with open(full_path) as pgn:
            while True:
                try:
                    game = chess.pgn.read_game(pgn)
                except:
                    failed += 1
                    continue
                if game is None:
                    break

                c += 1
                if c % 500 == 0:
                    print('parsing game #:', c, "failed:", failed)
                if c>1000:
                    break
                results = game.headers['Result']
                white_player = game.headers['White'].lower()
                black_player  = game.headers['Black'].lower()
                for player_name in [white_player, black_player]:
                    stats[player_name] = stats.get(player_name, {'wins':0, 'draws':0, 'losses':0})
                if results=='1-0':
                    stats[white_player]['wins'] = stats[white_player]['wins']+1
                    stats[black_player]['losses'] = stats[black_player]['losses'] +1
                elif results=='0-1':
                    stats[black_player]['wins'] = stats[black_player]['wins']+1
                    stats[white_player]['losses'] = stats[white_player]['losses'] +1
                elif results=='1/2-1/2':
                    stats[black_player]['draws'] = stats[black_player]['draws']+1
                    stats[white_player]['draws'] = stats[white_player]['draws'] +1
                else:
                    print('results:',results, game.headers)

        print('total:', c, file_name)
        return stats

def move_stats(path, file_name, stats, mod_index=0):
    # stats format:
    # {board : {move: {'wins':int, 'draws':int, 'losses':int}}} where board is the fen representation of the move and move is the str of move
    c = 0
    full_path = os.path.join(path, file_name)
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
            if c%500==0:
                print(c)
            results = game.headers['Result']

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

    print('total:', c, file_name)


def get_statistics(path, mod_index):
    files = os.listdir(path)
    # filter only pgn files in director
    files = [f for f in files if f[-4:]=='.pgn']
    stats = {}
    for f in files:
        full_path = os.path.join(path, f)
        winning_stats(path, f, stats)
        # move_stats(path, f, stats, mod_index)
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

