import pickle
from my_chess.shared import board_fen_to_hash, board_fen_to_hash384


for i in range(10):
    with open('/home/blacknight/Downloads/stat{}.pkl'.format(i), 'rb') as f:
        a = pickle.load(f)
        c = [{} for j in range(10)]
        for k, v in a.items():
            index = board_fen_to_hash(k) % 10
            c[index][k] = {}
            wins = 0
            for k2, v2 in v.items():
                wins += v2['wins']
            for k2, v2 in v.items():
                if v2['wins'] > 0.01 * wins:
                    c[index][k][k2] = {'w': v2['wins'], 'd': v2['draws'], 'l': v2['losses'],
                                       'r': float(v2['wins'] / (v2['wins'] + v2['losses']))}
            if len(c[index][k].keys())==0:
                del c[index][k]
    for j, d in enumerate(c):
        with open('/home/blacknight/Downloads/stat{0}_{1}.pkl'.format(i, j), 'wb') as f:
            pickle.dump(c[j], f)




for i in range(10):
    with open('/home/blacknight/Downloads/stat0_{0}.pkl'.format(i), 'rb') as f:
        a = pickle.load(f)
    for j in range(1, 10):
        with open('/home/blacknight/Downloads/stat{0}_{1}.pkl'.format(j, i), 'rb') as f:
            b = pickle.load(f)
        for k, v in b.items():
            if k in a:
                for k2, v2 in v.items():
                    if k2 in a[k]:
                        wins = a[k][k2]['w'] + v2['w']
                        losses = a[k][k2]['l'] + v2['l']
                        a[k][k2] = {'w': wins, 'd': a[k][k2]['d'] + v2['d'], 'l': losses,
                                    'r': float(wins) / (wins + losses)}
                    else:
                        a[k][k2] = v2
            else:
                a[k] = v
    with open('/home/blacknight/Downloads/cstat{0}.pkl'.format(i), 'wb') as f:
        pickle.dump(a, f)

# verify1
w, d, l = 0, 0, 0
for i in range(10):
    for j in range(10):
        with open('/home/blacknight/Downloads/stat{0}_{1}.pkl'.format(i, j), 'rb') as f:
            c = pickle.load(f)
            if b.fen() in c:
                w += c[b.fen()]['e2e4']['w']
                d += c[b.fen()]['e2e4']['d']
                l += c[b.fen()]['e2e4']['l']
                print(i, j)

#-5679190281720826738
#verify2
w, d, l = 0, 0, 0
for i in range(10):
    with open('/home/blacknight/Downloads/stat{0}.pkl'.format(i), 'rb') as f:
        c = pickle.load(f)
        w += c[b.fen()]['e2e4']['wins']
        d += c[b.fen()]['e2e4']['draws']
        l += c[b.fen()]['e2e4']['losses']


# repartition such that:
# dstat{0}_{1} indices: board_fen_to_hash % 10,board_fen_to_hash384%10
for i in range(10):
    with open('/home/blacknight/Downloads/cstat{0}.pkl'.format(i), 'rb') as f:
        a = pickle.load(f)
        b = [{} for j in range(10)]
        for k, v in a.items():
            index = board_fen_to_hash384(k) % 10
            b[index][k] = v
    for j in range(10):
        with open('/home/blacknight/Downloads/dstat{0}_{1}.pkl'.format(i, j), 'wb') as f:
            pickle.dump(b[j], f)
