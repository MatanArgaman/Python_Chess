import pickle
import tqdm
from multiprocessing import Pool
import multiprocessing
import chess
import numpy as np
from scipy.sparse import save_npz, csr_matrix

from predict import get_input_representation, get_output_representation
from shared.shared_functionality import board_fen_to_hash, board_fen_to_hash384, position_to_mirror_position, move_to_mirror_move
from shared.shared_functionality import OUTPUT_PLANES, INPUT_PLANES

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
            if len(c[index][k].keys()) == 0:
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

# -5679190281720826738
# verify2
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


# from scipy.sparse import save_npz
# save_npz('/home/blacknight/Downloads/t2.npz',o2)

# from scipy.sparse import load_npz
# o4=load_npz('/home/blacknight/Downloads/t2.npz')
# o3=o4.toarray().reshape([8,8,-1])
# (o==o3).all()

def create_input_output_representation(indices):
    index1, index2 = indices
    with open('/home/blacknight/Downloads/dstat{0}_{1}.pkl'.format(index1, index2), 'rb') as f:
        d = pickle.load(f)
    input_arr = np.zeros([8, 8, INPUT_PLANES * len(d.items())], dtype=np.float)
    output_arr = np.zeros([8, 8, OUTPUT_PLANES * len(d.items())], dtype=np.float)
    for i, (fen, value) in enumerate(d.items()):
        b = chess.Board(fen)
        moves_and_probabilities = [(k, v['r']) for k, v in value.items()]

        if not b.turn: # if black's turn then mirror board and moves
            b = b.mirror()
            moves = np.array([move_to_mirror_move(m[0]) for m in moves_and_probabilities])
        else:
            moves = np.array([m[0] for m in moves_and_probabilities])
        probabilities = np.array([m[1] for m in moves_and_probabilities])
        probabilities = np.square(probabilities)  # gives higher probabilities more preference
        probabilities /= probabilities.sum()  # normalize
        input_arr[..., i * INPUT_PLANES:(i + 1) * INPUT_PLANES] = get_input_representation(b, 0)
        output_arr[..., i * OUTPUT_PLANES:(i + 1) * OUTPUT_PLANES] = get_output_representation(moves, probabilities, b)
    sparse_output_arr = csr_matrix(output_arr.reshape([8,-1]))
    sparse_input_arr = csr_matrix(input_arr.reshape([8,-1]))
    save_npz('/home/blacknight/Downloads/estat{0}_{1}_i.npz'.format(index1, index2), sparse_input_arr)
    save_npz('/home/blacknight/Downloads/estat{0}_{1}_o.npz'.format(index1, index2), sparse_output_arr)


def create_input_output_representation_with_win_probability(indices):
    index1, index2 = indices
    with open('/home/blacknight/Downloads/dstat{0}_{1}.pkl'.format(index1, index2), 'rb') as f:
        d = pickle.load(f)
    input_arr = np.zeros([8, 8, INPUT_PLANES * len(d.items())], dtype=np.float)
    output_arr = np.zeros([8, 8, OUTPUT_PLANES * len(d.items())], dtype=np.float)
    value = np.zeros([len(d.items())], dtype=np.float)
    for i, (fen, item) in enumerate(d.items()):
        b = chess.Board(fen)
        moves_and_probabilities =[]
        wins =  0
        draws = 0
        losses = 0
        for k,v in item.items():
            moves_and_probabilities.append((k, v['r']))
            wins+= v['w']
            losses+= v['l']
            draws+= v['d']

        played  = wins + losses + draws
        value[i] = (float(wins-losses)/played) # range from -1 to 1

        if not b.turn: # if black's turn then mirror board and moves
            b = b.mirror()
            moves = np.array([move_to_mirror_move(m[0]) for m in moves_and_probabilities])
        else:
            moves = np.array([m[0] for m in moves_and_probabilities])
        probabilities = np.array([m[1] for m in moves_and_probabilities])
        probabilities = np.square(probabilities)  # gives higher probabilities more preference
        probabilities /= probabilities.sum()  # normalize
        input_arr[..., i * INPUT_PLANES:(i + 1) * INPUT_PLANES] = get_input_representation(b, 0)
        output_arr[..., i * OUTPUT_PLANES:(i + 1) * OUTPUT_PLANES] = get_output_representation(moves, probabilities, b)
    sparse_output_arr = csr_matrix(output_arr.reshape([8,-1]))
    sparse_input_arr = csr_matrix(input_arr.reshape([8,-1]))
    save_npz('/home/blacknight/Downloads/estat{0}_{1}_i.npz'.format(index1, index2), sparse_input_arr)
    save_npz('/home/blacknight/Downloads/estat{0}_{1}_o.npz'.format(index1, index2), sparse_output_arr)
    with open('/home/blacknight/Downloads/estat{0}_{1}_v.pkl'.format(index1, index2), 'wb') as f:
        pickle.dump(value, f)






indices = []
for i in range(10):
    for j in range(10):
        indices.append(([i,j]))
with Pool(multiprocessing.cpu_count()//2) as p:
  for _ in tqdm.tqdm(p.imap(create_input_output_representation, indices), total=len(indices)):
      pass
