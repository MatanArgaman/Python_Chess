
![alt text](https://media.licdn.com/dms/image/sync/D4D27AQGlB1bsrSDJbQ/articleshare-shrink_800/0/1697123995922?e=1697731200&v=beta&t=WI0w5sSiM9IIbFGFJMpDs5q-fhLThIDm2Ev0bEWmn_s)

Expert chess games taken from:
http://caissabase.co.uk/

Representation implemented according to alphazero article.

resnet architecture implemented as explained here(alphazero):

"A general reinforcement learning algorithm that
masters chess, shogi and Go through self-play"

https://discovery.ucl.ac.uk/id/eprint/10069050/
https://discovery.ucl.ac.uk/id/eprint/10069050/1/alphazero_preprint.pdf

run requirements:
install requirements.txt file.

<b><u>Playing</u></b>

Download the nn weights:<br>
https://drive.google.com/drive/folders/1iMF6H7JiasNJ-Db5j5LJPgaV3tOIJ-n-?usp=sharing

The elo ranking is currently ~2000 - tested against stockfish at https://chessui.com/.

<br>Set config.json "torch_nn_path" to the backbone .pth weights path (largest file).
Make sure the heads weights (_policy_network.pth, _value_network.pth)
are also in the same folder.

run play.py, standard run parameters to play against nn with mcts as white:<br>
--nn --mcts --whuman<br>
standard run parameters to play against nn with mcts as black:<br>
--nn --mcts --bhuman<br>

When playing your games are automatically saved to the my_chess/games folder, both an image of the board and its fen
are saved each turn.

You can resume a game from the saved fens by adding these parameters:<br>
-board  "&lt;your-fen&gt;"<br>
e.g:
-board  "8/4Q3/8/7k/5K2/8/8/8 w - - 0 1"
