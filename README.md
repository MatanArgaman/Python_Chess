

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

Download the nn weights:
https://drive.google.com/drive/folders/1iMF6H7JiasNJ-Db5j5LJPgaV3tOIJ-n-?usp=sharing
Set config.json "torch_nn_path" to the backbone .pth weights path (largest file).
Make sure the heads weights (_policy_network.pth, _value_network.pth)
are also in the same folder.

run play.py, standard run parameters to play against nn with mcts as white:
--nn --mcts --whuman
standard run parameters to play against nn with mcts as black:
--nn --mcts --bhuman

When playing your games are automatically saved to the my_chess/games folder, both an image of the board and its fen
are saved each turn.

You can resume a game from the saved fens by adding these parameters:
-board  "<your-fen>"
e.g:
-board  "8/4Q3/8/7k/5K2/8/8/8 w - - 0 1"
