

Expert chess games taken from:
http://caissabase.co.uk/

Representation implemented according to alphazero article.

resnet architecture implemented as explained here(alphazero):

"A general reinforcement learning algorithm that
masters chess, shogi and Go through self-play"

https://discovery.ucl.ac.uk/id/eprint/10069050/
https://discovery.ucl.ac.uk/id/eprint/10069050/1/alphazero_preprint.pdf

<b><u>Playing</u></b>

Set config.json "torch_nn_path" to the backbone .pth weights path.
Make sure the heads weights (_policy_network.pth, _value_network.pth)
are also in the same folder.

standard run parameters to play against nn with mcts:
--nn --mcts -whuman