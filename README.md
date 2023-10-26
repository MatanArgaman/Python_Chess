<h1><center>AlphaGo for Chess</center></h1>
<img src="https://repository-images.githubusercontent.com/330721108/4e6591b3-851e-4658-aa95-43eb8054a519" alt="AlphaGo for Chess">
<br>
<h4>AlphaGo for Chess is a chess AI agent that implements a deep neural network with the the combination of the MCTS algorithm. <br><br> This work follows the main ideas in the AlphaGo Zero article.<br><br>
The work resulted in an agent with ~2000 elo rating and was tested against stockfish (strongest available open source AI) manually.
</h4>
<br><br>
<b><h3>
Check out my blog explaining the work in detail  
<a href="https://wordpress.com/post/matanargaman.wordpress.com/191">here</a>

</h3></b>
<br><br><br>
Expert chess games taken from: http://caissabase.co.uk/

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
Make sure the heads' weights (_policy_network.pth, _value_network.pth)
are also in the same folder.

run play.py, standard run parameters to play against nn with mcts as white:<br>
--whuman<br>
standard run parameters to play against nn with mcts as black:<br>
--bhuman<br>

When playing your games are automatically saved to the my_chess/games folder, both an image of the board and its fen
are saved each turn.

You can resume a game from the saved fens by adding these parameters:<br>
-board  "&lt;your-fen&gt;"<br>
e.g:
-board  "8/4Q3/8/7k/5K2/8/8/8 w - - 0 1"


<br>You can undo your move by clicking the right mouse button. 
<br>Note that when playing against a player this will undo a the last move only but when
playing against the AI it will undo both the player's last move and the AI's.
