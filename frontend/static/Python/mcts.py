from __future__ import annotations
from random import shuffle
import sys
sys.path.append("static/Python/")
from transposition import TranspositionTable
from model import Model
from node import Node
import chess
import chess.polyglot
import chess.syzygy
import numpy as np

class MonteCarloEngine:
    def __init__(self, colour: int=chess.BLACK, temperature: float=0.5, iterations: int=25000, max_depth: int=20) -> None:
        """Initialize the Monte Carlo engine.

        Args:
            colour: the chess.WHITE or chess.BLACK value representing the side the engine is playing.
            temperature: a float value representing the temperature parameter for the softmax function.
            iterations: the number of iterations to run the Monte Carlo tree search.
            max_depth: the maximum depth to search in the tree.
        """
        self.model = Model()

        print(f"Creating AI of colour {colour}...")
        
        self.colour = colour
        self.max_depth = max_depth
        self.temperature = temperature
        self.iterations = iterations
        self.game_phase = 'opening'
        self.in_opening = True

        # Create a transposition table
        self.transposition_table = TranspositionTable()

    def search(self, node: Node) -> Node:
        """Perform a Monte Carlo tree search from the given node.

        Args:
            node: the root Node object to start the search from.
            model: the TensorFlow model to use for evaluating positions.

        Returns:
            The selected child Node object to continue the search from.
        """            
        for _ in range(self.iterations):
            leaf_node = self.select_leaf(node)
            value = self.evaluate(leaf_node)
            self.backpropagate(leaf_node, value)

        # Calculate the UCB1 scores for each child
        ucb_scores = [child.ucb1() for child in node.children]

        # Softmax the scores with the temperature parameter
        ucb_scores = np.array(ucb_scores)
        exp_scores = np.exp(ucb_scores / self.temperature)
        probs = exp_scores / sum(exp_scores)

        # Choose a child node according to the softmax probabilities
        index = np.argmax(probs)
        return node.children[index]

    def select_leaf(self, node: Node) -> Node:
        """Traverse the tree from the given node to a leaf node by selecting child nodes using the UCB1 algorithm.

        Args:
            node: the Node object to start the selection from.

        Returns:
            The leaf Node object that was selected.
        """
        while len(node.children) > 0:
            node = max(node.children, key=lambda c: c.ucb1())
            if node.depth == self.max_depth:
                return node
        return node

    def evaluate(self, node: Node) -> float:
        """Evaluate the given node using the TensorFlow model.

        Args:
            node: the Node object to evaluate.
            model: the TensorFlow model to use for evaluating positions.

        Returns:
            The evaluation of the node as a float value.
        """
        # Check if the position is in the transposition table
        value = self.transposition_table.get(node.board)
        
        # Check if it is checkmate
        if node.board.is_checkmate():
            if node.board.turn != self.colour:
                value = float('inf')  
        elif len(node.board.piece_map()) <= 5:
            print("Probing tablebase...")
            tb = chess.syzygy.open_tablebase("static/Python/tablebase")
            outcome = tb.get_wdl(node.board)
            tb.close()
            match outcome:
                case 2:
                    value = float('inf')
                case 1:
                    value = 100
                case 0:
                    value = 0
                case -1:
                    value = -100
                case -2:
                    value = -float('inf')
        elif value is None:
            self.game_phase = self.model.load_phase_weights(node.board)
            working_model = self.model.get_model()
        
            tensor = board_to_tensor(node.board).reshape(1, 8, 8, 12)
            value = working_model.__call__(tensor).numpy()[0][0]

        # Store the value in the transposition table
        self.transposition_table.put(node.board, value)

        if node.board.turn != self.colour:
            print("Inverting value...")
            return -value
        print("Not inverting value...")
        return value

    def backpropagate(self, node: Node, value: float) -> None:
        """Backpropagate the evaluation value through the tree to update the visit and win counts for each node.

        Args:
            node: the leaf Node object to start the backpropagation from.
            value: the evaluation value to backpropagate.
        """
        while node is not None:
            node.visits += 1
            if value > 0:
                node.wins += 1
            # node.value = node.wins / node.visits
            node.value = value
            node = node.parent

    def get_move(self, board: chess.Board) -> chess.Move:
        """Get the best move for the given board position according to the Monte Carlo tree search.

        Args:
            board: a chess.Board object representing the current board position.
            model: the TensorFlow model to use for evaluating positions.

        Returns:
            A chess.Move object representing the best move for the given board position.
        """
        if self.in_opening:
            with chess.polyglot.open_reader("static/Python/polyglot/gm2600.bin") as reader:
                move = reader.get(board)
                if move:
                    move = move.move
                    return move
                self.in_opening = False
                print(self.in_opening)
                print("No longer in opening preparation phase.")
                self.model.load_phase_weights(board)

        # Create a root node for the MCTS tree
        root_node = Node(board)

        # Add the legal moves for the position represented by the root node as children of the root node
        for move in root_node.board.legal_moves:
            root_node.add_child(move)
        shuffle(root_node.children)

        for child in root_node.children:
            if len(child.board.piece_map()) <= 5:
                print("Probing tablebase...")
                tablebase = chess.syzygy.open_tablebase("static/Python/tablebase")
                outcome = tablebase.get_wdl(child.board)
                tablebase.close()
                if self.colour == chess.WHITE:
                    target = 2
                else:
                    target = -2
                print(outcome)
                if outcome == target:
                    move = child.board.peek()
                    print(f"Playing move {move} to win the game from tablebase")
                    return move

        print("Getting move from neural network...")
        # Search the MCTS tree and choose the child node with the highest UCB1 score as the next move
        chosen_node = self.search(root_node)
        print(chosen_node.board.fen)
        move = chosen_node.board.peek()
        print("########## Move chosen ##########")
        print(f"AI made move: {move}. This was move number {list(root_node.board.legal_moves).index(move)} of {len(list(root_node.board.legal_moves))}")
        return move

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Convert the given board position to a tensor.

    Args:
        board: A chess.Board object representing the current board position.

    Returns:
        A numpy array representing the board position as a tensor.
    """
    # Convert the board to a tensor
    tensor = np.zeros((8, 8, 12))
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(i * 8 + j)
            if piece is not None:
                if piece.color:
                    tensor[i][j][piece.piece_type-1] = 1
                else:
                    tensor[i][j][piece.piece_type + 5] = 1
    return tensor