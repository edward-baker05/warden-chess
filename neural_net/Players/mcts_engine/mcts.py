from __future__ import annotations
import chess
import chess.polyglot
import chess.syzygy
import numpy as np
from Players.mcts_engine.transposition import TranspositionTable
from Players.mcts_engine.node import Node


class MonteCarloEngine:
    def __init__(self, colour: int, model_type: str, temperature: float = 0.2, iterations: int = 50000, max_depth: int = 25) -> None:
        """
        Initialize the Monte Carlo engine.

        Args:
            colour: the chess.WHITE or chess.BLACK value representing the side the engine is playing.
            temperature: a float value representing the temperature parameter for the softmax function.
            iterations: the number of iterations to run the Monte Carlo tree search.
            max_depth: the maximum depth to search in the tree.
        """
        # Create a TensorFlow model
        self.__model = self.__create_model(model_type)
        self.transposition_table = TranspositionTable()
        
        self.__colour = colour
        self.__max_depth = max_depth
        self.__temperature = temperature
        self.__iterations = iterations
        self.__in_opening = False
        
        
    def __create_model(self, model_type: str) -> tf.keras.Sequential:
        """
        Create a TensorFlow model based on the given model type.
        """
        # Match the model type to a function that creates the model
        match model_type:
            case "simple_small":
                from Players.mcts_engine.models.simple_small import create_model
            case "simple_large":
                from Players.mcts_engine.models.simple_large import create_model
            case "complex_small":
                from Players.mcts_engine.models.complex_small import create_model
            case "complex_large":
                from Players.mcts_engine.models.complex_large import create_model
            case _:
                raise ValueError("Invalid model name")
            
        # Return the model
        return create_model()

    def __search(self, node: Node) -> Node:
        """
        Run the Monte Carlo tree search from the given node.
        Args:
            node: the Node object to start the search from.
        Returns:
            The Node object that was selected.
        """
        for _ in range(self.__iterations):
            leaf_node = self.__select_leaf(node)
            value = self.__evaluate(leaf_node)
            self.__backpropagate(leaf_node, value)

        # Calculate the UCB1 scores for each child
        ucb_scores = [child.ucb1() for child in node.children]

        # Softmax the scores with the temperature parameter
        ucb_scores = np.array(ucb_scores)
        exp_scores = np.exp(ucb_scores / self.__temperature)
        probs = exp_scores / sum(exp_scores)

        # Choose a child node according to the softmax probabilities
        index = np.argmax(probs)
        return node.children[index]

    def __select_leaf(self, node: Node) -> Node:
        """
        Traverse the tree from the given node to a leaf node by selecting child nodes using the UCB1 algorithm.

        Args:
            node: the Node object to start the selection from.

        Returns:
            The leaf Node object that was selected.
        """
        while len(node.children) > 0:
            node = max(node.children, key=lambda c: c.ucb1())
            if node.depth == self.__max_depth:
                return node
        return node

    def __evaluate(self, node: Node) -> float:
        """
        Evaluate the given node using the TensorFlow model.

        Args:
            node: the Node object to evaluate.
            model: the TensorFlow model to use for evaluating positions.

        Returns:
            The evaluation of the node as a float value.
        """
        # Check if it is checkmate
        if node.board.is_checkmate():
            if node.board.turn != self.__colour:
                return float('inf')
            return -float('inf')

        # Check if the position is in the transposition table
        value = self.transposition_table.get(node.board)
        if value is not None:
            if node.board.turn != self.__colour:
                return -value
            return value

        working_model = self.__model

        tensor = board_to_tensor(node.board).reshape(1, 8, 8, 12)
        value = working_model.__call__(tensor, training=False).numpy()[0][0]

        # Store the value in the transposition table
        self.transposition_table.put(node.board, value)

        if node.board.turn != self.__colour:
            return -value
        return value

    def __backpropagate(self, node: Node, value: float) -> None:
        """
        Backpropagate the evaluation value through the tree to update the visit and win counts for each node.

        Args:
            node: the leaf Node object to start the backpropagation from.
            value: the evaluation value to backpropagate.
        """
        # Update the visit and win counts for each node
        while node is not None:
            node.visits += 1
            if value > 0:
                node.wins += 1
            node.value = node.wins / node.visits
            node = node.parent

    def get_move(self, board: chess.Board) -> chess.Move:
        """
        Get the best move for the given board position according to the Monte Carlo tree search.

        Args:
            board: a chess.Board object representing the current board position.
            model: the TensorFlow model to use for evaluating positions.

        Returns:
            A chess.Move object representing the best move for the given board position.
        """
        # Check if the position is in the opening book
        if self.__in_opening:
            with chess.polyglot.open_reader("neural_net/Players/mcts_engine/polyglot/DCbook_large.bin") as reader:
                move = reader.get(board)
                if move:
                    return move.move
                print("No longer in opening prep.")
                self.__in_opening = False
        
        # Check if the position is in the endgame tablebase
        if len(board.move_stack) < 5:
            options = []
            with chess.syzygy.open_tablebase("neural_net/Players/mcts_engine/syzygy") as tablebase:
                for move in board.legal_moves:
                    options.append(tablebase.probe_dtz(board))
            return list(board.legal_moves)[np.argmax(options)]
        
        # Create a root node for the MCTS tree
        root_node = Node(board)

        # Add the legal moves for the position represented by the root node as children of the root node
        for move in root_node.board.legal_moves:
            root_node.add_child(move)

        # Search the MCTS tree and choose the child node with the highest UCB1 score as the next move
        chosen_node = self.__search(root_node)
        move = chosen_node.board.peek()
        print(
            f"AI made move: {move}. This was move number {list(root_node.board.legal_moves).index(move)} of {len(list(root_node.board.legal_moves))}")
        return move


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Convert the given board position to a tensor.

    Args:
        board: A chess.Board object representing the current board position.

    Returns:
        A numpy array representing the board position as a tensor.
    """
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
