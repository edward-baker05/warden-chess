from typing import Optional
import tensorflow as tf
import chess

class Model:
    def __init__(self, model_type: str, phase: str='opening') -> None:
        self.create_model(model_type, phase)
        self.__phase = phase

    def create_model(self, model_type: str, phase: str='opening'):
        """Create and return a TensorFlow model for evaluating chess positions.
        Returns:
            A TensorFlow model.
        """
        if model_type == "complex_large":
            from models.complex_large import create
        elif model_type == "complex_small":
            from models.complex_small import create
        elif model_type == "simple_large":
            from models.simple_large import create
        elif model_type == "simple_small":
            from models.simple_small import create
        else:
            raise ValueError("Invalid model name")

        model = create()

        self.__model = model
    
    def load_phase_weights(self, board: chess.Board) -> str:
        """Check the weights of the given board position.
        Args:
            board: A chess.Board object representing the current board position.
            model: A TensorFlow model.
        Returns:
            A float representing the weight of the given board position.
        """
        game_phase = self.__get_phase(board)
        phase = {'opening': 'neural_net/Players/mcts_engine/weights_opening.h5', 
                'mid': 'neural_net/Players/mcts_engine/weights_mid.h5', 
                'end': 'neural_net/Players/mcts_engine/weights_end.h5'}
        
        if game_phase != self.__phase:
            self.__phase = game_phase
            try:
                print(f"Loading weights for phase: {game_phase}")
                self.__model.load_weights(phase[game_phase])
            except FileNotFoundError:
                print("No weights exist for this phase.")

        return game_phase
    
    def get_model(self) -> tf.keras.Model:
        """Get the model.
        Returns:
            A TensorFlow model.
        """
        return self.__model

    def __get_phase(self, board: chess.Board) -> str:
        """Get the current phase of the game.
        Args:
            board: A chess.Board object representing the current board position.
        Returns:
            A string representing the current phase of the game.
        """
        if len(board.piece_map()) <= 8:
            return 'end'
        elif len(board.move_stack) <= 10:
            return 'opening'
        else:
            return 'mid'