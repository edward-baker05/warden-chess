import tensorflow as tf
import chess


class Model:
    def __init__(self, model_type: str, phase: str = 'opening') -> None:
        self.__model = None
        self.__phase = phase
        self.create_model(model_type)

    def create_model(self, model_type: str):
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

        # new_model = create()
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(8, 8, 12)))
        model.add(tf.keras.layers.MaxPooling2D((2,2)))
        model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dense(units=1, activation='softmax'))
        optimiser = tf.keras.optimizers.Adam()
        loss = tf.keras.losses.MeanSquaredError()

        model.compile(optimizer=optimiser, loss=loss, metrics=['mse', 'mae', 'accuracy'])

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
