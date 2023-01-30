from typing import Optional
import tensorflow as tf
import chess

class Model:
    def __init__(self, phase: str='opening') -> None:
        self.create_model(phase)
        self.__phase = phase

    def create_model(self, phase: 'str'='opening'):
        """Create and return a TensorFlow model for evaluating chess positions.

        Returns:
            A TensorFlow model.
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu',  input_shape=(8, 8, 12)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=2))
        model.add(tf.keras.layers.Dropout(rate=0.2))

        model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=2))
        model.add(tf.keras.layers.Dropout(rate=0.2))

        model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=2))
        model.add(tf.keras.layers.Dropout(rate=0.2))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=1024, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(rate=0.2))
        model.add(tf.keras.layers.Dense(units=512, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(rate=0.2))
        model.add(tf.keras.layers.Dense(units=1))

        optimiser = tf.keras.optimizers.Adam()
        loss = tf.keras.losses.MeanSquaredError()

        model.compile(optimizer=optimiser, loss=loss, metrics=['accuracy'])
        try:
            model.load_weights(f"C:/Users/ed9ba/Documents/Coding/NEA/Warden/neural_net/Players/mcts_engine/weights_{phase}.h5")
        except FileNotFoundError:
            print("No weights file found. Creating new model.")
        print("Model created.")

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
