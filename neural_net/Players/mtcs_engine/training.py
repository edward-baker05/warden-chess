import chess
import numpy as np
import tensorflow as tf
from mtcs_engine import MonteCarloEngine, board_to_tensor, create_model

def train(nn_model: tf.keras.Model, games_num: int=100, epochs: int=10):
    # Define the optimizer and loss function for the neural network
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    engine = MonteCarloEngine()
    
    training_data = []
    outcomes = []

    try:
        for epoch in range(epochs):
            for i in range(games_num):
                # Initialize the chess board
                board = chess.Board()
                
                # Play a game between the engine and itself
                positions, outcome = play_game(board, engine)
                training_data.append(positions)
                outcomes.append(outcome)
                print(f"Game {i} complete")

            # Train the neural network on the positions and outcomes
            with tf.GradientTape() as tape:
                predictions = nn_model(training_data, training=True)
                loss_value = loss_fn(outcomes, predictions)
            grads = tape.gradient(loss_value, nn_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, nn_model.trainable_weights))
    except KeyboardInterrupt:
        pass

    # Save the new model weights
    nn_model.save_weights(r"neural_net\Players\mtcs_engine\weights.h5")
    print("Saved weights to disk")

def play_game(board: chess.Board, engine: MonteCarloEngine) -> tuple[list[np.ndarray], list[int]]:
    # Initialize the game result
    move_history = []
    one_hot_outcome = [0, 0]

    # Play the game until it is over
    while not board.is_game_over():
        # Get the move from the engine
        move = engine.get_move(board)

        # Make the move on the board
        board.push(move)

        # Add the position and the move to the game history
        move_history.append(board_to_tensor(board))

    if outcome := board.outcome() is not None:
        if outcome == "1-0":
            one_hot_outcome = [1, 0]
        elif outcome == "0-1":
            one_hot_outcome = [0, 1]
        
    return move_history, one_hot_outcome
    
def create_model() -> tf.keras.Model:
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
    model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
    
    optimiser = tf.keras.optimizers.Adam()
    
    model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['accuracy'])
    
    try:
        model.load_weights(r"neural_net\Players\mtcs_engine\weights.h5")
        print("Weights file found. Loading weights.")
    except FileNotFoundError:
        print("No weights file found. Training from scratch.")

    return model

train(create_model())