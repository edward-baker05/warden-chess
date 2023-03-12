import chess
import chess.pgn
import numpy as np
import pandas as pd
import tensorflow as tf
from random import choice

def create_model():
    print("Creating model: reinforcement")
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu', input_shape=(8, 8, 12)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

    model.add(tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

    model.add(tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

    model.add(tf.keras.layers.Conv2D(1024, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=2048, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1024, activation='relu'))
    model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
    
    optimiser = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.CategoricalCrossentropy()
    
    model.compile(optimizer=optimiser, loss=loss, metrics=['accuracy'])
    
    try:
        model.load_weights("/content/sample_data/reinforcement.h5")
    except OSError:
        pass

    return model

def display_board(board: chess.Board):
    for i, row in enumerate(board.unicode(invert_color=True).split("\n")):
        print(f"{8-i} {row}")
    print("  a b c d e f g h")

def play_game(model: tf.keras.Sequential):
    board = chess.Board()
    game = chess.pgn.Game()

    while not board.is_game_over():
        if board.fullmove_number > 100:
            break
        moves = list(board.legal_moves)
        if board.turn == chess.WHITE:
            scores = []

            for move in moves:
                board.push(move)
                scores.append(model.predict(board_to_tensor(board).reshape(1,8,8,12), verbose=0)[0])
                board.pop()

            win_probs = [score[0] for score in scores]
            move = moves[win_probs.index(max(win_probs))]
            node = game.add_variation(move)
        else:
            move = choice(moves)
            node = node.add_variation(move)
        
        board.push(move)
        display_board(board)
        print(board.fen())
        print()

    print(game)
    print(f"Winner of the game was {board.Outcome().winner}")
    # write_game(board)

def write_game(board: chess.Board):
    result = board.outcome().winner
    if result == chess.WHITE:
        winner = 'w'
    elif result == chess.BLACK:
        winner = 'b'
    else:
        winner = 'd'
    
    while board.move_stack:
        with open("/content/sample_data/self_play.csv", "a") as f:
            f.write(board.fen() + "," + winner + "\n")
        board.pop()
    print("Game written to file")

def board_to_tensor(board: chess.Board):
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

def train(model: tf.keras.Sequential):
    # for i in range(10):
    #     print(f"Playing game {i+1}")
    #     play_game(model)
    
    print("Starting training")
    training_positions, training_scores = get_data()
    
    try:
        model.fit(training_positions,
                            training_scores,
                            epochs=50, 
                            batch_size=64,
                            shuffle=True,
                            )
    except KeyboardInterrupt:
        pass

    print("\nTraining complete.")
    model.save_weights("/content/sample_data/reinforcement.h5")
    print("Saved weights to disk.")

def get_data() -> tuple[list[list[str]], list[list[str]]]:
    data = pd.read_csv("/content/sample_data/self_play.csv")

    training_positions = []
    training_outcomes = []
    
    for position in data.values.tolist():        
        board = chess.Board(position[0])
        board_as_tensor = board_to_tensor(board)

        outcome = position[1]
        if outcome == 'w':
            one_hot_outcome = [1,0]
        elif outcome == 'b':
            one_hot_outcome = [0,1]
        else:
            one_hot_outcome = [0,0]

        training_positions.append(board_as_tensor)
        training_outcomes.append(one_hot_outcome)
    
    return np.array(training_positions), np.array(training_outcomes)

model = create_model()
play_game(model)