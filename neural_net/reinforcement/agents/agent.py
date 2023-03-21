import random
import numpy as np
import tensorflow as tf
from agents.state import State


class ChessAgent:
    def __init__(self, lr, gamma, eps_initial, eps_decay, eps_min):
        self.lr = lr
        self.gamma = gamma
        self.eps = eps_initial
        self.eps_decay = eps_decay
        self.eps_min = eps_min

        # Define loss function and optimizer
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Create Q-network and target network
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.target_network.set_weights(self.q_network.get_weights())

        

        # Create replay buffer
        self.replay_buffer = []

    def _build_network(self):
        # Define input shape
        input_shape = (8, 8, 12)

        # Define model architecture
        inputs = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=128, activation='relu')(x)
        outputs = tf.keras.layers.Dense(units=1)(x)

        model = tf.keras.models.Sequential()

        x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
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

        # Define model
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        # Compile model
        model.compile(loss=self.loss_fn, optimizer=self.optimizer)

        return model

    def choose_action(self, state):
        if random.random() < self.eps:
            # Choose random action
            legal_moves = list(state.legal_moves)
            action_index = random.randint(0, len(legal_moves) - 1)
            action = legal_moves[action_index]
        else:
            # Choose action with highest Q-value
            state_input = state.board_to_state(state)
            q_values = self.q_network(state_input)
            legal_moves = state.get_legal_moves()
            legal_moves_input = [state.board_to_state(state.move(move)) for move in legal_moves]
            legal_q_values = self.target_network.predict(np.array(legal_moves_input))
            legal_q_values = np.squeeze(legal_q_values)
            legal_q_values_dict = dict(zip(legal_moves, legal_q_values))
            action = max(legal_q_values_dict, key=legal_q_values_dict.get)

        return action

    def train(self, batch_size):
        # Sample batch of transitions from replay buffer
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_input = np.array([state.board_to_state(state) for state in states])
        next_states_input = np.array([state.board_to_state(state) for state in next_states])

        # Compute targets
        q_values = self.q_network(states_input)
        next_q_values = self.target_network(next_states_input)
        next_q_values = np.squeeze(next_q_values)
        targets = np.array(rewards) + self.gamma * np.max(next_q_values, axis=1) * (1 - np.array(dones))
        targets = np.expand_dims(targets, axis=1)

        # Train Q-network on batch of transitions
        self.q_network.train_on_batch(states_input, targets)

    def update(self):
        # Update target network
        self.target_network.set_weights(self.q_network.get_weights())
        
        # Decay epsilon
        self.eps = max(self.eps * self.eps_decay, self.eps_min)

    def decay_epsilon(self):
        self.eps = max(self.eps * self.eps_decay, self.eps_min)
        
    def save_weights(self, path):
        self.q_network.save_weights(path)