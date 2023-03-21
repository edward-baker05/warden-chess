import os
from agents.replay_buffer import ReplayBuffer
from agents.agent import ChessAgent
from agents.state import State

# Define hyperparameters
gamma = 0.99
batch_size = 32
lr = 0.001
num_epochs = 10
buffer_size = 1000000
eps_initial = 1.0
eps_decay = 0.9999
eps_min = 0.1

# Create the replay buffer
replay_buffer = ReplayBuffer(buffer_size)

# Create the agent
agent = ChessAgent(lr, gamma, eps_initial, eps_decay, eps_min)

# Train the agent
for epoch in range(num_epochs):
    done = False
    state = State()
    while not done:
        # choose an action based on the current state
        action = agent.choose_action(state)

        # apply the chosen action to the current state to obtain the next state
        next_state = state.make_next_state(state, action)

        # calculate the reward for the current state and the chosen action
        reward, done = state.reward(action)

        # add the current state, action, reward, and next state to the replay buffer
        replay_buffer.add(state, action, reward, next_state, done)

        # update the current state
        state = next_state

        # Train agent on batch of transitions from replay buffer
        if len(replay_buffer) > batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            agent.train(states, actions, rewards, next_states, dones)

    # Decay epsilon
    agent.decay_epsilon()

    # Save model weights
    if epoch % 10 == 0:
        agent.save_weights(os.path.join('models', f'epoch{epoch}.h5'))

# Save final model weights
agent.save_weights(os.path.join('models', 'final.h5'))
