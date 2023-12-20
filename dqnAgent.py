import random
import numpy as np
import tensorflow as tf
from collections import deque

REPLAY_MEMORY_SIZE=50000
MIN_REPLAY_MEMORY=5000
BATCH_SIZE=64
DISCOUNT=0.97
UPDATE_TARGET_COUNTER=20

class DQN:

    def __init__(self, arq):

        # Create main and target model
        self.main_model =self.create_model(arq)
        self.target_model = self.create_model(arq)

        # Set weights of main to target
        self.target_model.set_weights(self.main_model.get_weights())

        # Set Replay Memory
        # (current_obs, action, resulting_obs, reward, done)
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Counter to track how many episode without update
        self.target_update_counter=0
    
    def create_model(self, model):
        n_actions=2
        
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    # Add step to buffer
    def append_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Train loop
    def train(self, terminal_state):

        # Check if replay buffer is big enough
        if len(self.replay_memory) < MIN_REPLAY_MEMORY:
            return 0

        # Get some replays from buffer
        batch=random.sample(self.replay_memory, BATCH_SIZE)

        # We use main model for current state
        # The purpose of the two models is predicting the Q-values
        current_state=np.array([step[0] for step in batch])
        current_q_list=self.main_model.predict(current_state, verbose=0)

        # We use target model for future state
        # We will use the future state to update the q_value of a chosen action in the current state
        future_state=np.array([step[2] for step in batch])
        future_q_list=self.target_model.predict(future_state, verbose=0)

        X=[]
        Y=[]

        for index, (current_obs, action, resulting_obs, reward, done) in enumerate(batch):
            # Expression (3) https://arxiv.org/pdf/1509.06461.pdf
            # Update q values using the future q_list
            if not done:
                max_future_q = np.max(future_q_list[index])
                # This the expression (3) for the q_value
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q=reward

            # When we have the new q_value for the chosen action we replace it
            current_q = current_q_list[index]
            current_q[action]=new_q

            # We want that our model predicts the q_value so we give the current state and the desired q_value for each action
            X.append(current_obs)
            Y.append(current_q)

        # We update the weights of the online model while keeping the target model the same
        self.main_model.fit(np.array(X), np.array(Y), batch_size=BATCH_SIZE, verbose=0, shuffle=False)

        # If terminal state is True the episode ended
        if terminal_state:
            self.target_update_counter+=1

        # We update the target network every UPDATE_TARGET_COUNTER
        if self.target_update_counter > UPDATE_TARGET_COUNTER:
            self.target_model.set_weights(self.main_model.get_weights())
            self.target_update_counter=0
    
    def get_qtable(self, obs):
        return self.main_model.predict(obs, verbose=0)[0]
    
    def evaluate_policy(self, env, render=True):
        current_state, _= env.reset()
        if render:
            env.render()
        total_reward=0

        done = False
        while not done:
            action=self.get_qtable(np.array([current_state]))
            action=np.argmax(action)
            current_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            if render:
                env.render()
        
        return total_reward
    
    def save(self, location):
        self.main_model.save(location)
    
    def load(self, location):
        self.main_model = tf.keras.models.load_model(location)
        self.target_model = tf.keras.models.load_model(location)
    

