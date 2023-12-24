from mazegen import MazeEnv
import random
from dqnAgent import DQN
import numpy as np
from tqdm import tqdm
import tensorflow as tf


model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(128,128,3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(4, activation = 'linear')
        ])

agent=DQN(model)

env=MazeEnv(size=4)
test_env=MazeEnv(size=4, render_mode="human")
N_ACTION=env.action_space.n

# Exploration
epsilon=1
EPSILON_DECAY=0.995
MIN_EPSILON=0.001

# Visualization
PREVIEW_TRAIN=False
PREVIEW_EVAL=False

# Episodes
N_EPISODES=2000
AGGREGATE_STATS_EVERY=250
LOAD_PATH=""

if LOAD_PATH != "":
    agent.load(LOAD_PATH)
current_ep_rewards=[]
ep_rewards=[]

agent.fill_min_memory(env)
for episode in tqdm(range(N_EPISODES), ascii=True, unit='episodes'):

    # Set reward to initial values
    episode_reward=0

    # Reset Environment
    current_state, _ =env.reset()

    # Iterate until episode end
    done = False
    while not done:

        # We use epsilon greedy policy, we take a random number and compare it to epsilon
        if random.random() > epsilon:
            # Choose action from q table
            action = np.argmax(agent.get_qtable(np.array([current_state])))
        else:
            # Random action
            action = random.choice([i for i in range(N_ACTION)])

        # Perform the step and get data
        new_state, reward, done, truncated, info = env.step(action)
        
        # Sum the reward
        episode_reward+=reward

        # Show preview if conditions are met
        if PREVIEW_TRAIN:
            env.render()

        # Every step we update replay memory
        agent.append_replay_memory((current_state, action, new_state, reward, done))
        agent.train(done)

        current_state = new_state

    current_ep_rewards.append(episode_reward)

    if episode % AGGREGATE_STATS_EVERY == 0:
        print("AGGREGATING EPISODE: ", episode) 
        # Get the average, minimum maximum reward across AGGREGATE_STATS_EVERY episodes
        average_reward = sum(current_ep_rewards) / len(current_ep_rewards)
        min_reward = min(current_ep_rewards)
        max_reward = max(current_ep_rewards)

        ep_rewards.append([average_reward, min_reward, max_reward])
        current_ep_rewards=[]

        # Evaluation
        print("Reward: ", agent.evaluate_policy(test_env, render=PREVIEW_EVAL))
        agent.save("best_model_maze.keras")
        

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

agent.save("best_model_maze.keras")