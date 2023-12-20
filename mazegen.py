import numpy as np
import random

import cv2

import gymnasium as gym
from gymnasium import spaces



class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, size=13, seed=None):
        super(MazeEnv, self).__init__()
        if seed is not None:
            random.seed(seed)

        # 2D Observation Space
        if size % 2:
            self.size = size # We need an odd size
        else:
            self.size = size + 1  # Square grid size

        self.window_size=(512,512,3)
        self.observation_dim = (36,36,3)
        self.grid = np.zeros((size, size))
        self.player = np.array([0,0])
        self.prev_position=np.array([[0,0]])
        self.score=0

        self.generate_maze()

        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=self.observation_dim[::-1], dtype=np.uint8) # Image stored as float values
        # Action Space
        self.action_space = spaces.Discrete(4) # [up, right, down, left]

        # Map number to direction
        self._action_to_direction = {
            0: np.array([-1, 0]),
            1: np.array([0, 1]),
            2: np.array([1,0]),
            3: np.array([0,-1])
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def step(self, action):

        # Apply direction
        direction= self._action_to_direction[action]

        # Reward System
        # Every move cost 2/size^2
        # Reward = 1 if goal is reached
        # Attempting to go into a wall or outside boundaries is -0.5
        # If the cell is visited -0.25
        # If score < -1, game is lost

        valid_dir = self.is_valid(self.player+direction)
        if valid_dir:
            self.player += direction
            reward=+0.05
            # Check if it's visited
            if any(np.array_equal(self.player, arr) for arr in self.prev_position):
                reward+= -0.20
            else:
                self.prev_position=np.vstack((self.prev_position, self.player))
        else:
            reward=-0.50

        # update score
        self.score += reward

        # Check if goal is reached or score is negative
        if np.array_equal(self.player, [self.size-1, self.size-1]):
            terminated = True
            reward=1
        elif self.score <= -1:
            terminated = True
        else:
            terminated = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def reset(self, seed = None, options=None):
        if seed is not None:
            random.seed(seed)
        self.generate_maze()
        self.player=np.array([0,0])
        self.prev_position=np.array([[0,0]])

        self.score = 0

        if self.render_mode == "human":
            cv2.imshow("Game", self.render_maze(self.grid.copy()))
            cv2.waitKey(1)


        observation=self._get_obs()
        info = self._get_info()

        return observation, info
    def _get_obs(self):
        observation = self.grid.copy()
        observation = self.render_maze(observation, self.observation_dim)

        # Observation of the neural network is the same space but rescaled to smaller dimension
        # cv2.imshow("Test", observation)
        # observation = observation / 255.
        observation = np.reshape(observation,self.observation_dim)
        return observation

    def _get_info(self):
        return {"x_dist": self.size-self.player[0] -1, "y_dist": self.size - self.player[1] -1}

    def render(self):
        if self.render_mode == "rgb_array":
            return self.grid
        elif self.render_mode == "human":
            cv2.imshow("Game", self.render_maze(self.grid.copy(), self.window_size))
            cv2.waitKey(1)

    def is_valid(self, pos):
        # check inside borders
        if pos[0] < 0 or pos[1] < 0 or pos[0] >= self.size or pos[1] >= self.size:
            return False
        # check if there is wall

        if self.grid[pos[0],pos[1]] == 1:
            return False
        return True
    def generate_maze(self):
        self.grid = self.create_maze(self.size+2)[1:-1, 1:-1]



    # Reference: https://medium.com/@msgold/using-python-to-create-and-solve-mazes-672285723c96
    def create_maze(self, dim):
        # Create a grid filled with walls
        maze = np.ones((dim, dim))

        # Define the starting point
        x, y = (0, 0)
        maze[2 * x + 1, 2 * y + 1] = 0

        # Initialize the stack with the starting point
        stack = [(x, y)]
        while len(stack) > 0:
            x, y = stack[-1]

            # Define possible directions
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if nx >= 0 and ny >= 0 and nx < dim // 2 and ny < dim // 2 and maze[2 * nx + 1, 2 * ny + 1] == 1:
                    maze[2 * nx + 1, 2 * ny + 1] = 0
                    maze[2 * x + 1 + dx, 2 * y + 1 + dy] = 0
                    stack.append((nx, ny))
                    break
            else:
                stack.pop()


        return maze
    def render_maze(self, maze_array, size=(512,512,3)):
        # Define colors
        wall_color = (0, 0, 0)  # Black for walls
        open_color = (255, 255, 255)  # White for open cells
        player_color = (0, 255, 0)  # Green for player position

        # Set player pos
        maze_array[self.player[0],self.player[1]]= 1/2

        # Get maze dimensions
        n = len(maze_array)

        # Create an empty image with a white background
        maze_image = np.ones(size, dtype=np.uint8) * 255

        # Calculate cell size in the image
        cell_size = size[0] // n

        # Draw maze cells
        for i in range(n):
            for j in range(n):
                x, y = j * cell_size, i * cell_size
                if maze_array[i, j] == 0:
                    # Open cell
                    maze_image[y:y + cell_size, x:x + cell_size] = open_color
                elif maze_array[i, j] == 1:
                    # Wall
                    maze_image[y:y + cell_size, x:x + cell_size] = wall_color
                elif maze_array[i, j] == 0.5:
                    # Player position
                    maze_image[y:y + cell_size, x:x + cell_size] = player_color

        return maze_image




if __name__ == "__main__":
    env = MazeEnv(size=5,render_mode="human", seed=393939)
    state, _ = env.reset()
    key_dict={119: 0, 100: 1, 115: 2, 97:3}
    done = False
    while not done:
        key_press=cv2.waitKey(0)
        state, reward, done, _, info = env.step(key_dict[key_press])
        print("State: \n", state)
        print("Reward: ",reward)
        print("Score: ", env.score)
        print("-----------")
        env.render()
"""
env = MazeEnv(size=5,render_mode="human", seed=393939)
while True:
    state, _ = env.reset()
    done = False
    while not done:
        action = random.choice([0,1,2,3])
        state, reward, done, _, info = env.step(action)
        print("State: \n", state)
        print("Reward: ",reward)
        print("Score: ", env.score)
        print("-----------")
        env.render()
"""