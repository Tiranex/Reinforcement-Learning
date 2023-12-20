from stable_baselines3.common.env_checker import check_env
from maze_gen import MazeEnv

env = MazeEnv()
check_env(env)