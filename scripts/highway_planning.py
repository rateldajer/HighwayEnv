# # Behavioural Planning for Autonomous Highway Driving
# 
# We plan a trajectory using the _Optimistic Planning for Deterministic systems_ ([OPD](https://hal.inria.fr/hal-00830182)) algorithm.


#@title Imports for env, agent, and visualisation.
# Environment
import gymnasium as gym
import highway_env

# Agent
from rl_agents.agents.common.factory import agent_factory

# Visualisation
import sys
from tqdm.notebook import trange
# get_ipython().system('pip install moviepy -U')
# get_ipython().system('pip install imageio_ffmpeg')
# get_ipython().system('pip install pyvirtualdisplay')
# get_ipython().system('apt-get install -y xvfb python-opengl ffmpeg')
# !git clone https://github.com/eleurent/highway-env.git
# sys.path.insert(0, './highway-env/scripts/')
from utils import record_videos, show_videos

#@title Run an episode

# Make environment
env = gym.make("highway-fast-v0", render_mode="rgb_array")
# env = gym.make("highway-fast-v0", render_mode="human")
env = record_videos(env)
(obs, info), done = env.reset(), False

# Make agent

agent_config = {"__class__": "<class 'rl_agents.agents.deep_q_network.pytorch.DQNAgent'>",
    "model": {
        "type": "MultiLayerPerceptron",
        "layers": [512, 512]
    },
    "gamma": 0.99,
    "n_steps": 10,
    "batch_size": 32,
    "memory_capacity": 50000,
    "target_update": 1,
    "exploration": {
        "method": "EpsilonGreedy",
        "tau": 50000,
        "temperature": 1.0,
        "final_temperature": 0.1
    }
}
# agent_config = {
#     "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
#     "env_preprocessors": [{"method":"simplify"}],
#     "budget": 50,
#     "gamma": 0.7,
# }
agent = agent_factory(env, agent_config)

# Run episode
for step in range(1000):
# for step in trange(env.unwrapped.config["duration"], desc="Running..."):
    action = agent.act(obs)
    # action = env.action_space.sample()  # agent policy that uses the observation and info
    obs, reward, done, truncated, info = env.step(action)
    
env.close()
show_videos()

