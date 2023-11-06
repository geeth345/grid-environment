import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Box

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.env import AgentID, ObsType

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class GridEnv(AECEnv):
    metadata = {
        "name": "grid_env",
    }

    def __init__(self, grid_size=28):
        super().__init__()

        # initialise the agents
        num_agents = 10
        self.agents = [f'agent_{i}' for i in range(num_agents)]
        self.agent_positions = {name: np.array([0, 0]) for name in self.agents}

        # set the grid state based on the mnist image
        self.grid_size = grid_size
        self.grid_state = np.zeros((grid_size, grid_size))

        self.reset()

    def observation_space(self, agent: AgentID):
        return Box(low=0, high=self.grid_size, shape=(2,), dtype=np.int32)

    def action_space(self, agent: AgentID):
        return Discrete(4)

    def reset(self, **kwargs):

        # agent positions all reset to centre of grid
        for id in self.agents:
            self.agent_positions[id] = np.array([self.grid_size // 2, self.grid_size // 2])

        # reset the other variables
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {'age': 0} for agent in self.agents}

        # empty grid state for now
        self.grid_state = np.zeros((self.grid_size, self.grid_size))

        # reset the agent selector
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):

        id = self.agent_selection

        if self.terminations[id] or self.truncations[id]:
            self.rewards[id] = 0
        else:

            if action == 0:  # up
                self.agent_positions[id] = self.agent_positions[id] - np.array([0, 1])
            elif action == 1:  # down
                self.agent_positions[id] = self.agent_positions[id] + np.array([0, 1])
            elif action == 2:  # left
                self.agent_positions[id] = self.agent_positions[id] - np.array([1, 0])
            elif action == 3:  # right
                self.agent_positions[id] = self.agent_positions[id] + np.array([1, 0])

            # Keep the agent within the grid bounds
            self.agent_positions[id] = np.clip(self.agent_positions[id], 0, self.grid_size - 1)

            # Updating done flag and reward for the agent (currently not doing anything)
            self._check_done(id)
            self.rewards[id] = 0

            # advance to the next agent
            self.agent_selection = self._agent_selector.next()

    def observe(self, agent: AgentID):
        # return a placeholder observation for now
        return self.agent_positions[agent]

    def _check_done(self, agent):
        self.infos[agent]['age'] += 1
        if self.infos[agent]['age'] >= 100:
            self.terminations[agent] = True

    def render(self, mode='human'):
        if not plt.get_fignums():
            plt.figure()
            plt.show(block=False)
        plt.clf()
        plt.imshow(self.grid_state)
        for agent in self.agents:
            x, y = self.agent_positions[agent]
            plt.plot(x, y, 'ro', markersize=5)
        plt.draw()
        plt.pause(0.2)


    def close(self):
        # TODO: implement this
        pass


# placeholder policy function
def policy():
    return np.random.randint(4)


# To show the environment updating in real time, we use pyplot interactive mode
plt.ion()

# testing the environment
env = GridEnv()
env.reset()

for agent in env.agent_iter():
    observation, _, terminated, truncated, info = env.last()
    if terminated:
        action = None
    else:
        action = policy()  # Update with actual policy
    env.step(action)
    env.render()
    if all(env.terminations.values()):
        print('All agents terminated')
        break

env.close()

plt.ioff()
plt.show()


