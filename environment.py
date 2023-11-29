import random

import numpy as np
from gymnasium.spaces import Discrete, Box, Tuple

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.env import AgentID, ObsType

import matplotlib.pyplot as plt

import pygame

from tensorflow.keras.datasets import mnist



class GridEnv(AECEnv):
    metadata = {
        "name": "grid_env",
    }

    def __init__(self, grid_size=28):
        super().__init__()

        # model parameters
        num_agents = 1
        self.vision_radius = 2  # how many squares beyond the current square can the agent see
        self.square_radius = True  # visibility window is a square
        self.confidence_decay = 0.995  # how much to decay confidence in historic observations
        self.binary_pixels = False  # whether to use binary or grayscale pixels
        self.max_age = 500  # how many steps before an agent is terminated (-1 for infinite)

        # visualisation parameters
        self.visualisation_type = 'pygame'
        self.render_wait_millis = 50

        self.agents = [f'agent_{i}' for i in range(num_agents)]
        self.agent_positions = {name: np.array([0, 0]) for name in self.agents}

        # set the grid state based on the mnist image
        self.grid_size = grid_size
        self.grid_state = np.zeros((grid_size, grid_size))

        self.reset()

    def observation_space(self, agent: AgentID):
        coord = Box(low=0, high=self.grid_size, shape=(2,), dtype=np.int32)
        num_visible_squares = ((2 * self.vision_radius) ** 2) - (2 * self.vision_radius) + 1
        return Tuple([coord, [Tuple(coord, Discrete(2)) for _ in range(num_visible_squares)]])

    def action_space(self, agent: AgentID):
        return Discrete(4)

    def reset(self, **kwargs):

        image = kwargs.get('image', np.zeros((self.grid_size, self.grid_size)))
        assert image.shape == (self.grid_size, self.grid_size), "Image must be of shape (grid_size, grid_size)"

        # agent positions all reset to centre of grid
        for id in self.agents:
            self.agent_positions[id] = np.array([self.grid_size // 2, self.grid_size // 2])

        # reset the other variables
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {
            'age': 0,
            'belief': np.zeros(image.shape),
            'confidence':np.zeros(image.shape)
        } for agent in self.agents}

        if self.binary_pixels:
            # load the image into the grid state, normalise the values, and then round the values to 0 or 1
            self.grid_state = np.round(image / 255)
        else:
            self.grid_state = image / 255

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

            # adjust confidence map for historic observations, currently just using a simple decay factor
            self.infos[id]['confidence'] = self.infos[id]['confidence'] * self.confidence_decay

            # update belief and confidence maps
            loc, obs = self.observe(id)
            for ob in obs:
                (x, y), val = ob
                self.infos[id]['belief'][x][y] = val
                self.infos[id]['confidence'][x][y] = 1


            # Updating done flag and reward for the agent (currently not doing anything)
            self._check_done(id)
            self.rewards[id] = 0

            # advance to the next agent
            self.agent_selection = self._agent_selector.next()

    def observe(self, agent: AgentID):
        coordinates = []
        x, y = self.agent_positions[agent]
        r = self.vision_radius
        for i in range(x - r, x + r + 1):
            for j in range(y - r, y + r + 1):
                if not ((0 <= i < self.grid_size) and (0 <= j < self.grid_size)):
                    continue
                if not self.square_radius and not (abs(x - i) + abs(y - j) <= r):
                    continue
                coordinates.append((i, j))

        grid_obs = []
        for coord in coordinates:
            grid_obs.append((coord, self.grid_state[coord]))

        return ((x, y), grid_obs)

    def _check_done(self, agent):
        self.infos[agent]['age'] += 1
        if (not self.max_age == -1) and (self.infos[agent]['age'] >= self.max_age):
            self.terminations[agent] = True

    def render(self, mode='human'):
        # if not plt.get_fignums():
        #     plt.figure()
        #     plt.show(block=False)
        # plt.clf()
        # plt.imshow(self.grid_state, cmap='gray')
        # for agent in self.agents:
        #     x, y = self.agent_positions[agent]
        #     plt.plot(x, y, 'ro', markersize=5)
        # plt.draw()
        # plt.pause(0.2)
        scale = 10
        window.fill((0, 0, 0))
        self.renderMatrix(self.grid_state, 0, scale)
        self.renderMatrix(self.infos[self.agent_selection]['belief'], 280, scale)
        self.renderMatrix(self.infos[self.agent_selection]['confidence'], 560, scale)
        pygame.display.update()
        pygame.time.delay(self.render_wait_millis)


    def renderMatrix(self, array, xOffset, scale):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                colour = array[i][j] * 255
                pygame.draw.rect(window, (colour, colour, colour), (j * scale + xOffset, i * scale, scale, scale))


    def close(self):
        # TODO: implement this
        pass





if __name__ == '__main__':
    # placeholder policy function
    def policy(currentPos, info):
        # is there a target? if not, set one

        if ('target' not in info) or (info['target'] == currentPos):
            info['target'] = (random.randint(0, 27), random.randint(0, 27))
        print(info['target'])
        print(currentPos)

        # move towards the target, semi-randomly
        target = info['target']
        xReached = (currentPos[0] == target[0])
        yReached = (currentPos[1] == target[1])
        if xReached:
            return 1 if currentPos[1] < target[1] else 0
        elif yReached:
            return 3 if currentPos[0] < target[0] else 2
        else:
            if np.random.rand() < 0.5:
                return 1 if currentPos[1] < target[1] else 0
            else:
                return 3 if currentPos[0] < target[0] else 2

    # import the mnist dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # # To show the environment updating in real time, we use pyplot interactive mode
    # plt.ion()
    pygame.init()
    pygame.display.set_caption('Grid Environment')
    window = pygame.display.set_mode((280 * 3, 280))

    # testing the environment
    env = GridEnv()
    env.reset(image=x_train[0])  # load the first image

    for agent in env.agent_iter():
        observation, _, terminated, truncated, info = env.last()

        print("\n")
        print(f"Observation for agent {agent}: ")
        print(f"Currently at position {observation[0]}")
        print(f"Visible squares: {observation[1]}")
        print(f"Num visible squares: {len(observation[1])}")

        if terminated:
            action = None
        else:
            action = policy(observation[0], info)  # Update with actual policy
        env.step(action)
        env.render()
        if all(env.terminations.values()):
            print('All agents terminated')
            break

    env.close()

    # plt.ioff()
    # plt.show()

    waiting_for_close = True
    while waiting_for_close:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting_for_close = False

    pygame.quit()