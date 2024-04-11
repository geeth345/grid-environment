import random

import numpy as np
from gymnasium.spaces import Discrete, Box, Tuple

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.env import AgentID, ObsType

import matplotlib.pyplot as plt

import pygame

from keras.datasets import mnist
from keras.models import load_model


class GridEnv(AECEnv):
    metadata = {
        "name": "grid_env",
    }

    def __init__(self, grid_size=28):
        super().__init__()

        # model parameters
        num_agents = 10
        self.vision_radius = 1  # how many squares beyond the current square can the agent see
        self.square_radius = True  # visibility window is a square
        self.confidence_decay = 1  # how much to decay confidence in historic observations
        self.binary_pixels = False  # whether to use binary or grayscale pixels
        self.max_age = 600  # how many steps before an agent is terminated (-1 for infinite)

        # visualisation parameters
        self.visualisation_type = 'pygame'
        self.render_wait_millis = 20
        self.generate_imgs_interval = 25
        self.print_info = False

        self.agents = [f'agent_{i}' for i in range(num_agents)]
        self.agent_positions = {name: np.array([0, 0]) for name in self.agents}

        # set the grid state based on the mnist image
        self.grid_size = grid_size
        self.grid_state = np.zeros((grid_size, grid_size))

        # load the generative model
        self.gen_model = load_model('../models_final/unet_mse/saved_model/gen.keras', compile=False)

        # load the cnn classifier model
        self.classifier = load_model('../models/mnist-cnn/mnist_cnn.h5')

        self.reset()

    def observation_space(self, agent: AgentID):
        coord = Box(low=0, high=self.grid_size, shape=(2,), dtype=np.int32)
        num_visible_squares = ((2 * self.vision_radius) ** 2) - (2 * self.vision_radius) + 1
        return Tuple([coord, [Tuple(coord, Discrete(2)) for _ in range(num_visible_squares)]])

    def action_space(self, agent: AgentID):
        return Discrete(4)

    def reset(self, **kwargs):

        image = kwargs.get('image', np.zeros((self.grid_size, self.grid_size)))
        self.label = kwargs.get('label', None)


        assert image.shape == (self.grid_size, self.grid_size), "Image must be of shape (grid_size, grid_size)"

        # agent positions all reset to a random position in the grid
        for id in self.agents:
            self.agent_positions[id] = np.array([np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)])

        # reset the other variables
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {
            'age': 0,
            'belief': -np.ones(image.shape),
            'confidence': np.zeros(image.shape),
            'cnn_gen_pred': -1,
            'cnn_naive_pred': -1,
            'gen_img': np.zeros((1, 28, 28, 1))
        } for agent in self.agents}

        # normalise the pixels
        self.grid_state = (image.astype(np.float32) - 127.5) / 127.5

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
            self.agent_positions[id] = np.clip(self.agent_positions[id], 1, self.grid_size - 2)

            # adjust confidence map for historic observations, currently just using a simple decay factor
            self.infos[id]['confidence'] = self.infos[id]['confidence'] * self.confidence_decay

            # update belief and confidence maps
            loc, obs = self.observe(id)
            for ob in obs:
                (x, y), val = ob
                self.infos[id]['belief'][x][y] = val
                self.infos[id]['confidence'][x][y] = 1

            if self.infos[id]['age'] % self.generate_imgs_interval == 0:
                # use the updated belief map to generate an image, and make predictions based on that
                # Add an extra dimension for the channel
                belief_map = np.expand_dims(self.infos[id]['belief'], axis=-1)
                confidence_map = np.expand_dims(self.infos[id]['confidence'], axis=-1)
                # Add an extra dimension for the batch size
                belief_map = np.expand_dims(belief_map, axis=0)
                confidence_map = np.expand_dims(confidence_map, axis=0)
                model_input = (belief_map, confidence_map)

                # print(f"Input shapes: {[x.shape for x in model_input]}")

                gen_img = self.gen_model.predict(model_input, verbose=0)

                # print(f"Generated image shape: {gen_img.shape}")

                cnn_gen_pred = np.argmax(self.classifier.predict(gen_img, verbose=0), axis=1)[0]
                cnn_naive_pred = np.argmax(self.classifier.predict(belief_map, verbose=0), axis=1)[0]

                self.infos[id]['cnn_gen_pred'] = cnn_gen_pred
                self.infos[id]['cnn_naive_pred'] = cnn_naive_pred
                self.infos[id]['gen_img'] = gen_img

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

        # classifications correct?
        gen_correct = self.infos[self.agent_selection]['cnn_gen_pred'] == self.label
        gen_colour = (0, 255, 0) if gen_correct else (255, 0, 0)
        naive_correct = self.infos[self.agent_selection]['cnn_naive_pred'] == self.label
        naive_colour = (0, 255, 0) if naive_correct else (255, 0, 0)

        # collect info
        # summarise agent's opinions
        opinions = [info['cnn_gen_pred'] for info in env.infos.values()]
        values, counts = np.unique(opinions, return_counts=True)
        average_opinion = values[np.argmax(counts)]
        opinions_naive = [info['cnn_naive_pred'] for info in env.infos.values()]
        values_naive, counts_naive = np.unique(opinions_naive, return_counts=True)
        average_opinion_naive = values_naive[np.argmax(counts_naive)]


        scale = 10
        window.fill((0, 0, 0))

        # draw the visualisations
        # row 1 - grid state, belief map ("masked image"), confidence map ("mask")
        self.renderMatrix(self.grid_state, 0, 0, scale)
        self.renderMatrix(self.infos[self.agent_selection]['belief'], 280, 0, scale)
        self.renderMatrix(self.infos[self.agent_selection]['confidence'], 560, 0, scale)

        # row 2 - generated image, cnn prediction, cnn naive prediction
        self.renderMatrix(self.infos[self.agent_selection]['gen_img'][0], 0, 280, scale, c=(0, 255, 255))
        self.renderDigit(str(self.infos[self.agent_selection]['cnn_gen_pred']), 280, 280, scale, c=gen_colour)
        self.renderDigit(str(self.infos[self.agent_selection]['cnn_naive_pred']), 560, 280, scale, c=naive_colour)


        # # debug
        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # ax1.imshow(self.infos[self.agent_selection]['gen_img'][0], cmap='gray')
        # ax2.imshow(self.infos[self.agent_selection]['belief'], cmap='gray')
        # ax3.imshow(self.infos[self.agent_selection]['confidence'], cmap='gray')
        # plt.show()
        
        
        # draw the agent position
        pygame.draw.rect(window, (255, 0, 0), (
        self.agent_positions[self.agent_selection][1] * scale, self.agent_positions[self.agent_selection][0] * scale,
        scale, scale))

        # draw the other agent's positions
        for agent in self.agents:
            if agent != self.agent_selection:
                pygame.draw.rect(window, (200, 128, 10), (
                self.agent_positions[agent][1] * scale, self.agent_positions[agent][0] * scale, scale, scale))

        # print some info at the bottom
        info = f"Agent: {self.agent_selection}, Age: {self.infos[self.agent_selection]['age']}, Correct: {gen_correct}, Naive Correct: {naive_correct}"
        info2 = f"All Opinions:      {opinions}, Modal Opinion: {average_opinion}"
        info3 = f"Naive Opinions: {opinions_naive}, Modal Opinion: {average_opinion_naive}"

        font = pygame.font.Font(None, 36)
        text = font.render(info, True, (255, 255, 255))
        text2 = font.render(info2, True, (255, 255, 255))
        text3 = font.render(info3, True, (255, 255, 255))
        window.blit(text, (0, 560))
        window.blit(text2, (0, 585))
        window.blit(text3, (0, 610))

        pygame.display.update()
        pygame.time.delay(self.render_wait_millis)

    def renderMatrix(self, array, xOffset, yOffset, scale, c=(255, 255, 255)):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                colour = (array[i][j] + 1.0) / 2
                pygame.draw.rect(window, (colour * c[0], colour * c[1], colour * c[2]),
                                 (j * scale + xOffset, i * scale + yOffset, scale, scale))

    def renderDigit(self, digit, xOffset, yOffset, scale, c=(255, 255, 255)):
        # Render the digit as text
        font = pygame.font.Font(None, scale * 25)
        digit_surface = font.render(digit, True, c)
        digit_rect = digit_surface.get_rect()
        digit_rect.center = (xOffset + (scale * 28) // 2, yOffset + (scale * 28) // 2)
        window.blit(digit_surface, digit_rect)

    def close(self):
        # TODO: implement this
        pass


if __name__ == '__main__':
    # placeholder policy function
    def policy(currentPos, info):

        # is there a previous direction? if not, set one as random
        if 'direction' not in info:
            info['direction'] = np.random.randint(0, 4)

        # semi-randomly change direction
        if np.random.uniform(0, 1) < 0.6:
            info['direction'] = np.random.randint(0, 4)

        # if we hit the edge of the grid, change direction so that we don't go over the edge
        # TODO: look at this more closely
        if currentPos[0] == 0 or currentPos[0] == 27 or currentPos[1] == 0 or currentPos[1] == 27:
            info['direction'] = np.random.randint(0, 4)

        # move in the direction
        move = info['direction']

        return move


    # import the mnist dataset
    _, (x_test, y_test) = mnist.load_data()

    # normalise the images (floats between -1 and 1)
    # x_test = (x_test.astype(np.float32) - 127.5) / 127.5

    # # To show the environment updating in real time, we use pygame
    # plt.ion()
    pygame.init()
    pygame.display.set_caption('Grid Environment')
    window = pygame.display.set_mode(((280 * 3), (280 * 2) + 75))

    # testing the environment
    env = GridEnv()

    for i in range(10):
        env.reset(image=x_test[i], label=y_test[i])  # load the image

        for agent in env.agent_iter():

            # only render if the agent is the first one
            isFirst = agent == 'agent_0'

            observation, _, terminated, truncated, info = env.last()

            if env.print_info:
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

            if isFirst:
                env.render()

            if all(env.terminations.values()):
                print('All agents terminated')
                break

            # if isFirst:
            #     # summarise agent's opinions
            #     opinions = [info['cnn_gen_pred'] for info in env.infos.values()]
            #     values, counts = np.unique(opinions, return_counts=True)
            #     average_opinion = values[np.argmax(counts)]
            #     print(f"Opinions of agents: {opinions}")
            #     # print(f"Counts: {counts}")
            #     print(f"Modal opinion: {average_opinion}")


        env.close()

    # plt.ioff()
    # plt.show()

    waiting_for_close = True
    while waiting_for_close:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting_for_close = False

    pygame.quit()
