import random

import numpy as np
from gymnasium.spaces import Discrete, Box, Tuple

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.env import AgentID, ObsType

import matplotlib.pyplot as plt

import pygame

import concurrent.futures

from keras.datasets import mnist
from keras.models import load_model


class GridEnv(AECEnv):
    metadata = {
        "name": "grid_env",
    }

    def __init__(self):
        super().__init__()

        # model parameters
        num_agents = 20
        grid_size = 28
        self.vision_radius = 1  # how many squares beyond the current square can the agent see
        self.square_radius = True  # visibility window is a square
        self.confidence_decay = 0.005  # how much to decay confidence in historic observations
        self.decay_mode = 'exponential'  # how to decay
        self.max_age = 400  # how many steps before an agent is terminated (-1 for infinite)
        self.image_switch_interval = 200  # how often to advance the image


        # visualisation parameters
        self.visualisation_type = 'pygame'
        self.render_wait_millis = 0
        self.generate_imgs_interval = 1
        self.print_info = False

        self.agents = [f'agent_{i}' for i in range(num_agents)]
        self.agent_positions = {name: np.array([0, 0]) for name in self.agents}

        # set the grid state based on the mnist image
        self.grid_size = grid_size
        self.grid_state = np.zeros((grid_size, grid_size))

        # fields to store the images and labels
        self.images = []
        self.labels = []
        self.current_image = 0

        # load the generative model
        # self.gen_model = load_model('../models_final/unet_mse/saved_model/gen.keras', compile=False)

        # load the cnn classifier model
        self.classifier = load_model('../models/mnist-cnn/mnist_cnn.h5')

        self.reset()

    def observation_space(self, agent: AgentID):
        coord = Box(low=0, high=self.grid_size, shape=(2,), dtype=np.int32)
        num_visible_squares = ((2 * self.vision_radius) ** 2) - (2 * self.vision_radius) + 1
        return Tuple([coord, [Tuple(coord, Discrete(2)) for _ in range(num_visible_squares)]])

    def action_space(self, agent: AgentID):
        return Discrete(4)

    def set_model(self, model_path):
        self.gen_model = load_model(model_path, compile=False)

    def set_decay(self, mode, decay):
        self.decay_mode = mode
        self.confidence_decay = decay

    def reset(self, **kwargs):
        self.images = []
        self.labels = []

        # load the images and labels
        self.images.append(kwargs.get('image1', np.zeros((self.grid_size, self.grid_size))))
        self.images.append(kwargs.get('image2', np.zeros((self.grid_size, self.grid_size))))
        self.labels.append(kwargs.get('label1', -1))
        self.labels.append(kwargs.get('label2', -1))
        self.current_image = 0

        for image in self.images:
            assert image.shape == (self.grid_size, self.grid_size), "Image must be of shape (grid_size, grid_size)"

        # agent positions all reset to a random position in the grid
        for id in self.agents:
            self.agent_positions[id] = np.array(
                [np.random.randint(2, self.grid_size - 3), np.random.randint(2, self.grid_size - 3)])

        # reset the other variables
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {
            'age': 0,
            'belief': -np.ones(self.images[0].shape),
            'confidence': np.zeros(self.images[0].shape),
            'cnn_gen_pred': -1,
            'cnn_naive_pred': -1,
            'gen_img': np.zeros((1, 28, 28, 1))
        } for agent in self.agents}

        # normalise the pixels
        self.grid_state = (self.images[0].astype(np.float32) - 127.5) / 127.5

        # reset the agent selector
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def firstStep(self):
        # use belief and confidence maps to generated predictions
        # vectorise implementation - infer all agents at once to cut down on processing time
        belief_maps = np.expand_dims(np.array([self.infos[agent]['belief'] for agent in self.agents]), axis=-1)
        confidence_maps = np.expand_dims(np.array([self.infos[agent]['confidence'] for agent in self.agents]), axis=-1)
        generated_images = self.gen_model.predict((belief_maps, confidence_maps), verbose=0)
        predictions = np.argmax(self.classifier.predict(generated_images, verbose=0), axis=1)
        naive_predictions = np.argmax(self.classifier.predict(belief_maps, verbose=0), axis=1)

        # update the agent info
        for i, agent in enumerate(self.agents):
            self.infos[agent]['cnn_gen_pred'] = predictions[i]
            self.infos[agent]['cnn_naive_pred'] = naive_predictions[i]
            self.infos[agent]['gen_img'] = generated_images[i]

    def advance_image(self):
        self.current_image += 1
        if self.current_image >= len(self.images):
            self.current_image = 0
        self.grid_state = (self.images[self.current_image].astype(np.float32) - 127.5) / 127.5


    def step(self, action):

        id = self.agent_selection

        if id == 'agent_0':
            self.firstStep()

        if self.terminations[id] or self.truncations[id]:
            self.rewards[id] = 0
        else:

            if action == 0:  # up
                self.agent_positions[id] = self.agent_positions[id] - np.array([0, 1])
            elif action == 2:  # down
                self.agent_positions[id] = self.agent_positions[id] + np.array([0, 1])
            elif action == 1:  # left
                self.agent_positions[id] = self.agent_positions[id] - np.array([1, 0])
            elif action == 3:  # right
                self.agent_positions[id] = self.agent_positions[id] + np.array([1, 0])

            # Keep the agent within the grid bounds
            self.agent_positions[id] = np.clip(self.agent_positions[id], 1, self.grid_size - 2)

            # adjust confidence map for historic observations
            if self.decay_mode == 'exponential':
                self.infos[id]['confidence'] = self.infos[id]['confidence'] * (1 - self.confidence_decay)
            elif self.decay_mode == 'linear':
                self.infos[id]['confidence'] = np.maximum(self.infos[id]['confidence'] - self.confidence_decay, 0)

            # update belief and confidence maps
            loc, obs = self.observe(id)
            for ob in obs:
                (x, y), val = ob
                self.infos[id]['belief'][x][y] = val
                self.infos[id]['confidence'][x][y] = 1

            # if self.infos[id]['age'] % self.generate_imgs_interval == 0:
            #     # use the updated belief map to generate an image, and make predictions based on that
            #     # Add an extra dimension for the channel
            #     belief_map = np.expand_dims(self.infos[id]['belief'], axis=-1)
            #     confidence_map = np.expand_dims(self.infos[id]['confidence'], axis=-1)
            #     # Add an extra dimension for the batch size
            #     belief_map = np.expand_dims(belief_map, axis=0)
            #     confidence_map = np.expand_dims(confidence_map, axis=0)
            #     model_input = (belief_map, confidence_map)
            #
            #     # print(f"Input shapes: {[x.shape for x in model_input]}")
            #
            #     gen_img = self.gen_model.predict(model_input, verbose=0)
            #
            #     print(f"Generated image shape: {gen_img.shape}")
            #
            #     cnn_gen_pred = np.argmax(self.classifier.predict(gen_img, verbose=0), axis=1)[0]
            #     cnn_naive_pred = np.argmax(self.classifier.predict(belief_map, verbose=0), axis=1)[0]
            #
            #     self.infos[id]['cnn_gen_pred'] = cnn_gen_pred
            #     self.infos[id]['cnn_naive_pred'] = cnn_naive_pred
            #     self.infos[id]['gen_img'] = gen_img

            # TODO: hmm this doesn't look nice, i should really be using a ParallelEnv
            if id == 'agent_0' and self.infos[id]['age'] % self.image_switch_interval == 0:
                self.advance_image()

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

    def getCurrentLabel(self):
        return self.labels[self.current_image]

    def render(self, mode='human'):

        # has pygame been initialised?
        if not pygame.get_init():
            pygame.init()
            pygame.display.set_caption('Grid Environment')
            self.window = pygame.display.set_mode(((280 * 3), (280 * 2) + 75))

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
        gen_correct = self.infos[self.agent_selection]['cnn_gen_pred'] == self.labels[self.current_image]
        gen_colour = (0, 255, 0) if gen_correct else (255, 0, 0)
        naive_correct = self.infos[self.agent_selection]['cnn_naive_pred'] == self.labels[self.current_image]
        naive_colour = (0, 255, 0) if naive_correct else (255, 0, 0)


        # collect info
        # summarise agent's opinions
        opinions = [info['cnn_gen_pred'] for info in self.infos.values()]
        values, counts = np.unique(opinions, return_counts=True)
        average_opinion = values[np.argmax(counts)]
        opinions_naive = [info['cnn_naive_pred'] for info in self.infos.values()]
        values_naive, counts_naive = np.unique(opinions_naive, return_counts=True)
        average_opinion_naive = values_naive[np.argmax(counts_naive)]

        scale = 10
        self.window.fill((0, 0, 0))

        # draw the visualisations
        # row 1 - grid state, belief map ("masked image"), confidence map ("mask")
        self.renderMatrix(self.grid_state, 0, 0, scale)
        self.renderMatrix(self.infos[self.agent_selection]['belief'], 280, 0, scale)
        self.renderMatrix((self.infos[self.agent_selection]['confidence'] * 2) - 1, 560, 0, scale)

        # row 2 - generated image, cnn prediction, cnn naive prediction
        self.renderMatrix(self.infos[self.agent_selection]['gen_img'], 0, 280, scale, c=(0, 255, 255))
        self.renderDigit(str(self.infos[self.agent_selection]['cnn_gen_pred']), 280, 280, scale, c=gen_colour)
        self.renderDigit(str(self.infos[self.agent_selection]['cnn_naive_pred']), 560, 280, scale, c=naive_colour)

        # # debug
        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # ax1.imshow(self.infos[self.agent_selection]['gen_img'][0], cmap='gray')
        # ax2.imshow(self.infos[self.agent_selection]['belief'], cmap='gray')
        # ax3.imshow(self.infos[self.agent_selection]['confidence'], cmap='gray')
        # plt.show()

        # draw the agent position
        pygame.draw.rect(self.window, (255, 0, 0), (
            self.agent_positions[self.agent_selection][1] * scale,
            self.agent_positions[self.agent_selection][0] * scale,
            scale, scale))

        # draw the other agent's positions
        for agent in self.agents:
            if agent != self.agent_selection:
                pygame.draw.rect(self.window, (200, 128, 10), (
                    self.agent_positions[agent][1] * scale, self.agent_positions[agent][0] * scale, scale, scale))

        # print some info at the bottom
        info = f"Agent: {self.agent_selection}, Age: {self.infos[self.agent_selection]['age']}, Correct: {gen_correct}, Naive Correct: {naive_correct}"
        info2 = f"All Opinions:      {opinions}, Modal Opinion: {average_opinion}"
        info3 = f"Naive Opinions: {opinions_naive}, Modal Opinion: {average_opinion_naive}"

        font = pygame.font.Font(None, 36)
        text = font.render(info, True, (255, 255, 255))
        text2 = font.render(info2, True, (255, 255, 255))
        text3 = font.render(info3, True, (255, 255, 255))
        self.window.blit(text, (0, 560))
        self.window.blit(text2, (0, 585))
        self.window.blit(text3, (0, 610))

        pygame.display.update()
        pygame.time.delay(self.render_wait_millis)

    def renderMatrix(self, array, xOffset, yOffset, scale, c=(255, 255, 255)):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                colour = (array[i][j] + 1.0) / 2
                pygame.draw.rect(self.window, (colour * c[0], colour * c[1], colour * c[2]),
                                 (j * scale + xOffset, i * scale + yOffset, scale, scale))

    def renderDigit(self, digit, xOffset, yOffset, scale, c=(255, 255, 255)):
        # Render the digit as text
        font = pygame.font.Font(None, scale * 25)
        digit_surface = font.render(digit, True, c)
        digit_rect = digit_surface.get_rect()
        digit_rect.center = (xOffset + (scale * 28) // 2, yOffset + (scale * 28) // 2)
        self.window.blit(digit_surface, digit_rect)



    def close(self):
        # TODO: implement this
        pass


if __name__ == '__main__':
    multi_thread = True

    # policy function
    def policy(currentPos, info):


        # is there a previous direction? if not, set one as random
        if 'direction' not in info:
            info['direction'] = np.random.randint(0, 4)

        # semi-randomly change direction
        if np.random.uniform(0, 1) < 0.7:
            info['direction'] = np.random.randint(0, 4)

        # if we hit the edge of the grid, turn around so that we don't go over the edge
        # TODO: look at this more closely
        if currentPos[0] == 1 or currentPos[0] == 26 or currentPos[1] == 1 or currentPos[1] == 26:
            info['direction'] = (info['direction'] + 2) % 4

        # move in the direction
        move = info['direction']

        return move


    # import the mnist dataset
    _, (x_test, y_test) = mnist.load_data()

    ix1 = np.random.randint(0, x_test.shape[0], 1000)
    ix2 = np.random.randint(0, x_test.shape[0], 1000)
    x_test1 = x_test[ix1]
    x_test2 = x_test[ix2]
    labels1 = y_test[ix1]
    labels2 = y_test[ix2]


    # normalise the images (floats between -1 and 1)
    # x_test = (x_test.astype(np.float32) - 127.5) / 127.5

    # # # To show the environment updating in real time, we use pygame
    # # plt.ion()
    # pygame.init()
    # pygame.display.set_caption('Grid Environment')
    # window = pygame.display.set_mode(((280 * 3), (280 * 2) + 75))

    # testing the environment
    # env = GridEnv()

    models = {
        'u-net': '../models_final/unet_mse/saved_model/gen.keras',
        #'u-net_dynamic': '../models_dynamic/unet_mse/saved_model/gen.keras',
        #'acgan': '../models_final/unet_acgan/saved_model/gen_6000.keras',
        #'acgan_dynamic': '../models_dynamic/unet_acgan/saved_model/gen_10000.keras',
    }

    num_runs = 50


    def process_model(model_name, model_path, render=True):
        env = GridEnv()
        # create file to write to
        f = open(f'run_data_dynamic/{model_name}.csv', 'w')
        f.write('run,label,iter,agent,prediction,naive_prediction\n')

        env.set_model(model_path)

        if model_name[-8:] == '_dynamic':
            env.set_decay('exponential', 0.008)
        else:
            env.set_decay('exponential', 0.0)

        for run in range(num_runs):
            print(f"Testing agents with {model_name} on image {run}, label {y_test[run]}")

            env.reset(image1=x_test1[run], image2=x_test2[run], label1=labels1[run], label2=labels2[run])  # load the image


            for agent in env.agent_iter():
                observation, _, terminated, truncated, info = env.last()

                if terminated:
                    action = None
                else:
                    action = policy(observation[0], info)

                if render and agent == 'agent_0':
                    env.render()

                env.step(action)
                # write to file
                f.write(f"{run},{env.getCurrentLabel()},{info['age']},{agent},{info['cnn_gen_pred']},{info['cnn_naive_pred']}\n")

                if all(env.terminations.values()):
                    print('All agents terminated')
                    break

        f.close()
        env.close()


    if multi_thread:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks to the executor.
            future_to_model = {executor.submit(process_model, model_name, model_path, False): model_name for
                               model_name, model_path in models.items()}

            # Wait for the tasks to complete and handle any exceptions.
            for future in concurrent.futures.as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"{model_name} generated an exception: {exc}")

    else:
        for model_name, model_path in models.items():
            process_model(model_name, model_path, render=True)


    waiting_for_close = True
    while waiting_for_close:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting_for_close = False

    pygame.quit()
