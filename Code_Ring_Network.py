import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Ring_Layer import RingLayer
from Code_Layer import CodeLayer
from Duration_Layer import DurationLayer
from Map_Layer import MapLayer
from utilities import exponential_decay

class CodeRingNetwork:
    def __init__(self, num_ring_units: int, num_code_units: int, code_factor: int, num_dur_units: int,
                 map_d1: int, map_d2: int, init_map_lr: float = 0.1, init_map_nhood: float = 3, 
                 activity_scale: float = 0.2, init_delta: float = 0.99) -> None:
        
        self.ring_layer = RingLayer(num_ring_units)
        self.code_layer = CodeLayer(num_map_units=(map_d1*map_d2), num_ring_units=num_ring_units, 
                                    num_code_units=num_code_units, code_factor=code_factor)
        self.duration_layer = DurationLayer(num_dur_units, (map_d1*map_d2))
        self.map_layer = MapLayer(map_d1, map_d2, num_ring_units, num_code_units, 
                                  init_map_lr, init_map_nhood, weight_scale=activity_scale)
        self.activity_scale = activity_scale
        # delta is the effect of code noise vs the map activity in determining code activity
        self.delta = init_delta

    def train(self, num_epochs: int, t_max: int, t_steps: int, plot_gif: bool = False) -> list:
        '''
        Trains the code-ring network by generating inputs consisting of 
            random activity on the map and code layers, determining the output
            of the ring layer based on that activity, and updating the 
            bidirectional weights between the code and map layers based on the 
            classical SOFM learning algorithm.
        
        :param num_epochs int: the number of iterations to train the model over
        :param t_max int: the maximum time to integrate each iteration over
        :param t_steps int: the number of timesteps to integrate over from [0, t_max]
        :param plot_gif bool: whether a GIF video should be saved by the doodling process

        :returns scores list: list of the scores of the doodles over time      
        '''
        durations = 0.2
        map_size = int(self.map_layer.d1 * self.map_layer.d1)
        scores = []
        for epoch in tqdm(range(num_epochs)):
            # get map activity by choosing a random winning neuron
            map_winner_idx = np.random.choice(map_size)
            # set activity matrix to all zeros
            map_activity = np.zeros((self.map_layer.d1, self.map_layer.d2))
            # then turn coordinates of winner in the activity matrix into a 1
            map_winner = self.map_layer.convert_to_coord(map_winner_idx)
            map_activity = self.map_layer.neighborhood(map_winner)
            weighted_map_activity = self.map_layer.weights_to_code_from_map @ map_activity.reshape(map_size, 1)

            # apply random babbling signal into code layer
            code_noise = np.random.rand(self.code_layer.num_code_units, 1) * self.activity_scale
            # get combined input into code layer by applying delta to babbling noise vs the weighted map activity
            code_input = self.delta * code_noise + (1 - self.delta) * weighted_map_activity
           
            # determine output of code layer (input into ring layer)
            ring_input = self.code_layer.weights_to_ring_from_code @ code_input

            # determine activity of duration layer
            # TODO: right now, this is just a constant
            dur_output = self.duration_layer.activate(durations)

            # determine quality of the output drawing
            doodle_score, (xs, ys) = self.ring_layer.activate(ring_input, dur_output, t_max=t_max, t_steps=t_steps, epoch=epoch, plot_gif=plot_gif)
            
            # determine the most similar neuron to the activity of the code layer
            map_winner = self.map_layer.forward(code_input)
            # update the weights (bidirectionally, so both weight matrices M<->C) based on quality of the output
            self.map_layer.update_weights(code_input, map_winner, doodle_score)

            # increase the influence of the map as opposed to the code babbling signal
            self.delta = exponential_decay(epoch, decay_rate=0.02, init_val=1.0)

            print(f'Epoch {epoch}: {doodle_score}')
            scores += [doodle_score]

        return scores
    
if __name__ == '__main__':
    r = 36
    c = 72
    cf = 2
    d = 36
    m_d1 = 12
    m_d2 = 12
    init_lr = 0.1
    init_map_sigma = 2
    initial_delta = 1.0
    num_epochs = 1000
    crn = CodeRingNetwork(r, c, cf, d, m_d1, m_d2, init_lr, init_map_sigma, initial_delta)
    scores = crn.train(num_epochs, 30, 300, plot_gif=True)
    plt.plot(range(num_epochs), scores)
    pass
