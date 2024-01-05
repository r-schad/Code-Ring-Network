import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Ring_Layer import RingLayer
from Code_Layer import CodeLayer
from Duration_Layer import DurationLayer
from Map_Layer import MapLayer
from utilities import exponential, sigmoid
from datetime import datetime as dt
import os

class CodeRingNetwork:
    def __init__(self, num_ring_units: int, num_code_units: int, code_factor: int, num_dur_units: int,
                 map_d1: int, map_d2: int, init_map_lr: float = 0.1, init_map_nhood: float = 3,
                 activity_scale: float = 0.2, init_delta: float = 0.99) -> None:
        '''
        The class for the Code Ring Network. The architecture is as follows:
            - Map Layer: A Self-Organizing Feature Map (SOFM) for encoding sequences
                NOTE: The Map Layer has bidirectional weights <-> the code layer. These weights
                are equivalent and remain equivalent throughout the learning process.
            - Code Layer: A sequence-encoding layer for generating sequences for producing drawings
            - Duration Layer: currently not implemented, right now just returns an appropriately-sized list
                of whatever duration values are provided to the init() function
            - Ring Layer: a spiking neural controller with dynamics for generating drawings 
                based on inputs provided by the Code (and Duration) layers
        
        :param num_ring_units int: number of neurons in ring layer
        :param num_code_units int: total number of neurons in code layer
        :param code_factor int: width of the ring-shape in the code layer, i.e. how thick the strip is
            NOTE: num_code_units / num_ring_units must equal code_factor
        :param num_dur_units int: total number of neurons in duration layer. 
            NOTE: Right now, num_dur_units should equal num_ring_units
        :param map_d1 int: number of rows in the SOFM
        :param map_d2 int: number of columns in the SOFM
        :param init_map_lr float: initial SOFM learning rate (eta)
            NOTE: this parameter is currently static throughout training
        :param init_map_nhood float: inital SOFM neighborhood range (sigma_M)
            NOTE: this parameter is currently static throughout training
        :param activity_scale float: the scaling factor of the Map <-> Code weights, 
            and the noise generated on the Code layer
        :param init_delta float: the inital weighting value of the noise generated on the code layer 
            as opposed to the weighting of the map signal propagated forward to the code layer

        :returns: None
        '''
        
        self.id_string = str(dt.now()).replace(':', '').replace('.','')
        print(f'ID string: {self.id_string}')
        if not os.path.isdir('output'):
            os.mkdir('output')

        self.folder_name = f'output\\{self.id_string}'
        os.mkdir(self.folder_name)
        
        self.ring_layer = RingLayer(num_ring_units)
        self.code_layer = CodeLayer(num_map_units=(map_d1*map_d2), num_ring_units=num_ring_units,
                                    num_code_units=num_code_units, code_factor=code_factor,
                                    code_ring_spread=0.02)
        self.duration_layer = DurationLayer(num_dur_units, (map_d1*map_d2))
        self.map_layer = MapLayer(map_d1, map_d2, num_ring_units, num_code_units,
                                  init_map_lr, init_map_nhood, weight_scale=activity_scale)
        self.activity_scale = activity_scale
        # delta is the effect of code noise vs the map activity in determining code activity
        self.init_delta = init_delta
        self.delta = init_delta

    def train(self, num_epochs: int, durations: float, t_max: int, t_steps: int, plot_gif: bool = False) -> list:
        '''
        Trains the code-ring network by generating inputs consisting of 
            random activity on the map and code layers, determining the output
            of the ring layer based on that activity, and updating the 
            bidirectional weights between the code and map layers based on the 
            classical SOFM learning algorithm.
        
        :param num_epochs int: the number of iterations to train the model over
        :param durations float: the duration value output from the duration layer for each neuron
            FIXME: this is temporary, will need removed once duration layer is changed
        :param t_max int: the maximum time to integrate each iteration over
        :param t_steps int: the number of timesteps to integrate over from [0, t_max]
        :param plot_gif bool: whether a GIF video should be saved by the doodling process

        :returns scores list: list of the scores of the doodles over time      
        '''
        map_size = int(self.map_layer.d1 * self.map_layer.d1)
        scores = []
        for epoch in tqdm(range(num_epochs)):
            # get map activity by choosing a random winning neuron
            rand_map_winner_idx = np.random.choice(map_size)
            # then activate the neighborhood around the winner
            rand_map_winner = self.map_layer.convert_to_coord(rand_map_winner_idx)
            map_signal = self.map_layer.neighborhood(rand_map_winner)
            
            # propagate map signal forward
            weighted_map_signal = self.map_layer.weights_to_code_from_map @ map_signal.reshape(map_size, 1)
            map_activation = weighted_map_signal / np.sum(map_signal) # TODO: figure out how to normalize this

            # apply random babbling signal into code layer
            code_noise = np.random.rand(self.code_layer.num_code_units, 1) * self.activity_scale
            # get combined input into code layer by applying delta to babbling noise vs the map activation
            code_input = self.delta * code_noise + (1 - self.delta) * map_activation
           
            # determine output of code layer (input into ring layer)
            ring_input = self.code_layer.weights_to_ring_from_code @ code_input

            # determine activity of duration layer
            # TODO: right now, this is just a constant
            dur_output = self.duration_layer.activate(durations)

            # determine quality of the output drawing
            doodle_score, (_xs, _ys), _intersec_pts = self.ring_layer.activate(ring_input, dur_output, t_max=t_max, t_steps=t_steps,
                                                              folder_name=self.folder_name, epoch=epoch,
                                                              plot_results=True, plot_gif=plot_gif, 
                                                              vars_to_plot={'v': False, 'u': False, 'z': True, 'I_prime': False})
            
            # determine the most similar neuron to the activity of the code layer
            map_winner = self.map_layer.forward(code_input)

            # update the weights (bidirectionally, so both weight matrices M<->C) based on quality of the output
            self.map_layer.update_weights(code_input, map_winner, doodle_score)

            # decrease the neighborhood range
            self.map_layer.nhood_range = exponential(epoch, -0.002, self.map_layer.init_nhood_range) # TODO: parameterize decay rate

            # decrease the influence of the code babbling signal
            self.delta = exponential(epoch, rate=-0.002, init_val=self.init_delta) # TODO: parameterize decay rate

            print(f'Epoch {epoch}: {doodle_score}')
            scores += [doodle_score]

        return scores
    
    def show_map_results(self, filename, durations: float, t_max: int, t_steps: int) -> None:
        '''
        Saves the outputted drawing from each map neuron in one overall figure.

        :param filename str: the filename of the outputted plot
        :param durations float: the durations inputted into the code layer (this is temporary until duration layer is implemented)
        :param t_max int: the maximum time to integrate each iteration over
        :param t_steps int: the number of timesteps to integrate over from [0, t_max]

        :returns: None
        '''
        print('Saving Map Results Plot')
        map_size = self.map_layer.d1 * self.map_layer.d2
        fig, axs = plt.subplots(self.map_layer.d1, self.map_layer.d2, figsize=(self.map_layer.d1,self.map_layer.d2), sharex=True, sharey=True)

        activity_matrix = np.zeros((self.map_layer.d1, self.map_layer.d2))
        for i in tqdm(range(map_size)):
            (r, c) = self.map_layer.convert_to_coord(i)
            activity_matrix[r, c] = 1.0
            weighted_map_signal = self.map_layer.weights_to_code_from_map @ activity_matrix.reshape(map_size, 1)
            map_activation = weighted_map_signal / np.sum(activity_matrix)

            # determine output of code layer (input into ring layer)
            ring_input = self.code_layer.weights_to_ring_from_code @ map_activation

            # determine activity of duration layer
            # TODO: right now, this is just a constant
            dur_output = self.duration_layer.activate(durations)

            # determine quality of the output drawing
            doodle_score, (xs, ys), intersec_pts = self.ring_layer.activate(ring_input, dur_output, t_max=t_max, t_steps=t_steps,
                                                              folder_name=self.folder_name, epoch=-1,
                                                              plot_results=False, plot_gif=False, 
                                                              vars_to_plot={'v': False, 'u': False, 'z': False, 'I_prime': False})
            
            # generate drawing for current neuron
            self.ring_layer.plot_final_doodle(ax=axs[r][c], xs=xs, ys=ys, intersec_pts=intersec_pts, individualize_plot=False)
            axs[r][c].set_xlabel(f'{np.round(doodle_score,3)}')
            activity_matrix[r, c] = 0.0

        plt.tight_layout()
        fig.savefig(filename)
        plt.close()


    
if __name__ == '__main__':
    r = 36
    cf = 1
    c = cf*r
    d = 36
    durs = 0.2
    m_d1 = 8
    m_d2 = 8
    init_lr = 0.05
    init_map_sigma = 2
    initial_delta = 0.8
    num_epochs = 500
    t_max = 30
    t_steps = 300
    activity_scale = 0.5
    crn = CodeRingNetwork(num_ring_units=r,
                          num_code_units=c,
                          code_factor=cf,
                          num_dur_units=d,
                          map_d1=m_d1, map_d2=m_d2,
                          init_map_lr=init_lr,
                          init_map_nhood=init_map_sigma,
                          activity_scale=activity_scale,
                          init_delta=initial_delta)
    
    id_string = crn.folder_name.split('\\')[-1]

    sigmoid_mu = np.mean(crn.map_layer.neighborhood((0,0)) * crn.map_layer.d1 * crn.map_layer.d2) * activity_scale

    crn.show_map_results(f'{crn.folder_name}\\initial_outputs_{id_string}.png', durs, t_max, t_steps)
    scores = crn.train(num_epochs, durs, t_max, t_steps, plot_gif=False)
    crn.show_map_results(f'{crn.folder_name}\\final_outputs_{id_string}.png', durs, t_max, t_steps)
    plt.plot(range(num_epochs), scores)
    plt.title(f'Scores Over Time {id_string}')
    plt.savefig(f'{crn.folder_name}\\all_scores_{id_string}.png')
    pass
