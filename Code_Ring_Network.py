import numpy as np
import matplotlib
from scipy.signal import find_peaks
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tqdm import tqdm
import imageio
import os
from datetime import datetime as dt
import pandas as pd

from Ring_Layer import RingLayer
from Code_Layer import CodeLayer
from Duration_Layer import DurationLayer
from Map_Layer import MapLayer
from utilities import get_color_range, exponential, write_params, dist, bimodal_gaussian_noise

class CodeRingNetwork:
    def __init__(self, num_ring_units: int, num_code_units: int, code_factor: int, num_dur_units: int,
                 map_d1: int, map_d2: int, **init_kwargs) -> None:
        '''
        The class for the Code Ring Network. The architecture is as follows:
            - Map Layer: A Self-Organizing Feature Map (SOFM) for encoding sequences
                NOTE: The Map Layer has bidirectional weights <-> the code layer. These weights
                are equivalent and remain equivalent throughout the learning process.
            - Code Layer: A sequence-encoding layer for generating sequences for producing drawings
            - Duration Layer: currently not implemented, right now just returns an appropriately-sized list
                of whatever duration values are provided to the init() function
            - Ring Layer: a spiking neural model with dynamics for generating drawings 
                based on inputs provided by the Code (and Duration) layers
        
        :param num_ring_units int: number of neurons in ring layer
        :param num_code_units int: total number of neurons in code layer
        :param code_factor int: width of the ring-shape in the code layer, i.e. how thick the strip is
            NOTE: num_code_units / num_ring_units must equal code_factor
        :param num_dur_units int: total number of neurons in duration layer. 
            NOTE: Right now, num_dur_units should equal num_ring_units
        :param map_d1 int: number of rows in the SOFM
        :param map_d2 int: number of columns in the SOFM
        :param init_nhood_range float: inital SOFM neighborhood range (sigma_M)
        :param weight_min float: the minimum value of the M<->C weights
        :param weight_max float: the maximum value of the M<->C weights
        :param code_ring_spread float: the standard deviation of the Code->Ring weights Gaussian

        :returns: None
        '''
        # administrative/utility variables
        self.id_string = str(dt.now()).replace(':', '').replace('.','')
        print(f'ID string: {self.id_string}')
        if not os.path.isdir('output'):
            os.mkdir('output')
        self.folder_name = f'output\\{self.id_string}'
        os.mkdir(self.folder_name)

        self.vars_to_plot = {'z': True, 'v': False, 'u': False, 'r': False}
        self.COLOR_RANGE = get_color_range(num_ring_units, map_name='hsv')

        # layer initalizations
        self.ring_layer = RingLayer(num_ring_units)
        self.code_layer = CodeLayer(num_map_units=(map_d1*map_d2), num_ring_units=num_ring_units,
                                    num_code_units=num_code_units, code_factor=code_factor,
                                    code_ring_spread=init_kwargs['code_ring_spread'])
        self.duration_layer = DurationLayer(num_dur_units, (map_d1*map_d2))
        self.map_layer = MapLayer(map_d1, map_d2, num_ring_units, num_code_units, 
                                  **init_kwargs)
                       
    def train(self, num_epochs: int, activation_nhood: float,
              init_delta: float, delta_decay_rate: float,
              init_learning_nhood: float, learning_nhood_decay: float,
              init_learning_rate: float, learning_rate_decay: float,
              durations: float, t_max: int, max_t_steps: int,
              plot_results: bool = False, plot_gif: bool = False, **metric_kwargs) -> (list, list):
        '''
        Trains the code-ring network by generating inputs consisting of 
            random activity on the code layer, determining the output
            of the ring layer based on that activity, and updating the 
            bidirectional weights between the code and map layers based on the 
            SOFM learning algorithm given the quality of the outputted doodle.
        
        :param num_epochs int: the number of iterations to train the model over
        :param activation_nhood float: the std. dev. (sigma) of the neighborhood for map layer activity
            NOTE: this stays static throughout learning so the weights don't have to adapt to the changing
                distribution of map signals
        :param init_delta float: the initial value of the influence of the noise on the code layer
        :param delta_decay_rate float: the exponential decay rate of the influence of the noise on the code layer
        :param init_learning_nhood float: the initial value of the the std. dev. (sigma) 
            of the Gaussian neighborhood for the map weight update equation
        :param learning_nhood_decay float: the exponential decay rate of the weight update Gaussian neighborhood sigma
        :param init_learning_rate float: the initial value of learning_rate
        :param learning_rate_decay float: the exponential decay rate of learning_rate
            NOTE: if this value is 0, a static learning rate is being used
        :param durations float: the duration value output from the duration layer for each neuron
        :param t_max int: the maximum time to integrate each iteration over
        :param max_t_steps int: the max number of timesteps to integrate over from [0, t_max].
            NOTE: the true value of timesteps can be less if integration is halted once all effectors
                run out of resource
        :param plot_results bool: whether the final doodle and the timeseries for all variables
            passed in the kwargs should be saved
        :param plot_gif bool: whether a GIF video should be saved by the doodling process

        :returns strictest_scores, scores (list, list): list of the scores of the doodles over time,
            first, when using hte strictest version of the metric, then with using the effective version
            of the metric at that current epoch      
        '''
        map_size = int(self.map_layer.d1 * self.map_layer.d2)
        map_neuron_idxs = np.arange(map_size)

        strictest_scores = np.ndarray((num_epochs, map_size))
        scores = np.ndarray((num_epochs, map_size))
        activity_counts = np.zeros((num_epochs, map_size))
        winner_counts = np.zeros((num_epochs, map_size))

        learning_rate = init_learning_rate
        learning_nhood = init_learning_nhood
        delta = init_delta

        # initialize the strictness of the quality metric
        metric_kwargs['sigma_Q'] = metric_kwargs['sigma_Q_init']

        self.show_map_results(f'{self.folder_name}\\map_train_begin{self.id_string}.png', durations, t_max, max_t_steps, False, **metric_kwargs)
        plt.matshow(self.map_layer.weights_to_code_from_map, vmin=0, vmax=1)
        plt.colorbar()
        plt.title('Weights at Training Beginning')
        plt.xlabel('Map Neurons')
        plt.ylabel('Code Neurons')
        plt.savefig(f'{self.folder_name}\\weights_train_begin_{self.id_string}.png')
        plt.close()

        num_active_ideal = len(metric_kwargs['ideal_seq'])
        
        for epoch in tqdm(range(num_epochs)):
            # randomly shuffle the order of map neuron activations
            np.random.shuffle(map_neuron_idxs)
            for iteration, rand_active_idx in enumerate(tqdm(map_neuron_idxs)):
                # ordered-shape generator
                if metric_kwargs['method'] == 'o':
                    sigma_delta = metric_kwargs['sigma_delta_init'] # static sigma_delta
                    code_output = self.ordered_shape_generator(metric_kwargs['ideal_seq'], sigma_delta)

                # unordered-shape generator (AKA noise generator)
                elif metric_kwargs['method'] == 'u':
                    sigma_delta = metric_kwargs['sigma_delta_init'] # static sigma_delta
                    code_output = self.unordered_shape_generator(metric_kwargs['ideal_seq'], sigma_delta)

                else: # method uses map influence
                    rand_active_neuron = self.map_layer.convert_to_coord(rand_active_idx)

                    # apply static neighborhood range for activation neighborhood
                    # to activate the neighborhood around the winner
                    map_signal = self.map_layer.neighborhood(rand_active_neuron, sigma=activation_nhood)
                    
                    # propagate map signal forward
                    map_activation = self.map_layer.weights_to_code_from_map @ map_signal.reshape(map_size, 1)

                    if metric_kwargs['noise_type'] == 'bmgs': # bimodal gaussian, shuffled
                        code_noise = bimodal_gaussian_noise(num_low=metric_kwargs['noise_num_low'],
                                                            num_high=metric_kwargs['noise_num_high'],
                                                            mean_low=metric_kwargs['noise_mean_low'],
                                                            mean_high=metric_kwargs['noise_mean_high'],
                                                            sigma_low=metric_kwargs['noise_sigma_low'],
                                                            sigma_high=metric_kwargs['noise_sigma_high'],
                                                            shuffle=True,
                                                            clip_01=True).reshape(self.code_layer.num_code_units, 1)
                    
                    elif metric_kwargs['noise_type'] == 'bmgu': # bimodal gaussian, unshuffled
                        code_noise = bimodal_gaussian_noise(num_low=metric_kwargs['noise_num_low'],
                                                            num_high=metric_kwargs['noise_num_high'],
                                                            mean_low=metric_kwargs['noise_mean_low'],
                                                            mean_high=metric_kwargs['noise_mean_high'],
                                                            sigma_low=metric_kwargs['noise_sigma_low'],
                                                            sigma_high=metric_kwargs['noise_sigma_high'],
                                                            shuffle=False,
                                                            clip_01=True).reshape(self.code_layer.num_code_units, 1)

                    # additive map influence
                    if metric_kwargs['method'] == 'a':
                        code_output = self.additive_map_influence(map_activation, code_noise, delta, num_active_ideal)

                    # mixture map influence
                    elif metric_kwargs['method'] == 'm':
                        code_output = self.mixture_map_influence(map_activation, code_noise, delta, num_active_ideal)

                    # rotational map influence
                    elif metric_kwargs['method'] == 'r':
                        sigma_delta = (metric_kwargs['sigma_delta_init'] * delta) + 1
                        code_output = self.rotational_map_influence(map_activation, delta, num_active_ideal)

                # determine output of code layer (input into ring layer)
                ring_input = (self.code_layer.weights_to_ring_from_code @ code_output).squeeze()

                # determine activity of duration layer
                # TODO: right now, this is just a constant
                dur_output = self.duration_layer.activate(durations)

                # integrate ring layer model over time
                v_series, z_series, u_series, r_series = self.ring_layer.activate(ring_input, dur_output, t_max=t_max, t_steps=max_t_steps)

                # limit t_steps to end of integration
                t_steps = z_series.shape[1]
                
                # apply model results to the drawing system
                x_series, y_series = self.ring_layer.create_drawing(z_series, t_steps)

                # evaluate drawing
                score = self.angle_metric(z_series, x_series=x_series, y_series=y_series, use_final=False, **metric_kwargs)
                strictest_score = self.angle_metric(z_series, x_series=x_series, y_series=y_series, use_final=True, **metric_kwargs)
                intersec_pts = np.ndarray((0,2)) # TODO: these aren't needed anymore
                intersec_times = np.array([])

                # determine the most similar neuron to the activity of the code layer
                map_winner = self.map_layer.forward(code_output)
                winner_idx = self.map_layer.convert_to_index(map_winner)

                # update the weights (bidirectionally, so both weight matrices M<->C)
                # update is based on output score, current nhood range, and learning rate
                self.map_layer.update_weights(code_output, map_winner,
                                            nhood_sigma=learning_nhood, learning_rate=learning_rate,
                                            score=score)

                print(f'Training Epoch {epoch} | Active Neuron {rand_active_idx} | Winner Neuron {winner_idx}: {score}')

                activity_counts[epoch][rand_active_idx] += 1
                winner_counts[epoch][winner_idx] += 1
                strictest_scores[epoch][rand_active_idx] = strictest_score
                scores[epoch][rand_active_idx] = score

                # plot ring layer variables over time
                plot_v = v_series if self.vars_to_plot['v'] else []
                plot_u = u_series if self.vars_to_plot['u'] else []
                plot_z = z_series if self.vars_to_plot['z'] else []
                plot_r = r_series if self.vars_to_plot['r'] else []
                if plot_results:
                    self.plot_results(x_series, y_series, intersec_pts,
                                ring_inputs=ring_input,
                                v=plot_v, u=plot_u, z=plot_z, r=plot_r,
                                folder_name=self.folder_name,
                                epoch=epoch, iteration=iteration,
                                active_idx=rand_active_idx, winner_idx=winner_idx,
                                score=strictest_score, plot_gif=plot_gif)

                if plot_gif:
                    self.create_gif(x_series, y_series, t_steps, intersec_pts, intersec_times, self.folder_name, epoch)

            # decrease the influence of the code babbling signal
            delta = exponential(epoch, rate=delta_decay_rate, init_val=init_delta)

            # calculate new sigma_Q
            metric_kwargs['sigma_Q'] = metric_kwargs['sigma_Q_init'] + (epoch * (metric_kwargs['sigma_Q_final'] - metric_kwargs['sigma_Q_init']) / num_epochs)
            
            # decrease the neighborhood range
            learning_nhood = exponential(epoch, rate=learning_nhood_decay, init_val=init_learning_nhood)

            # decrease the learning rate (if learning_rate_decay == 0, use static learning rate)
            learning_rate = exponential(epoch, rate=learning_rate_decay, init_val=init_learning_rate)
 
            # plot results every 100 epochs
            if not(epoch % 100):
                self.show_map_results(f'{self.folder_name}\\map_train_epoch{epoch}_{self.id_string}.png', durations, t_max, max_t_steps, False, **metric_kwargs)
                plt.matshow(self.map_layer.weights_to_code_from_map, vmin=0, vmax=1)
                plt.colorbar()
                plt.title(f'Weights at Epoch {epoch}')
                plt.xlabel('Map Neurons')
                plt.ylabel('Code Neurons')
                plt.savefig(f'{self.folder_name}\\weights_train_epoch{epoch}_{self.id_string}.png')
                plt.close()
        
        self.show_map_results(f'{self.folder_name}\\map_train_final{self.id_string}.png', durations, t_max, max_t_steps, False, **metric_kwargs)

        plt.matshow(self.map_layer.weights_to_code_from_map, vmin=0, vmax=1)
        plt.colorbar()
        plt.title('Weights at Training Final')
        plt.xlabel('Map Neurons')
        plt.ylabel('Code Neurons')
        plt.savefig(f'{self.folder_name}\\weights_train_final_{self.id_string}.png')
        plt.close()

        plt.matshow(winner_counts.T)
        plt.colorbar()
        plt.title(f'Map Win Counts {self.id_string}')
        plt.xlabel('Epochs')
        plt.ylabel('Winning Map Neuron')
        plt.savefig(f'{self.folder_name}\\map_win_counts_heatmap_{self.id_string}.png')
        plt.close()

        return strictest_scores, scores
    
    def save_model_params(self, params, filename):
        '''
        Saves out the weight matrix saved in `params`.

        :param params np.array: the array of parameters
        :param filename str: the filename of the outputted excel file
        '''
        df = pd.DataFrame(params)
        df.to_excel(filename, index=False, header=False)
    
    def load_model_params(self, filename):
        '''
        Loads in an excel file of model parameters and returns the numpy array.

        :param filename str: the filename of the inputted excel file
        '''
        return pd.read_excel(filename, header=None).to_numpy()
    
    def show_map_results(self, filename, durations: float, t_max: int, max_t_steps: int, plot_results: bool, **metric_kwargs) -> None:
        '''
        Saves the outputted drawing from each map neuron in one overall figure.

        :param filename str: the filename of the outputted plot
        :param durations float: the durations inputted into the code layer (this is temporary until duration layer is implemented)
        :param t_max int: the maximum time to integrate each iteration over
        :param max_t_steps int: the maximum number of timesteps to integrate over. The actual quantity may be lower 
            if the resource in the effector layer falls below the mimimum threshold for any activity to occur.
        :param plot_results bool: whether to save out the individual output of each map neuron in its own folder. This will
            increase runtime but provide greater detail into visualizing the map's learnings.

        :returns: None
        '''
        print('Saving Map Results Plot')
        map_size = self.map_layer.d1 * self.map_layer.d2
        fig, axs = plt.subplots(self.map_layer.d1, self.map_layer.d2,
                                figsize=(self.map_layer.d1,self.map_layer.d2), sharex=True, sharey=True)

        activity_matrix = np.zeros((self.map_layer.d1, self.map_layer.d2))
        for i in tqdm(range(map_size)):
            (r, c) = self.map_layer.convert_to_coord(i)
            activity_matrix[r, c] = 1.0
            code_input = self.map_layer.weights_to_code_from_map @ activity_matrix.reshape(map_size, 1)

            code_output = np.zeros((self.code_layer.num_code_units,1))
            
            # get top n indexes in input. only keep those, zero out the others for the code output
            n = len(metric_kwargs['ideal_seq'])
            top_n_idxs = np.argpartition(code_input.squeeze(), -n)[-n:]
            code_output[top_n_idxs] = code_input[top_n_idxs]

            # determine output of code layer (input into ring layer)
            ring_input = self.code_layer.weights_to_ring_from_code @ code_output

            # determine activity of duration layer
            # TODO: right now, this is just a constant
            dur_output = self.duration_layer.activate(durations)

            # integrate ring layer model over time
            v_series, z_series, u_series, r_series = self.ring_layer.activate(ring_input, dur_output, t_max=t_max, t_steps=max_t_steps)

            # limit t_steps to end of integration
            t_steps = z_series.shape[1]

            # apply model results to the drawing system
            x_series, y_series = self.ring_layer.create_drawing(z_series, t_steps)

            # evaluate drawing
            score = self.angle_metric(z_series, x_series=x_series, y_series=y_series, use_final=True, **metric_kwargs)
            intersec_pts = np.ndarray((0,2))
            intersec_times = np.array([])

            # determine the most similar neuron to the activity of the code layer
            map_winner = self.map_layer.forward(code_output)
            winner_idx = self.map_layer.convert_to_index(map_winner)

            # plot ring layer variables over time
            plot_v = v_series if self.vars_to_plot['v'] else []
            plot_u = u_series if self.vars_to_plot['u'] else []
            plot_z = z_series if self.vars_to_plot['z'] else []
            plot_r = r_series if self.vars_to_plot['r'] else []
            if plot_results:
                self.plot_results(xs=x_series, ys=y_series, intersec_pts=intersec_pts,
                            ring_inputs=ring_input,
                            v=plot_v, u=plot_u, z=plot_z, r=plot_r,
                            folder_name=self.folder_name,
                            epoch=filename.split('\\')[-1].split('.')[0], iteration=i,
                            active_idx=i, winner_idx=winner_idx,
                            score=score, plot_gif=False, idx_folders=True)
                                            
            # generate drawing for current neuron
            self.plot_final_doodle(ax=axs[r][c], xs=x_series, ys=y_series, intersec_pts=intersec_pts, individualize_plot=False)
            axs[r][c].set_xlim([-20,20])
            axs[r][c].set_ylim([-20,20])
            axs[r][c].set_xlabel(f'{np.round(score,3)}')
            axs[r][c].set_box_aspect(1)

            # reset active neuron to inactive
            activity_matrix[r, c] = 0.0

        plt.tight_layout()
        fig.savefig(filename)
        plt.close(fig)

    def plot_results(self, xs: np.array, ys: np.array, intersec_pts: np.ndarray,
                     ring_inputs: np.array, v: np.ndarray, u: np.ndarray,
                     z: np.ndarray, r: np.ndarray,
                     folder_name: str,
                     epoch: int, iteration: int, active_idx: int, winner_idx: int,
                     score: float, plot_gif=False, idx_folders=False) -> None:
        '''
        Plots the final resulting doodle and the variable activity graph of the ring layer. 
            The plots are saved to directory: `folder_name`\\`epoch` if plot_gif is True.
            Else, they're just saved in `folder_name`.
        
        :param xs np.array: array of x-coordinates over time
        :param ys np.array: array of y-coordinates over time
        :param intersec_pts np.ndarray: array of shape (t_steps, 2) with 
            the [x, y] coordinates of each intersection point of the doodle
        :param ring_inputs np.array: the array of inputs into the ring layer
        :param v np.ndarray: array of shape (num_ring_neurons, t_steps) of the series of v (activation) values of the ring layer. 
            If not being plotted, will be [].    
        :param u np.ndarray: array of shape (num_ring_neurons, t_steps) of the series of u (deactivation) values of the ring layer. 
            If not being plotted, will be [].        
        :param z np.ndarray: array of shape (num_ring_neurons, t_steps) of the series of z (output) values of the ring layer. 
            If not being plotted, will be [].        
        :param r np.ndarray: array of shape (num_ring_neurons, t_steps) of the series of r (resource) values of the ring layer. 
            If not being plotted, will be [].
        :param folder_name str: the model instance's corresponding folder name
        :param epoch int: the current epoch (just for plotting purposes)
        :param iteration int: the current iteration within the current epoch (just for plotting purposes)
        :param active_idx int: the index of the active map neuron (just for plotting purposes)
        :param winner_idx int: the index of the winner map neuron (just for plotting purposes)
        :param score float: the score of the outputted doodle (just for plotting purposes)
        :param plot_gif bool: whether to plot a GIF for each episode. This is used to determine the trial's folder structure
        :param idx_folders bool: whether each active map neuron's iterations should be kept in a separate folder or not.
            NOTE: this gives greater clarity in the way one neuron may learn over time.

        :returns: None
        '''
        f, axs = plt.subplots(1, 2)
        self.plot_final_doodle(axs[0], xs, ys, intersec_pts)
        self.plot_activity(axs[1], ring_inputs, v, u, z, r)

        f.suptitle(f'Epoch {epoch}, Iteration {iteration}\n\
                   Score = {np.round(score,2)} | Active = {active_idx} | Winner = {winner_idx}',
                   fontsize=12)
        f.tight_layout()
        
        if plot_gif:
            if not os.path.isdir(f'{folder_name}\\{epoch}'):
                os.makedirs(f'{folder_name}\\{epoch}')
            f.savefig(f'{folder_name}\\{epoch}\\plot_{self.id_string}_epoch{epoch}')
        elif idx_folders:
            if not os.path.isdir(f'{folder_name}\\neurons\\{active_idx}'):
                os.makedirs(f'{folder_name}\\neurons\\{active_idx}')
            f.savefig(f'{folder_name}\\neurons\\{active_idx}\\plot_{self.id_string}_idx{active_idx}_epoch{epoch}')
        else:
            if not os.path.isdir(f'{folder_name}\\plots'):
                os.makedirs(f'{folder_name}\\plots')
            f.savefig(f'{folder_name}\\plots\\plot_epoch{epoch}_iteration{iteration}_{self.id_string}')

        plt.close(f)

    def plot_final_doodle(self, ax: plt.axis,
                          xs: np.array, ys: np.array,
                          intersec_pts: np.ndarray,
                          individualize_plot: bool = True) -> None:
        '''
        Plots the final doodle.

        :param ax matplotlib.pyplot.axis: the axis object to plot on
        :param xs np.array: array of x-coordinates over time
        :param ys np.array: array of y-coordinates over time
        :param intersec_pts np.ndarray: array of shape (t_steps, 2) with 
            the [x, y] coordinates of each intersection point of the doodle
        :param individualize_plot bool: whether the plot should be individualized to fit
            that specific doodle's range of outputs, include legend, etc. This should be 
            True for most cases, but False when using CodeRingNetwork.show_results().

        :returns: None
        '''
        # plot lines
        ax.plot(xs, ys, alpha=0.5, c='black')

        if individualize_plot:
            # plot initial pen point
            ax.scatter(xs[0], ys[0], alpha=0.8, marker = 'o', c='black', label='Origin')
            # organize plot
            ax.set_xlim([-20,20])
            ax.set_xlabel('x', fontsize = 14)
            ax.set_ylim([-20,20])
            ax.set_ylabel('y', fontsize = 14)
            ax.set_box_aspect(1)
            ax.set_title('Final Output')
            ax.legend()
            intersec_point_size = 20
            
        else:
            intersec_point_size = 4

        # plot all intersection points (if any)
        if intersec_pts.any():
            ax.scatter(intersec_pts[:,0], intersec_pts[:,1],
                       color='red', marker='o', s=intersec_point_size,
                       label='Intersections')

    def plot_activity(self, ax: plt.axis,
                      ring_inputs: np.ndarray, v: np.ndarray = [], u: np.ndarray = [],
                      z: np.ndarray = [], r: np.ndarray = [], max_t_steps: int = 700) -> None:
        '''
        Plots the time series of the variables involved with the ring layer.

        :param ax matplotlib.pyplot.axis: the axis object to plot on
        :param ring_inputs np.ndarray: "r^0"; the array of inputs into the ring layer
            ring_inputs determine the order of activation of the ring neurons
        :param v np.ndarray: the activation (v) series of each ring neuron
            If not being plotted, will be [].   
        :param u np.ndarray: the deactivation (u) series of each ring neuron
            If not being plotted, will be [].   
        :param z np.ndarray: the output (z) series of each ring neuron
            If not being plotted, will be [].   
        :param r np.ndarray: the resource (r) series of each ring neuron
            If not being plotted, will be [].   

        :returns: None
        '''
        # include 8 most active ring neurons in legend
        sorted_inputs = np.flip(np.argsort(ring_inputs.squeeze()))
        for i in sorted_inputs[:4]:
            color = self.COLOR_RANGE[i]
            if np.any(v):
                plt.plot(v[i], label=f'v_{i}', c=color, linestyle='dashed')
            if np.any(u):
                plt.plot(u[i], label=f'u_{i}', c=color, linestyle='dotted')
            if np.any(r):
                plt.plot(r[i], label=f"r_{i}", c=color, linestyle='dashdot')
            if np.any(z):
                plt.plot(z[i], label=f'z_{i}', c=color, linestyle='solid')
        
        # add '_' to beginning of these labels in the legend so they're ignored
        # we want to ignore the later half of inputs for visual clarity so legend isn't too big
        for i in sorted_inputs[4:]:
            color = self.COLOR_RANGE[i]
            if np.any(v):
                plt.plot(v[i], label=f'_v_{i}', c=color, linestyle='dashed')
            if np.any(u):
                plt.plot(u[i], label=f'_u_{i}', c=color, linestyle='dotted')
            if np.any(r):
                plt.plot(r[i], label=f"_r_{i}", c=color, linestyle='dashdot')
            if np.any(z):
                plt.plot(z[i], label=f'_z_{i}', c=color, linestyle='solid')

        # ax.legend(loc='upper right')
        ax.set_xlim([0, max_t_steps])
        tick_range = np.arange(0,max_t_steps+100,100)
        ax.set_xticks(tick_range)
        ax.set_xticklabels(tick_range / 10)
        ax.set_ylim([0, 1])
        ax.set_xlabel('t')
        ax.set_title('Dynamics Plot')
        plt.axhline(y=0.0, c="black", linewidth=0.05)

    def create_frame(self, xs: np.array, ys: np.array,
                     t: int, epoch: int, ax: plt.axis, pen_color: str,
                     intersec_pts: np.ndarray, intersec_times: np.ndarray) -> None:
        '''
        Creates the frame of the doodle up to (and including) timestep t.

        :param xs np.array: array of the x-series of the doodle
        :param ys np.array: array of the y-series of the doodle
        :param t int: the current timestep in the doodle
        :param epoch int: the current epoch
        :param ax plt.axis: axis to plot on
        :param pen_color string: pyplot string for color of the pen on the plot
        :param intersec_pts np.ndarray: array of all intersection points encountered before timestep `t`
        :param intersec_times np.ndarray: array of times before `t` where an intersection occurred

        :returns: None
        '''
        assert len(xs) == len(ys), "xs and ys shape doesn't match!"
    
        # plot lines up to current timestep
        ax.plot(xs[:t+1], ys[:t+1], color=pen_color, alpha=0.5, label=f'Epoch {epoch}')
        # plot current pen point
        ax.scatter(xs[t], ys[t], color=pen_color, alpha=0.8, marker = 'o')

        # plot all intersection points up to current timestep (if any)
        intersec_idxs_to_plot = np.where(intersec_times <= t)[0]
        if np.any(intersec_idxs_to_plot):
            ax.scatter(intersec_pts[intersec_idxs_to_plot,0],
                    intersec_pts[intersec_idxs_to_plot,1],
                    color='red', marker='o')

        # organize plot
        ax.set_xlim([-1 * np.max(np.abs([xs, ys])), np.max(np.abs([xs, ys]))])
        ax.set_xlabel('x', fontsize = 14)
        ax.set_ylim([-1 * np.max(np.abs([xs, ys])), np.max(np.abs([xs, ys]))])
        ax.set_ylabel('y', fontsize = 14)
        ax.set_title(f'Step {t}', fontsize=14)
        ax.legend()

    def create_gif(self, x_series: np.array, y_series: np.array, t_steps: int,
                   intersec_pts: np.ndarray, intersec_times: np.ndarray, folder_name: str,
                   epoch: int, pen_color: str = 'black') -> None:
        '''
        Creates a GIF video of the doodling process, with intersection points showing in red.
            The GIF is saved in `folder_name`\\`epoch`.
        
        :param x_series np.ndarray: array of the x-values of the pen over time
        :param y_series np.ndarray: array of the y-values of the pen over time
        :param t_steps int: number of values of `t` over which the model will be integrated
        :param intersec_pts np.ndarray: array of all intersection points encountered before timestep `t`
        :param intersec_times np.ndarray: array of times before `t` where an intersection occurred
        :param folder_name str: the model instance's corresponding folder name
        :param epoch int: the current epoch
        :param pen_color string: pyplot string for color of the pen on the plot

        :returns None:
        '''
        # create directories
        if not os.path.isdir(f'{folder_name}\\{epoch}\\img'):
            os.makedirs(f'{folder_name}\\{epoch}\\img')

        self.id_string = folder_name.split('\\')[-1]

        frames = []
        print('Creating GIF...')
        for t in tqdm(range(t_steps)):
            f, ax = plt.subplots()

            self.create_frame(x_series, y_series, t, epoch, ax, pen_color, intersec_pts=intersec_pts, intersec_times=intersec_times)
            
            f.savefig(f'{folder_name}\\{epoch}\\img\\img_{t}.png')
            plt.close()
            image = imageio.v2.imread(f'{folder_name}\\{epoch}\\img\\img_{t}.png')
            frames.append(image)

        # duration is 1/100th seconds, per frame
        imageio.mimsave(f"{folder_name}\\{epoch}\\GIF_{self.id_string}.gif", frames, **{'duration':2.5})

    def detect_intersection(self, xs: np.array, ys: np.array, t: int) -> tuple[np.ndarray, float]:
        '''
        Given a drawing, detects the intersection points of the drawing using the formula found here:
            https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line_segment
        
        :param xs np.ndarray: a np.ndarray of the x-series of the doodle
        :param ys np.ndarray: a np.ndarray of the y-series of the doodle
        :param t int: the current timestep in the doodle

        :returns (new_intersec np.ndarray, t float): intersection point encountered at timestep `t`    
        '''
        assert t >= 3, 'There must be at least 4 points to have any intersections'
        # get point 1 (time t-1) and point 2 (time t) coordinates
        x1 = xs[t-1]
        x2 = xs[t]
        y1 = ys[t-1]
        y2 = ys[t]

        # get point 3s (times t=0,...,t-3) and point 4s (times t=1,...,t-2) coordinates
            # NOTE: we don't include the line segment from time t-2 to time t-1 because that's just going to intersect
            # with point 1 because of time t-1
        x3s = xs[:t-2]
        y3s = ys[:t-2]
        x4s = np.roll(xs, -1)[:t-2]
        y4s = np.roll(ys, -1)[:t-2]

        # find where two line segments intersect (if they do)
        numerator_t = ((x1 - x3s) * (y3s - y4s)) - ((y1 - y3s) * (x3s - x4s))
        numerator_u = ((x1 - x3s) * (y1 - y2)) - ((y1 - y3s) * (x1 - x2))
        denom = ((x1 - x2) * (y3s - y4s)) - ((y1 - y2) * (x3s - x4s))

        # (denom can't equal 0) and (0 <= n_t/d <= 1)
        intersec_t_idxs = np.nonzero((0. != denom) & (0 <= numerator_t / denom) & (numerator_t / denom <= 1))[0]

        # (denom can't equal 0) and (0 <= n_u/d <= 1)
        intersec_u_idxs = np.nonzero((0. != denom) & (0 <= numerator_u / denom) & (numerator_u / denom <= 1))[0]

        # get indexes where both t and u are between 0 and 1 - these points have intersections between the two line segments
        intersec_idxs = np.intersect1d(intersec_t_idxs, intersec_u_idxs)

        if intersec_idxs.size != 0:
            intersec_idx = intersec_idxs[0]
        else:
            return (np.empty((0,2)), None)

        # get the t-value for the intersection point
        intersec_t = numerator_t[intersec_idx] / denom[intersec_idx]
        
        # apply the t-value to the line equation to get the actual intersection coordinates
        intersec_x = x1 + (intersec_t * (x2 - x1))
        intersec_y = y1 + (intersec_t * (y2 - y1))

        # if there is no intersection, new_intersec will be shape (0,2), so just an empty list
        # return that empty list, and use None for t_return to indicate no intersection
        new_intersec = np.array([intersec_x, intersec_y]).reshape(-1,2)
        
        return new_intersec, t
    
    def calc_doodle_len(self, xs: np.array, ys: np.array) -> float:
        '''
        Calculates the length of the entire doodle, segment by segment.
        :param xs np.array: list of all x-coordiates throughout the doodle
        :param ys np.array: list of all y-coordiates throughout the doodle

        :returns doodle_len float: total length of doodle
        '''
        x1s = xs[:-1]
        x2s = np.roll(xs, -1)[:-1]
        y1s = ys[:-1]
        y2s = np.roll(ys, -1)[:-1]
        dists = dist(x1s, y1s, x2s, y2s)
        doodle_len = np.sum(dists)
        return doodle_len
        
    def angle_metric(self, z_series, **metric_kwargs):
        '''
        Calculates the quality score of a doodle based on how close each subsequent angle in the sequence is 
            to the corresponding angle in an ideal sequence. 
        
        If the nominal metric is being used, the angle is calculated based on the tuned angle of each effector.
        If the empirical (non-nominal) metric is used, the actual angle of doodle is calculated at the peak point of each effector. 

        :param z_series np.ndarray: the series of z-values for each ring neuron

        :returns score float: the quality score of the doodle
        '''
        peak_times = np.array([])

        # first calculate when each effector peaks
        for i in range(z_series.shape[0]):
            peak_time, _ = find_peaks(z_series[i,:], height=0.4) # return peak_times for each peak (if more than 1) and peak_properties
            if peak_time.shape[0] == 1:
                peak_times = np.concatenate((peak_times, peak_time.astype('int')))
            elif peak_time.shape[0] == 0:
                peak_times = np.concatenate((peak_times, [-1]))
            else: # more than 1 peak for effector i
                print(f'Multiple peaks for neuron {i}: {peak_time}')
                return 0.0
        
        n = z_series.shape[0]
        angle_diff = 360 / n
        num_active_ideal = len(metric_kwargs['ideal_seq'])

        # nominal metric - angles used are those that each effector is tuned to
        if metric_kwargs['nominal']:
            effector_seq = np.argsort(peak_times) # sorts ascending, with all inactive effector idxs coming first
            active_angles = np.argwhere(peak_times > 0).flatten()
            angle_seq = effector_seq[np.isin(effector_seq, active_angles)] * 360 / n
        
        # empirical metric - angles used are those drawn at each timestep that an effector peaks
        else:
            actual_peaktimes = np.sort(peak_times[np.where(peak_times >= 0)[0]]).astype('int')
            x1s = metric_kwargs['x_series'][actual_peaktimes]
            y1s = metric_kwargs['y_series'][actual_peaktimes]
            
            x2s = metric_kwargs['x_series'][actual_peaktimes+1]
            y2s = metric_kwargs['y_series'][actual_peaktimes+1]

            angle_seq = (np.degrees(np.arctan2((y2s - y1s), (x2s - x1s))) + 360) % 360

        # if there's an incorrect number of line segments, score the doodle 0.0
        if len(angle_seq) != len(metric_kwargs['ideal_seq']):
            return 0.0
        
        # since we score all rotations of the ideal shape as perfect, we need many variations of the ideal sequence
        # (i.e., we need one for each ring neuron)
        offsets = np.broadcast_to(np.arange(0,360,angle_diff), (num_active_ideal,n)).T
        base_shapes = np.broadcast_to(metric_kwargs['ideal_seq'], (n, num_active_ideal))
        all_shapes = (base_shapes + offsets) % 360

        # calculate how far (in circular distance) each angle in the sequence is from all possible rotated ideal sequences
        abs_dist = np.abs(angle_seq - all_shapes)
        circ_dist = np.minimum(abs_dist, 360 - abs_dist)

        # if we want to score based on the final (most strict) version of the metric, we choose a different value of sigma_Q
        if metric_kwargs['use_final']:
            subscores = np.exp(-np.square(circ_dist) / (2 * (metric_kwargs['sigma_Q_final'] ** 2)))
        else:
            subscores = np.exp(-np.square(circ_dist) / (2 * (metric_kwargs['sigma_Q'] ** 2)))

        # give each angle in the sequence a weight based on how heavily we want to punish largely inaccurate angles
        subscore_weights = np.exp(-metric_kwargs['penalty_factor'] * (subscores - 1)) / np.sum(np.exp(-metric_kwargs['penalty_factor'] * (subscores - 1)), axis=1).reshape(-1,1)
        
        # then calculate the weighted average
        all_scores = np.sum(subscores * subscore_weights, axis=1)

        # and take the max value across all rotations of the ideal shape
        # i.e., we only score based on the best matched ideal sequence with the actual sequence
        score = np.max(all_scores)

        return score
    
    def ordered_shape_generator(self, ideal_seq: np.ndarray, 
                                sigma_delta: float) -> np.ndarray:
        '''
        ordered shape generator: for each neuron i, in a (randomly-rotated)
          ideal sequence, select the active neuron in the code pattern from
          a gaussian distribution centered at i. i is chosen in order of the ideal sequence,
          so the shapes generated will have angles near the ideal angle for each line segment.

        :param ideal_seq np.ndarray: the ideal sequence of angles in the given shape
        :param sigma_delta float: the amount of noise in the generated shapes
        '''
        num_active_ideal = ideal_seq.shape[0]
        code_neuron_idxs = np.arange(0, self.code_layer.num_code_units)
        angle_diff = 360 / self.code_layer.num_code_units

        # randomly rotate the ideal shape
        r = np.random.randint(0, self.code_layer.num_code_units)
        base_shape = (ideal_seq + (r * angle_diff)) % 360
        high_idxs = (base_shape / angle_diff).astype('int')
        shape_signal = np.zeros(self.code_layer.num_code_units)
        shape_signal[high_idxs] = np.linspace(1, 0.25, len(high_idxs))
        highest_shape_idxs = np.argpartition(shape_signal.flatten(), -num_active_ideal)[-num_active_ideal:]

        # select each neuron from its own individual gaussian
        selected = []
        for i in highest_shape_idxs:
            abs_dist = np.abs(code_neuron_idxs - i)
            circ_dist = np.minimum(abs_dist, self.code_layer.num_code_units - abs_dist)
            idvdl_gaussian = (np.exp(-np.square(circ_dist) / (2 * (sigma_delta**2))))
            prob_selection = idvdl_gaussian / np.sum(idvdl_gaussian)
            selected += [np.random.choice(code_neuron_idxs, 1, replace=False, p=prob_selection)]

        active_code_idxs = np.array(selected).reshape(num_active_ideal)
        # give those k neurons a value from the ideal signal (i.e., evenly-spaced values)
        code_output = np.zeros(self.code_layer.num_code_units)
        code_output[active_code_idxs] = shape_signal[highest_shape_idxs].flatten()
        return code_output

    def unordered_shape_generator(self, ideal_seq: np.ndarray, 
                                  sigma_delta: float) -> np.ndarray:
        '''
        unordered shape generator: select k neurons from a 
            k-modal gaussian distribution with each mode centered at
            the (randomly-rotated) active neurons in the ideal sequence

        :param ideal_seq np.ndarray: the ideal sequence of angles in the given shape
        :param sigma_delta float: the amount of noise in the generated shapes
        '''
        num_active_ideal = ideal_seq.shape[0]
        code_neuron_idxs = np.arange(0, self.code_layer.num_code_units)
        angle_diff = 360 / self.code_layer.num_code_units

        # randomly rotate the ideal shape
        r = np.random.randint(0, self.code_layer.num_code_units)
        base_shape = (ideal_seq + (r * angle_diff)) % 360
        high_idxs = (base_shape / angle_diff).astype('int')
        shape_signal = np.zeros(self.code_layer.num_code_units)
        shape_signal[high_idxs] = np.linspace(1, 0.25, len(high_idxs))
        highest_shape_idxs = np.argpartition(shape_signal.flatten(), -num_active_ideal)[-num_active_ideal:]

        probs = np.zeros((self.code_layer.num_code_units))
        for i in highest_shape_idxs:
            abs_dist = np.abs(code_neuron_idxs - i)
            circ_dist = np.minimum(abs_dist, self.code_layer.num_code_units - abs_dist)
            idvdl_gaussian = np.exp(-np.square(circ_dist) / (2 * (sigma_delta**2)))
            probs = probs + idvdl_gaussian
        # normalize probabilities so they sum to 1
        probs = probs / np.sum(probs)
        # select k random neurons from the multimodal distribution
        active_code_idxs = np.random.choice(code_neuron_idxs, num_active_ideal, replace=False, p=probs)

        # give those k neurons a value from the ideal signal (i.e., evenly-spaced values)
        code_output = np.zeros(self.code_layer.num_code_units)
        code_output[active_code_idxs] = shape_signal[highest_shape_idxs].flatten()
        return code_output
    
    def additive_map_influence(self, map_signal: np.ndarray,
                               code_noise: np.ndarray, delta: float, limit: int) -> np.ndarray:
        '''
        Used if we want to combine the map's influence in the generated shapes using additive noise

        :param map_signal np.ndarray: the map activation vector
        :param code_noise np.ndarray: the code noise activation vector
        :param delta float: the weighting of noise in the map influence
        :param limit int: whether to limit the number of outputs above 0
            to be only the top `limit` values
        '''
        code_input = (delta * code_noise) + ((1 - delta) * map_signal)
        if limit:
            high_idxs = np.argpartition(code_input.flatten(), -limit)[-limit:]
            code_output = np.zeros(self.code_layer.num_code_units)
            code_output[high_idxs] = code_input[high_idxs].flatten()
        else:
            code_output = code_input
        return code_output

    def mixture_map_influence(self, map_signal: np.ndarray,
                              code_noise: np.ndarray, delta: float, limit: int) -> np.ndarray:
        '''
        Used if we want to combine the map's influence in the generated shapes using mixture noise

        :param map_signal np.ndarray: the map activation vector
        :param code_noise np.ndarray: the code noise activation vector
        :param delta float: the weighting of noise in the map influence
        :param limit int: whether to limit the number of outputs above 0
            to be only the top `limit` values
        '''
        code_neuron_idxs = np.arange(0, self.code_layer.num_code_units)
        # get the number of values in vector that will come from noise
        num_noise = int(np.round(self.code_layer.num_code_units * delta, 0))
        # get random indexes to be noise
        noise_idxs = np.random.choice(code_neuron_idxs, size=num_noise, replace=False)
        # get all other (non-noise) indexes
        # these indexes in the final code activity vector will come from the map influence
        map_activity_idxs = np.setdiff1d(code_neuron_idxs, noise_idxs)
        # define the code activity vector
        code_input = np.ndarray((self.code_layer.num_code_units, 1))
        # fill in the noise indexes in code activity
        code_input[noise_idxs] = code_noise[noise_idxs]
        # fill in the map influence indexes in code activity
        code_input[map_activity_idxs] = map_signal[map_activity_idxs]

        if limit:
            high_idxs = np.argpartition(code_input.flatten(), -limit)[-limit:]
            code_output = np.zeros(self.code_layer.num_code_units)
            code_output[high_idxs] = code_input[high_idxs].flatten()
        else:
            code_output = code_input
        return code_output

    def rotational_map_influence(self, map_signal: np.ndarray,
                                 sigma_delta: float,  num_active: int) -> np.ndarray:
        '''
        Used if we want to combine the map's influence in the generated shapes using rotational noise
            Using the signal from the active map neuron, select k neurons from a k-modal 
            gaussian distribution with each peak centered at the k highest weights from the
            map signal.

        :param map_signal np.ndarray: the map activation vector
        :param sigma_delta float: the amount of noise in how the map vector is scrambled
        :param num_active float: whether to limit the number of outputs above 0
            to be only the top `limit` values
        '''
        code_neuron_idxs = np.arange(0, self.code_layer.num_code_units)
        highest_map_idxs = np.argpartition(map_signal.flatten(), -num_active)[-num_active:]

        probs = np.zeros((self.code_layer.num_code_units))
        for i in highest_map_idxs:
            abs_dist = np.abs(code_neuron_idxs - i)
            circ_dist = np.minimum(abs_dist, self.code_layer.num_code_units - abs_dist)
            idvdl_gaussian = np.exp(-np.square(circ_dist) / (2 * (sigma_delta**2)))
            probs = probs + idvdl_gaussian
        # normalize probabilities so they sum to 1
        probs = probs / np.sum(probs)
        # select k random neurons from the multimodal distribution
        active_code_idxs = np.random.choice(code_neuron_idxs, num_active, replace=False, p=probs)

        # give those k neurons a value from the map
        code_output = np.zeros(self.code_layer.num_code_units)
        code_output[active_code_idxs] = map_signal[highest_map_idxs].flatten()
        return code_output
    

if __name__ == '__main__':
    # define model arguments
    ring_neurons = 36
    weight_RC_spread = 0.00001

    code_factor = 1
    code_neurons = code_factor*ring_neurons
    
    duration_neurons = 36
    durs = 0.2

    map_neurons_d1 = 12
    map_neurons_d2 = 12
    weight_MC_min = 0.0
    weight_MC_max = 1.0
    map_activity_sigma = 0.0001

    tmax = 70
    tsteps = 700

    # define training arguments
    train_epochs = 1000
    train_init_lr = 0.1
    train_lr_decay = -0.0000
    train_init_map_sigma = 2
    train_nhood_decay = -0.0015
    train_init_delta = 1.0
    delta_exp_decay_rate = 0.00001

    # define metric arguments
    metric_sigma_Q_init = 10
    metric_sigma_Q_final = 3
    metric_penalty_factor = 3
    metric_nominal = False
    metric_ideal_seq = np.array([0,10,20,30])

    # define noise arguments
    sigma_delta_init = 1
    noise_num_high = len(metric_ideal_seq)
    noise_num_low = ring_neurons - noise_num_high
    noise_mean_low = 0.1
    noise_mean_high = 0.6
    noise_sigma_low = 0.1
    noise_sigma_high = 0.4
    noise_type = 'bmgs'
    method = 'u'

    # initialize model
    crn = CodeRingNetwork(num_ring_units=ring_neurons,
                        num_code_units=code_neurons,
                        code_factor=code_factor,
                        num_dur_units=duration_neurons,
                        map_d1=map_neurons_d1, map_d2=map_neurons_d2,
                        code_ring_spread=weight_RC_spread,
                        noise_num_high=noise_num_high,
                        noise_num_low=noise_num_low,
                        noise_mean_low=noise_mean_low,
                        noise_mean_high=noise_mean_high,
                        noise_sigma_low=noise_sigma_low,
                        noise_sigma_high=noise_sigma_high)

    # train model
    train_strictest_scores, train_scores = crn.train(train_epochs, map_activity_sigma, train_init_delta, delta_exp_decay_rate,
                            train_init_map_sigma, train_nhood_decay,
                            train_init_lr, train_lr_decay,
                            durs, tmax, tsteps, plot_gif=False, plot_results=False,
                            noise_num_high=noise_num_high,
                            noise_num_low=noise_num_low,
                            noise_mean_low=noise_mean_low,
                            noise_mean_high=noise_mean_high,
                            noise_sigma_low=noise_sigma_low,
                            noise_sigma_high=noise_sigma_high,
                            sigma_Q_init=metric_sigma_Q_init,sigma_Q_final=metric_sigma_Q_final, penalty_factor=metric_penalty_factor,
                            nominal=metric_nominal, ideal_seq=metric_ideal_seq,
                            sigma_delta_init=sigma_delta_init,
                            method=method,
                            noise_type=noise_type)

    # plot adjusted scores heatmap
    plt.matshow(train_scores.T, vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f'Adjusted Scores Heatmap {crn.id_string}')
    plt.xlabel('Epoch')
    plt.ylabel('Iteration')
    plt.savefig(f'{crn.folder_name}\\scores_heatmap_adjusted_{crn.id_string}.png')
    plt.close()

    # plot strictest scores heatmap
    plt.matshow(train_strictest_scores.T, vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f'Strictest Scores Heatmap {crn.id_string}')
    plt.xlabel('Epoch')
    plt.ylabel('Iteration')
    plt.savefig(f'{crn.folder_name}\\scores_heatmap_strictest_{crn.id_string}.png')
    plt.close()
    
    epoch_scores = np.mean(train_scores, axis=1)
    epoch_strictest_scores = np.mean(train_strictest_scores, axis=1)

    # plot adjusted scores scatter plot
    for idx in range(map_neurons_d1*map_neurons_d2):
        plt.scatter(range(train_epochs), train_scores[:,idx])
    plt.plot(epoch_scores, label='Avg.Adj. Epoch Score', c='black')
    plt.title(f'Adjusted Scores Over Time {crn.id_string}')
    plt.legend()
    plt.savefig(f'{crn.folder_name}\\scores_adjusted_{crn.id_string}.png')
    plt.close()

    # plot strictest scores scatter plot
    for idx in range(map_neurons_d1*map_neurons_d2):
        plt.scatter(range(train_epochs), train_strictest_scores[:,idx])
    plt.plot(epoch_strictest_scores, label='Avg. Strictest Epoch Score', c='black')
    plt.title(f'Strictest Scores Over Time {crn.id_string}')
    plt.legend()
    plt.savefig(f'{crn.folder_name}\\scores_strictest_{crn.id_string}.png')
    plt.close()

    # write the parameters to a text file
    write_params(f'{crn.folder_name}\\params_{crn.id_string}.txt', **locals())
    crn.save_model_params(crn.map_layer.weights_to_code_from_map, f'{crn.folder_name}\\weights_{crn.id_string}.xlsx')

    pass