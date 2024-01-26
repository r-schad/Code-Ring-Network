import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
import os
from datetime import datetime as dt

from Ring_Layer import RingLayer
from Code_Layer import CodeLayer
from Duration_Layer import DurationLayer
from Map_Layer import MapLayer
from utilities import get_color_range, curvature, gaussian, exponential, sigmoid, write_params, dist

class CodeRingNetwork:
    def __init__(self, num_ring_units: int, num_code_units: int, code_factor: int, num_dur_units: int,
                 map_d1: int, map_d2: int, weight_min: float = 0.0, weight_max: float = 1.0,
                 code_ring_spread: float = 0.02) -> None:
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

        self.vars_to_plot = {'z': True, 'v': False, 'u': False, 'I_prime': False}
        self.COLOR_RANGE = get_color_range(num_ring_units, map_name='hsv')

        # layer initalizations
        self.ring_layer = RingLayer(num_ring_units)
        self.code_layer = CodeLayer(num_map_units=(map_d1*map_d2), num_ring_units=num_ring_units,
                                    num_code_units=num_code_units, code_factor=code_factor,
                                    code_ring_spread=code_ring_spread)
        self.duration_layer = DurationLayer(num_dur_units, (map_d1*map_d2))
        self.map_layer = MapLayer(map_d1, map_d2, num_ring_units, num_code_units,
                                  weight_min, weight_max)
               
        # metric-specific variables
        self.ideal_curvature = 2
        self.curvature_sd = 0.8
        self.intersec_rate = -1.5
        # self.metric_sigmoid_growth = 10
        # self.metric_sigmoid_center = 0.75
        self.doodle_len_beta = 50
        self.min_doodle_len = 5

    def pretrain(self, num_epochs: int, learning_sigma: float, learning_rate: float,
                 durations: float, t_max: int, t_steps: int,
                 plot_results: bool = False, plot_gif: bool = False) -> list:
        '''
        Trains the code-ring network by generating inputs consisting of 
            random activity on only the code layer, determining the output
            of the ring layer based on that activity, and updating the 
            bidirectional weights between the code and map layers based on the 
            classical SOFM learning algorithm.
        
        :param num_epochs int: the number of iterations to pretrain the model over
        :param learning_sigma float: the std. dev. (sigma) of the 
            Gaussian neighborhood for the map weight update equation
            NOTE: this stays static throughout pretraining
        :param learning_rate float: the learning rate during the pretraining stage
            NOTE: this stays static throughout pretraining
        :param durations float: the duration value output from the duration layer for each neuron
            FIXME: this is temporary, will need removed once duration layer is changed
        :param t_max int: the maximum time to integrate each iteration over
        :param t_steps int: the number of timesteps to integrate over from [0, t_max]
        :param plot_results bool: whether the final doodle and the vars_to_plot timeseries should be saved
        :param plot_gif bool: whether a GIF video should be saved by the doodling process

        :returns scores list: list of the scores of the doodles over time      
        '''
        scores = []
        for epoch in tqdm(range(num_epochs)):
            # apply random babbling signal into code layer
            uniform_code_noise = np.random.uniform(low=0, high=1, size=(self.code_layer.num_code_units, 1))
            code_input = np.where(uniform_code_noise < 0.9, 0.0, uniform_code_noise) # TODO: change to a distribution
         
            # determine output of code layer (input into ring layer)
            ring_input = (self.code_layer.weights_to_ring_from_code @ code_input).squeeze()

            # determine activity of duration layer
            # TODO: right now, this is just a constant
            dur_output = self.duration_layer.activate(durations)

            # integrate ring layer model over time
            v_series, z_series, u_series, I_prime_series = self.ring_layer.activate(ring_input, dur_output, t_max=t_max, t_steps=t_steps)
            
            # apply model results to the drawing system
            x_series, y_series = self.ring_layer.create_drawing(z_series, t_steps)

            # evaluate drawing
            score, curvatures, intersec_pts, intersec_times = self.evaluate(x_series, y_series, t_steps)

            # plot ring layer variables over time
            plot_v = v_series if self.vars_to_plot['v'] else []
            plot_u = u_series if self.vars_to_plot['u'] else []
            plot_z = z_series if self.vars_to_plot['z'] else []
            plot_I_prime = I_prime_series if self.vars_to_plot['I_prime'] else []
            if plot_results:
                self.plot_results(x_series, y_series, intersec_pts,
                            ring_inputs=ring_input,
                            v=plot_v, u=plot_u, z=plot_z, I_prime=plot_I_prime,
                            folder_name=self.folder_name, epoch=epoch, score=score, plot_gif=plot_gif)

            if plot_gif:
                self.create_gif(x_series, y_series, t_steps, intersec_pts, intersec_times, self.folder_name, epoch)
                
            # determine the most similar neuron to the activity of the code layer
            map_winner = self.map_layer.forward(code_input)

            # update the weights (bidirectionally, so both weight matrices M<->C) based on quality of the output
            self.map_layer.update_weights(code_input, map_winner,
                                          nhood_sigma=learning_sigma, learning_rate=learning_rate,
                                          score=score)

            print(f'Pretraining Epoch {epoch}: {score}')
            scores += [score]

        return scores
        

    def train(self, num_epochs: int, activation_nhood: float,
              init_delta: float, delta_decay_rate: float,
              init_learning_nhood: float, learning_nhood_decay: float,
              init_learning_rate: float, learning_rate_decay: float,
              durations: float, t_max: int, t_steps: int,
              plot_results: bool = False, plot_gif: bool = False) -> list:
        '''
        Trains the code-ring network by generating inputs consisting of 
            random activity on the map and code layers, determining the output
            of the ring layer based on that activity, and updating the 
            bidirectional weights between the code and map layers based on the 
            classical SOFM learning algorithm.
        
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
            FIXME: this is temporary, will need removed once duration layer is changed
        :param t_max int: the maximum time to integrate each iteration over
        :param t_steps int: the number of timesteps to integrate over from [0, t_max]
        :param plot_results bool: whether the final doodle and the vars_to_plot timeseries should be saved
        :param plot_gif bool: whether a GIF video should be saved by the doodling process

        :returns scores list: list of the scores of the doodles over time      
        '''
        map_size = int(self.map_layer.d1 * self.map_layer.d1)
        scores = []
        
        learning_rate = init_learning_rate
        learning_nhood = init_learning_nhood
        delta = init_delta
        for epoch in tqdm(range(num_epochs)):
            # get map activity by choosing a random winning neuron
            rand_map_winner_idx = np.random.choice(map_size)
            # then activate the neighborhood around the winner
            rand_map_winner = self.map_layer.convert_to_coord(rand_map_winner_idx)

            # apply static neighborhood range for activation neighborhood
            # and normalize activation neighborhood so map_signal sums to 1.0
            map_signal = (self.map_layer.neighborhood(rand_map_winner, sigma=activation_nhood) /
                          np.sum(self.map_layer.neighborhood(rand_map_winner, sigma=activation_nhood)))
            
            # propagate map signal forward
            map_activation = self.map_layer.weights_to_code_from_map @ map_signal.reshape(map_size, 1)

            # apply random babbling signal into code layer
            uniform_code_noise = np.random.uniform(low=0, high=1, size=(self.code_layer.num_code_units, 1))
            code_noise = np.where(uniform_code_noise < 0.9, 0.0, uniform_code_noise) # TODO: change to a distribution

            # get combined input into code layer by applying delta to babbling noise vs the map activation
            code_input = delta * code_noise + (1 - delta) * map_activation
           
            # determine output of code layer (input into ring layer)
            ring_input = (self.code_layer.weights_to_ring_from_code @ code_input).squeeze()

            # determine activity of duration layer
            # TODO: right now, this is just a constant
            dur_output = self.duration_layer.activate(durations)

            # integrate ring layer model over time
            v_series, z_series, u_series, I_prime_series = self.ring_layer.activate(ring_input, dur_output, t_max=t_max, t_steps=t_steps)
            
            # apply model results to the drawing system
            x_series, y_series = self.ring_layer.create_drawing(z_series, t_steps)

            # evaluate drawing
            score, curvatures, intersec_pts, intersec_times = self.evaluate(x_series, y_series, t_steps)

            # plot ring layer variables over time
            plot_v = v_series if self.vars_to_plot['v'] else []
            plot_u = u_series if self.vars_to_plot['u'] else []
            plot_z = z_series if self.vars_to_plot['z'] else []
            plot_I_prime = I_prime_series if self.vars_to_plot['I_prime'] else []
            if plot_results:
                self.plot_results(x_series, y_series, intersec_pts,
                            ring_inputs=ring_input,
                            v=plot_v, u=plot_u, z=plot_z, I_prime=plot_I_prime,
                            folder_name=self.folder_name, epoch=epoch, score=score, plot_gif=plot_gif)

            if plot_gif:
                self.create_gif(x_series, y_series, t_steps, intersec_pts, intersec_times, self.folder_name, epoch)
                
            # determine the most similar neuron to the activity of the code layer
            map_winner = self.map_layer.forward(code_input)

            # update the weights (bidirectionally, so both weight matrices M<->C)
            # update is based on output score, current nhood range, and learning rate
            self.map_layer.update_weights(code_input, map_winner,
                                          nhood_sigma=learning_nhood, learning_rate=learning_rate,
                                          score=score)

            # decrease the influence of the code babbling signal
            delta = exponential(epoch, rate=delta_decay_rate, init_val=init_delta)
            
            # decrease the neighborhood range
            learning_nhood = exponential(epoch, rate=learning_nhood_decay, init_val=init_learning_nhood)

            # decrease the learning rate (if learning_rate_decay == 0, use static learning rate)
            learning_rate = exponential(epoch, rate=learning_rate_decay, init_val=init_learning_rate)

            print(f'Training Epoch {epoch}: {score}')
            scores += [score]

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
        fig, axs = plt.subplots(self.map_layer.d1, self.map_layer.d2, 
                                figsize=(self.map_layer.d1,self.map_layer.d2), sharex=True, sharey=True)

        activity_matrix = np.zeros((self.map_layer.d1, self.map_layer.d2))
        for i in tqdm(range(map_size)):
            (r, c) = self.map_layer.convert_to_coord(i)
            activity_matrix[r, c] = 1.0
            map_activation = self.map_layer.weights_to_code_from_map @ activity_matrix.reshape(map_size, 1)

            # determine output of code layer (input into ring layer)
            ring_input = self.code_layer.weights_to_ring_from_code @ map_activation

            # determine activity of duration layer
            # TODO: right now, this is just a constant
            dur_output = self.duration_layer.activate(durations)

            # integrate ring layer model over time
            v_series, z_series, u_series, I_prime_series = self.ring_layer.activate(ring_input, dur_output, t_max=t_max, t_steps=t_steps)
            
            # apply model results to the drawing system
            x_series, y_series = self.ring_layer.create_drawing(z_series, t_steps)

            # evaluate drawing
            score, curvatures, intersec_pts, intersec_times = self.evaluate(x_series, y_series, t_steps)
                                            
            # generate drawing for current neuron
            self.plot_final_doodle(ax=axs[r][c], xs=x_series, ys=y_series, intersec_pts=intersec_pts, individualize_plot=False)
            axs[r][c].set_xlabel(f'{np.round(score,3)}')
            activity_matrix[r, c] = 0.0

        plt.tight_layout()
        fig.savefig(filename)
        plt.close()

    def plot_results(self, xs: np.array, ys: np.array, intersec_pts: np.ndarray,
                     ring_inputs: np.array, v: np.ndarray, u: np.ndarray,
                     z: np.ndarray, I_prime: np.ndarray,
                     folder_name: str, epoch: int, score: float, plot_gif=False) -> None:
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
        :param I_prime np.ndarray: array of shape (num_ring_neurons, t_steps) of the series of I_prime (resource) values of the ring layer. 
            If not being plotted, will be [].
        :param folder_name str: the model instance's corresponding folder name
        :param epoch int: the current epoch
        :param score float: the score of the outputted doodle
        :param plot_gif bool: whether to plot a GIF for each episode. This is used to determine the trial's folder structure.

        :returns: None
        '''
        self.id_string = folder_name.split('\\')[-1]
        f, axs = plt.subplots(1, 2)
        self.plot_final_doodle(axs[0], xs, ys, intersec_pts)
        self.plot_activity(axs[1], ring_inputs, v, u, z, I_prime)

        f.suptitle(f'Epoch {epoch} - Score = {np.round(score,3)}', fontsize=14)
        if plot_gif:
            if not os.path.isdir(f'{folder_name}\\{epoch}'):
                os.makedirs(f'{folder_name}\\{epoch}')
            f.savefig(f'{folder_name}\\{epoch}\\plot_{self.id_string}_epoch{epoch}')
        else:
            if not os.path.isdir(f'{folder_name}'):
                os.mkdir(f'{folder_name}')
            f.savefig(f'{folder_name}\\plot_{self.id_string}_epoch{epoch}')

        plt.close()

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
            # plot final pen point
            ax.scatter(xs[-1], ys[-1], alpha=0.8, marker = 'o', c='black', label='Final Point')
            # organize plot
            ax.set_xlim([-1 * np.max(np.abs([xs, ys])), np.max(np.abs([xs, ys]))])
            ax.set_xlabel('x', fontsize = 14)
            ax.set_ylim([-1 * np.max(np.abs([xs, ys])), np.max(np.abs([xs, ys]))])
            ax.set_ylabel('y', fontsize = 14)
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
                      z: np.ndarray = [], I_prime: np.ndarray = []) -> None:
        '''
        Plots the time series of the variables involved with the ring layer.

        :param ax matplotlib.pyplot.axis: the axis object to plot on
        :param ring_inputs np.ndarray: "I"; the array of inputs into the ring layer
            ring_inputs determine the order of activation of the ring neurons
        :param v np.ndarray: the activation (v) series of each ring neuron
            If not being plotted, will be [].   
        :param u np.ndarray: the deactivation (u) series of each ring neuron
            If not being plotted, will be [].   
        :param z np.ndarray: the output (z) series of each ring neuron
            If not being plotted, will be [].   
        :param I_prime np.ndarray: the "effective input" (I') series of each ring neuron
            If not being plotted, will be [].   

        :returns: None
        '''
        # include 8 most active ring neurons in legend
        sorted_inputs = np.flip(np.argsort(ring_inputs))
        for i in sorted_inputs[:8]:
            color = self.COLOR_RANGE[i]
            if np.any(v):
                plt.plot(v[i], label=f'v_{i}', c=color, linestyle='dashed')
            if np.any(u):
                plt.plot(u[i], label=f'u_{i}', c=color, linestyle='dotted')
            if np.any(I_prime):
                plt.plot(I_prime[i], label=f"I'_{i}", c=color, linestyle='dashdot')
            if np.any(z):
                plt.plot(z[i], label=f'z_{i}', c=color, linestyle='solid')
        
        # add '_' to beginning of these labels in the legend so they're ignored
        # we want to ignore the later half of inputs for visual clarity so legend isn't too big
        for i in sorted_inputs[8:]:
            color = self.COLOR_RANGE[i]
            if np.any(v):
                plt.plot(v[i], label=f'_v_{i}', c=color, linestyle='dashed')
            if np.any(u):
                plt.plot(u[i], label=f'_u_{i}', c=color, linestyle='dotted')
            if np.any(I_prime):
                plt.plot(I_prime[i], label=f"_I'_{i}", c=color, linestyle='dashdot')
            if np.any(z):
                plt.plot(z[i], label=f'_z_{i}', c=color, linestyle='solid')

        ax.legend(loc='upper right')
        ax.set_ylim([0, 1])
        ax.set_xlabel('t')
        ax.set_title('Variable Plots')
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
        # TODO: we very occassionally still get random dots showing up on y=x. Doesn't seem like
        # TODO: they're taken into account for metric scoring though, but not 100% sure.
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
        # TODO: get desired GIF durations to work
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

        # get the t-value for the intersection point
        intersec_t = numerator_t[intersec_idxs] / denom[intersec_idxs]
        
        # apply the t-value to the line equation to get the actual intersection coordinates
        intersec_x = x1 + (intersec_t * (x2 - x1))
        intersec_y = y1 + (intersec_t * (y2 - y1))

        # if there is no intersection, new_intersec will be shape (0,2), so just an empty list
        # return that empty list, and use None for t_return to indicate no intersection
        new_intersec = np.array([intersec_x, intersec_y]).reshape(-1,2)
        if new_intersec.shape[0] == 0:
            t_return = None
        else:
            t_return = t
        
        return new_intersec, t_return
    
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
    
    def evaluate(self, x_series: np.ndarray, y_series: np.ndarray, t_steps: int) -> tuple[float, np.array, np.ndarray, np.array]:
        '''
        Gets the intersection points, curvature values, and total metric score of a doodle,
            and returns the 4-tuple of score, curvatures, intersection points, intersection times.

        :param x_series np.array: array of x-coordinates over time
        :param y_series np.array: array of y-coordinates over time
        :param t_steps int: the number of timesteps integrated over

        :returns (score: float, curvatures: np.array, intersec_pts: np.ndarray, intersec_times: np.array):
            a tuple containing the metric score, a list of curvature scores,
            an array of [x, y] intersection points, and an array of the corresponding timesteps of intersection
        '''
        # get intersection points and curvatures of outputted drawing
        curvatures = []
        intersec_pts = np.ndarray((0,2))
        intersec_times = np.array([])
        # must have at least 4 points for intersections, so start at index 3
        for t_cur in range(3, t_steps, 1):
            new_intersec, intersec_t = self.detect_intersection(x_series, y_series, t_cur)
            if intersec_t:
                intersec_pts = np.concatenate((intersec_pts, new_intersec), axis=0)
                intersec_times = np.append(intersec_times, intersec_t)

            curvatures += [curvature(x1=x_series[t_cur-2], y1=y_series[t_cur-2],
                                        x2=x_series[t_cur-1], y2=y_series[t_cur-1],
                                        x3=x_series[t_cur], y3=y_series[t_cur])]
            
        curvatures = np.array(curvatures)
            
        doodle_length = self.calc_doodle_len(x_series, y_series)
        score = self.metric(curvatures, len(intersec_times), doodle_length)
        return score, curvatures, intersec_pts, intersec_times

    def metric(self, curvatures: np.array, num_intersecs: int, doodle_length: float) -> float:
        '''
        The metric based on which doodles are evaluated. Combines an average curvature score 
            with the ratio of intersection points to determine an overall quality score of a doodle.

        :param curvatures np.array: the curvature values for each discretized line segment of the doodle
        :param num_intersecs int: the number of intersection points occurring in the doodle
        :param doodle_length float: total length of doodle

        :returns score float: the combined metric score of the given doodle
        '''
        avg_curv_subscore = (np.sum(gaussian(curvatures,
                                mean=self.ideal_curvature, sd=self.curvature_sd)) /
                                len(curvatures))
        intersec_score = exponential(num_intersecs, rate=self.intersec_rate, init_val=1)
        length_score = sigmoid(doodle_length, beta=self.doodle_len_beta, mu=self.min_doodle_len) # TODO: parameterize these
        score = avg_curv_subscore * intersec_score * length_score
        return score
    
if __name__ == '__main__':
    ring_neurons = 36
    weight_RC_spread = 0.002

    code_factor = 1
    code_neurons = code_factor*ring_neurons
    
    duration_neurons = 36
    durs = 0.2

    map_neurons_d1 = 8
    map_neurons_d2 = 8
    weight_CR_min = 0.0
    weight_CR_max = 1.0
    map_activity_sigma = 0.8

    tmax = 30
    tsteps = 300

    pretrain_epochs = 5000
    pretrain_lr = 0.05
    pretrain_nhood_sigma = map_neurons_d1 / 3

    train_epochs = 45000
    train_init_lr = 0.01
    train_lr_decay = 0.0
    train_init_map_sigma = map_neurons_d1 / 3
    train_nhood_decay = -1 * np.log(train_init_map_sigma) / 5000
    train_init_delta = 1.0
    delta_exp_decay_rate = -0.0001

    for doodle_length in [1,2,5,6,8,10]:
        crn = CodeRingNetwork(num_ring_units=ring_neurons,
                        num_code_units=code_neurons,
                        code_factor=code_factor,
                        num_dur_units=duration_neurons,
                        map_d1=map_neurons_d1, map_d2=map_neurons_d2,
                        weight_min=weight_CR_min, weight_max=weight_CR_max,
                        code_ring_spread=weight_RC_spread)
        
        crn.ideal_doodle_len = doodle_length
            
        crn.id_string = crn.folder_name.split('\\')[-1] + f'doodle_len-{doodle_length}'
        plt.ioff()

        crn.show_map_results(f'{crn.folder_name}\\initial_outputs_{crn.id_string}.png', durs, tmax, tsteps)

        pretrain_scores = crn.pretrain(pretrain_epochs, pretrain_nhood_sigma, pretrain_lr, durs, tmax, tsteps, plot_results=False, plot_gif=False)
        # plot pretraining map
        crn.show_map_results(f'{crn.folder_name}\\pretraining_outputs_{crn.id_string}.png', durs, tmax, tsteps)
        # plot pretraining weights
        plt.matshow(crn.map_layer.weights_to_map_from_code)
        plt.colorbar()
        plt.savefig(f'{crn.folder_name}\\pretraining_weights_{crn.id_string}.png')
        plt.close()

        train_scores = crn.train(train_epochs, map_activity_sigma, train_init_delta, delta_exp_decay_rate,
                        train_init_map_sigma, train_nhood_decay,
                        train_init_lr, train_lr_decay,
                        durs, tmax, tsteps, plot_gif=False, plot_results=False)
        crn.show_map_results(f'{crn.folder_name}\\final_outputs_{crn.id_string}.png', durs, tmax, tsteps)
        
        total_epochs = pretrain_epochs + train_epochs
        all_scores = pretrain_scores + train_scores
        running_mean_window = int(total_epochs / 100)

        # plot results timeseries
        # scatter all scores
        plt.scatter(range(total_epochs), all_scores, label='Doodle Scores', s=1)
        # plot running mean
        plt.plot(np.convolve(all_scores, np.ones(running_mean_window)/running_mean_window, mode='valid'),
                c='#ff7f0e', label='Running Mean', linewidth=2)
        # show where map activity starts having an impact
        plt.axvline(pretrain_epochs, c='black', label='Beginning of Map Influence')
        plt.title(f'Scores Over Time {crn.id_string}')
        plt.legend()
        plt.savefig(f'{crn.folder_name}\\all_scores_{crn.id_string}.png')
        plt.close()

        # plot weights
        plt.matshow(crn.map_layer.weights_to_map_from_code)
        plt.colorbar()
        plt.savefig(f'{crn.folder_name}\\final_weights_{crn.id_string}.png')
        plt.close()

    curv_min = crn.low_curv
    curv_max = crn.high_curv
    curvature_sd = crn.curvature_sd
    intersec_rate = crn.intersec_rate
    sigmoid_growth = crn.metric_sigmoid_growth
    sigmoid_center = crn.metric_sigmoid_center

    write_params(f'{crn.folder_name}\\params_{crn.id_string}.txt', **locals())
    pass