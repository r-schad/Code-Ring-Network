import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from tqdm import tqdm

from scipy.integrate import solve_ivp
from scipy.stats import norm

from utilities import sigmoid, COLORS, COLOR_NAMES, curvature, gaussian, exponential

class RingLayer:
    '''
    A class implementing a ring layer that can draw an output given an input signal.
    '''
    def __init__(self, num_ring_units: int = 36,
                 tau: float = 1.0, lambda_: float = 20, psi: float = 1.2, # activation
                 rho: float = 0.1, gamma: float = 0.1, # deactivation
                 phi: float = 0.5, # resource depletion
                 alpha: float = 0.9, # drawing momentum
                 beta: float = 50.0, mu: float = 0.1, # output sigmoid transform
                 epsilon: float = 0.00001) -> None:
        '''
        :param num_ring_units int: number of neurons in ring layer
        # TODO: describe each parameter's function in the model
        :param tau float:
        :param lambda_ float:
        :param psi float:
        :param rho float:
        :param gamma float:
        :param beta float:
        :param mu float:
        :param phi float:
        :param alpha float:
        :param epsilon float:

        :returns: None
        '''
        self.num_ring_units = num_ring_units
        self.tau=tau
        self.lambda_=lambda_
        self.psi=psi
        self.rho=rho
        self.gamma=gamma
        self.beta=beta
        self.mu=mu
        self.phi=phi
        self.alpha=alpha
        self.epsilon=epsilon

        self.directions = np.array([np.deg2rad(i * 360 / self.num_ring_units) for i in range(self.num_ring_units)]) # radians
        self.headings = np.array([[np.cos(dir), np.sin(dir)] for dir in self.directions])
    
    def activate(self, input_from_code: np.ndarray, dur_outputs: np.ndarray,
                 t_max: int, t_steps: int, folder_name: str, epoch: int,
                 vars_to_plot: dict = {'v':False,'u':False,'z':True,'I_prime':False},
                 plot_gif: bool = False) -> float:
        '''
        Applies the outputs of the code and duration layers into the ring layer to determine outputs of the ring layer.

        :param input_from_code np.ndarray: "I"; the array of outputs of the code layer, and inputs into the ring layer
            code_outputs determine the order of activation of the ring neurons        
        :param dur_outputs np.ndarray: "c"; the array of outputs of the duration layer, and inputs into the ring layer
            dur_outputs determine the duration that each ring neuron is activated
        :param t_max int: the maximum timestep to integrate to
        :param t_steps int: the number of timesteps to integrate over from [0, t_max]
        :param folder_name str: the model instance's corresponding folder name
        :param epoch int: the current epoch
        :param vars_to_plot dict: which of the 4 series should be plotted (v, u, z, I_prime)
            keys=the four series listed above, 
            values=bool indicating if each corresponding series should be plotted
        :param plot_gif bool: whether a GIF should be created of the drawing

        :returns score float: the metric score of the outputted drawing based on the metric function.
        '''
        # initialize activations and de-activations to 0
        v = np.zeros(self.num_ring_units)
        u = np.zeros(self.num_ring_units)

        input_from_code = input_from_code.squeeze()

        # create state vector out of the 3 vectors v, u, I so it's 1-D to work with solve_ivp()
        # state is initialized as [v_0, ... , v_N-1, u_0, ..., u_N-1, I_0, ..., I_N-1]
        # so, I' is initialized to I (AKA input_from_code)
        state = np.array((v, u, input_from_code)).reshape(3*self.num_ring_units)

        # define our discretized timesteps
        t = np.linspace(0, t_max, t_steps)

        # integrate model over discretized timesteps
        result = solve_ivp(fun=lambda t, state: self.doodle(t, input_from_code, dur_outputs, state), t_span=(min(t), max(t)), t_eval=t, y0=state)
        if not result.success:
            return 0
        v_series = result.y[:self.num_ring_units,]
        z_series = sigmoid(v_series, self.beta, self.mu)
        u_series = result.y[self.num_ring_units:2*self.num_ring_units,]
        I_prime_series = result.y[2*self.num_ring_units:,]

        # apply model results to the drawing system
        x_series, y_series = self.create_drawing(z_series, t_steps)

        # evaluate drawing
        score, curvatures, intersec_pts, intersec_times = self.evaluate(x_series, y_series, t_steps, ideal_curv=2.0, curv_sd=0.6) # FIXME: make these a variable

        # plot variables over time
        plot_v = v_series if vars_to_plot['v'] else []
        plot_u = u_series if vars_to_plot['u'] else []
        plot_z = z_series if vars_to_plot['z'] else []
        plot_I_prime = I_prime_series if vars_to_plot['I_prime'] else []
        self.plot_results(x_series, y_series, intersec_pts,
                          ring_inputs=input_from_code,
                          v=plot_v, u=plot_u, z=plot_z, I_prime=plot_I_prime,
                          folder_name=folder_name, epoch=epoch, score=score, plot_gif=plot_gif)

        if plot_gif:
            self.create_gif(x_series, y_series, t_steps, intersec_pts, intersec_times, epoch)

        return (score, (x_series, y_series))

    def create_drawing(self, z_series: np.ndarray, t_steps: int) -> tuple:
        '''
        Applies the outputs of the model to create a doodle produced by the drawing mechanism.

        :param z_series np.ndarray: the series of z-values for each ring neuron
        :param t_steps int: the number of timesteps we integrated over

        :returns (xs, ys) tuple: tuple of the x_series and y_series of the pen (including momentum) 
        '''
        # calculate the activity in each direction over each timestep
        dir_series = z_series.T @ self.headings # does not include momentum
        momentum_term = np.roll(dir_series, 1, axis=0) # roll time series forward one step
        momentum_term[0, :] = np.array([0., 0.]) # set first momentum step to 0

        # get array of alphas to increasing powers
        # [0, alpha, alpha^2, alpha^3, ...]
        alphas = np.cumprod([self.alpha] * (t_steps - 1))
        alphas = np.array([0] + list(alphas))

        # recurrence relation boils down to the following momentum term
        # convolution(N, M) gives a result of n + m - 1 elements. we only need the first t_steps
        # TODO: turn the recurrence relation back to a loop. or maybe at least compare performance/time between the two
        dir_series_with_momentum_x = ((1 - self.alpha) * (z_series.T @ self.headings).T[0,:] +
                                     (1 - self.alpha) * np.convolve((z_series.T @ self.headings).T[0,:], alphas)[:t_steps])
        
        dir_series_with_momentum_y = ((1 - self.alpha) * (z_series.T @ self.headings).T[1,:] +
                                     (1 - self.alpha) * np.convolve((z_series.T @ self.headings).T[1,:], alphas)[:t_steps])

        # scale x and y distances by 1/10 to keep drawings on the page
        # TODO: should 1/10 be a variable?
        xs_with_momentum = (1 / 10) * dir_series_with_momentum_x
        ys_with_momentum = (1 / 10) * dir_series_with_momentum_y

        # calculate cumulative location of pen over time
        x_series_with_momentum = np.cumsum(xs_with_momentum)
        y_series_with_momentum = np.cumsum(ys_with_momentum)

        return (x_series_with_momentum, y_series_with_momentum)
    
    def plot_results(self, xs: np.array, ys: np.array, intersec_pts: np.ndarray,
                     ring_inputs: np.array, v: np.ndarray, u: np.ndarray,
                     z: np.ndarray, I_prime: np.ndarray,
                     folder_name: str, epoch: int, score: float, plot_gif=False) -> None:
        '''
        Plots the final resulting doodle and the variable activity graph of the ring layer. 
            The plots are saved to directory: `folder_name`\\`epoch`.
        
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
        id_string = folder_name.split('\\')[-1]
        f, axs = plt.subplots(1, 2)
        self.plot_final_doodle(axs[0], xs, ys, intersec_pts)
        self.plot_activity(axs[1], ring_inputs, v, u, z, I_prime)

        f.suptitle(f'Epoch {epoch} - Score = {np.round(score,3)}', fontsize=14)
        if plot_gif:
            if not os.path.isdir(f'{folder_name}\\{epoch}'):
                os.makedirs(f'{folder_name}\\{epoch}')
            f.savefig(f'{folder_name}\\{epoch}\\plot_{id_string}_epoch{epoch}')
        else:
            if not os.path.isdir(f'{folder_name}'):
                os.mkdir(f'{folder_name}')
            f.savefig(f'{folder_name}\\plot_{id_string}_epoch{epoch}')

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

        # plot all intersection points (if any)
        if intersec_pts.any():
            ax.scatter(intersec_pts[:,0], intersec_pts[:,1], color='red', marker='o', label='Intersections')

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
        # TODO: fix this so code units corresponding to same ring unit are same color (also use gradient ring for color scheme?)
        for i in np.argsort(ring_inputs):
            color = COLORS[i % len(COLORS)]
            if np.any(v):
                plt.plot(v[i], label=f'v_{i}', c=color, linestyle='dashed')
            if np.any(u):
                plt.plot(u[i], label=f'u_{i}', c=color, linestyle='dotted')
            if np.any(I_prime):
                plt.plot(I_prime[i], label=f"I'_{i}", c=color, linestyle='dashdot')
            if np.any(z):
                plt.plot(z[i], label=f'z_{i}', c=color, linestyle='solid')
            plt.axhline(y=0.0, c="black", linewidth=0.05)

        ax.set_ylim([0, 1])
        ax.set_xlabel('t')
        ax.set_title('Variable Plots')
    
    def doodle(self, t: np.ndarray, code_values: np.ndarray,
               dur_values: np.ndarray, state: np.ndarray) -> np.ndarray:
        '''
        Calculates the rate of change of the ring layer variables over time. This is the 
            function inputted into solve_ivp.
        NOTE: Because we can't provide a vectorized state (i.e. state can't 
            be more than 1-d in solve_ivp()), we hide the three vectors in state, 
            so state is a vector of [v, u, I'], where v, u, and I' are all 
            vectors of length `num_ring_units`.
    
            Then, we can handle the change in v, u, and I' separately, 
            and concat them back together to be returned as the new state.

        :param t np.ndarray: array of the timestep values to integrate over
            (per solve_ivp() documentation, t is a necessary parameter for the function being applied to solve_ivp().
            but should not be used in the function)
        :param code_values np.ndarray: input values provided by the code layer
        :param dur_values np.ndarray: input values provided by the duration layer
        :param state np.ndarray: array of shape (3*N) containing v, u, and I' vectors
            state is defined by [v_0, ..., v_N-1, u_0, ..., u_N-1, I'_0, ..., I'_N-1]
            this is done because solve_ivp() does not allow more than 1-D arrays in it, so we compensate by combining
            our 3 vectors into one long array and then split them apart for actual use inside the function being
            integrated by solve_ivp()

        :returns new_state np.ndarray: array of shape (3*N) containing the updated v, u, and I' vectors after the current timestep
            new_state is defined in identical fashion to state: [v_0, ..., v_N-1, u_0, ..., u_N-1, I'_0, ..., I'_N-1]
        '''
        # split state into 3 vectors
        v = state[0:self.num_ring_units]
        u = state[self.num_ring_units:2*self.num_ring_units]
        I_prime = state[2*self.num_ring_units:3*self.num_ring_units]

        assert set([v.shape[0], u.shape[0], I_prime.shape[0]]) == set([self.num_ring_units]), f"State's shapes don't match! {v.shape, u.shape, I_prime.shape}"

        z = sigmoid(v, self.beta, self.mu)
    
        # calculate dv/dt, du/dt, DI'/dt
        inhibition_vec = 1 - (self.psi * np.dot(z, 1 - np.eye(self.num_ring_units))) # multiply by the sum of *other* neuron's outputs
        dv = (1 / self.tau) * ((-1 * self.lambda_ * u * v) + (I_prime * inhibition_vec))
        du = (-1 * self.rho * u) + (self.gamma * I_prime * (z) / (dur_values + self.epsilon))
        dI_prime = -1 * self.phi * code_values * z
        
        # join v, u, and I' back together to be returned
        new_state = np.array((dv, du, dI_prime)).reshape(3*self.num_ring_units)

        return new_state

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
                   intersec_pts: np.ndarray, intersec_times: np.ndarray,  folder_name: str,
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

        id_string = folder_name.split('\\')[-1]

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
        imageio.mimsave(f"{folder_name}\\{epoch}\\GIF_{id_string}.gif", frames, **{'duration':2.5})

    def evaluate(self, x_series: np.ndarray, y_series: np.ndarray, t_steps: int,
                 ideal_curv: float = 2, curv_sd: float = 0.6, intersec_growth: float = 0.2) -> tuple[float, list, np.ndarray, np.array]:
        '''
        Gets the intersection points, curvature values, and total metric score of a doodle,
            and returns the 4-tuple of score, curvatures, intersection points, intersection times.

        :param x_series np.array: array of x-coordinates over time
        :param y_series np.array: array of y-coordinates over time
        :param t_steps int: the number of timesteps integrated over
        :param ideal_curv float: the ideal curvature based on which to evaluate the doodle
        :param curv_sd float: the standard deviation of the curvature Gaussian curve 
            used to evaluate the doodle
        :param intersec_growth float: the exponential growth rate of the intersection penalty

        :returns (score: float, curvatures: list, intersec_pts: np.ndarray, intersec_times: np.array)
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
        score = self.metric(t_steps, np.array(curvatures), len(intersec_times), ideal_curv=ideal_curv, curv_sd=curv_sd, intersec_growth=intersec_growth)
        return score, curvatures, intersec_pts, intersec_times

    def metric(self, t_steps: int, curvatures: list, num_intersecs: int,
               ideal_curv: float = 2, curv_sd: float = 0.6, intersec_growth: float = 0.2):
        '''
        The metric based on which doodles are evaluated. Combines an average curvature score 
            with the ratio of intersection points to determine an overall quality score of a doodle.

        :param t_steps int: the number of timesteps over which the doodle was drawn.
        :param curvatures list: the list of curvatures for each discretized line segment of the doodle
        :param num_intersecs int: the number of intersection points occurring in the doodle
        :param ideal_curv float: the desired curvature for each line segment of the doodle. This serves
            as the center of the Gaussian curve.
        :param curv_sd float: the standard deviation of the curvature Gaussian curve
        :param intersec_growth float: the exponential growth rate of the intersection penalty
        
        :returns score float: the combined metric score of the given doodle
        '''
        avg_curv_subscore = (1 / (t_steps - 3)) * np.sum((-1 * gaussian(curvatures, mean=ideal_curv, sd=curv_sd)) + 1)
        intersec_subscore = exponential(num_intersecs, rate=intersec_growth, init_val=1) - 1
        score = avg_curv_subscore + intersec_subscore
        return score
