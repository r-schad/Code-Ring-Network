import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from tqdm import tqdm
from datetime import datetime as dt

from scipy.integrate import solve_ivp

from utilities import sigmoid, COLORS, COLOR_NAMES

class RingLayer:
    '''
    A class implementing a simple ring layer that can draw an output given a sequence of activation steps.
    '''
    def __init__(self, num_units=36, tau=1.0, lambda_=20, eta=1.2, rho=0.1, gamma=0.1, beta=50.0, mu=0.1, phi=0.5, alpha=0.9, epsilon=0.00001) -> None:
        '''
        :param num_units int: number of neurons in ring layer
        # TODO: describe each parameter's function in the model
        :param tau float:
        :param lambda_ float:
        :param eta float:
        :param rho float:
        :param gamma float:
        :param beta float:
        :param mu float:
        :param phi float:
        :param alpha float:
        :param epsilon float:

        :returns: None
        '''
        self.id_string = str(dt.now()).replace(':', '').replace('.','')
        print(f'ID string: {self.id_string}')
        if not os.path.isdir('output'):
            os.mkdir('output')

        self.folder_name = f'output\\{self.id_string}'
        os.mkdir(self.folder_name)

        self.num_units = num_units
        self.tau=tau
        self.lambda_=lambda_
        self.eta=eta
        self.rho=rho
        self.gamma=gamma
        self.beta=beta
        self.mu=mu
        self.phi=phi
        self.alpha=alpha
        self.epsilon=epsilon

        self.directions = np.array([np.deg2rad(i * 360 / self.num_units) for i in range(self.num_units)]) # radians
        self.headings = np.array([[np.cos(dir), np.sin(dir)] for dir in self.directions])
    

    def activate(self, code_outputs, dur_outputs, t_max: int, t_steps: int, vars_to_plot={'v':False,'u':False,'z':True,'I_prime':False}, plot_gif=False) -> None:
        '''
        Applies the outputs of the code and duration layers into the ring layer to determine outputs of the ring layer.

        :param code_outputs np.ndarray: "I"; the array of outputs of the code layer, and inputs into the ring layer
            code_outputs determine the order of activation of the ring neurons
        
        :param dur_outputs np.ndarray: "c"; the array of outputs of the duration layer, and inputs into the ring layer
            dur_outputs determine the duration that each ring neuron is activated
        :param t_max int: the maximum timestep to integrate to
        :param t_steps int: the number of timesteps to integrate over from [0, t_max]
        :param vars_to_plot dict: which of the 4 series should be plotted (v, u, z, I')
            keys=the four series listed above, values=bool indicating if each corresponding series should be plotted
        :param plot_gif bool: whether a GIF should be created of the drawing

        :return None:
        '''
        # initialize activations and de-activations to 0
        v = np.zeros(self.num_units)
        u = np.zeros(self.num_units)

        # create state vector out of the 3 vectors v, u, I so it's 1-D to work with solve_ivp()
        # state is initialized as [v_0, ... , v_N-1, u_0, ..., u_N-1, I_0, ..., I_N-1]
        # so, I' is initialized to I (AKA code_outputs)
        state = np.array((v, u, code_outputs)).reshape(3*self.num_units)

        # define our discretized timesteps
        t = np.linspace(0, t_max, t_steps)

        # integrate model over discretized timesteps
        result = solve_ivp(fun=lambda t, state: self.doodle(t, code_outputs, dur_outputs, state), t_span=(min(t), max(t)), t_eval=t, y0=state)
        v_series = result.y[:self.num_units,]
        z_series = sigmoid(v_series, self.beta, self.mu)
        u_series = result.y[self.num_units:2*self.num_units,]
        I_prime_series = result.y[2*self.num_units:,]

        # apply model results to the drawing system
        x_series, y_series = self.create_drawing(z_series, t_steps)

        if any(vars_to_plot.values()):
            self.plot_activity(code_outputs, t, v_series, u_series, z_series, I_prime_series, vars_to_plot)

        if plot_gif:
            self.create_gif(x_series, y_series, t_steps)


    def create_drawing(self, z_series, t_steps) -> tuple:
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
        xs_with_momentum = (1 / 10) * dir_series_with_momentum_x
        ys_with_momentum = (1 / 10) * dir_series_with_momentum_y

        # calculate cumulative location of pen over time
        x_series_with_momentum = np.cumsum(xs_with_momentum)
        y_series_with_momentum = np.cumsum(ys_with_momentum)

        return (x_series_with_momentum, y_series_with_momentum)


    def plot_activity(self, code_outputs, t, v, u, z, I_prime, vars_to_plot) -> None:
        '''
        :param code_outputs np.ndarray: "I"; the array of outputs of the code layer, and inputs into the ring layer
            code_outputs determine the order of activation of the ring neurons
        :param t np.ndarray: the discretized timesteps to plot over
        :param v np.ndarray: the activation (v) series of each ring neuron
        :param u np.ndarray: the deactivation (u) series of each ring neuron
        :param z np.ndarray: the output (z) series of each ring neuron
        :param I_prime np.ndarray: the "effective input" (I') series of each ring neuron
        :param vars_to_plot dict: which of the 4 series should be plotted (v, u, z, I')
            keys=the four series listed above, values=bool indicating if each corresponding series should be plotted

        :returns None:
        '''
        fig, axs = plt.subplots()

        for i in np.argsort(code_outputs):
            color = COLORS[i % len(COLORS)]
            if vars_to_plot['v']: 
                plt.plot(t, v[i], label=f'v_{i}', c=color, linestyle='dashed')
            if vars_to_plot['u']: 
                plt.plot(t, u[i], label=f'u_{i}', c=color, linestyle='dotted')
            if vars_to_plot['I_prime']: 
                plt.plot(t, I_prime[i], label=f"I'_{i}", c=color, linestyle='dashdot')
            if vars_to_plot['z']: 
                plt.plot(t, z[i], label=f'z_{i}', c=color, linestyle='solid')    
            plt.axhline(y=0.0, c="black", linewidth=0.05)

        plt.ylim([0, 1])
        plt.xlabel('t')

        fig.savefig(f'{self.folder_name}\\plot_{self.id_string}')
        plt.close()
    
    def doodle(self, t, inputs, dur_values, state) -> np.ndarray:
        '''
        Because we can't provide a vectorized state (i.e. state can't be more than 1-d in solve_ivp()),
        we hide the three vectors in state, so state is a vector of [v, u, I'], 
        where v, u, and I' are all vectors of length `num_units`.
        
        Then, we can handle the change in v, u, and I' separately, 
        and concat them back together to be returned as the new state.

        :param t np.ndarray: array of the timestep values to integrate over 
            (per solve_ivp() documentation, t is a necessary parameter for the function being applied to solve_ivp().
            but should not be used in the function)
        :param inputs np.ndarray: input values provided by the code layer
        :param state np.ndarray: array of shape (3*N) containing v, u, and I' vectors
            state is defined by [v_0, ..., v_N-1, u_0, ..., u_N-1, I'_0, ..., I'_N-1]
            this is done because solve_ivp() does not allow more than 1-D arrays in it, so we compensate by combining
            our 3 vectors into one long array and then split them apart for actual use inside the function being
            integrated by solve_ivp()
        :param p dict: dictionary of the parameters being applied to this run of the model

        :returns new_state np.ndarray: array of shape (3*N) containing the updated v, u, and I' vectors after the current timestep
            new_state is defined in identical fashion to state: [v_0, ..., v_N-1, u_0, ..., u_N-1, I'_0, ..., I'_N-1]
        '''
        # split state into 3 vectors
        v = state[0:self.num_units]
        u = state[self.num_units:2*self.num_units]
        I_prime = state[2*self.num_units:3*self.num_units]

        assert set([v.shape[0], u.shape[0], I_prime.shape[0]]) == set([self.num_units]), f"State's shapes don't match! {v.shape, u.shape, I_prime.shape}"

        z = sigmoid(v, self.beta, self.mu)
    
        # calculate dv/dt, du/dt, DI'/dt
        inhibition_vec = 1 - (self.eta * np.dot(z, 1 - np.eye(self.num_units))) # multiply by the sum of *other* neuron's outputs
        dv = (1 / self.tau) * ((-1 * self.lambda_ * u * v) + (I_prime * inhibition_vec))
        du = (-1 * self.rho * u) + (self.gamma * I_prime * (z) / (dur_values + self.epsilon))
        dI_prime = -1 * self.phi * inputs * z
        
        # join v, u, and I' back together to be returned
        new_state = np.array((dv, du, dI_prime)).reshape(3*self.num_units)

        return new_state


    def detect_intersection(self, xs, ys, t) -> np.ndarray:
        '''
        :param xs list: a list of the x-series of the doodle
        :param ys list: a list of the y-series of the doodle
        :param t int: the current timestep in the doodle

        :returns new_intersec np.ndarray: intersection point(s) encountered at timestep `t`    
        '''
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
        # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line_segment
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
        new_intersec = np.array([intersec_x, intersec_y]).reshape(-1,2)

        return new_intersec


    def create_frame(self, xs, ys, t, ax, pen_label, pen_color, detect_intersec, intersec_pts) -> np.ndarray:
        '''
        :param xs list: a list of the x-series of the doodle
        :param ys list: a list of the y-series of the doodle
        :param t int: the current timestep in the doodle
        :param ax plt.axis: axis to plot on
        :param pen_label string: string to label the pen on the plot
        :param pen_color string: pyplot string for color of the pen on the plot
        :param detect_intersec bool: boolean indicating whether intersections should be detected
        :param intersec_pts np.ndarray: array of all intersection points encountered before timestep `t`

        :returns intersec_pts np.ndarray: array of all intersection points encountered up to (and including) timestep `t`
        '''
        assert len(xs) == len(ys), "xs and ys shape doesn't match!"
        
        # plot lines up to current timestep
        ax.plot(xs[:t+1], ys[:t+1], color=pen_color, alpha=0.5, label=pen_label)
        # plot current pen point
        ax.scatter(xs[t], ys[t], color=pen_color, alpha=0.8, marker = 'o')

        # collect any intersections from the current timestep's line segment
        # (intersections can only occur if we have drawn 4 points)
        if detect_intersec and t >= 3:
            new_intersec = self.detect_intersection(xs, ys, t)
            intersec_pts = np.concatenate((intersec_pts, new_intersec), axis=0)
        
        # plot all intersection points up to current timestep (if any)
        if intersec_pts.any():
            ax.scatter(intersec_pts[:,0], intersec_pts[:,1], color='red', marker='o')

        # organize plot
        ax.set_xlim([-1 * np.max(np.abs([xs, ys])), np.max(np.abs([xs, ys]))])
        ax.set_xlabel('x', fontsize = 14)
        ax.set_ylim([-1 * np.max(np.abs([xs, ys])), np.max(np.abs([xs, ys]))])
        ax.set_ylabel('y', fontsize = 14)
        ax.set_title(f'Step {t}', fontsize=14)
        ax.legend()

        return intersec_pts


    def create_gif(self, x_series, y_series, t_steps, pen_label='Output', pen_color='black', detect_intersec=True) -> None:
        '''
        :param x_series np.ndarray: array of the x-values of the pen over time
        :param y_series np.ndarray: array of the y-values of the pen over time
        :param t_steps int: number of values of `t` over which the model will be integrated
        :param pen_label string: string to label the pen on the plot
        :param pen_color string: pyplot string for color of the pen on the plot
        :param detect_intersec bool: boolean indicating whether intersections should be detected

        :returns None:
        '''
        # create directories
        if not os.path.isdir(f'{self.folder_name}\\img'):
            os.mkdir(f'{self.folder_name}\\img')

        frames = []
        intersections = np.ndarray((0, 2))
        print('Creating GIF...')
        for t in tqdm(range(t_steps)):
            f, ax = plt.subplots()

            intersections = self.create_frame(x_series, y_series, t, ax, pen_label, pen_color, detect_intersec, intersec_pts=intersections)
            
            f.savefig(f'{self.folder_name}\\img\\img_{t}.png')
            plt.close()
            image = imageio.v2.imread(f'{self.folder_name}\\img\\img_{t}.png')
            frames.append(image)

        # duration is 1/100th seconds, per frame
        # TODO: get desired GIF durations
        imageio.mimsave(f"{self.folder_name}\\GIF_{self.id_string}.gif", frames, **{'duration':2.5})

if __name__ == "__main__":
    r = RingLayer()
    r.activate(np.random.rand(r.num_units), np.random.rand(r.num_units), t_max=40, t_steps=400, plot_gif=True)

