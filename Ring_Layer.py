import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from utilities import sigmoid

def event_attr():
    def decorator(func):
        func.terminal = True
        return func
    return decorator

class RingLayer:
    '''
    A class implementing a ring layer that can draw an output given an input signal.
    '''
    def __init__(self, num_ring_units: int = 36,
                 tau: float = 1.0, lambda_: float = 20, psi: float = 1.2, # activation
                 rho: float = 0.1, gamma: float = 0.1, # deactivation
                 phi: float = 1.2, # resource depletion
                 alpha: float = 0.5, # drawing momentum
                 beta: float = 200.0, mu: float = 0.1, # output sigmoid transform
                 epsilon: float = 0.00001) -> None:
        '''
        :param num_ring_units int: number of ring neurons
        :param tau float = 1.0: time constant for the effector system
        :param lambda_ float = 20: gain parameter on the activation/deactivation product
        :param psi float = 1.2: gain parameter on the lateral inhibition from active neuron
        :param rho float = 0.1: gain parameter for the decrease in deactivation signal
        :param gamma float = 0.1: gain parameter for the increase in deactivation signal
        :param beta float = 200: steepness of the v->z sigmoid 
        :param mu float = 0.1: center of the v->z sigmoid
        :param phi float = 1.2: gain parameter on the resource depletion rate 
        :param alpha float = 0.5: momentum constant on the doodling process
        :param epsilon float = 0.00001: small value to avoid divide by zero errors

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
                 t_max: int, t_steps: int, cutoff: bool = True) -> float:
        '''
        Applies the outputs of the code and duration layers into the ring layer to determine outputs of the ring layer.

        :param input_from_code np.ndarray: "r^0"; the array of outputs of the code layer, and inputs into the ring layer
            code_outputs determine the order of activation of the ring neurons        
        :param dur_outputs np.ndarray: "c"; the array of outputs of the duration layer, and inputs into the ring layer
            dur_outputs determine the duration that each ring neuron is activated
        :param t_max int: the maximum timestep to integrate to
        :param t_steps int: the number of timesteps to integrate over from [0, t_max]
        :param folder_name str: the model instance's corresponding folder name
        :param epoch int: the current epoch
        :param vars_to_plot dict: which of the 4 series should be plotted (v, u, z, r)
            keys=the four series listed above, 
            values=bool indicating if each corresponding series should be plotted
        :param plot_results bool: whether a plot of the results should be created
        :param plot_gif bool: whether a GIF should be created of the drawing

        :returns score float: the metric score of the outputted drawing based on the metric function.
        '''
        # initialize activations and de-activations to 0
        v = np.zeros(self.num_ring_units)
        u = np.zeros(self.num_ring_units)

        input_from_code = input_from_code.squeeze()

        # create state vector out of the 3 vectors v, u, r so it's 1-D to work with solve_ivp()
        # state is initialized as [v_0, ... , v_N-1, u_0, ..., u_N-1, r^0_0, ..., r^0_N-1]
        # so, r is initialized to r^0 (AKA input_from_code)
        state = np.array((v, u, input_from_code)).reshape(3*self.num_ring_units)

        # define our discretized timesteps
        t = np.linspace(0, t_max, t_steps)

        # integrate model over discretized timesteps
        if cutoff:
            result = solve_ivp(fun=lambda t, state: self.doodle(t, input_from_code, dur_outputs, state), t_span=(min(t), max(t)), dense_output=True, y0=state, event=self.stop_drawing)
        else:
            result = solve_ivp(fun=lambda t, state: self.doodle(t, input_from_code, dur_outputs, state), t_span=(min(t), max(t)), t_eval=t, y0=state)
            
        if not result.success:
            return 0
        v_series = result.y[:self.num_ring_units,]
        z_series = sigmoid(v_series, self.beta, self.mu)
        u_series = result.y[self.num_ring_units:2*self.num_ring_units,]
        r_series = result.y[2*self.num_ring_units:,]

        return v_series, z_series, u_series, r_series
    
    @event_attr()
    def stop_drawing(self, t, state):
        '''
        Utility function that returns a value below 0 when all effectors
            have remaining resource below 0.05. When scipy detects the resource
            value crossing the x-axis, it stops integration (solve_ivp). 
        NOTE: scipy requires t to be given as the first argument in any event function,
            even if it's not used.
        :param t int: time variable required by scipy for terminal functions, but not actually used
        :param state np.ndarray: array of shape (3*N) containing v, u, and r vectors
        '''
        r_series = state[2*self.num_ring_units:]
        return np.max(r_series) - 0.05
        
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

        # recurrence relation for momentum boils down to the following terms
        # convolve(N, M) gives a result of n + m - 1 elements. we only need the first t_steps
        xs_with_momentum = ((1 - self.alpha) * (z_series.T @ self.headings).T[0,:] +
                                     (1 - self.alpha) * np.convolve((z_series.T @ self.headings).T[0,:], alphas)[:t_steps])
        
        ys_with_momentum = ((1 - self.alpha) * (z_series.T @ self.headings).T[1,:] +
                                     (1 - self.alpha) * np.convolve((z_series.T @ self.headings).T[1,:], alphas)[:t_steps])

        # calculate cumulative location of pen over time
        x_series_with_momentum = np.cumsum(xs_with_momentum)
        y_series_with_momentum = np.cumsum(ys_with_momentum)

        return (x_series_with_momentum, y_series_with_momentum)
        
    def doodle(self, t: np.ndarray, ring_input: np.ndarray,
               dur_values: np.ndarray, state: np.ndarray) -> np.ndarray:
        '''
        Calculates the rate of change of the ring layer variables over time. This is the 
            function inputted into solve_ivp.
        NOTE: Because we can't provide a vectorized state (i.e. state can't 
            be more than 1-d in solve_ivp()), we hide the three vectors in state, 
            so state is a vector of [v, u, r], where v, u, and r are all 
            vectors of length `num_ring_units`.
    
            Then, we can handle the change in v, u, and r separately, 
            and concat them back together to be returned as the new state.

        :param t np.ndarray: array of the timestep values to integrate over
            (per solve_ivp() documentation, t is a necessary parameter for the function being applied to solve_ivp().
            but may not be used in the function)
        :param ring_input np.ndarray: input values provided by the code layer
        :param dur_values np.ndarray: input values provided by the duration layer
        :param state np.ndarray: array of shape (3*N) containing v, u, and r vectors
            state is defined by [v_0, ..., v_N-1, u_0, ..., u_N-1, r_0, ..., r_N-1]
            this is done because solve_ivp() does not allow more than 1-D arrays in it, so we compensate by combining
            our 3 vectors into one long array and then split them apart for actual use inside the function being
            integrated by solve_ivp()

        :returns new_state np.ndarray: array of shape (3*N) containing the updated v, u, and r vectors after the current timestep
            new_state is defined in identical fashion to state: [v_0, ..., v_N-1, u_0, ..., u_N-1, r_0, ..., r_N-1]
        '''
        # split state into 3 vectors
        v = state[0:self.num_ring_units]
        u = state[self.num_ring_units:2*self.num_ring_units]
        r = state[2*self.num_ring_units:3*self.num_ring_units]

        assert set([v.shape[0], u.shape[0], r.shape[0]]) == set([self.num_ring_units]), f"State's shapes don't match! {v.shape, u.shape, r.shape}"

        z = sigmoid(v, self.beta, self.mu)
        # calculate dv/dt, du/dt, Dr/dt
        inhibition_vec = 1 - (self.psi * np.dot(z, 1 - np.eye(self.num_ring_units))) # multiply by the sum of *other* neuron's outputs
        dv = (1 / self.tau) * ((-1 * self.lambda_ * u * v) + (r * inhibition_vec))
        du = (-1 * self.rho * u) + (self.gamma * r * (z) / (dur_values + self.epsilon))
        dr = -1 * self.phi * ring_input * z
        
        # join v, u, and r back together to be returned
        new_state = np.array((dv, du, dr)).reshape(3*self.num_ring_units)

        return new_state

