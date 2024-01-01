import numpy as np
from scipy.stats import norm

class CodeLayer:
    '''
    A class implementing a code layer, organized in a ring-shape, that can take in an input, 
    and produces a sequence of activations to be passed into the ring layer.
    '''
    def __init__(self, num_map_units: int, num_ring_units: int,
                 num_code_units: int, code_factor: int,
                 code_ring_spread: float = 0.02) -> None:
        '''
        :param num_map_units int: number of neurons in the map layer.
        :param num_ring_units int: number of neurons in the ring layer.
        :param num_code_units int: total number of neurons in the code layer.
        :param code_factor int: the  width of the code layer strip, 
            i.e., the number of neurons in code layer per ring neuron.
        :param code_ring_spread float: the standard deviation of the Gaussian curve 
            used for determining the strength of weights from Code->Ring. 
            Default of 0.02 will give a weight of ~0.4 to the immediate neighbors of the corresponding
            ring unit, and a weight of ~0.02 to the neighbors 2-steps away from the peak ring unit for a
            given code unit, and basically ~0.00 to all further neurons.

        :returns: None
        '''
        self.num_map_units = num_map_units
        self.num_code_units = num_code_units
        assert num_code_units % code_factor == 0, 'Uneven number of code units.'
        self.num_code_units_per_circle = int(num_code_units / code_factor)
        self.code_factor = code_factor
        self.num_ring_units = num_ring_units

        self.weights_to_ring_from_code = self.get_gauss_ring_weights(scale=code_ring_spread)

    def get_gauss_ring_weights(self, scale: float = 0.02):
        '''
        Defines the weights from Code->Ring, which are either shaped in a circular strip,
        with width=`self.code_factor`. These weights are based on a Gaussian curve, with 
        each code unit's weights peaking at its corresponding ring unit. Depending on the value of 
        `scale`, a given code unit may or may not correspond to a neighborhood of ring units, based
        on the Gaussian curve used.

        :param scale float: the standard deviation of the Gaussian curve used for determining the strength
                            of weights from Code->Ring. 
                            Default of 0.02 will give a weight of ~0.4 to the immediate neighbors of the corresponding
                            ring unit, and a weight of ~0.02 to the neighbors 2-steps away from the peak ring unit for a
                            given code unit.

        :returns cov_mat_strip np.ndarray: the array of weights to ring from code layer, 
            with shape (num_ring_units, num_code_units, code_factor).
        '''
        # define input into gaussian pdf
        temp_gauss_arr = np.linspace(0, 1.0, self.num_ring_units)
        # get gaussian pdf values - with standard deviation `scale`
        weight_vals = norm.pdf(temp_gauss_arr, loc=temp_gauss_arr[len(temp_gauss_arr)//2], scale=scale)
        # scale weight vals down so peak is 1.0
        weight_vals_std = weight_vals / np.sum(weight_vals)
        # roll weight vals so peak is at index 0
        weight_vals_rolled = np.roll(weight_vals_std, -1 * np.argmax(weight_vals_std))

        # create covariance matrix of code->ring weights, using thin code layer (we expand into a strip later)
        cov_mat = np.ndarray((self.num_ring_units, self.num_code_units_per_circle))

        # store the weight values in each row of the covariance matrix, offset by 1 for each column
        for i in range(self.num_ring_units):
            cov_mat[:,i] = np.roll(weight_vals_rolled, i)

        # duplicate the rows onto a new axis with dimension=code_factor
        cov_mat_strip = np.tile(cov_mat, reps=self.code_factor)

        return cov_mat_strip
