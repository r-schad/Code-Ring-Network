import numpy as np
import matplotlib.pyplot as plt

class CodeLayer:
    '''
    A class implementing a code layer, organized in a ring-shape, that can take in an input, 
    and produces a sequence of activations to be passed into the ring layer.
    '''
    def __init__(self, code_factor, ring_units) -> None:
        '''
        :param code_factor int: number of neurons in code layer per ring neuron
        :param ring_units int: number of neurons in ring layer
        :returns: None
        '''
        self.code_factor = code_factor
        self.ring_units = ring_units

        self.code_units = self.code_factor * self.ring_units
        
        # create covariance matrix of code->ring weights
        cov_mat = np.ndarray((self.code_units, self.ring_units))
        
        # get all weight values that will exist in the covariance matrix
        weight_vals = np.linspace(0.0, 1.0, int(np.ceil(self.ring_units / 2))) # TODO consider changing linspace to use mexican-hat or steeper gaussian

        # reflect weight values so the weight array is symmetric
        weight_arr = np.hstack((weight_vals, np.flip(weight_vals[:-1])))

        # store the weight values in each row of the covariance matrix, offset by 1 for each row
        for i in range(self.ring_units):
            cov_mat[i] = np.roll(weight_arr, i)

        # duplicate the rows so we have self.code_factor rows centered at each ring neuron
        cov_mat = np.repeat(cov_mat[0:self.ring_units], repeats=self.code_factor, axis=0)
        self.weights = np.random.rand(self.code_units, self.ring_units) * cov_mat     


    def get_activations(self, input) -> np.ndarray:
        '''
        :param input np.ndarray: the input signal provided to the network
        :returns: the sequence of activation values to be sent to the ring layer
        '''

        pass

c = CodeLayer(3, 9)
