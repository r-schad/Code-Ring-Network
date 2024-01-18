import numpy as np

class DurationLayer:
    '''
    A class implementing a code layer, organized in a ring-shape, that can take in an input, 
    and produces a sequence of activations to be passed into the ring layer.
    '''
    def __init__(self, num_ring_units, num_map_units) -> None:
        '''
        :param code_factor int: number of neurons in code layer per ring neuron
        :param ring_units int: number of neurons in ring layer

        :returns: None
        '''
        self.num_ring_units = num_ring_units
        self.num_dur_units = num_ring_units
        self.num_map_units = num_map_units
        self.weights_to_map_from_dur = np.ones((self.num_map_units, self.num_dur_units))

    def activate(self, value: float) -> np.array:
        '''
        returns an array `value`s, with length self.num_ring_units as a placeholder.

        :param value (float, np.ndarray): the input value

        :returns np.array: array of value, with length num_ring_units
        '''
        return np.array([value] * self.num_ring_units)
