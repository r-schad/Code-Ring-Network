import numpy as np
import matplotlib.pyplot as plt

from utilities import bimodal_exponential_noise, bimodal_gaussian_noise

class MapLayer():
    def __init__(self, map_d1: int, map_d2: int, num_ring_units: int, num_code_units: int,
                 **init_kwargs) -> None:
        '''
        A class implementing a Self-Organizing Feature Map (SOFM) as the foundation of the
          Code Ring Network. 
        This layer has bidirectional weights between Map Layer <-> Code Layer, with 
            W_CM[i, j] == W_MC[j, i].
        
        :param map_d1 int: dimension 1 of the SOFM grid.
        :param map_d2 int: dimension 2 of the SOFM grid.
        :param num_ring_units int: number of neurons in the Ring layer.
        :param num_code_units int: number of neurons in the Code layer.
        :param init_learning_rate float: the initial learning rate of the weights of the SOFM.
        :param init_nhood_range float: the initial standard deviation of the SOFM 
            when computing the activation of the Gaussian neighborhood around the 
            winning neuron for a given input.
        :param weight_scale float: the initial scaling of the weights to define their initial range.

        :returns: None        
        '''
        self.d1 = map_d1
        self.d2 = map_d2
        self.num_ring_units = num_ring_units
        self.num_code_units = num_code_units

        self.neurons = np.array([[[i,j] for j in range(self.d1)] for i in range(self.d2)])
        self.dist_arrays = self.get_distances_for_all_neurons()
        # TODO: determine method of weight init. good trials were just uniform [0,1] but pretraining dragged everything towards 0 anyways
        # self.weights_to_code_from_map = np.random.uniform(0.0, 0.2, size=(num_code_units, int(map_d1*map_d2)))
        # self.weights_to_code_from_map = np.random.uniform(low=weight_min, high=weight_max, size=(num_code_units, int(map_d1*map_d2)))
        self.weights_to_code_from_map = np.ndarray((num_code_units, int(map_d1*map_d2)))
        # initialize weights with exponential distribution (same as noise generating process)
        for m in range(int(map_d1*map_d2)):
            code_noise = bimodal_gaussian_noise(num_low=init_kwargs['noise_num_low'],
                                                num_high=init_kwargs['noise_num_high'],
                                                mean_low=init_kwargs['noise_mean_low'],
                                                mean_high=init_kwargs['noise_mean_high'],
                                                sigma_low=init_kwargs['noise_sigma_low'],
                                                sigma_high=init_kwargs['noise_sigma_high'],
                                                shuffle=False,
                                                clip_01=True).reshape(self.num_code_units, 1)
            self.weights_to_code_from_map[:,m] = code_noise.flatten()
        # defines a `view` of original array - they point to same memory - between W_CM and W_MC
        self.weights_to_map_from_code = self.weights_to_code_from_map.T

    def get_distances_for_all_neurons(self) -> np.ndarray:
        '''
        Initializes a ((d1*d2) x d1 x d2) array of the Euclidian norms of 
            each neuron for every possible winner. (This avoids having to compute these 
            values for each input example. Instead, we just do it for all possible neurons
            once at the start of the current training stage.)

        :returns dist_arrs np.ndarray: the array of shape (d1*d2, d1, d2) of all 
            neurons' distances from each neuron.
        '''
        dist_arrs = np.ndarray((self.d1*self.d2, self.d1, self.d2))
        for r in range(self.d1):
            for c in range(self.d2):
                i = self.convert_to_index((r,c))
                dist_arrs[i] = self.calc_distances_from_winner((r,c))

        return dist_arrs
    
    def calc_distances_from_winner(self, winner: tuple) -> np.ndarray:
        '''
        Takes in a (i, j) coordinate of a neuron from which to compute the distances.
            Returns the (d1 x d2) array of toroidal (wrap-around) distances of each 
            neuron from the winner, using the distance formula.

        :param winner tuple: the neuron from which the distances will be computed.

        :returns dists: the array of shape (d1, d2) with the toroidal distances of 
            each neuron from `winner`.
        '''
        dx = np.abs(np.subtract(self.neurons[:,:,0], winner[0]))
        dy = np.abs(np.subtract(self.neurons[:,:,1], winner[1]))

        dx_toroid = np.minimum(dx, np.array(self.d2 - dx))
        dy_toroid = np.minimum(dy, self.d1 - dy)

        dists = np.sqrt(np.square(dx_toroid) + np.square(dy_toroid))

        return dists
    
    def convert_to_coord(self, i: (int, np.integer)) -> tuple:
        '''
        Takes in an integer index i, and returns its tuple coordinate 
            based on the dimensions of the SOFM.

        :param i (int, np.int): integer index of the neuron
        
        :returns: (i, j) coordinates of the given integer index 
        '''
        assert isinstance(i, (int, np.integer)), 'Index must be type int'
        # convert from index to coordinates
        return (i // self.d2, i % self.d2)

    def convert_to_index(self, coords: tuple) -> int:
        '''
        Takes in a tuple coordinate, and returns its integer index 
            based on the dimensions of the SOFM.

        :param coords tuple: the (i, j) coordinates of the neuron

        :returns: integer index of the neuron
        '''
        assert isinstance(coords, tuple), 'Coordinates must be type tuple'
        # convert from coordinates to index
        return (coords[0] * self.d2) + coords[1]
    
    def neighborhood(self, winner: tuple, sigma: float) -> np.ndarray:
        '''
        Takes in a winning neuron and current epoch and returns the (d1, d2) array
            of the Gaussian neighborhood scaling factor (with std. dev. sigma)
            for each neuron centered around the winner.

        :param winner tuple: the (i, j) coordinate of the winning neuron.
        :param sigma float: the range of the neighboerhood

        :returns nhood np.ndarray: the (d1, d2) array of values of each neuron based 
            on the Gaussian neighborhood around `winner`. 
        '''
        winner_i = self.convert_to_index(winner)
        # get the dist_array for the winner neuron
        dists = self.dist_arrays[winner_i]
        top = np.negative(np.square(dists))
        bottom = 2 * sigma ** 2
        nhood = np.exp(np.divide(top, bottom))
        return nhood
    
    def update_weights(self, input_vec: np.array, winner: tuple,
                       nhood_sigma: float, learning_rate: float, score: float) -> None:
        '''
        Takes in a single input vector, winning neuron, and the score of the output,
            and updates both of the model's weight matrices in-place. Uses learning_rate in the 
            weight update calculation.

        :param input_vec np.ndarray: the input vector to compare the weights against
        :param winner tuple: the winning neuron whose weights are closest to `input_vec`
        :param nhood_sigma float: the std. dev. of the Gaussian neighborhood for the map weight update equation
        :param learning_rate float: the maximum learning rate applied to this weight update
            NOTE: the effective learning rate is calculated by scaling learning_rate by the score of the output
        :param score float: the score of the output, as determined by the metric function

        :returns: None
        '''
        nhood_scores = self.neighborhood(winner, sigma=nhood_sigma).reshape(self.d1*self.d2, 1) #- 0.02 # antihebbian learning

        effective_lr = learning_rate * score
        weight_changes = (nhood_scores * effective_lr *
                          np.subtract(input_vec.T, self.weights_to_map_from_code) *
                          self.weights_to_map_from_code * (1 - self.weights_to_map_from_code))
        
        self.weights_to_map_from_code += weight_changes # this updates both weight matrices

    def forward(self, code_activity: np.array) -> tuple:
        '''
        Takes in a single input vector, and returns the winning neuron as a coordinate.

        :param code_activity np.array: array of activity in the code layer, which serves as
            the input vector with which we compare the map neurons' weights

        :returns winner_coords tuple: the (i, j) coordinates of the winning neuron
        '''
        norms = np.linalg.norm(code_activity.T - self.weights_to_map_from_code, axis=1)
        winner_index = int(np.argmin(norms))
        winner_coords = self.convert_to_coord(winner_index)

        return winner_coords
