import numpy as np
import matplotlib.pyplot as plt

class MapLayer():
    def __init__(self, map_d1: int, map_d2: int, num_ring_units: int, num_code_units: int, 
                 init_learning_rate: float, init_nhood_range: float, weight_scale: float, epsilon=0.00001) -> None:
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
        self.epsilon = epsilon

        self.learning_rate = init_learning_rate
        self.nhood_range = init_nhood_range
        self.neurons = np.array([[[i,j] for j in range(self.d1)] for i in range(self.d2)])
        self.dist_arrays = self.get_distances_for_all_neurons()
        self.weights_to_code_from_map = np.random.rand(num_code_units, int(map_d1*map_d2)) * weight_scale
        # define a `view`` of original array - they point to same memory - between W_CM and W_MC
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
    
    def neighborhood(self, winner: tuple) -> np.ndarray:
        '''
        Takes in a winning neuron and current epoch and returns the (d1, d2) array
            of the Gaussian neighborhood scaling factor for each neuron centered 
            around the winner. Uses nhood_range in determining the Gaussian curve.

        :param winner tuple: the (i, j) coordinate of the winning neuron.

        :returns nhood np.ndarray: the (d1, d2) array of values of each neuron based 
            on the Gaussian neighborhood around `winner`. 
        '''
        winner_i = self.convert_to_index(winner)
        # get the dist_array for the winner neuron
        dists = self.dist_arrays[winner_i]
        top = np.negative(np.square(dists))
        bottom = 2 * self.nhood_range ** 2
        nhood = np.exp(np.divide(top, bottom))
        return nhood
    
    def update_weights(self, input_vec: np.array, winner: tuple, grad: float, score: float) -> None:
        '''
        Takes in a single input vector, winning neuron, and the score of the output,
            and updates both of the model's weight matrices in-place. Uses learning_rate in the 
            weight update calculation.

        :param input_vec np.ndarray: the input vector to compare the weights against
        :param winner tuple: the winning neuron whose weights are closest to `input_vec`
        :param score float: the score of the output, as determined by the metric function

        :returns: None
        '''
        input_vec = input_vec.squeeze()
        nhood_scores = self.neighborhood(winner).reshape(self.d1*self.d2,1)
        weight_changes = ((1 / (score + self.epsilon)) * nhood_scores * self.learning_rate *
                          np.subtract(self.weights_to_map_from_code, input_vec))
        self.weights_to_map_from_code += weight_changes # this updates both weight matrices

    def forward(self, code_activity: np.array) -> tuple:
        '''
        Takes in a single input vector, and returns the winning neuron as a coordinate.

        :param code_activity np.array: array of activity in the code layer, which serves as
            the input vector with which we compare the map neurons' weights

        :returns winner_coords tuple: the (i, j) coordinates of the winning neuron
        '''
        # TODO: do we need to reshape here?
        weights_reshaped = self.weights_to_code_from_map.reshape(self.num_code_units, self.d1*self.d2)
        norms = np.linalg.norm(weights_reshaped - code_activity, axis=0)
        winner_index = int(np.argmin(norms))
        winner_coords = self.convert_to_coord(winner_index)

        return winner_coords
    
    # def visualize_weights(self, filename): # TODO: we'll need this
    #     '''
    #     Given a filename, plots the weights of all neurons in the SOFM in a grid of shape (self.d1, self.d2).

    #     NOTE: Due to the sheer amount of data to plot, 
    #     this function takes a very significant amount of time to complete (up to an hour).
    #     '''
    #     print('Creating neuron visualization plot...')
    #     try:
    #         fig, axs = plt.subplots(self.d1, self.d2, figsize=(self.d1,self.d2), sharex=True, sharey=True)
    #     except NotImplementedError:
    #         fig, axs = plt.subplots(self.d1, self.d2, figsize=(self.d1,self.d2), sharex=True, sharey=True)

    #     for r in range(self.d1):
    #         print(f'{round(100 * r / self.d1)}%', end='\r')
    #         for c in range(self.d2):
    #             i = self.convert_to_index((r,c))
    #             ax = axs[r][c]

    #             # plot image on subplot
    #             ax.imshow(self.weights[i].reshape(self.image_dims[0], self.image_dims[1]), cmap='gray', vmin=0, vmax=1)
                
    #             ax.set_xbound([0,self.image_dims[1]])

    #     plt.tight_layout()
    #     fig.savefig(filename)
    #     plt.close()
