import numpy as np
import matplotlib.pyplot as plt

class MapLayer():
    def __init__(self, map_d1, map_d2, num_ring_units, num_code_units, code_factor, init_learning_rate, init_nhood_range, weight_scale) -> None:
        '''
        A class implementing a Self-Organizing Feature Map (SOFM) as the foundation of the Code Ring Network. 
        This layer has bidirectional weights between Map Layer <-> Code Layer, with 
            W_CM[i, j, k] == W_MC[k, j, i].
        
        :param map_d1 int: dimension 1 of the SOFM grid.
        :param map_d2 int: dimension 2 of the SOFM grid.
        :param num_ring_units int: number of neurons in the Ring layer, which is equal to the number of neurons 
            on a given ring of the Code layer, where there can be multiple rings depending on `code_factor`.
        :param code_factor int: number of rings around the code layer,
            i.e., the number of Code neurons per Ring Neuron.
        :param init_lr float: the initial learning rate of the weights of the SOFM.
        :param init_nhood_range float: the initial standard deviation of the SOFM 
            when computing the activation of the Gaussian neighborhood around the 
            winning neuron for a given input.
        '''
        self.d1 = map_d1
        self.d2 = map_d2
        self.num_ring_units = num_ring_units
        self.num_code_units = num_code_units
        self.code_factor = code_factor

        self.learning_rate = init_learning_rate
        self.nhood_range = init_nhood_range
        self.neurons = np.array([[[i,j] for j in range(self.d1)] for i in range(self.d2)])
        self.dist_arrays = self.get_distances_for_all_neurons()
        self.weights_to_code_from_map = np.random.rand(num_code_units, int(map_d1*map_d2)) * weight_scale
        self.weights_to_map_from_code = self.weights_to_code_from_map.T # returns a `view`` of original array; they point to same memory

    def get_distances_for_all_neurons(self):
        '''
        Initializes a ((d1*d2) x d1 x d2) array of the Euclidian norms of each neuron for every possible winner.
        (This avoids having to compute these values for each input example;
        instead, we just do it for all possible neurons once at the start of the current training stage.)
        '''
        dist_arrs = np.ndarray((self.d1*self.d2, self.d1, self.d2))
        for r in range(self.d1):
            for c in range(self.d2):
                i = self.convert_to_index((r,c))
                dist_arrs[i] = self.calc_distances_from_winner((r,c))

        return dist_arrs
    
    def calc_distances_from_winner(self, winner):
        '''
        Takes in a (d1 x d2) array of the row index of each neuron,
        a (d1 x d2) array of the column index of each neuron,
        and a winning neuron.
        Returns the (d1 x d2) array of distances of each neuron from the winner,
        using the distance formula.
        '''
        dx = np.abs(np.subtract(self.neurons[:,:,0], winner[0]))
        dy = np.abs(np.subtract(self.neurons[:,:,1], winner[1]))

        dx_toroid = np.minimum(dx, np.array(self.d2 - dx))
        dy_toroid = np.minimum(dy, self.d1 - dy)

        dists = np.sqrt(np.square(dx_toroid) + np.square(dy_toroid))

        return dists
    
    def convert_to_coord(self, i):
        '''
        Takes in an integer index i, and returns its tuple coordinate based on the dimensions of the SOFM
        '''
        assert isinstance(i, (int, np.integer)), 'Index must be type int' # convert from index to coordinates
        return (i // self.d2, i % self.d2)

    def convert_to_index(self, coords):
        '''
        Takes in a tuple coordinate, and returns its integer index based on the dimensions of the SOFM
        '''
        assert isinstance(coords, tuple), 'Coordinates must be type tuple' # convert from coordinates to index
        return (coords[0] * self.d2) + coords[1]
    
    def neighborhood(self, winner):
        '''
        Takes in a winning neuron and current epoch and returns a 2d array (n x n)
        of the Gaussian neighborhood scaling factor for each neuron centered around the winner.
        '''
        winner_i = self.convert_to_index(winner)
        # get the dist_array for the winner neuron
        dists = self.dist_arrays[winner_i]
        top = np.negative(np.square(dists))
        bottom = 2 * self.nhood_range ** 2
        return np.exp(np.divide(top, bottom))
    
    def update_weights(self, input_vec, winner, score):
        '''
        Takes in a single input vector, winning neuron, neighborhood range, and learning rate,
        and updates the model's weights in-place.
        '''
        input_vec = input_vec.squeeze()
        nhood_scores = self.neighborhood(winner).reshape(self.d1*self.d2,1)
        weight_changes = score * nhood_scores * self.learning_rate * np.subtract(self.weights_to_map_from_code, input_vec)
        self.weights_to_map_from_code += weight_changes # this updates both weight matrices

    def forward(self, code_activity):
        '''
        Takes in a single input vector, 
        and returns the winning neuron as a coordinate
        '''
        weights_reshaped = self.weights_to_code_from_map.reshape(self.num_code_units, self.d1*self.d2)
        norms = np.linalg.norm(weights_reshaped - code_activity, axis=0)
        winner_index = int(np.argmin(norms))

        return self.convert_to_coord(winner_index)
    
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
