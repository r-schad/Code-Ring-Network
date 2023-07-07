import numpy as np
import matplotlib.pyplot as plt
import imageio

class RingLayer:
    '''
    A class implementing a simple ring layer that can draw an output given a sequence of activation steps.
    '''
    def __init__(self, num_units: int) -> None:
        '''
        :param num_units int: number of neurons in ring layer
        :returns: None
        '''
        self.num_units = num_units
        self.directions = np.array([np.deg2rad(i * 360 / self.num_units) for i in range(self.num_units)]) # radians
        self.headings = np.array([[np.cos(dir), np.sin(dir)] for dir in self.directions])
        

    def draw(self, activations: np.ndarray, draw_duration: float, start_point: np.ndarray = [0,0], save_gif: bool = False, gif_duration: int = 5) -> np.ndarray:
        '''
        :param activations np.ndarray: array of size n with activation values for each neuron at each index
        :param duration int: length of the lines to be drawn TODO: change this to take a sequence of durations
        :param start_point np.ndarray: array of size 2, containing the starting point of the drawing. TODO: assert that this contains floats in array

        :returns doodle np.ndarray: TODO returns the outputted doodle 
        TODO make gif optional
        '''
        xs = [start_point[0]]
        ys = [start_point[1]]
        for act in activations:
            dir_vec = act @ self.headings
            linex, liney = draw_duration * dir_vec
            xs.append(xs[-1] + linex)
            ys.append(ys[-1] + liney)

        if save_gif:
            frames = []
            for t in range(len(activations)):
                self.create_frame(xs, ys, t)
                image = imageio.v2.imread(f'.\\img\\img_{t}.png')
                frames.append(image)

            imageio.mimsave('example.gif', frames, duration=gif_duration)         

    def create_frame(self, xs, ys, t) -> None:
        '''
        :param xs list: a list of the x-values of the doodle
        :param ys list: a list of the y-values of the doodle
        :param t int: the current timestep in the doodle

        TODO file naming
        '''
        fig = plt.figure(figsize=(6, 6))
        plt.plot(xs[:(t+1)], ys[:(t+1)], color = 'gray' )
        plt.plot(xs[t], ys[t], color = 'black', marker = 'o' )
        plt.xlim([-1, 1])
        plt.xlabel('x', fontsize = 14)
        plt.ylim([-1, 1])
        plt.ylabel('y', fontsize = 14)
        plt.title(f'Relationship between x and y at step {t}',
                fontsize=14)
        plt.savefig(f'.\\img\\img_{t}.png',
                    transparent = False,
                    facecolor = 'white'
                )
        plt.close()

NUM_UNITS = 4
r = RingLayer(NUM_UNITS)

# draw random doodle as example
r.draw(activations=[[np.random.rand() for i in range(NUM_UNITS)] for a in range(100)], draw_duration=0.05, save_gif=True)
pass
