import numpy as np
import matplotlib.colors as mcolors

def sigmoid(v: (float, np.ndarray), beta: float = 50.0, mu: float = 0.1) -> (float, np.ndarray):
    '''
    Returns a sigmoid output with steepness beta centered at mu.

    :param v (float, np.ndarray): input value(s)
    :param beta float: steepness
    :param mu float: center of curve

    :returns output (float, np.ndarray): value or array of sigmoid output
    '''

    try:
        output = 1 / (1 + (np.exp((-1*beta) * (v - mu))))
    except RuntimeWarning:
        print(f'Overflow - v={v}')
    return output

def gaussian(x: (float, np.ndarray), mean: float = 0.2, sd: float = 0.2, peak: float = 1.0) -> (float, np.ndarray):
    '''
    Reurns a Gaussian output of the input varianle based at center mean, with standard deviation sd. The curve's peak height is determined by peak.

    :param x (float, np.ndarray): input value(s)
    :param mean float: the mean, or center of the Gaussian curve
    :param sd float: the standard deviation of the Gaussian curve
    :param peak float: the peak height of the Gaussian curve


    :returns output (float, np.ndarray): value or array of Gaussian output
    '''
    return peak * np.exp(-1 * np.square(x - mean) / (2 * (sd ** 2)))

def exponential_decay(t: (float, np.ndarray), decay_rate: float = 0.02, init_val:float = 1.0) -> (float, np.ndarray):
    '''
    Returns an exponential decay output of input variable t, with initial value init_val. The decay rate is determined by parameter decay_rate.

    :param t (float, np.ndarray): input value(s)
    :param decay_rate float: decay rate of the exponential curve
    :param init_val float: initial value of the exponential curve

    :returns output (float, np.ndarray): value or array of exponential decay output
    '''
    output = init_val * np.exp(-1 * decay_rate * t)
    return output

def area_of_triangle(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> float:
    '''
    Given 3 (x, y) coordinate points, returns the area of the triangle between them.

    :param x1 float: first x-coordinate
    :param y1 float: first y-coordinate
    :param x2 float: second x-coordinate
    :param y2 float: second y-coordinate
    :param x3 float: third x-coordinate
    :param y3 float: third y-coordinate

    :returns area float: the area of the triangle created by the 3 given points
    '''
    area = 0.5 * (
        (x1 * (y2 - y3)) +
        (x2 * (y3 - y1)) +
        (x3 * (y1 - y2))
    )
    return area

def dist(xi: float, yi: float, xj: float ,yj: float) -> float:
    '''
    Computes distance between 2 (x, y) coordinate points.

    :param xi float: first x-coordinate
    :param yi float: first y-coordinate
    :param xj float: second x-coordinate
    :param yj float: second y-coordinate

    :returns distance float: distance between the 2 points.
    '''
    distance = np.sqrt(np.square(xi - xj) + np.square(yi - yj))
    return distance

def curvature(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> float:
    '''
    Computes the Menger Curvature, which is 1/radius of the circle connecting the 3 given
      (x, y) coordinate points. See https://en.wikipedia.org/wiki/Menger_curvature.
    
    :param x1 float: first x-coordinate
    :param y1 float: first y-coordinate
    :param x2 float: second x-coordinate
    :param y2 float: second y-coordinate
    :param x3 float: third x-coordinate
    :param y3 float: third y-coordinate

    :returns curv float: the Menger Curvature of the 3 given points.
    '''
    curv = (4 * area_of_triangle(x1, y1, x2, y2, x3, y3) /
        (dist(x1, y1, x2, y2) * dist(x2, y2, x3, y3) * dist(x3, y3, x1, y1))
    )
    return curv

# Define all colors to be used
COLOR_NAMES = [k.replace('tab:', '') for k in mcolors.TABLEAU_COLORS.keys()]
COLORS = list(mcolors.TABLEAU_COLORS.values())