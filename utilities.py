import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

EPSILON = np.finfo(np.float64).eps

def sigmoid(v: (float, np.ndarray), beta: float = 50.0, mu: float = 0.1) -> (float, np.ndarray):
    '''
    Returns a sigmoid output with steepness beta centered at mu.

    :param v (float, np.ndarray): input value(s)
    :param beta float: steepness
    :param mu float: center of curve

    :returns output (float, np.ndarray): value or array of sigmoid output
    '''
    v = np.where(v < EPSILON, EPSILON, v)
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

def exponential(t: (float, np.ndarray), rate: float, init_val: float, center: float = 0.0) -> (float, np.ndarray):
    '''
    Returns an exponential output of input variable t, with initial value init_val. 
        The growth/decay rate (depending on sign) is determined by parameter `rate`.

    :param t (float, np.ndarray): input value(s)
    :param rate float: growth/decay rate of the exponential curve; 
        if rate > 0, the function is exponential growth; if rate < 0, the function is exponential decay
    :param init_val float: initial value of the exponential curve
    :param center float: the central/starting x-value of the exponential, i.e. where the y-value is init_val

    :returns output (float, np.ndarray): value or array of exponential decay output
    '''
    output = init_val * np.exp(rate * (t - center))
    return output

def bimodal_exponential_noise(num_low, num_high, noise_rate_low, noise_rate_high, shuffle=True, clip_01=True):
    '''
    Returns a rolled or shuffled vector of noise values clipped to [0,1]. Returned vector is in the shape (num_low + num_high, 1).
    '''
    noise1 = np.random.exponential(1 / noise_rate_low, num_low)
    noise2 = 1 - np.random.exponential(1 / noise_rate_high, num_high)
    noise = np.concatenate((noise1, noise2))
    if shuffle:
        np.random.shuffle(noise)
    else:
        random_roll = np.random.rand(0, num_low+num_high)
        noise = np.roll(noise, random_roll)
    if clip_01:
        noise = np.clip(noise, 0, 1)

    return noise

def bimodal_gaussian_noise(num_low, num_high, mean_low, mean_high, sigma_low, sigma_high, shuffle=True, clip_01=True):
    noise1 = np.random.normal(loc=mean_low, scale=sigma_low, size=num_low)
    noise2 = np.random.normal(loc=mean_high, scale=sigma_high, size=num_high)
    noise = np.concatenate((noise1, noise2))
    if shuffle:
        np.random.shuffle(noise)
    else:
        random_roll = np.random.randint(0, num_low+num_high)
        noise = np.roll(noise, random_roll)

    if clip_01:
        high_idxs = np.argwhere(noise > 1.0).flatten()
        noise[high_idxs] = np.clip(noise[high_idxs], 0, 1)
        noise[high_idxs] -= np.random.rand(len(high_idxs)) * 0.1

        low_idxs = np.argwhere(noise < 0.0).flatten()
        noise[low_idxs] = np.clip(noise[low_idxs], 0, 1)
        noise[low_idxs] += np.random.rand(len(low_idxs)) * 0.1

        noise = np.clip(noise, 0, 1)

    return noise

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
    area = np.abs(0.5 * (
        (x1 * (y2 - y3)) +
        (x2 * (y3 - y1)) +
        (x3 * (y1 - y2))
    ))
    return area

def dist(xi: float, yi: float, xj: float, yj: float) -> float:
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

def diff_curvature(xs, ys):
    dx = np.diff(xs, prepend=0)
    dx2 = np.diff(xs, 2, prepend=[0,0])
    dy = np.diff(ys, prepend=0)
    dy2 = np.diff(ys, 2, prepend=[0,0])
    top = np.abs((dx * dy2) - (dy * dx2))
    bottom = np.power((np.square(dx) + np.square(dy)), 3/2)
    return np.abs(np.diff(top / bottom, prepend=0))

def write_params(filename, **kwargs):
    with open(filename, 'w') as f:
        for var, val in kwargs.items():
            f.write(f'{var}: {val}\n')
        f.close()

# Define all colors to be used
COLOR_NAMES = [k.replace('tab:', '') for k in mcolors.TABLEAU_COLORS.keys()]
COLORS = list(mcolors.TABLEAU_COLORS.values())
def get_color_range(num_colors, map_name='hsv'):
    cmap = plt.get_cmap(map_name, num_colors)
    color_range = cmap(np.arange(0,num_colors))
    return color_range
