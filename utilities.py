import numpy as np
import matplotlib.colors as mcolors

def sigmoid(v, beta, mu):
    try:
        return 1 / (1 + (np.e ** ((-1*beta) * (v - mu))))
    except RuntimeWarning:
        print(f'Overflow - v={v}')
        return 1 / (1 + (np.e ** ((-1*beta) * (v - mu))))

COLOR_NAMES = [k.replace('tab:', '') for k in mcolors.TABLEAU_COLORS.keys()]
COLORS = list(mcolors.TABLEAU_COLORS.values())