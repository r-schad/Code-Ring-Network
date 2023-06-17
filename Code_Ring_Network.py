import numpy as np
import matplotlib.pyplot as plt

from Ring_Layer import Ring_Layer
from Code_Layer import Code_Layer

class Code_Ring_Network:
    def __init__(self, ring_layer: Ring_Layer, code_layer: Code_Layer) -> None:
        self.ring_layer = ring_layer
        self.code_layer = code_layer

