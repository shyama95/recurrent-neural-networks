import numpy as np

class RNN:
    Whh = []
    Wxh = []
    Why = []
    output = []

    def __init__(self, M, N):
        self.Whh = np.zeroes(M, N)
        self.Wxh = np.zeroes(M, N)
        self.Why = np.zeroes(M, N)

    def forward(self, input):
        ...
        # computes and returns the output after the last time step and also saves it in the state variable output
        self.output = []
        return self.output

    def backward(self):
        ...
        # unrolls the RNN and performs back propagation through time, computes and updates the state variables
        self.Whh = []
        self.Wxh = []
        self.Why = []
