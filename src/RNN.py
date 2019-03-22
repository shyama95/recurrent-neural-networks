import numpy as np

symbol_table = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 19, 20, 21, 26, 27, 30, 33, 37, 38, 39, 40, 41, 42, 43, 45, 54, 57, 60, 63, 64, 65, 66, 75, 77, 78, 83, 85, 91, 93, 94, 96, 97, 99, 102, 104, 110, 114, 117, 118, 119, 120, 122, 125, 128, 132, 133, 140, 141, 142, 143, 144, 146, 148, 155, 156, 157, 158, 159, 160, 162, 163, 168, 172, 173, 174, 175, 176, 179, 180, 183, 184, 185, 191, 192, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 211, 212, 213, 214, 219, 220, 221, 224, 226, 228, 229, 230, 231, 233, 234, 240, 242, 243, 252, 254, 255, 256, 258, 259, 260, 264, 265, 266, 268, 269, 270, 272, 289, 292, 293, 295, 298, 300, 301, 307, 308, 309, 311, 314, 320, 322, 324, 331, 332, 340]


class RNN:
    Whh = []
    Wxh = []
    Why = []
    h = []
    y = []
    V = 153
    H = 64

    def __init__(self, h, v):
        self.H = h
        self.V = v
        self.Whh = np.random.rand(h, h) * 0.01
        self.Wxh = np.random.rand(h, v) * 0.01
        self.Why = np.random.rand(1, h) * 0.01

    def forward(self, input):
        self.h = np.zeros(self.H, dtype=float)

        for inp in input:
            x = self.input_layer(inp)
            self.next_state(x)

        self.compute_y()
        return self.y

    def backward(self):
        ...
        # unrolls the RNN and performs back propagation through time, computes and updates the state variables
        self.Whh = []
        self.Wxh = []
        self.Why = []

    def next_state(self, x):
        self.h = np.tanh(np.dot(self.Whh, self.h) + np.dot(self.Wxh, x))

    def compute_y(self):
        self.y = np.dot(self.Why, self.h)

    def input_layer(self, input):
        x = np.zeros(self.V)
        x[symbol_table.index(int(input))] = 1
        return x
