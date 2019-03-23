import numpy as np

symbol_table = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 19, 20, 21, 26, 27, 30, 33, 37, 38, 39, 40, 41, 42, 43, 45, 54, 57, 60, 63, 64, 65, 66, 75, 77, 78, 83, 85, 91, 93, 94, 96, 97, 99, 102, 104, 110, 114, 117, 118, 119, 120, 122, 125, 128, 132, 133, 140, 141, 142, 143, 144, 146, 148, 155, 156, 157, 158, 159, 160, 162, 163, 168, 172, 173, 174, 175, 176, 179, 180, 183, 184, 185, 191, 192, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 211, 212, 213, 214, 219, 220, 221, 224, 226, 228, 229, 230, 231, 233, 234, 240, 242, 243, 252, 254, 255, 256, 258, 259, 260, 264, 265, 266, 268, 269, 270, 272, 289, 292, 293, 295, 298, 300, 301, 307, 308, 309, 311, 314, 320, 322, 324, 331, 332, 340]


class RNN:
    Whh = []
    Wxh = []
    Why = []
    bias = []
    h = []
    output = 0
    V = 153
    H = 64
    D = 100
    h_all = []

    def __init__(self, h, v):
        self.H = h
        self.V = v
        range_low = -np.sqrt(1/self.V)
        range_high = np.sqrt(1 / self.V)
        self.Whh = np.random.uniform(range_low, range_high, (h, h))
        self.Wxh = np.random.uniform(range_low, range_high, (h, v))
        self.Why = np.random.uniform(range_low, range_high, (1, h))
        self.bias = np.random.uniform(range_low, range_high)

    def forward(self, input):
        self.h = np.zeros(self.H, dtype=float)
        self.h_all.append(self.h)

        for inp in input:
            x = self.input_layer(inp)
            self.next_state(x)

        self.compute_output()
        return self.output

    def backward(self, input, target):
        N = 10
        eta = 0.001
        initial_loss = self.total_loss(input, target)
        for epoch in range(N):
            j = 0
            print('Epoch : ' + str(epoch + 1))
            print('Initial loss = ' + str(initial_loss))

            Wxh = self.Wxh.copy()
            Whh = self.Whh.copy()
            Why = self.Why.copy()
            bias = self.bias

            for value in input:
                input_values = value.split(' ')
                input_values.remove('')
                input_values = list(map(int, input_values))
                [dL_dWxh, dL_dWhh, dL_dWhy, dL_dbias] = self.gradient_computation(input_values, target[j])
                self.Wxh = self.Wxh + eta * dL_dWxh
                self.Whh = self.Whh + eta * dL_dWhh
                self.Why = self.Why + eta * dL_dWhy
                self.bias = self.bias + eta * dL_dbias
                j += 1

            new_loss = self.total_loss(input, target)

            if new_loss > initial_loss:
                self.Wxh = Wxh
                self.Whh = Whh
                self.Why = Why
                self.bias = bias
                print('Loss set to : ' + str(self.total_loss(input, target)))
                break
            else:
                initial_loss = new_loss

        return [self.Wxh, self.Whh, self.Why, self.bias]

    def next_state(self, x):
        self.h = np.tanh(np.dot(self.Whh, self.h) + np.dot(self.Wxh, x))
        self.h_all.append(self.h)

    def compute_output(self):
        y = np.dot(self.Why, self.h) + self.bias
        output = self.sigmoid(y)
        self.output = 0 if output < 0.5 else 1

    def input_layer(self, input):
        x = np.zeros(self.V)
        x[symbol_table.index(int(input))] = 1
        return x

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def cross_entropy_loss(self, target, predicted):
        loss = target * np.log(predicted + 1e-10) + (1 - target) * np.log(1 - predicted + 1e-10)
        return loss

    def total_loss(self, input, target):
        total_loss = 0
        i = 0
        for value in input:
            input_values = value.split(' ')
            input_values.remove('')
            predicted = self.forward(input_values)
            total_loss -= self.cross_entropy_loss(target[i], predicted)
            i += 1
        total_loss = total_loss / len(input)
        return total_loss

    def gradient_computation(self, input, target):
        predicted = self.forward(input)
        dL_dWhy = np.transpose((target - predicted) * self.h)
        dL_dbias = target - predicted

        dL_dWxh = np.zeros(self.Wxh.shape)
        dL_dWhh = np.zeros(self.Whh.shape)

        input_length = len(input)
        h_list = self.h_all.copy()
        delta_t = (target - predicted) * (np.eye(self.H) -
                                          np.matmul(h_list[input_length], np.transpose(h_list[input_length])))

        for t in range(input_length-1, input_length - self.D, -1):
            xt = self.input_layer(input[t - 1]).reshape((1, self.V))
            dL_dWxh += np.matmul(delta_t, np.transpose(self.Why)) * xt
            dL_dWhh += delta_t * h_list[t]
            delta_t = delta_t * self.Whh * (np.eye(self.H) - np.matmul(h_list[t], np.transpose(h_list[t])))

        dL_dWhh = dL_dWhh * self.Why

        return [dL_dWxh, dL_dWhh, dL_dWhy, dL_dbias]
