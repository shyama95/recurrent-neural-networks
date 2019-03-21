class Model:
    # number of layers in the model
    nLayers = 1
    # H => RNN size, i.e. hidden dimension
    H = 64
    # V => vocabulary size
    V = 153
    # D => Word vector size
    D = 1
    isTrain = False

    def forward(self, input):
        ...
        output = []
        return output

    def backward(self):
        ...
        # Sequentially calls the backward function for the layers contained in the model
