# implements the cross-entropy loss function
class Criterion:
    def forward(self, input, target):
        ...
        # computes the average cross-entropy loss over the batch

    def backward(self, input, target):
        ...
        # computes and returns the gradient of the Loss with respect to the input to this layer
