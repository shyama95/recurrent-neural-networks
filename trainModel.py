import numpy as np
from src.Model import *
from src.RNN import *

modelName = 'models'
data = 'trainingData/train_data.txt'
target = 'trainingData/train_labels.txt'
model = Model()
rnn = RNN(model.H, model.V)

trainingData = ''
with open(data, 'r') as file:
    trainingData = file.read()
trainingData = trainingData.splitlines(keepends=False)

targetData = ''
with open(target, 'r') as file:
    targetData = file.read()
targetData = targetData.splitlines(keepends=False)
targetData = list(map(int, targetData))

V = model.V

for value in trainingData:
    input_values = value.split(' ')
    input_values.remove('')
    output = rnn.forward(input_values)
    print(output)
