import numpy as np
import time

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

testData = ''
with open('testData/test_data.txt', 'r') as file:
    testData = file.read()
testData = testData.splitlines(keepends=False)

V = model.V

# for value in trainingData:
#     input_values = value.split(' ')
#     input_values.remove('')
#     output = rnn.forward(input_values)
#     print(output)

# print(rnn.total_loss(trainingData, targetData))
start = time.time()
rnn.backward(trainingData, targetData)
start = time.time() - start
print('Total time taken = ' + str(start))

output_csv = ''
for value in testData:
    input_values = value.split(' ')
    input_values.remove('')
    output = rnn.forward(input_values)
    output_csv += str(output) + ','

print(output_csv)
with open('bestModel/output.csv', 'w') as file:
    file.write(output_csv)
