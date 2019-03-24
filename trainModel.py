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

V = model.V

start = time.time()
[Wxh, Whh, Why, bias] = rnn.backward(trainingData, targetData)
start = time.time() - start
print('Total time taken = ' + str(start))

np.save(modelName + '/Wxh', Wxh)
np.save(modelName + '/Whh', Whh)
np.save(modelName + '/Why', Why)
np.save(modelName + '/bias', bias)

print('Model written to ' + modelName)
