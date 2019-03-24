from src.RNN import *
from src.Model import *

modelName = 'models'
data = 'testData/test_data.txt'
model = Model()
rnn = RNN(model.H, model.V)

# reading test data
testData = ''
with open(data, 'r') as file:
    testData = file.read()
testData = testData.splitlines(keepends=False)

# reading model data
rnn.Wxh = np.load(modelName + '/Wxh.npy')
rnn.Whh = np.load(modelName + '/Whh.npy')
rnn.Why = np.load(modelName + '/Why.npy')
rnn.bias = np.load(modelName + '/bias.npy')

# computing output
output_csv = ''
output_csv2 = 'id,label\n'

i = 0
for value in testData:
    input_values = value.split(' ')
    input_values.remove('')
    output = rnn.forward(input_values)
    output_csv += str(output) + '\n'
    output_csv2 += str(i) + ',' + str(output) + '\n'
    i += 1

output_csv = output_csv[:-1]
output_csv2 = output_csv2[:-1]
with open('testPrediction.csv', 'w+') as file:
    file.write(output_csv)

with open('bestModel/output.csv', 'w+') as file:
    file.write(output_csv2)

print('Result saved to testPrediction.csv and output.csv')
