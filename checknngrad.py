# Written by bdelta for Python 3.7
# https://github.com/bdelta
# Used to check that the backpropagation gradients in 
# nn_784_50_10.py are correct

from nn_784_50_10 import *
import copy

in_size = 10
hidden = 20
output = 5
epsilon = 0.5

# Create random weights
def initWeight(col, row, epsilon):
    weight = np.random.rand(col, row)
    weight = weight*epsilon - 2*epsilon
    weight += np.ones((col, row))
    return weight

theta1 = initWeight(hidden, in_size + 1, epsilon)
theta2 = initWeight(output, hidden + 1, epsilon)

weights = [theta1, theta2]

test_input = np.ones((1, 10))
test_output = np.zeros((1, 5))
test_output[:, 3] = 1

# Compare only the first 10 partial derivatives
# First 10 partial derivatives obtained by gradient estimation
diff = 0.001
grads = np.zeros(10)
for i in range(10):
    temp = copy.deepcopy(weights)
    temp[0][0, i] += diff
    nnUpper = NNModel(epsilon, temp)
    upper = nnUpper.cost(nnUpper.feedforward(test_input)[0], test_output, 0)
    temp[0][0, i] -= 2*diff
    nnLower = NNModel(epsilon, temp)
    lower = nnLower.cost(nnLower.feedforward(test_input)[0], test_output, 0)

    grads[i] = (upper-lower)/(2*diff)
    
print(grads)

# First 10 partial derivatives obtained by backprop
nn = NNModel(epsilon, weights)
print(nn.feedandbackprop(test_input, test_output, 0)[0][0, 0:10])