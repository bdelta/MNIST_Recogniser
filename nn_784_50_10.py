# Written by bdelta for Python 3.7
# https://github.com/bdelta
# Neural network written from scratch (matrix multiplications)

from __future__ import division
import numpy as np
import pickle

in_size = 784
hidden = 50
output = 10

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_grad(x):
    return sigmoid(x)*(1 - sigmoid(x))

class NNModel(object):

    def __init__ (self, epsilon, weight=None):
        self.activation = sigmoid
        self.activation_grad = sigmoid_grad
        if weight == None:
            self.weights = [self._initWeight(hidden, in_size + 1, epsilon), \
                self._initWeight(output, hidden + 1, epsilon)]
        else:
            self.weights = weight

    def predict(self, data):
        out = self.feedforward(data)[0]
        return np.argmax(out, axis=1)

    def train(self, data, labels, batch, rate, epochs, reg):
        for n in range(epochs):
            batches = int(np.floor(data.shape[0]/batch))
            excess = data.shape[0] % batch
            for i in range(batches):
                batched_data = data[i*batch:(i+1)*batch, :]
                batched_labels = labels[i*batch:(i+1)*batch, :]
                grad_1, grad_2 = self.feedandbackprop(batched_data, batched_labels, reg)

                # Gradient descent
                self.weights[0] -= rate*grad_1
                self.weights[1] -= rate*grad_2

                if i%(batches/20) == 0:
                    print("Finished batch " + str(i))

            if excess > 0:
                excess_data = data[-excess: , :]
                excess_labels = labels[-excess: , :]
                grad_1, grad_2 = self.feedandbackprop(excess_data, excess_labels, reg)

                # Gradient descent
                self.weights[0] -= rate*grad_1
                self.weights[1] -= rate*grad_2

            hypo = self.feedforward(data)[0]
            cost = self.cost(hypo, labels, 0)
            print("Epoch number: " + str(n) + " has cost of " + str(cost))



    def feedforward(self, data):
        a_1 = np.append(np.ones((data.shape[0], 1)), data, 1) #For the bias term
        z_2 = np.matmul(a_1,self.weights[0].T)
        a_2 = self.activation(z_2)
        a_2 = np.append(np.ones((a_2.shape[0], 1)), a_2, 1)
        z_3 = np.dot(a_2, self.weights[1].T)
        hypo = self.activation(z_3)
        return hypo, a_1, a_2, z_2, z_3

    def feedandbackprop(self, data, labels, reg):
        m = data.shape[0]
        grad_1 = np.zeros(self.weights[0].shape)
        grad_2 = np.zeros(self.weights[1].shape)
        for i in range(m):
            hypo, a_1, a_2, z_2, z_3 = self.feedforward(data[i, :].reshape((1, -1)))
            label = labels[i, :].reshape((1, -1))
            if hypo.shape != label.shape:
                raise Exception("Hypothesis and labels don't match!")
            delta_3 = hypo - label
            delta_2 = np.matmul(delta_3, self.weights[1])*self.activation_grad(np.append([[1]], z_2, 1))
            grad_1 += np.matmul(delta_2.T, a_1)[1:, :]
            grad_2 += np.matmul(delta_3.T, a_2)

            # Regularization
            reg_weight_1 = np.append(self.weights[0][:, 1:], np.zeros((self.weights[0].shape[0], 1)), 1)
            reg_weight_2 = np.append(self.weights[1][:, 1:], np.zeros((self.weights[1].shape[0], 1)), 1)
            grad_1 += reg*(self.weights[0])
            grad_2 += reg*(self.weights[1])
        return grad_1/m, grad_2/m

    def cost(self, hypo, label, reg):
        if hypo.shape != label.shape:
            raise Exception("Hypothesis " + str(hypo.shape) \
                + " and labels " + str(label.shape) + " don't match!")
        m = hypo.shape[0]
        J = -label*np.log(hypo) - (1 - label)*np.log(1-hypo)
        J = np.sum(J)

        # Regularization
        if reg != 0:
            theta1 = self.weights[0][:, 1:]
            theta2 = self.weights[1][:, 1:]
            J += (reg/2)*(np.sum(theta1*theta1) + \
                np.sum(theta2*theta2))
        return J/m

    def loadWeight(self, weights):
        self.weights = weights

    def saveWeight(self, filepath):
        pickle.dump(self.weights, open(filepath, 'wb'))

    def _initWeight(self, col, row, epsilon):
        weight = np.random.rand(col, row)
        weight = weight*epsilon - 2*epsilon
        return weight