# Written by bdelta for Python 3.7
# https://github.com/bdelta
# Neural network using tensorflow

import tensorflow as tf
import numpy as np

in_size = 784
hidden1 = 800
hidden2 = 800
output = 10

class NNModel(object):

    def __init__ (self, weight_path=None):
        # Define the NN model
        self.x = tf.placeholder('float', [None, in_size])
        self.y = tf.placeholder('float', [None, output])

        # Tensorflow weights
        self.input_weights = tf.Variable(tf.random_normal([in_size, hidden1]))
        self.input_biases = tf.Variable(tf.random_normal([hidden1]))
        self.hidden1_weights = tf.Variable(tf.random_normal([hidden1, hidden2]))
        self.hidden1_biases = tf.Variable(tf.random_normal([hidden2]))
        self.hidden2_weights = tf.Variable(tf.random_normal([hidden2, output]))
        self.hidden2_biases = tf.Variable(tf.random_normal([output]))
        self.weights_saver = tf.train.Saver()

        self.activation = tf.nn.relu
        prediction = self.feedforward(self.x)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(self.y, \
                prediction))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost) 

        # Create computation graph
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        self.saved_weights = None
        if weight_path != None:
            self.saved_weights = weight_path
            self.weights_saver.restore(self.sess, self.saved_weights)
            print("Loaded weights!")

    def predict(self, data):          
        out = self.feedforward(self.x)
        output = self.sess.run(out, feed_dict={self.x:data})
        output = np.argmax(output, axis=1)
        print(output)
        return output 

    def train(self, data, labels, batch, epochs):
        for n in range(epochs):
            batches = int(np.floor(data.shape[0]/batch))
            excess = data.shape[0] % batch
            epoch_cost = 0

            for i in range(batches):
                epoch_x = data[i*batch:(i+1)*batch, :]
                epoch_y = labels[i*batch:(i+1)*batch, :]
                
                _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.x:epoch_x, self.y:epoch_y})
                epoch_cost += c

            if excess > 0:
                epoch_x = data[-excess: , :]
                epoch_y = labels[-excess: , :]
                _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.x:epoch_x, self.y:epoch_y})
                epoch_cost += c

            
            print("Epoch number: " + str(n) + " has cost of " + str(epoch_cost))

        

    def feedforward(self, data):
        z_2 = tf.matmul(data, self.input_weights) + self.input_biases
        a_2 = self.activation(z_2)
        z_3 = tf.matmul(a_2, self.hidden1_weights) + self.hidden1_biases
        a_3 = self.activation(z_3)
        z_4 = tf.matmul(a_3, self.hidden2_weights) + self.hidden2_biases
        return z_4

    def saveWeight(self, filepath):
        self.weights_saver.save(self.sess, filepath)
        self.saved_weights = filepath
        
    def closeSession(self):
        self.sess.close()