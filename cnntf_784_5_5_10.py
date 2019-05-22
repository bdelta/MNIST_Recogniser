# Written by bdelta for Python 3.7
# https://github.com/bdelta
# Convolutional Neural network using tensorflow

import tensorflow as tf
import numpy as np

in_size = 784
output = 10

class NNModel(object):

    def __init__ (self, weight_path=None):
        # Define the CNN model
        self.x = tf.placeholder('float', [None, in_size])
        self.y = tf.placeholder('float', [None, output])

        # Tensorflow weights
        self.conv1_weights = tf.Variable(tf.random_normal([5, 5, 1, 32]))
        self.conv2_weights = tf.Variable(tf.random_normal([5, 5, 32, 64]))
        self.fc_weights = tf.Variable(tf.random_normal([7*7*64, 1024]))
        self.out_weights = tf.Variable(tf.random_normal([1024, output]))

        self.conv1_biases = tf.Variable(tf.random_normal([32]))
        self.conv2_biases = tf.Variable(tf.random_normal([64]))
        self.fc_biases = tf.Variable(tf.random_normal([1024]))
        self.out_biases = tf.Variable(tf.random_normal([output]))

        self.weights_saver = tf.train.Saver()

        # Setting up the model
        self.activation = tf.nn.relu
        self.z = tf.reshape(self.x, shape=[-1, 28, 28, 1])
        self.prediction = self.feedforward(self.z)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(self.y, \
                self.prediction))
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
        output = self.sess.run(self.prediction, feed_dict={self.x:data})
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
        conv1 = self.conv2d(data, self.conv1_weights) + self.conv1_biases
        conv1 = self.maxpool2d(conv1)

        conv2 = self.conv2d(conv1, self.conv2_weights) +self.conv2_biases
        conv2 = self.maxpool2d(conv2)

        fc = tf.reshape(conv2, [-1, 7*7*64])
        fc = tf.matmul(fc, self.fc_weights) + self.fc_biases
        fc = self.activation(fc)

        output = tf.matmul(fc, self.out_weights) + self.out_biases

        return output

    def saveWeight(self, filepath):
        self.weights_saver.save(self.sess, filepath)
        self.saved_weights = filepath
        
    def closeSession(self):
        self.sess.close()

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def maxpool2d(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')