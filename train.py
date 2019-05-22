# Written by bdelta for Python 3.7
# https://github.com/bdelta
# Training the neural network made from scratch with MNIST

from extract_data import *
from nn_784_50_10 import *
import matplotlib.pyplot as plt

# One hot encoding of MNIST
def encodeLabel(label, N):
    y = np.zeros((1, N))
    y[:, label] = 1
    return y

path_train_im = "./MNIST_dataset/train-images/train-images.idx3-ubyte"
path_train_labels = "./MNIST_dataset/train-labels/train-labels.idx1-ubyte"
path_test_im = "./MNIST_dataset/test-images/t10k-images.idx3-ubyte"
path_test_labels = "./MNIST_dataset/test-labels/t10k-labels.idx1-ubyte"
path_weights = "nn_784_50_10.p"

learning_rate = 0.001
epochs = 4
reg = 0.005
epsilon = 10e-4 # Random intialization range

# Open the training files
la_f = FileExtractor()
la_f.openfile(path_train_labels)
print("File has type " + str(la_f.type))
print("There are " + str(la_f.num) + " samples")

im_f = FileExtractor()
im_f.openfile(path_train_im)
print("File has type " + str(im_f.type))
print("There are " + str(im_f.num) + " samples")

# Extract label + image in data arrays
im_arr = np.zeros((im_f.num, im_f.imcols*im_f.imrows))
lab_arr = np.zeros((la_f.num, 10))
for i in range(im_f.num):
    la = la_f.extractLabel()
    lab_arr[i, :] = encodeLabel(la, 10)
    x = im_f.extractImage()
    x = normalizeImage(x)
    im_arr[i, :] = x.reshape(1, im_f.imcols*im_f.imrows) 
    if i == 0:
        print("Press q to close image")
        plt.imshow(x)
        plt.show()

im_f.closefile()
la_f.closefile()

# Create neural network
weights = pickle.load(open(path_weights, "rb"))
nn = NNModel(epsilon, weights)
nn.train(im_arr, lab_arr, 200, learning_rate, epochs, reg)
nn.saveWeight(path_weights)

# Test set accuracy
# Loading the test files
la_f = FileExtractor()
la_f.openfile(path_test_labels)
print("File has type " + str(la_f.type))
print("There are " + str(la_f.num) + " samples")

im_f = FileExtractor()
im_f.openfile(path_test_im)
print("File has type " + str(im_f.type))
print("There are " + str(im_f.num) + " samples")

la_arr = np.zeros(la_f.num, dtype=int)
test_arr = np.zeros((im_f.num, im_f.imcols*im_f.imrows))
for i in range(im_f.num):
    la = la_f.extractLabel()
    la_arr[i] = la
    x = im_f.extractImage()
    x = normalizeImage(x)
    test_arr[i, :] = x.reshape(1, im_f.imcols*im_f.imrows) 
    if i == 0:
        print("Press q to close image")
        plt.imshow(x)
        plt.show()
        

im_f.closefile()
la_f.closefile()

prediction = nn.predict(test_arr)
print(la_arr)
print("Accuracy = " + str(100*(np.sum(prediction==la_arr)/float(prediction.size))) + "%")

