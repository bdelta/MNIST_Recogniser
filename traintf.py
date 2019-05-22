# Written by bdelta for Python 3.7
# https://github.com/bdelta
# Training the neural network made in tensorflow with MNIST dataset
# Select a model by importing it

from extract_data import *
# from nntf_784_800_10 import *
# from nntf_784_800_800_10 import *
from cnntf_784_5_5_10 import *
import matplotlib.pyplot as plt

# One hot encoding of MNIST
def encodeLabel(label, N):
    y = np.zeros((1, N))
    y[:, label] = 1.0
    return y

path_train_im = "./MNIST_dataset/train-images/train-images.idx3-ubyte"
path_train_labels = "./MNIST_dataset/train-labels/train-labels.idx1-ubyte"
path_test_im = "./MNIST_dataset/test-images/t10k-images.idx3-ubyte"
path_test_labels = "./MNIST_dataset/test-labels/t10k-labels.idx1-ubyte"

# Change weight path according to model
# path_weights = "./nntf_784_800_10/nntf_784_800_10.ckpt"
# path_weights = "./nntf_784_800_800_10/nntf_784_800_800_10.ckpt"
path_weights = "./cnntf_784_5_5_10/cnntf_784_5_5_10.ckpt"

# If no weights have been trained yet set this to true
first_train = False

epochs = 20
batch = 1000

# Open the training files
la_f = FileExtractor()
la_f.openfile(path_train_labels)
print("File has type " + str(la_f.type))
print("There are " + str(la_f.num) + " samples")

im_f = FileExtractor()
im_f.openfile(path_train_im)
print("File has type " + str(im_f.type))
print("There are " + str(im_f.num) + " samples")

#Extract label + image in data arrays
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

# Test set accuracy
# Loading the test files
tla_f = FileExtractor()
tla_f.openfile(path_test_labels)
print("File has type " + str(tla_f.type))
print("There are " + str(tla_f.num) + " samples")

tim_f = FileExtractor()
tim_f.openfile(path_test_im)
print("File has type " + str(tim_f.type))
print("There are " + str(tim_f.num) + " samples")

tla_arr = np.zeros((tla_f.num, 10))
test_arr = np.zeros((tim_f.num, tim_f.imcols*tim_f.imrows))
for i in range(tim_f.num):
    la = tla_f.extractLabel()
    tla_arr[i, :] = encodeLabel(la, 10)
    x = tim_f.extractImage()
    x = normalizeImage(x)
    test_arr[i, :] = x.reshape(1, tim_f.imcols*tim_f.imrows) 
    if i == 0:
        print("Press q to close image")
        plt.imshow(x)
        plt.show()
        

tim_f.closefile()
tla_f.closefile()

# Create neural network

if first_train:
    nn = NNModel(None)
else:
    nn = NNModel(path_weights)
nn.train(im_arr, lab_arr, batch, epochs)
nn.saveWeight(path_weights)

prediction = nn.predict(test_arr)
tla_arr = np.argmax(tla_arr, axis=1)
print("Accuracy = " + str(100*(np.sum(prediction==tla_arr)/float(prediction.size))) + "%")

nn.closeSession()