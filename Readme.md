# MNIST Digit Recogniser

A python 3 application using the tkinter UI and tensorflow to recognize digits trained on the MNIST dataset.

There are four different models supplied:
* A self-made neural network using only numpy multrix multiplications
* A tensorflow neural network with 1 hidden layer of 800 nodes
* A tensorflow neural network with 2 hidden layer of 800 nodes
* A convolutional neural network with 2 convolutional layers

An in-depth writeup of this project is at https://thenextepoch.blogspot.com/

Youtube demo:
https://www.youtube.com/watch?v=kHgDBAbkma0

### Requirements
```
python 3

Modules:
    tensorflow (GPU edition if available)
    numpy
    Pillow
    tkinter
    matplotlib
    copy
    pickle

```

## Running

Clone the repo to your local working directory.
Install modules using pip in a virtual environment.
Edit main.py to specify model.
Run main.py in virtual environment, draw a digit and predict!
Run train.py to train the self made neural network.
Run traintf.py to train any of the tensorflow neural networks.

## Acknowledgments

* Andrew Ng Machine Learning course on [coursera](https://www.coursera.org/learn/machine-learning)
* [Sentdex](https://www.youtube.com/user/sentdex) Machine Learning with tensorflow series on youtube

* [Yann Lecun](http://yann.lecun.com/exdb/mnist/) MNIST database and his research in neural networks