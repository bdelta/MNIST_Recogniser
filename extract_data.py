# Written by bdelta for Python 3.7
# https://github.com/bdelta
# Used to extract from the MNIST dataset
# provided by Yann LeCun 
# http://yann.lecun.com/exdb/mnist/
# Extract from ubyte file format and view

import numpy as np

def normalizeImage(x):
    return x/255.0

class FileExtractor(object):

    def __init__(self, filepath=None):
        self._f = None
        self._type = None
        self._num = None
        self._imrows = None
        self._imcols = None
        self.openfile(filepath)

    def openfile(self, filepath):
        if filepath:
            try:
                self._f = open(filepath, "rb")
            except Exception as e:
                print(e)
                self._f = None
            else:
                print("Successfully opened file!")
                self._readHeader()
        else:
            self._f = None

    def closefile(self):
        if self._f:
            self._f.close()

    def readByte(self, bytes):
        if self._f:
            return self._f.read(bytes)

    def readInt(self):
        dat = self.readByte(4)
        return int.from_bytes(dat, byteorder='big')

    def extractImage(self):
        if self._type == 'image':
            im = self._f.read(self._imrows*self._imcols)
            im = np.fromstring(im, dtype='B').reshape((self._imrows,self._imcols))
            return im
        else:
            print("This file does not contain images!")
            return None

    def extractLabel(self):
        if self._type == 'label':
            la = self.readByte(1)
            return int.from_bytes(la, byteorder='big')
        else:
            print("This file does not contain labels!")
            return None

    def _readHeader(self):
        header = self.readInt()
        self._num = self.readInt()
        if header == 2049:
            self._type = 'label'
        elif header == 2051:
            self._type = 'image'
            self._imrows = self.readInt()
            self._imcols = self.readInt()

    @property
    def file(self):
        return self._f
    
    @property
    def type(self):
        return self._type
    
    @property
    def num(self):
        return self._num
    
    @property
    def imrows(self):
        return self._imrows

    @property
    def imcols(self):
        return self._imcols
    