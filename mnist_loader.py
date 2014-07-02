import os
import struct
from array import array
import numpy as np
from PIL import Image

'''
    This class loads and formats the data from MNIST Handwritten database.
    It is also capable of showing and saving individual images, for debuging purposes.
    More info in the database and the it's format is available on:
    http://yann.lecun.com/exdb/mnist/

    Based on work done by Richard Marko
    https://github.com/sorki/python-mnist

'''
class MNIST(object):
    def __init__(self, path='.'):
        self.path = path

        self.test_img_fname = 't10k-images-idx3-ubyte'
        self.test_lbl_fname = 't10k-labels-idx1-ubyte'

        self.train_img_fname = 'train-images-idx3-ubyte'
        self.train_lbl_fname = 'train-labels-idx1-ubyte'

        self.test_images = []
        self.test_labels = []

        self.train_images = []
        self.train_labels = []


    def load_testing(self):
        ims, labels = self.load(os.path.join(self.path, self.test_img_fname),
                         os.path.join(self.path, self.test_lbl_fname))

        data = self.format_data(ims, labels)
        return data

    def load_training(self):
        ims, labels = self.load(os.path.join(self.path, self.train_img_fname), os.path.join(self.path, self.train_lbl_fname))
        data = self.format_data(ims, labels)

        return data

    def load(self, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                    'got %d' % magic)

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                    'got %d' % magic)

            image_data = array("B", file.read())

        images = []
        for i in xrange(size):
            images.append([0]*rows*cols)

        for i in xrange(size):
            images[i][:] = image_data[i*rows*cols : (i+1)*rows*cols]

        return images, labels


    '''
        data is a list of tuples (x, y), with x corresponding to a list of 784 (28 * 28) integers,
        each corresponding to the gray level of a specific pixel in the image, and y is the digit that the image represents.
        This method takes that and returns it in a numpy/linear algebra friendly format.
        x becomes to a numpy column vector, with 784 entries,
        and y becomes a numpy column vector, with 10 entries, with a 1 on the correct digit index (zero-based) and zero everywhere else.
    '''
    def format_data(self, images, labels):
        data = []
        for i, l in zip(images, labels):
            ni = np.array([i]).T/255.0
            nl = np.zeros((10, 1))
            nl[l] = 1.0
            data.append((ni, nl))

        return data


    def show_image(self, data):
        size = 28, 28

        t = data.tolist()
        data = []
        for i in t:
            data.append(int(i[0]*255))


        #print data

        im = Image.new("L", size, 255)
        imdata = list(im.getdata())
        #print imdata

        imdata = data
        im.putdata(imdata)
        im.show()
        im.save("oi.png")





'''
from mnist import MNIST
from PIL import Image
mndata = MNIST('.')
data = mndata.load_training()

images = data[0]

def save_to_image(data):
    size = 28, 28

    im = Image.new("L", size, 255)
    imdata = list(im.getdata())
    imdata = data
    im.putdata(imdata)
    im.show()


save_to_image(images[0])
'''
