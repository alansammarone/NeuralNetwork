'''
	This implements a feedforward neural network. 
	Learning is achieved through backpropagation and stochastic gradient descent.

'''

import math
import json
import ast
import numpy as np
import pickle
import sys
import random
from mnist_loader import MNIST


def activation(z):
	return 1.0/(1 + math.exp(-z))

def activation_prime(z):
	return activation(z) * (1 - activation(z))


activation_vectorized = np.vectorize(activation)	
activation_prime_vectorized = np.vectorize(activation_prime)


class NeuralNetwork:

	activation_vectorized = np.vectorize(activation)
	activation_prime_vectorized = np.vectorize(activation_prime)


	def cost(self, y, a):
		r = 0
		for i in range(len(y)):
			q = np.absolute(y[i] - a[i])

			r += q.dot(np.transpose(q))

		return r
	
	def __init__(self, initialization):

		self.layers = []
		self.biases = []
		self.weights = [] 

		

		if type(initialization) is list: # We should initialize the net with random parameters
			self.layers = initialization
			self.initialize_with_random_parameters()
		elif type(initialization) is str: # We should initialize the net with parameters saved in a json
			self.initialize_with_saved_preset(initialization)


	def initialize_with_random_parameters(self):
		self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]
		self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.layers[:-1], self.layers[1:])]
		# Divide the weights by sqrt(x) to make the standart deviation smaller, reducing the chance that neurons will saturate.


	def initialize_with_saved_preset(self, preset_name):
		neural_net_parameters = self.load_parameters(preset_name)
		self.layers = neural_net_parameters["layers"]
		self.weights = [np.matrix(w) for w in neural_net_parameters["weights"]]
		self.biases = [np.matrix(b) for b in neural_net_parameters["biases"]]

	def feed_forward(self, a):
	
		for weight, bias in zip(self.weights, self.biases):
			a = self.activation_vectorized(np.dot(weight,a) + bias)

		return a

	def validate_data(self, validation_data):
		measure_error = False
		count = 0
		if measure_error is True:
			e = 0
			for x, y in validation_data:
				e += self.cost(y, self.feed_forward(x))

			e /= len(validation_data)

			return " average error %s " % e
		else:
			for x, y in validation_data:
				a = self.feed_forward(x)
				r = np.zeros((10, 1))
				r[np.argmax(a)] = 1.0

				if (r == y).all():
					count += 1

			return " got %s / %s right. That's %.2f" % (count, len(validation_data), count*100.0/len(validation_data)) + "%."


	def stochastic_gradient_descent(self, training_data, test_data=None, epochs=10, batch_size=10, learning_rate=3.0):

		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.epochs = epochs

		for e in range(self.epochs):
			
			random.shuffle(training_data)
			mini_batches = [training_data[n:n+self.batch_size] for n in range(0, len(training_data), self.batch_size)]

			for batch in mini_batches: self.learn_from_batch(batch)
			if test_data is not None: # Let's see how good we are doing. Behold, that slows down the process considerably.
				validation_measure = self.validate_data(test_data)	
				print "Epoch %s: %s" % (e, validation_measure)
			


	def learn_from_batch(self, data):


		delta_nabla_w = [np.zeros(w.shape) for w in self.weights]
		delta_nabla_b = [np.zeros(b.shape) for b in self.biases]

		for x, y in data:
			this_nabla_w, this_nabla_b = self.get_cost_gradient(x, y)
			delta_nabla_w = [dnw+tnw for dnw, tnw in zip(delta_nabla_w, this_nabla_w)]
			delta_nabla_b = [dnb+tnb for dnb, tnb in zip(delta_nabla_b, this_nabla_b)]

		self.weights = [w-((self.learning_rate/len(data))*dnw) for w, dnw in zip(self.weights, delta_nabla_w)]
		self.biases = [b-((self.learning_rate/len(data))*dnb) for b, dnb in zip(self.biases, delta_nabla_b)]


	def get_cost_gradient(self, x, y):


		weighted_inputs = []
		activation = x
		activations = [x]
		errors = []
		nabla_w, nabla_b = [], []

		for weight, bias in zip(self.weights, self.biases):

			weighted_inputs.append(np.dot(weight,activation) + bias)
			activation = self.activation_vectorized(weighted_inputs[-1])
			activations.append(activation)


		for l in range(1, len(self.layers)):
			if l is 1:
				error = (activations[-l] - y) * self.activation_prime_vectorized(weighted_inputs[-l])
			else:
				error = self.weights[1-l].T.dot(nabla_b[0]) * self.activation_prime_vectorized(weighted_inputs[-l])

			nabla_w.insert(0, error.dot(activations[-l-1].T))
			nabla_b.insert(0, error)

		return nabla_w, nabla_b



	def save_parameters(self, preset_name):
		net_parameters = {"layers": self.layers, "weights": [w.tolist() for w in self.weights], "biases": [b.tolist() for b in self.biases]}
		j = open("presets/" + preset_name + ".json", "w")
		json.dump(net_parameters, j)
		j.close()


	def load_parameters(self, preset_name):
		raw_net_parameters = open("presets/" + preset_name + ".json")
		net_parameters = json.load(raw_net_parameters)
		raw_net_parameters.close()
		return net_parameters



'''
NN = NeuralNetwork([784, 50, 10])

handwritten_digits = MNIST("data/mnist")
training_data = handwritten_digits.load_training()
np.random.shuffle(training_data)
NN.stochastic_gradient_descent(training_data[:57000], training_data[57000:], learning_rate=.5)
NN.save_parameters("my_cool_preset")

'''





