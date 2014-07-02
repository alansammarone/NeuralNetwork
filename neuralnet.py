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

		self.weights = [np.random.randn(y, x) for x, y in zip(self.layers[:-1], self.layers[1:])]


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

			return " %s / %s" % (count, len(validation_data))




	def stochastic_gradient_descent(self, training_data, test_data=None):


		self.batch_size = 10
		self.learning_rate = 3.0
		self.epochs = 5




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












cfg = [784, 196, 10]
#cfg = [2, 1]


net = NeuralNetwork(cfg)

mnist = MNIST("data/mnist/")
mnist_data = mnist.load_training()

training_data = mnist_data[0:50000]
validation_data = mnist_data[50000:60000]



#net.weights = np.array([_w])
#net.biases = b


#x = [1, 0]

#print net.feed_forward(np.array([x]).T)


#training_data = []
'''
for i in range(100):
	x = [random.choice([0, 1]) for i in range(cfg[0])]
	#y = [random.random() for i in range(cfg[-1])]
	y = _w.dot(x) + b
	y = 1.0/(1 + math.exp(-y))


	x = np.array([x]).T
	y = np.array([y]).T
	
	training_data.append((x, y))

'''
#validation_data = training_data[30:60]


net.stochastic_gradient_descent(training_data, validation_data)



#print net.feed_forward(np.array([[0,0]]).T)
#print net.feed_forward(np.array([[1,0]]).T)
#print net.feed_forward(np.array([[0,1]]).T)
#print net.feed_forward(np.array([[1,1]]).T)


random.shuffle(mnist_data)



corrects = []
wrongs = []
for x, y in mnist_data[0:10000]:
	r = net.feed_forward(x)
	if np.argmax(r) == np.argmax(y):
		corrects.append((x, y))
	else:
		wrongs.append((x, y))




i = 34
print np.argmax(net.feed_forward(wrongs[i][0]))
mnist.show_image(wrongs[i][0])





net.save_parameters("mnist_recognizer2")































