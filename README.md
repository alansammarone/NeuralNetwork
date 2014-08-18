# Feed-forward neural network

This is an implementantion in Python of a simple model of a neural network. It uses the sigmoid function as the activation function, stochastic gradient descent for learning, and quadratic cost function (cross-entropy should be available soon). Weights and bias are initialized from a gaussian distribution (when not loaded), and are choosen using a low standart deviation to prevent neurons from being initially saturated.

## Usage

### Instanciation 

You can initialize the network with random weights and biases, and *i* input neurons, *h* hidden layers and *o* output layers like so (any number of hidden layers is allowed):

```python
NN = NeuralNetwork([i, h, o])
```
You can also initialize the net with a set of previously saved weights and bias by passing a *preset name* to the constructor (which should be present in the *presets* folder):

```python
NN = NeuralNetwork("my_cool_preset")
```

### Learning 

Learning can be achieved calling the `stochastic_gradient_descent` method: 

```python
NN.stochastic_gradient_descent(training_data, test_data, epochs=10, batch_size=10, learning_rate=3.0)
```

*training_data* (and *test_data*, if needed) should be a list of of tuples *(x, y)*. *x* should be a numpy column vector with *i* elements, *i* being the number of  neurons in the input layers, and *y* should also be a column vector with *o* elements, where *o* is the number of neurons in the output layer.

If *test_data* is provided, at each epoch the method will print the number of itens in *test_data* that the network got correct.

### Output

One can compute the output of the network as easily as 

```python
NN.feed_forward(x)
```

*x* is the input, and should be a numpy column vector with the same number of entries and the number of input neurons in the input layer.

### Misc

It is possible to load the MNIST data using the following methods:

```python
handwritten_digits = MNIST("data/mnist")
training_data = handwritten_digits.load_training()
testing_data = handwritten_digits.load_testing()
```

You can also save the current set of weights and biases in a external file, in the *presets* folder: 

```
NN.save_parameters("my_cool_preset")
```