# import the necessary packages
import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # initialize the list of weights matrices, then store the network architecture and learning rate
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # start looping from the index of the first layer but stop before we reach the last two layers
        for i in np.arange(0,len(layers) -2):
            # randomly initialize a weight matrix connecting the number of nodes in each respective layer together, adding an extra node for the bias
            w = np.random.randn(layers[i] +1, layers[i +1] +1)
            self.W.append(w / np.sqrt(layers[i]))

        # the last two layers are a special case where the input connections need a bias term but the output does not
        w = np.random.randn(layers[-2] +1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # construct and return a string that represents the network architecture
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        # compute and return the sigmoid activation value for a given input value
        return 1.0/ (1+ np.exp(-x))

    def sigmoid_deriv(self, x):
        # compute the derivative of the sigmoid function ASSUMING that ‘x‘ has already been passed through the ‘sigmoid‘ function
        return x * (1- x)

    def fit(self, X, y, epochs=1000, displayUpdate=100):
        # insert a column of 1’s as the last entry in the feature matrix -- this little trick allows us to treat the bias as a trainable parameter within the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]
        
        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            # loop over each individual data point and train our network on it
            for(x, target)in zip(X, y):
                self.fit_partial(x, target)
                
                # check to see if we should display a training update
                if epoch ==0 or(epoch +1) % displayUpdate ==0:
                    loss =self.calculate_loss(X, y)
                    print("[INFO] epoch={}, loss={:.7f}".format(epoch +1, loss))

    def fit_partial(self, x, y):
        # construct our list of output activations for each layer as our data point flows through the network; the first activation is a special case -- it’s just the input feature vector itself
        A = [np.atleast_2d(x)]

        # FEEDFORWARD:
        # loop over the layers in the network
        for layer in np.arange(0,len(self.W)):
            # feedforward the activation at the current layer by taking the dot product between the activation and the weight matrix -- this is called the "net input" to the current layer
            net = A[layer].dot(self.W[layer])
            
            # computing the "net output" is simply applying our nonlinear activation function to the net input
            out =self.sigmoid(net)
            
            # once we have the net output, add it to our list of activations
            A.append(out)

        # BACKPROPAGATION
        # the first phase of backpropagation is to compute the difference between our *prediction* (the final output activation in the activations list) and the true target value