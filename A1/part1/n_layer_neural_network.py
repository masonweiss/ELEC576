# Mason Weiss
# ELEC 576 - Introduction to Deep Learning
# Prof. Ankit Patel, Rice University
# Due: October 10, 2024
# adapted from code snippet three_layer_neural_network by tan_nguyen

import numpy as np
import matplotlib.pyplot as plt
from three_layer_neural_network import NeuralNetwork, generate_data, plot_decision_boundary

class DeepNeuralNetwork(NeuralNetwork):
    """
    This class builds and trains an n-layer neural network
    """

    def __init__(self, nn_input_dim, nn_output_dim, actFun_type='Tanh', reg_lambda=0.01, seed=0,
                 num_hidden_layers=1, hidden_layer_sizes=None):
        """
        :param nn_input_dim: input dimension
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        :param num_hidden_layers: number of hidden layers to add (total number is this + 2)
        :param hidden_layer_sizes: a list of the number of nodes in each hidden layer, must have the same length, or if
        just length-1, then consistent hidden layer sizing throughout
        """
        # initialize the first weight and bias of the network through nn class
        super().__init__(nn_input_dim, hidden_layer_sizes[0], nn_output_dim, actFun_type, reg_lambda, seed)

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [3]  # default hidden layer size is 3 nodes for all layers

        if len(hidden_layer_sizes) == 1:
            hidden_layer_sizes = hidden_layer_sizes * num_hidden_layers  # convert to list

        layer_sizes = [nn_input_dim] + hidden_layer_sizes + [nn_output_dim]  # provide dims for first and last layer

        self.layers = []

        for layer_idx in range(num_hidden_layers+1):
            self.layers.append(Layer(layer_sizes[layer_idx], layer_sizes[layer_idx + 1],
                                     reg_lambda=self.reg_lambda, seed=seed,
                                     actFun=lambda x: self.actFun(x, type=self.actFun_type),
                                     diff_actFun=lambda x: self.diff_actFun(x, type=self.actFun_type)))

    def feedforward(self, X, actFun):
        """
        feedforward builds an n-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        """
        self.layers[0].feedforward(X, actFun=actFun)

        for idx in range(len(self.layers)-1):  # iterate through n-1 times
            self.layers[idx+1].feedforward(self.layers[idx].a, actFun=actFun)

        exp_scores = np.exp(self.layers[-1].z)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return None

    def calculate_loss(self, X, y):
        """
        calculate_loss computes the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction and the current accuracy as a tuple
        """
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))

        data_loss = 0
        num_correct = 0
        for n in range(num_examples):
            data_loss -= np.dot([(y[n] + 1) % 2, y[n]], np.log(self.probs[n]))
            # compute accuracy
            num_correct += int(y[n] == np.argmax(self.probs[n]))

        reg_loss = 0
        for layer in self.layers:
            reg_loss += np.sum(np.square(layer.W))

        # Add regularization term to loss
        data_loss += self.reg_lambda / 2 * reg_loss
        return (1. / num_examples) * data_loss, num_correct/num_examples*100

    def predict(self, X):
        """
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        """
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        """
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW for each layer, and dL/dB for each layer
        """

        num_examples = len(X)
        delta3 = self.probs.copy()
        delta3[range(num_examples), y] -= 1

        dLdz = delta3  # derivative of loss with respect to output from penultimate layer is delta3

        dW = []  # list of derivatives of weight matrices for each layer
        db = []

        dW.append(np.matmul(dLdz.T, self.layers[-2].a).T)
        db.append(np.sum(dLdz, axis=0))

        dLda = np.matmul(dLdz, self.layers[-1].W.T)  # derivative wrt activation at layer l-1 for next backprop step

        for layer_idx in range(len(self.layers)-2, -1, -1):
            if layer_idx == 0:
                a_minus = X
            else:
                a_minus = self.layers[layer_idx-1].a

            dLda, dLdW, dLdb = self.layers[layer_idx].backprop(a_minus, dLda,
                                                               lambda x: self.diff_actFun(x, type=self.actFun_type))

            dW.insert(0, dLdW)  # insert the computed derivative at first position
            db.insert(0, dLdb)

        return dW, db

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        """
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        """
        # Gradient descent.
        loss = []
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Backpropagation
            dW, db = self.backprop(X, y)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            for layer_idx in range(len(self.layers)):
                dW[layer_idx] += self.reg_lambda * self.layers[layer_idx].W
                self.layers[layer_idx].W += -epsilon * dW[layer_idx]
                self.layers[layer_idx].b += -epsilon * db[layer_idx]

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                curr_loss, curr_accuracy = self.calculate_loss(X, y)
                loss.append(curr_loss)
                print("Loss after iteration %i: %f | accuracy: %f" % (i, curr_loss, curr_accuracy))

        return loss

    def visualize_decision_boundary(self, X, y, figname):
        """
        visualize_decision_boundary plots the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :param figname: name of figure to save
        """
        plot_decision_boundary(lambda x: self.predict(x), X, y, figname)


class Layer(object):
    def __init__(self, input_dim, output_dim, actFun, diff_actFun, reg_lambda=0.01, seed=0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.actFun = actFun
        self.diff_actFun = diff_actFun
        self.reg_lambda = reg_lambda

        np.random.seed(seed)
        self.W = np.random.randn(self.input_dim, self.output_dim) / np.sqrt(self.input_dim)
        self.b = np.zeros((1, self.output_dim))

    def feedforward(self, a_minus, actFun):
        """
        feedforward propagates the inputs through this layer
        :param a_minus: input data to layer
        :param actFun: activation function
        :return:
        """
        self.z = np.dot(self.W.T, a_minus.T).T + self.b
        self.a = actFun(self.z)

        return None

    def backprop(self, a_minus, dLda, diff_actFun):
        """
        backprop implements backpropagation to compute the gradients used to update the
        parameters in the backward step for just this layer
        :param a_minus: activation from previous layer / input to this layer
        :param dLda: derivative of loss with respect to the activation at this layer
        :return: dLda_minus, dLdW, dLdb
        """
        dLdz = np.multiply(dLda, diff_actFun(self.z))
        dLda_minus = np.matmul(dLdz, self.W.T)  # derivative wrt activation at layer l-1 for next backprop step
        dLdW = np.matmul(dLdz.T, a_minus).T
        dLdb = np.sum(dLdz, axis=0)

        return dLda_minus, dLdW, dLdb

def main():
    # MAKE MOONS
    X, y = generate_data()
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()

    activations = ['Tanh', 'Sigmoid', 'ReLU']
    architectures = [[3], [8], [5, 3], [8, 8], [8, 5, 3]]

    for act in activations:
        losses = []
        for archi in architectures:
            model = DeepNeuralNetwork(nn_input_dim=2, nn_output_dim=2, actFun_type=act, num_hidden_layers=len(archi),
                                      hidden_layer_sizes=archi, seed=42)
            loss = model.fit_model(X, y, epsilon=(0.01 if len(archi) == 1 else 0.002))
            losses.append(loss)
            model.visualize_decision_boundary(X, y, f'1f_{act}_{str(archi)}_nodes.png')

        plt.figure()
        for idx in range(len(losses)):
            plt.plot(np.arange(0, 20000, 1000), losses[idx], label=str([2] + architectures[idx] + [2]))
        plt.yscale('log')
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.title(f'Loss with {act} for Various Model Architectures - 1f)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'1f_loss_{act}.png')
        plt.show()

    # MAKE CIRCLES
    X, y = generate_data(data_name='make_circles')
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.savefig('1f2_visualize_data.png')
    plt.show()

    for act in activations:
        losses = []
        for archi in architectures:
            model = DeepNeuralNetwork(nn_input_dim=2, nn_output_dim=2, actFun_type=act, num_hidden_layers=len(archi),
                                      hidden_layer_sizes=archi, seed=42)
            loss = model.fit_model(X, y, epsilon=(0.01 if len(archi) == 1 else 0.002))
            losses.append(loss)
            model.visualize_decision_boundary(X, y, f'1f2_{act}_{str(archi)}_nodes.png')

        plt.figure()
        for idx in range(len(losses)):
            plt.plot(np.arange(0, 20000, 1000), losses[idx], label=str([2] + architectures[idx] + [2]))
        plt.yscale('log')
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.title(f'Loss with {act} for Various Model Architectures - 1f2')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'1f2_loss_{act}.png')
        plt.show()

if __name__ == "__main__":
    main()