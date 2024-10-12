# Mason Weiss
# ELEC 576 - Introduction to Deep Learning
# Prof. Ankit Patel, Rice University
# Due: October 10, 2024
# adapted from code snippet three_layer_neural_network by tan_nguyen

import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


def generate_data(data_name=None):
    """
    generate data
    :param data_name: the name of the dataset to render
    :return: X: input data, y: given labels
    """
    np.random.seed(0)
    if data_name == 'make_circles':
        X, y = datasets.make_circles(n_samples=200, noise=0.03)
    else:
        X, y = datasets.make_moons(n_samples=200, noise=0.20)

    return X, y


def plot_decision_boundary(pred_func, X, y, figname):
    """
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :param figname: name of figure to save
    :return:
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.savefig(figname)
    plt.show()


########################################################################################################################
########################################################################################################################
# YOUR ASSSIGMENT STARTS HERE
# FOLLOW THE INSTRUCTION BELOW TO BUILD AND TRAIN A 3-LAYER NEURAL NETWORK
########################################################################################################################
########################################################################################################################
class NeuralNetwork(object):
    """
    This class builds and trains a neural network
    """

    def __init__(self, nn_input_dim, nn_hidden_dim, nn_output_dim, actFun_type='Tanh', reg_lambda=0.01, seed=0):
        """
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        """
        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        # initialize the weights and biases in the network
        np.random.seed(seed)

        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, self.nn_hidden_dim))

        self.W2 = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_hidden_dim)
        self.b2 = np.zeros((1, self.nn_output_dim))


    def actFun(self, z, type):
        """
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        """
        if type == 'Tanh':
            a = (np.exp(2*z)-1)/(np.exp(2*z)+1)
        elif type == 'Sigmoid':
            a = 1/(1+np.exp(-1*z))
        elif type == 'ReLU':
            a = np.where(z < 0, 0, z)
        else:
            assert False, "Type must be {'Tanh', 'Sigmoid', 'ReLU'}"

        return a

    def diff_actFun(self, z, type):
        """
        diff_actFun computes the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        """
        if type == 'Tanh':
            da = 4/(np.exp(z)+np.exp(-z))**2
        elif type == 'Sigmoid':
            da = (np.exp(-1*z))/(1+np.exp(-1*z))**2
        elif type == 'ReLU':
            da = np.where(z > 0, 1, 0)
        else:
            assert False, "Type must be {'Tanh', 'Sigmoid', 'ReLU'}"

        return da

    def feedforward(self, X, actFun):
        """
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        """
        # requires multiple transposes to maintain designated order of Wx + b with the dimensions provided
        self.z1 = np.dot(self.W1.T, X.T).T + self.b1
        self.a1 = actFun(self.z1)
        self.z2 = np.dot(self.W2.T, self.a1.T).T + self.b2
        exp_scores = np.exp(self.z2)
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

        # Add regularization term to loss
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
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
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        """

        # IMPLEMENT YOUR BACKPROP HERE

        num_examples = len(X)
        delta3 = self.probs.copy()
        delta3[range(num_examples), y] -= 1

        dW2 = np.matmul(delta3.T, self.a1).T  # dW2 = dL/dW2
        db2 = np.sum(delta3, axis=0)  # db2 = dL/db2

        dLdz1 = np.multiply(np.matmul(delta3, self.W2.T), self.diff_actFun(self.z1, self.actFun_type)).T
        dW1 = np.matmul(dLdz1, X).T  # dW1 = dL/dW1
        db1 = np.sum(dLdz1, axis=1)  # db1 = dL/db1

        return dW1, dW2, db1, db2

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
            dW1, dW2, db1, db2 = self.backprop(X, y)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.reg_lambda * self.W2
            dW1 += self.reg_lambda * self.W1

            # Gradient descent parameter update
            self.W1 += -epsilon * dW1
            self.b1 += -epsilon * db1
            self.W2 += -epsilon * dW2
            self.b2 += -epsilon * db2

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


def main():
    # 1(a): generate and visualize Make-Moons dataset
    X, y = generate_data()
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.savefig('1a_visualize_data.png')
    plt.show()

    # 1(e)(i): compare different activations
    fig1e = ['Tanh', 'Sigmoid', 'ReLU']
    losses_1e = []

    for act in fig1e:
        model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=3, nn_output_dim=2, actFun_type=act)
        loss = model.fit_model(X,y)
        losses_1e.append(loss)
        model.visualize_decision_boundary(X, y, f'1e1_{act}_3_nodes.png')

    plt.figure()
    for idx in range(len(fig1e)):
        plt.plot(np.arange(0, 20000, 1000), losses_1e[idx], label=fig1e[idx])
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('Loss for Various Activation Functions - 1e(i))')
    plt.legend()
    plt.grid(True)
    plt.savefig('1e1_loss_diff_activations.png')
    plt.show()

    # 1(e)(ii): compare different numbers of nodes in hidden layer for tanh
    fig1f = [5, 8, 20]
    losses_1f = []

    for h_nodes in fig1f:
        model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=h_nodes, nn_output_dim=2, actFun_type='Tanh', seed=20)
        loss = model.fit_model(X, y)
        losses_1f.append(loss)
        model.visualize_decision_boundary(X, y, f'1e2_Tanh_{h_nodes}_nodes.png')

    plt.figure()
    for idx in range(len(fig1e)):
        plt.plot(np.arange(0, 20000, 1000), losses_1f[idx], label=f'number of nodes = {fig1f[idx]}')
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('Loss with Tanh for varying # of hidden layer nodes - 1e(ii)')
    plt.legend()
    plt.grid(True)
    plt.savefig('1e2_loss_hidden_size.png')
    plt.show()

if __name__ == "__main__":
    main()