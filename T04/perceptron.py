import numpy as np


class Perceptron:  # Perceptron class

    # Perceptron constructor
    def __init__(self, n_input, n_ite_max=500, learning_rate=0.25, weight=None, bias=None, seed=1):
        self.n_ite_max = n_ite_max  # max number of training iterations
        self.learning_rate = learning_rate  # training learning rate

        np.random.seed(seed)  # lock random seed
        self.bias = bias if bias is not None else np.random.random()  # bias
        self.weight = weight if weight is not None else np.random.random(n_input)  # weight

    # predict method
    def predict(self, x):
        z = self.weight @ x + self.bias
        return (z >= 0).astype(int)  # return activation

    # training method
    def train(self, training_input, label):

        # training iterations
        for n_ite in range(self.n_ite_max):
            sum_abs_err = 0

            # display bias and weights history
            print(f'ite: {n_ite:d}, b = {self.bias:g}, w = ', self.weight)

            # loop on samples (i)
            for i in range(len(label)):

                # define error
                err = label[i] - self.predict(training_input[:, i])
                sum_abs_err += abs(err)

                # update weights and bias
                self.weight += self.learning_rate * err * training_input[:, i]
                self.bias += self.learning_rate * err

            # stop condition
            if sum_abs_err == 0:
                break
