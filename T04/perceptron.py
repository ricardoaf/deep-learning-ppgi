import numpy as np


class Perceptron:  # Perceptron class

    # Perceptron constructor
    def __init__(self, n_input, n_ite_max=500, learning_rate=0.01, weight=None):
        self.n_ite_max = n_ite_max  # max number of training iterations
        self.learning_rate = learning_rate  # training learning rate
        self.weight = weight if weight is not None else np.zeros(n_input+1)  # bias and weights

    # predict method
    def predict(self, x):
        z = np.dot(self.weight[1:], x) + self.weight[0]
        return int(z >= 0)  # return activation

    # training method
    def train(self, training_input, label):

        # training iterations
        for n_ite in range(self.n_ite_max):
            sum_abs_err = 0

            # display bias and weights history
            print(f'ite: {n_ite:d}, [b, w] = ', self.weight)

            # loop on samples (i)
            for i in range(len(label)):

                # define error
                err = label[i] - self.predict(training_input[:, i])
                sum_abs_err += abs(err)

                # update weights and bias
                self.weight[1:] += self.learning_rate * err * training_input[:, i]
                self.weight[0] += self.learning_rate * err

            # stop condition
            if sum_abs_err == 0:
                break
