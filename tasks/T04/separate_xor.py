import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

# generate input data
###############################################################################

# generate data
x1 = np.array([0, 1, 0, 1])
x2 = np.array([0, 0, 1, 1])
data = np.vstack([x1, x2])

# generate labels
y_AND = np.array([0, 0, 0, 1])
y_NOT_AND = np.array([1, 1, 1, 0])
y_OR = np.array([0, 1, 1, 1])
y_XOR = y_OR & y_NOT_AND

# train perceptrons
###############################################################################

# train perceptron (AND)
p_AND = Perceptron(2)
p_AND.train(data, y_AND)

# train perceptron (OR)
p_OR = Perceptron(2)
p_OR.train(data, y_OR)

# train perceptron (NOT_AND)
p_NOT_AND = Perceptron(2)
p_NOT_AND.train(data, y_NOT_AND)

# XOR logical function
###############################################################################

def p_xor_predict(data):  # ANN behavior: XOR prediction
    a_OR = p_OR.predict(data)
    a_NOT_AND = p_NOT_AND.predict(data)
    a = np.vstack([a_OR, a_NOT_AND])
    return p_AND.predict(a)

# display results
y_XOR_predicted = p_xor_predict(data)
print('XOR labels:     ', y_XOR)
print('XOR prediction: ', y_XOR_predicted)

# plot data
###############################################################################

# plot XOR data
plt.plot(x1[y_XOR == 0], x2[y_XOR == 0], 'ro')
plt.plot(x1[y_XOR == 1], x2[y_XOR == 1], 'b+')
plt.xlabel('x1'), plt.ylabel('x2'), plt.grid(True), plt.title('XOR logical function')

def plot_separation_line(w, b, style):  # plot separation line
    left, right = plt.xlim()
    plt.plot([left, right], [(-w[0]*left-b)/w[1], (-w[0]*right-b)/w[1]], style)

plot_separation_line(p_OR.weight, p_OR.bias, 'k-')
plot_separation_line(p_NOT_AND.weight, p_NOT_AND.bias, 'k-')
plt.show()
