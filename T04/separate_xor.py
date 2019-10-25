import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

# generate data and labels
data = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
y_OR, y_NAND, y_XOR = np.array([0, 1, 1, 1]), np.array([1, 1, 1, 0]), np.array([0, 1, 1, 0])

# plot data
x0, y0 = data[0, y_XOR == 0], data[1, y_XOR == 0]
x1, y1 = data[0, y_XOR == 1], data[1, y_XOR == 1]
plt.plot(x0, y0, 'ro', x1, y1, 'b+')
plt.xlabel('x1'), plt.ylabel('x2'), plt.grid(True)
left, right = plt.xlim()

# train perceptron (OR)
p_OR = Perceptron(2, learning_rate=0.4, weight=np.array([0., 1., 1.]))
p_OR.train(data, y_OR)
w, b = p_OR.weight[1:], p_OR.weight[0]

# plot 1st line (OR)
plt.plot([left, right], [(-w[0]*left-b)/w[1], (-w[0]*right-b)/w[1]], 'k-')

# train perceptron (NAND)
p_NAND = Perceptron(2, learning_rate=0.4, weight=np.array([0., 0.1, 0.1]))
p_NAND.train(data, y_NAND)
w, b = p_NAND.weight[1:], p_NAND.weight[0]

# plot 2nd line (NAND)
plt.plot([left, right], [(-w[0]*left-b)/w[1], (-w[0]*right-b)/w[1]], 'k-')
plt.title('XOR logical function')
plt.show()

# display predictions
for i in range(data.shape[1]):
    x = data[:, i]
    print(p_OR.predict(x) & p_NAND.predict(x))
