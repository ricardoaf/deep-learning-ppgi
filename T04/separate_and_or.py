import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

# generate data
x1, x2 = np.meshgrid([0, 1], [0, 1])
x1, x2 = np.hstack(x1), np.hstack(x2)
data = np.vstack([x1, x2])

######################################################################################

# generate labels and train perceptron
y = x1 & x2  # AND
p = Perceptron(2, learning_rate=0.4, weight=np.array([0., 1., 1.]))
p.train(data, y)
w, b = p.weight[1:], p.weight[0]

# plot data
plt.plot(x1[y == 0], x2[y == 0], 'ro', x1[y == 1], x2[y == 1], 'b+')
plt.xlabel('x1'), plt.ylabel('x2'), plt.grid(True)

# plot separation line
left, right = plt.xlim()
plt.plot([left, right], [(-w[0]*left-b)/w[1], (-w[0]*right-b)/w[1]], 'k-')

######################################################################################

# generate labels and train perceptron
y = x1 | x2  # OR
p = Perceptron(2, learning_rate=0.4, weight=np.array([0., 1., 1.]))
p.train(data, y)
w, b = p.weight[1:], p.weight[0]

# plot data
plt.subplots()
plt.plot(x1[y == 0], x2[y == 0], 'ro', x1[y == 1], x2[y == 1], 'b+')
plt.xlabel('x1'), plt.ylabel('x2'), plt.grid(True)

# plot separation line
left, right = plt.xlim()
plt.plot([left, right], [(-w[0]*left-b)/w[1], (-w[0]*right-b)/w[1]], 'k-')
plt.show()
