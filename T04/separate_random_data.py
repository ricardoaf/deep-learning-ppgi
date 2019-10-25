import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

# lock random seed
np.random.seed(1)

# generate random data
ns = 50
xc = np.random.normal(loc=0, size=ns)
yc = np.random.normal(loc=0, size=ns)
xp = np.random.normal(loc=3, size=ns)
yp = np.random.normal(loc=4, size=ns)

# plot random data
plt.plot(xc, yc, 'ro', xp, yp, 'b+')
plt.xlabel('x1'), plt.ylabel('x2'), plt.grid(True)

# classify data
data = np.vstack([np.hstack([xc, xp]), np.hstack([yc, yp])])
category = np.hstack([len(xc)*[1], len(xp)*[0]])

# apply perceptron training
pcpt = Perceptron(2, learning_rate=0.5)
pcpt.train(data, category)
w, b = pcpt.weight[1:], pcpt.weight[0]

# plot separation line
left, right = plt.xlim()
plt.plot([left, right], [(-w[0]*left-b)/w[1], (-w[0]*right-b)/w[1]], 'k-')
plt.show()
