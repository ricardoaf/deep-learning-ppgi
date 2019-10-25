import numpy as np
import matplotlib.pyplot as plt


class Data:  # define Data class
    def __init__(self, name, mean, mark):
        self.name = name,
        self.mean = mean
        self.mark = mark

    def generate_random_sample(self, n_samples):
        xs = np.random.normal(self.mean[0], 1, n_samples)
        ys = np.random.normal(self.mean[1], 1, n_samples)
        return xs, ys


c = Data('o', [0, 0], 'ro')  # circle instance
p = Data('+', [3, 4], 'b+')  # plus instance

# lock random seed
np.random.seed(1)

# generate samples
xc, yc = c.generate_random_sample(50)
xp, yp = p.generate_random_sample(50)

# plot samples
plt.plot(xc, yc, c.mark, label=c.name)
plt.plot(xp, yp, p.mark, label=p.name)

# set options and show plot
plt.xlabel('x1'), plt.ylabel('x2'), plt.grid(True)
plt.legend(loc='lower right')
plt.show()
