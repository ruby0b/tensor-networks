import random

import matplotlib.pyplot as plt
from scipy.io import loadmat


x7 = loadmat('./examples/mnist/mnist-7x7.mat')['trainX']
x7.shape = (x7.shape[0], 7, 7)
x14 = loadmat('./examples/mnist/mnist-14x14.mat')['trainX']
x14.shape = (x14.shape[0], 14, 14)
x28 = loadmat('./examples/mnist/mnist-28x28.mat')['trainX']
x28.shape = (x28.shape[0], 28, 28)

labels = loadmat('./examples/mnist/mnist-7x7.mat')['trainY'][0]

i = random.randrange(len(x28))

print(labels[i])
fig, axs = plt.subplots(3)
axs[0].imshow(x28[i])
axs[1].imshow(x14[i])
axs[2].imshow(x7[i])

plt.show()
