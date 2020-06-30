import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from mayavi import mlab
from mpl_toolkits.mplot3d import Axes3D


ZP = np.load("orderbook.npy")

X, Y = np.mgrid[0 : ZP.shape[0] : 1, 0 : ZP.shape[1] : 1]

# Z = np.array(Z).T

# print(Z, Z.shape)

# Z = Z / np.max(np.abs(Z).flatten())

print(X.shape, Y.shape, ZP.shape)


s = mlab.barchart(X, Y, ZP)
mlab.show()

# ax.plot_surface(X, Y, ZP, cmap=plt.cm.Spectral)


# plt.show()
