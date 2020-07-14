import numpy as np
from mayavi import mlab


ZP = np.load("orderbook_shorter.npy")
X, Y = np.mgrid[0 : ZP.shape[0] : 1, 0 : ZP.shape[1] : 1]
print(X.shape, Y.shape, ZP.shape)

s = mlab.barchart(X, Y, ZP)
mlab.show()
