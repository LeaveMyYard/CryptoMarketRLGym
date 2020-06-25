import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

with open("data.pickle", "rb") as f:
    ord_data = pickle.load(f)

print(len(ord_data))

exit()

ord_data = ord_data[:300]

price_filter_range = [8000, 11000]

Z = []
ZP = []

val = ord_data[0]

total_min_price = np.inf
total_max_price = -np.inf

for val in ord_data:
    min_price = np.floor(min(val.keys()))
    max_price = np.ceil(max(val.keys()))

    total_max_price = max(max_price, total_max_price)
    total_min_price = min(min_price, total_min_price)

    prices = {int(v): 0 for v in np.arange(min_price, max_price + 1)}

    for price, volume in val.items():
        prices[round(price)] += volume

    Z.append(prices)

for val in Z:
    initial_dict = {
        i: 0 for i in np.arange(price_filter_range[0], price_filter_range[1] + 1)
    }
    for price, volume in val.items():
        if price >= price_filter_range[0] and price <= price_filter_range[1]:
            initial_dict[price] = volume

    ZP.append(list(initial_dict.values()))

ZP = np.array(ZP).T

X, Y = np.mgrid[
    price_filter_range[0] : price_filter_range[1] + 1 : 1, 0 : len(ord_data) : 1
]

# Z = np.array(Z).T

# print(Z, Z.shape)

# Z = Z / np.max(np.abs(Z).flatten())

from mayavi import mlab

s = mlab.barchart(X, Y, ZP)
mlab.show()

# ax.plot_surface(X, Y, ZP, cmap=plt.cm.Spectral)


# plt.show()

