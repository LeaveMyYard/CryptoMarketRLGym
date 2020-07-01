import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from mayavi import mlab
from mpl_toolkits.mplot3d import Axes3D

orderbook_dir = r"C:\Users\blackbox1\Documents\GitHub\ArbitrageTrader\orderbook"
price_points = 50
price_bounds = (3000, 12000)

print("Gathering files...")

files = [
    f
    for f in os.listdir(orderbook_dir)[:10000]
    if os.path.isfile(os.path.join(orderbook_dir, f))
]

ord_data = []

s = len(files)

print("Loading all orderbook data...")

for i, filename in enumerate(sorted(files)):
    ord_data.append(pickle.load(open(os.path.join(orderbook_dir, filename), mode="rb")))
    print(
        f"{i + 1}/{s}  [{round(100 * i / s, 2)}%]      ",
        end=("\r" if i != s - 1 else "   ... finished \n"),
    )

p = sorted({price for shot in ord_data for price in shot.keys()})


Z = []
ZP = []

val = ord_data[0]

total_min_price = np.inf
total_max_price = -np.inf

print("Processing loaded data...")

for i, val in enumerate(ord_data):
    min_price = np.floor(min(val.keys()))
    max_price = np.ceil(max(val.keys()))

    prices_cut = sorted(val.keys())[len(val) // 16 : -len(val) // 16]

    total_max_price = max(np.ceil(max(prices_cut)), total_max_price)
    total_min_price = min(np.floor(min(prices_cut)), total_min_price)

    prices = {int(v): 0 for v in np.arange(min_price, max_price + 1)}

    for price, volume in val.items():
        prices[round(price)] += volume

    Z.append(prices)

    print(
        f"{i + 1}/{s}  [{round(100 * i / s, 2)}%]      ",
        end=("\r" if i != s - 1 else "   ... finished \n"),
    )

del ord_data

# last_price = (
#     min([k for k, v in Z[-1].items() if v < 0])
#     + max([k for k, v in Z[-1].items() if v > 0])
# ) // 2

# price_filter_range = [last_price - price_points, last_price + price_points - 1]

price_filter_range = (np.floor(total_min_price), np.ceil(total_max_price))

print(f"Cutting data from {price_filter_range[0]} to {price_filter_range[1]} ...")

for i, val in enumerate(Z):
    initial_dict = {
        i: 0 for i in np.arange(price_filter_range[0], price_filter_range[1] + 1)
    }
    for price, volume in val.items():
        if price >= price_filter_range[0] and price <= price_filter_range[1]:
            initial_dict[price] = volume

    ZP.append(list(initial_dict.values()))

    print(
        f"{i + 1}/{s}  [{round(100 * i / s, 2)}%]      ",
        end=("\r" if i != s - 1 else "   ... finished \n"),
    )

ZP = np.array(ZP, dtype=np.float64).T
ZP = ZP / np.max(np.abs(ZP))

# X, Y = np.mgrid[
#     price_filter_range[0] : price_filter_range[1] + 1 : 1, 0 : len(ord_data) : 1
# ]

# Z = np.array(Z).T

# print(Z, Z.shape)

# Z = Z / np.max(np.abs(Z).flatten())

# print(X.shape, Y.shape, ZP.shape)

print("Saving data")

np.save("orderbook_shorter.npy", ZP)

print("Finished")

print("Final data shape:", ZP.shape)


# s = mlab.barchart(X, Y, ZP)
# mlab.show()

# ax.plot_surface(X, Y, ZP, cmap=plt.cm.Spectral)


# plt.show()
