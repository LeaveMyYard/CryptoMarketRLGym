import os
import numpy as np
import sys
import threading
import sqlite3
from params import get_params

if __name__ == "__main__":

    datasize = 10000
    batchsize = 10000

    gpus = [0, 1, 2, 3]
    threads = []

    conn = sqlite3.connect("research.db")
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS BatchesData")
    cursor.execute(
        "CREATE TABLE BatchesData (batchID INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE, start INTEGER NOT NULL, end INTEGER NOT NULL, finished INTEGER DEFAULT 0)"
    )

    paramssize = len(get_params())

    for i in range(0, paramssize - batchsize + 1, batchsize):
        cursor.execute(
            "INSERT INTO BatchesData(start, end) VALUES(?, ?)",
            (i, min(i + batchsize, paramssize)),
        )

    cursor.execute("DROP TABLE IF EXISTS SimulationResults")
    cursor.execute(
        "CREATE TABLE SimulationResults ( \
            simulationID INTEGER PRIMARY KEY, \
            result FLOAT NOT NULL, \
            order_pairs_param INTEGER NOT NULL,\
            order_start_size_param INTEGER NOT NULL,\
            order_step_size_param INTEGER NOT NULL,\
            interval_param INTEGER NOT NULL,\
            min_spread_param INTEGER NOT NULL,\
            min_position_param INTEGER NOT NULL,\
            max_position_param INTEGER NOT NULL \
        )"
    )

    conn.commit()

    for gpu in gpus:
        comm = (
            sys.executable
            + " "
            + os.path.join(os.getcwd(), "gym_trading", "envs", "trading_sim_cuda.py")
            + f' --gpu {gpu} --data "hist_data/bitmex_1m.csv" --datasize {datasize} --batchsize {batchsize}'
        )

        thread = threading.Thread(target=os.system, args=[comm])
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
