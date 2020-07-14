import os
import sys
import threading


if __name__ == "__main__":
    gpus = [0, 1, 2]
    threads = []

    for gpu in gpus:
        comm = (
            sys.executable
            + " "
            + os.path.join(os.getcwd(), "gym_trading", "envs", "trading_sim_cuda.py")
            + f' --gpu {gpu} --gpucount {len(gpus)} --data "hist_data/bitmex_1m.csv" --datasize 50000'
        )

        thread = threading.Thread(target=os.system, args=[comm])
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
