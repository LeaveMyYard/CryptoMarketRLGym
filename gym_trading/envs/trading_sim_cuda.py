import pycuda.driver as drv
import pycuda.tools

# import pycuda.autoinit
import numpy as np
import pandas as pd
import itertools
import argparse
import sys
import typing
import threading
import time
import sqlite3
from params import get_params
from pycuda.compiler import SourceModule
from pycuda._driver import Context


class gpuThread(threading.Thread):
    def __init__(self, gpuid: int, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        drv.init()
        self.ctx = drv.Device(gpuid).make_context()
        self.device = self.ctx.get_device()

    def run(self):
        self.ctx.push()
        try:
            super().run()
        finally:
            self.ctx.pop()


class gpuCalculationThreaded(gpuThread):
    def __init__(
        self,
        gpuid: int,
        code: str,
        pre_input: typing.List[np.ndarray] = None,
        *args,
        **kwargs,
    ):
        kwargs["target"] = self.target_func
        super().__init__(gpuid, *args, **kwargs)

        self.mod = SourceModule(open(f"cuda/{code}").read())
        self.simulate = self.mod.get_function("start")
        t = time.time()
        self.pre_input = []
        print(f"GPU[{gpuid}] Copying pre-input memory to device...")
        if pre_input is not None:
            for a in pre_input:
                a_bytes = a.size * a.dtype.itemsize
                a_gpu = drv.mem_alloc(a_bytes)
                drv.memcpy_htod(a_gpu, a)
                self.pre_input.append(a_gpu)
        print(f"GPU[{gpuid}] Finished after {round(time.time() - t, 2)} seconds")
        self.args = args
        self.kwargs = kwargs

    def target_func(
        self,
        inp: typing.List[np.ndarray],
        output: typing.List[np.ndarray],
        block: typing.Tuple[int, int, int],
        grid: typing.Tuple[int, int],
    ):
        self.simulate(
            *[drv.Out(o) for o in output],
            *[drv.In(i) for i in inp],
            *self.pre_input,
            block=block,
            grid=grid,
        )


def main():
    inp_total = get_params()

    parser = argparse.ArgumentParser(description="GPU trading simulation tester")

    parser.add_argument("--gpu", action="store", dest="gpuid", type=int)
    parser.add_argument("--data", action="store", dest="data_location", type=str)
    parser.add_argument("--datasize", action="store", dest="data_size", type=int)
    parser.add_argument("--batchsize", action="store", dest="batch_size", type=int)

    args = parser.parse_args()

    print(f"GPU[{args.gpuid}] Loading data")
    data = pd.read_csv(args.data_location).values[-args.data_size :, 2:]
    print(f"GPU[{args.gpuid}] Loaded", len(data))

    outside_global_data = (data * 10000).astype(np.int64)
    data_size = np.array([data.shape[0]]).reshape((1,))

    gpu_threads = {}

    def simulate_gpu(inp, gpuid):
        nonlocal gpu_threads

        result = np.ones((inp.shape[0]), dtype=np.int32)
        if gpuid not in gpu_threads:
            thread = gpuCalculationThreaded(
                gpuid=gpuid,
                code="gpu_grid_simulation.cu",
                pre_input=[outside_global_data, data_size],
                args=[],
            )
            gpu_threads[gpuid] = thread
        else:
            thread = gpu_threads[gpuid]

        thread.target_func([inp], [result], (1, 1, 1), (1, inp.shape[0]))
        return result.reshape(inp.shape[0]) / 10000000

    conn = sqlite3.connect("research.db")
    cursor = conn.cursor()

    print(f"GPU[{args.gpuid}] Starting a simulation...")

    cursor.execute("SELECT COUNT(batchID) FROM BatchesData")

    (total,) = cursor.fetchone()

    st = time.time()

    while True:
        t = time.time()

        try:
            cursor.executescript("PRAGMA locking_mode = EXCLUSIVE; BEGIN EXCLUSIVE;")
        except sqlite3.OperationalError:
            continue

        conn.commit()

        cursor.execute(
            "SELECT batchID, start, end FROM BatchesData WHERE finished = 0 LIMIT 1"
        )
        fetch = cursor.fetchone()
        if fetch is None:
            break

        batch_id, start, end = fetch

        cursor.execute(
            "UPDATE BatchesData SET finished = 1 WHERE batchID = ?", (batch_id,)
        )
        cursor.execute("PRAGMA locking_mode = NORMAL")
        conn.commit()

        print(
            f"GPU[{args.gpuid}] Taken batch {batch_id} ({start}:{end}) [{round(100 * batch_id / total, 2)}%]"
        )

        res = simulate_gpu(inp_total[start:end], args.gpuid)

        print(
            f"GPU[{args.gpuid}] Batch {batch_id} took {round(time.time() - t, 2)} seconds"
        )

        for i, result in enumerate(res):
            cursor.execute(
                "INSERT OR IGNORE INTO SimulationResults VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    i + start,
                    result,
                    str(inp_total[i + start, 0]),
                    str(inp_total[i + start, 1]),
                    str(inp_total[i + start, 2]),
                    str(inp_total[i + start, 3] / 1000000),
                    str(inp_total[i + start, 4] / 1000000),
                    str(inp_total[i + start, 5]),
                    str(inp_total[i + start, 6]),
                ),
            )

            if i % 100 == 0:
                conn.commit()

        conn.commit()

    print(
        f"GPU[{args.gpuid}] Took a total of {round(time.time() - st, 2)} seconds to process the data"
    )


if __name__ == "__main__":
    main()

Context.pop()
