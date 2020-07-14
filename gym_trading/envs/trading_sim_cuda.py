import pycuda.driver as drv
import pycuda.tools

# import pycuda.autoinit
import numpy as np
import pandas as pd
import itertools
import argparse
import typing
import threading
import time
import sqlite3
from pycuda.compiler import SourceModule


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
    def __init__(self, gpuid: int, code: str, *args, **kwargs):
        kwargs["target"] = self.target_func
        super().__init__(gpuid, *args, **kwargs)

        self.mod = SourceModule(open(f"cuda/{code}").read())
        self.simulate = self.mod.get_function("start")

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
            block=block,
            grid=grid,
        )


def _arange(start: float, stop: float, step: float) -> np.ndarray:
    return np.arange(start=start, stop=stop + step, step=step)


def main():
    order_pairs_param = _arange(1, 30, 1)
    order_start_size_param = _arange(25, 150, 25)
    order_step_size_param = _arange(0, 300, 25)
    interval_param = _arange(0.0025, 0.02, 0.0025) * 1000000
    min_spread_param = _arange(0.0025, 0.02, 0.0025) * 1000000
    min_position_param = np.array([-1000])  # _arange(-4500, 0, 500)
    max_position_param = np.array([1000])  # _arange(4500, 0, -500)

    inp_total = np.array(
        list(
            itertools.product(
                order_pairs_param,
                order_start_size_param,
                order_step_size_param,
                interval_param,
                min_spread_param,
                min_position_param,
                max_position_param,
            )
        ),
        dtype=np.int64,
    )

    parser = argparse.ArgumentParser(description="GPU trading simulation tester")

    parser.add_argument("--gpu", action="store", dest="gpuid", type=int)
    parser.add_argument("--gpucount", action="store", dest="gpucount", type=int)
    parser.add_argument("--data", action="store", dest="data_location", type=str)
    parser.add_argument("--datasize", action="store", dest="data_size", type=int)

    args = parser.parse_args()

    print("Loading data")
    data = pd.read_csv(args.data_location).values[-args.data_size :, 2:]
    print("Loaded", len(data))

    outside_global_data = (data * 10000).astype(np.int64)
    data_size = np.array([data.shape[0]]).reshape((1,))

    def simulate_gpu(inp, gpuid):
        result = np.ones((inp.shape[0]), dtype=np.int32)
        thread = gpuCalculationThreaded(
            gpuid=gpuid,
            code="gpu_grid_simulation.cu",
            args=[
                [inp, outside_global_data, data_size],
                [result],
                (1, 1, 1),
                (1, inp.shape[0]),
            ],
        )
        thread.start()
        thread.join()

        return result.reshape(inp.shape[0]) / 10000000

    # conn = sqlite3.connect("research.db")

    start = args.gpuid * inp_total.shape[0] // args.gpucount
    end = (args.gpuid + 1) * inp_total.shape[0] // args.gpucount

    print(f"Running a GPU[{args.gpuid}] simulation...")
    b = []

    for sr in range(start, end, 10000):
        t = time.time()

        b += list(simulate_gpu(inp_total[sr : min(sr + 10000, end)], args.gpuid))

        print(
            f"GPU[{args.gpuid}]",
            "It took:",
            round(time.time() - t, 2),
            "s",
            f"| {len(b)}/{(end - start)} [{round(100 * len(b) / (end - start), 2)}/100.00%]",
        )
    best = max(b)
    print(
        f"GPU[{args.gpuid}] The best iteration got {best}, with the settings: {inp_total[b.index(best)]}; average result: {round(sum(b) / len(b), 6)}"
    )
    print(np.array(b))


if __name__ == "__main__":
    main()
