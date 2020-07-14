import numpy as np
import itertools


def _arange(start: float, stop: float, step: float) -> np.ndarray:
    return np.arange(start=start, stop=stop + step, step=step)


def get_params():
    order_pairs_param = _arange(1, 30, 1)
    order_start_size_param = _arange(25, 150, 25)
    order_step_size_param = _arange(0, 300, 25)
    interval_param = _arange(0.0025, 0.02, 0.0025) * 1000000
    min_spread_param = _arange(0.0025, 0.02, 0.0025) * 1000000
    min_position_param = _arange(-4000, 0, 1000)
    max_position_param = _arange(4000, 0, -1000)

    return np.array(
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
