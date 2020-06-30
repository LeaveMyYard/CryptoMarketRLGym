"""
Module containing orderbook marketmaker gym environment
"""

import cv2
import gym
import typing
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt


class OrderbookMarketmaker(gym.Env):
    """
    OrderbookMarketmaker - Gym environment for marketmaking based on orderbook.

    You need to pre-load orderbook data from an exchange and then to pre-process it (read the documentation for that).
    """

    metadata = {"render.modes": ["human", "rgb_array", "grayscale_array"]}

    def __init__(
        self, orderbook: str, orderbook_depth: int = 50, orderbook_hist: int = 500
    ):
        """
        __init__ Constructor

        Create a gym.Environment based on the orderbook, loaded from .npy file (pre-processed orderbook).

        Args:
            orderbook (str): file, from where to read the data.
            orderbook_depth (int, optional): How much price ticks in asks and bids to process. Defaults to 50.
            orderbook_hist (int, optional): How much historical data to process. Defaults to 500.
        """
        super().__init__()
        self.orderbook_depth = orderbook_depth * 2
        self.orderbook_hist = orderbook_hist
        self.data = np.load(orderbook)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.orderbook_depth, self.orderbook_hist, 1),
            dtype=np.float64,
        )

    def step(self, action: int) -> typing.Tuple[object, float, bool, dict]:
        """
        step

        Perform a step on an environment.

        Args:
            action (int): An action, that is in spaces.Discrete(5).
                0 - cancel all orders.
                1 - add 1 buy and 1 sell order in both parts of an orderbook in distance of 1 point from each other
                2 - add 1 buy and 1 sell order in both parts of an orderbook in distance of 2 point from each other
                3 - add 1 buy and 1 sell order in both parts of an orderbook in distance of 3 point from each other
                4 - add 1 buy and 1 sell order in both parts of an orderbook in distance of 4 point from each other

        Returns:
            typing.Tuple[object, float, bool, dict]
                ob (object): an environment-specific object representing your observation of
                    the environment.
                reward (float): amount of reward achieved by the previous action. The scale
                    varies between environments, but the goal is always to increase
                    your total reward.
                episode_over (bool): whether it's time to reset the environment again. Most (but not
                    all) tasks are divided up into well-defined episodes, and done
                    being True indicates the episode has terminated. (For example,
                    perhaps the pole tipped too far, or you lost your last life.)
                info (dict): diagnostic information useful for debugging. It can sometimes
                    be useful for learning (for example, it might contain the raw
                    probabilities behind the environment's last state change).
                    However, official evaluations of your agent are not allowed to
                    use this for learning.
        """
        self._current_timestep += 1
        self._current_data = self.data[
            :, self._current_timestep - self.orderbook_hist : self._current_timestep
        ]

        return (
            self._current_data,
            0.0,
            self._current_timestep == self.data.shape[1],
            {"balance": 0},
        )

    def reset(self):
        """
        reset

        Reset the state of the environment to an initial state
        """
        self._current_timestep = self.orderbook_hist
        self._current_data = self.data[
            :, self._current_timestep - self.orderbook_hist : self._current_timestep
        ]

    def render(self, mode="human", close=False):
        """
        render [summary]

        [extended_summary]

        Args:
            mode (str, optional): [description]. Defaults to "human".
            close (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        if mode == "human":
            cv2.imshow("env", self.render(mode="rgb_array"))
            cv2.waitKey(1)
        elif mode == "rgb_array":
            cm = plt.get_cmap("bwr")
            img = cm(np.abs(self._current_data) * 255)[:, :, :3]
            return img
        elif mode == "grayscale_array":
            return self._current_data

    def close(self):
        ...


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = OrderbookMarketmaker("orderbook_shorter.npy")
    env.reset()
    i = 0

    while True:
        obs, reward, done, info = env.step(0)
        env.render()
        print(i)
        i += 1
        # print(env.render(mode="rgb_array"), end=" ")
        # plt.imshow(env.render(mode="rgb_array"))
        # plt.show()
        if done:
            env.reset()
