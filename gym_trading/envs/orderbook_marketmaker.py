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

    It is made to be used on BTC/USDT data. You can find an example of data in orderbook.npy

    Args:
        orderbook (str): file, from where to read the data.
        orderbook_depth (int, optional): How much price ticks in asks and bids to process. Defaults to 50.
        orderbook_hist (int, optional): How much historical data to process. Defaults to 500.
        starting_price (int, optional): price of the top of an orderbook. Defaults to 9000.
        starting_balance (int, optional): balance in quote asset (BTC). Defaults to 10.
        maker_fee (float, optional): fee of creating an order. Defaults to -0.00025.

    Note:
        You need to pre-load orderbook data from an exchange and then to pre-process it (read the documentation for that).
    """

    metadata = {"render.modes": ["human", "rgb_array", "grayscale_array"]}

    class Order:
        def __init__(self, price: float, volume: float):
            self.price = price
            self.volume = volume

        def is_filled(self, orderbook_spread: typing.Tuple[int, int]) -> bool:
            return (
                self.volume < 0
                and self.price <= orderbook_spread[0]
                or self.volume > 0
                and self.price >= orderbook_spread[1]
            )

    def __init__(
        self,
        orderbook: str,
        orderbook_depth: int = 40,
        orderbook_hist: int = 500,
        starting_price: int = 9000,
        starting_balance: float = 10,
        maker_fee: float = -0.00025,
    ):
        """
        __init__ Constructor

        Create a gym.Environment based on the orderbook, loaded from .npy file (pre-processed orderbook).

        Args:
            orderbook (str): file, from where to read the data.
            orderbook_depth (int, optional): How much price ticks in asks and bids to process. Defaults to 50.
            orderbook_hist (int, optional): How much historical data to process. Defaults to 500.
            starting_price (int, optional): price of the top of an orderbook. Defaults to 9000.
            starting_balance (int, optional): balance in quote asset (BTC). Defaults to 10.
            maker_fee (float, optional): fee of creating an order. Defaults to -0.00025.
        """
        super().__init__()
        self.orderbook_depth = orderbook_depth
        self.orderbook_hist = orderbook_hist
        self.starting_price = starting_price
        self.starting_balance = starting_balance
        self.maker_fee = maker_fee
        self.data = np.load(orderbook)
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.orderbook_depth * 2, self.orderbook_hist, 1),
            dtype=np.float64,
        )

    def step(self, action: int) -> typing.Tuple[object, float, bool, dict]:
        """
        step - Perform a step on an environment.

        Standart Gym function to make an action on environment.

        Args:
            action (int): An action, that is in spaces.Discrete(6).
                0 - cancel all orders.
                1 - add 1 buy and 1 sell order in both parts of an orderbook in distance of 1 point from spread
                2 - add 1 buy and 1 sell order in both parts of an orderbook in distance of 2 points from spread
                3 - add 1 buy and 1 sell order in both parts of an orderbook in distance of 3 points from spread
                4 - add 1 buy and 1 sell order in both parts of an orderbook in distance of 4 points from spread
                5 - do nothing.

        Returns:
            typing.Tuple[object, float, bool, dict]:
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
        self._calculate_current_data()

        self._make_action(action)

        last_balance = self._balance

        for order in self._current_orders:
            if order.is_filled(
                (
                    self._current_price[0] + self.starting_price,
                    self._current_price[1] + self.starting_price,
                )
            ):
                self._calculate_order_filled(order)
                self._current_orders.remove(order)

        price = (
            self._current_price[0] + self._current_price[1]
        ) / 2 + self.starting_price
        unr_profit = (
            self._current_position["volume"]
            * (1 / self._current_position["average_price"] - 1 / price)
            if self._current_position["volume"] != 0
            else 0
        )

        if self._current_position["volume"] != 0:
            bankruptcy_price = (
                1
                - 1
                / (
                    1
                    + (
                        self._current_position["volume"]
                        / (self._balance * self._current_position["average_price"])
                    )
                )
            ) * self._current_position["average_price"]

            if (
                self._current_position["volume"] >= 0
                and price <= bankruptcy_price
                and bankruptcy_price > 0
                or self._current_position["volume"] <= 0
                and price >= bankruptcy_price
                and bankruptcy_price > 0
            ):
                return (
                    self._current_data,
                    0,
                    True,
                    {
                        "balance": self._balance,
                        "unr_profit": unr_profit,
                        "orders_created": self.orders_created,
                        "orders_filled": self.orders_filled,
                        "orders_canceled": self.orders_canceled,
                        "position": self._current_position,
                        "info": "Liquidated",
                        "price": price,
                        "bankruptcy_price": bankruptcy_price,
                    },
                )

        return (
            self._current_data,
            self._balance - last_balance,
            self._current_timestep == self.data.shape[1],
            {
                "balance": self._balance,
                "unr_profit": unr_profit,
                "orders_created": self.orders_created,
                "orders_filled": self.orders_filled,
                "orders_canceled": self.orders_canceled,
                "position": self._current_position,
            },
        )

    def _calculate_current_data(self):
        last_snapshot = self.data[:, self._current_timestep - 1]

        # Stepping on left of the orderbook.
        # Finding the first positive order.
        for i in range(self._current_price[0], -1, -1):
            if last_snapshot[i] > 0:
                self._current_price[0] = i
                break

        last_p = self._current_price[0]

        # Then stepping on right.
        # Finding the first positive order next to the negative one
        for i in range(self._current_price[0], len(last_snapshot)):
            if last_snapshot[i] > 0:
                last_p = i
            elif last_snapshot[i] < 0:
                self._current_price[0] = last_p
                break

        # Stepping on right of the orderbook.
        # Finding the first negative order.
        for i in range(self._current_price[1], len(last_snapshot)):
            if last_snapshot[i] < 0:
                self._current_price[1] = i
                break

        last_p = self._current_price[1]

        # Then stepping on left.
        # Finding the first negative order next to the positive one
        for i in range(self._current_price[1], -1, -1):
            if last_snapshot[i] < 0:
                last_p = i
            elif last_snapshot[i] > 0:
                self._current_price[1] = last_p
                break

        center_price = (self._current_price[0] + self._current_price[1]) // 2

        self._current_data = self.data[
            center_price - self.orderbook_depth : center_price + self.orderbook_depth,
            self._current_timestep - self.orderbook_hist : self._current_timestep,
        ]

        if self._current_data.shape[0] != self.orderbook_depth * 2:
            raise ValueError(
                f"Not enough orderbook for depth {self.orderbook_depth}. "
                + f"Current spread prices: {self._current_price}. "
                + f"Total orderbook data size: {self.data.shape[0]}. "
                + "Make the depth lower or load a wider orderbook."
            )

    def _make_action(self, action: int):
        if action == 0:
            self.orders_canceled += len(self._current_orders)
            self._current_orders = []
        elif action != 5:
            volume = (
                self._balance
                * (
                    self._current_price[0]
                    + self._current_price[1]
                    + 2 * self.starting_price
                )
                / 2
                / 1000
            )
            self._current_orders.append(
                OrderbookMarketmaker.Order(
                    self._current_price[0] - action + self.starting_price, volume
                )
            )
            self._current_orders.append(
                OrderbookMarketmaker.Order(
                    self._current_price[1] + action + self.starting_price, -volume
                )
            )
            self.orders_created += 2

    def _calculate_order_filled(self, order: "OrderbookMarketmaker.Order"):
        if self._current_position["volume"] == 0:
            self._current_position["volume"] = order.volume
            self._current_position["average_price"] = order.price
        elif np.sign(self._current_position["volume"]) == np.sign(order.volume):
            self._current_position["average_price"] = (
                order.price * order.volume
                + self._current_position["average_price"]
                * self._current_position["volume"]
            )
            self._current_position["volume"] = (
                order.volume + self._current_position["volume"]
            )
            self._current_position["average_price"] /= self._current_position["volume"]
        elif self._current_position["volume"] >= order.volume:
            self._current_position["volume"] += order.volume
            self._balance += order.volume * (
                -1 / self._current_position["average_price"] + 1 / order.price
            )
        else:
            self._balance += self._current_position["volume"] * (
                1 / self._current_position["average_price"] - 1 / order.price
            )
            self._current_position["volume"] -= order.volume
            self._current_position["average_price"] = order.price

        self._balance -= self.maker_fee * abs(order.volume)
        self.orders_filled += 1

    def reset(self):
        """
        reset

        Reset the state of the environment to an initial state.
        """
        self._current_timestep = self.orderbook_hist
        self._current_price = [self.data.shape[0] // 2, self.data.shape[0] // 2]
        self._calculate_current_data()
        self._current_orders: typing.List[self.Order] = []
        self._balance = self.starting_balance  # Balance is in BTC
        self._current_position: typing.Dict[str, float] = {
            "volume": 0,
            "average_price": 0,
        }
        self.orders_created = 0
        self.orders_filled = 0
        self.orders_canceled = 0

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
            cm = plt.get_cmap("seismic")
            img = cm((self._current_data / 2 + 0.5))[:, :, :3]
            return img
        elif mode == "grayscale_array":
            return self._current_data

    def close(self):
        ...


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import datetime

    env = OrderbookMarketmaker("orderbook_shorter.npy")
    env.reset()
    i = 0
    t = datetime.datetime.now()

    while True:
        obs, reward, done, info = env.step(env.action_space.sample())
        env.render()
        i += 1

        if (datetime.datetime.now() - t).total_seconds() >= 1:
            # print("FPS:", i)
            i = 0
            t = datetime.datetime.now()
        # print(env.render(mode="rgb_array"), end=" ")
        # plt.imshow(env.render(mode="rgb_array"))
        # plt.show()
        if done:
            print(info)
            print("Episode finished, reset.")
            env.reset()
        else:
            print(info, end="                      \r")

