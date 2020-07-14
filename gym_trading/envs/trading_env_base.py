"""
Module containing orderbook marketmaker gym environment
"""

import cv2
import gym
import typing
import numpy as np
from gym import spaces
from collections import namedtuple
import matplotlib.pyplot as plt


class BaseTradingEnv(gym.Env):
    metadata: typing.Dict[str, list] = {"render.modes": []}

    class Order:
        def __init__(self, price: float, volume: float):
            self.price = price
            self.volume = volume

        def is_filled(
            self,
            *,
            orderbook_spread: typing.Optional[typing.Tuple[int, int]] = None,
            candle_data: typing.Optional[typing.Dict[str, int]] = None,
        ) -> bool:
            """
            is_filled checks if the order was filled

            You have to pass either orderbook_spread or candle_data to process.
            They are both kwargs.

            Args:
                candle_data (typing.Optional[typing.Dict[str, int]]): A dictionary, that has to contain "High" and "Low" value. Defaults to None.
                orderbook_spread (typing.Optional[typing.Tuple[int, int]], optional): A tuple, where the first element is an ask price, the second is a bid price. Defaults to None.

            Raises:
                ValueError: Either orderbook_spread or candle_data has to be passed.

            Returns:
                bool: True if the order was filled, False otherwise
            """
            if not orderbook_spread and not candle_data:
                raise ValueError(
                    "Either orderbook_spread or candle_data has to be passed."
                )
            if orderbook_spread is not None:
                return (
                    self.volume < 0
                    and self.price <= orderbook_spread[0]
                    or self.volume > 0
                    and self.price >= orderbook_spread[1]
                )
            elif candle_data is not None:
                return (
                    self.price >= candle_data["High"]
                    and self.price <= candle_data["Low"]
                )

            raise ValueError

    def __init__(
        self, starting_balance: float = 10, fee: float = -0.00025,
    ):
        super().__init__()
        self.starting_balance = starting_balance
        self.fee = fee
        self.__current_data: typing.Any

    def step(self, action: int) -> typing.Tuple[object, float, bool, dict]:
        self._current_timestep += 1
        self._current_data = self._calculate_current_data()

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
            self._current_position.volume
            * (1 / self._current_position.average_price - 1 / price)
            if self._current_position.volume != 0
            else 0
        )

        if self._current_position.volume != 0:
            bankruptcy_price = (
                1
                - 1
                / (
                    1
                    + (
                        self._current_position.volume
                        / (self._balance * self._current_position.average_price)
                    )
                )
            ) * self._current_position.average_price

            if (
                self._current_position.volume >= 0
                and price <= bankruptcy_price
                and bankruptcy_price > 0
                or self._current_position.volume <= 0
                and price >= bankruptcy_price
                and bankruptcy_price > 0
            ):
                return (
                    self._current_data.reshape(
                        (self.orderbook_depth * 2, self.orderbook_hist, 1)
                    ),
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

    def __check_all_orders

    def _calculate_current_data(self) -> typing.Any:
        raise NotImplementedError

    def _make_action(self, action: int) -> typing.Any:
        raise NotImplementedError

    def add_order(self, price: float, volume: float) -> "BaseTradingEnv.Order":
        order = BaseTradingEnv.Order(price, volume)
        self._current_orders.append(order)
        self.orders_created += 1
        return order

    def cancel_order(self, order: "BaseTradingEnv.Order") -> None:
        self._current_orders.remove(order)
        self.orders_canceled += 1

    def fill_order(self, order: "BaseTradingEnv.Order") -> None:
        if self._current_position.volume == 0:
            self._current_position.volume = order.volume
            self._current_position.average_price = order.price
        elif np.sign(self._current_position.volume) == np.sign(order.volume):
            self._current_position.average_price = (
                order.price * order.volume
                + self._current_position.average_price * self._current_position.volume
            )
            self._current_position.volume = order.volume + self._current_position.volume
            self._current_position.average_price /= self._current_position.volume
        elif abs(self._current_position.volume) >= abs(order.volume):
            self._current_position.volume += order.volume
            self._balance += order.volume * (
                -1 / self._current_position.average_price + 1 / order.price
            )
        else:
            self._balance += self._current_position.volume * (
                1 / self._current_position.average_price - 1 / order.price
            )
            self._current_position.volume += order.volume
            self._current_position.average_price = order.price

        self._balance -= self.fee * abs(order.volume) / order.price
        self.orders_filled += 1

    def reset(self):
        """
        reset

        Reset the state of the environment to an initial state.
        """
        self._current_timestep = self.orderbook_hist
        self._current_price = [self.data.shape[0] // 2, self.data.shape[0] // 2]
        self._current_data = self._calculate_current_data()
        self._current_orders: typing.List[self.Order] = []
        self._balance = self.starting_balance  # Balance is in BTC
        self._current_position: typing.NamedTuple[float, float] = namedtuple(
            "Position", ["volume", "average_price"]
        )
        self.orders_created = 0
        self.orders_filled = 0
        self.orders_canceled = 0

        return self._current_data

    def render(self, mode="human", close=False):
        ...

    def close(self):
        ...


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import datetime

    env = OrderbookMarketmaker("orderbook.npy")
    env.reset()
    i = 0
    tests = []
    t = datetime.datetime.now()

    while True:
        obs, reward, done, info = env.step(env.action_space.sample())
        # env.render()
        i += 1

        if (datetime.datetime.now() - t).total_seconds() >= 1:
            print("FPS:", i)
            i = 0
            t = datetime.datetime.now()
        # print(env.render(mode="rgb_array"), end=" ")
        # plt.imshow(env.render(mode="rgb_array"))
        # plt.show()
        if done:
            print(info)
            plt.clf()
            plt.plot(list(range(len(tests))), tests)
            plt.gcf().savefig("plot.png")
            tests.append(info["balance"] + info["unr_profit"])
            # print("Episode finished, reset.")
            env.reset()
        # else:
        # print(info, end="                      \r")

