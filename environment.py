"""
Просто какой-то рофл я хз вообще
"""

import gym
import test_import
from gym import spaces


class OrderbookMarketmaker(gym.Env):
    """
    This environment is using an orderbook data to do marketmaking.

    Args:
        gym ([type]): [description]

    Raises:
        ValueError: If you are a Loh, then sorry lol
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(3)
        # Example for using image as input:
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8
        )

        self.a = test_import.OrderbookMarketmakerv2()

    def step(self, action: int):
        """
        step [summary]

        Execute one time step within the environment

        Args:
            action (int): [description]

        Raises:
            ValueError: [description]
        """
        raise ValueError

    def reset(self):
        """
        reset [summary]

        Reset the state of the environment to an initial state
        """
        ...

    def render(self, mode="human", close=False):
        """
        render [summary]

        Render the environment to the screen

        Args:
            mode (str, optional): [description]. Defaults to "human".
            close (bool, optional): [description]. Defaults to False.
        """
