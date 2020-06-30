from gym.envs.registration import register

register(
    id="orderbook-marketmaker-v0", entry_point="gym_trading.envs:OrderbookMarketmaker",
)
