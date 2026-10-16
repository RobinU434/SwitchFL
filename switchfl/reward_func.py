from abc import ABC, abstractmethod
from typing import Tuple
from switchfl.observer import compute_delay
from switchfl.utils.flatland_compat import apply_action
from flatland.envs.rail_env import RailEnvActions
import numpy as np


class _RewardFunction(ABC):
    """Reward of the env's own loop, computed when a switch acts and credited to the switch the train
    reaches next. Pass a subclass instance as ASyncSwitchEnv(..., reward_func=...).

    For learning prefer the transitions of switchfl.core.SwitchCore, whose reward terms also see
    what happened until the train's next decision (arrival, truncation, tau).
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, train, train_actions, train_to_last_node, port_blocked) -> Tuple[float, float]:
        """Returns (reward, current delay of the train); the delay is stored for the train's next decision."""
        raise NotImplementedError


class StandardRewardFunction(_RewardFunction):
    def __init__(self, rail_env, stop_penalty: float = 1300):
        super().__init__()
        self.rail_env = rail_env
        self.stop_penalty = stop_penalty

    def __call__(self, train, train_actions, train_to_last_node, port_blocked) -> Tuple[float, float]:
        """
        Delay difference since the train's last switch, minus stop_penalty for STOP while a port was free.
        Override `extra_reward` to add bonuses or further penalties.

        Parameters
        ----------
        train : TrainAgent
            The train agent for which the reward is being calculated.
        train_actions : List[int]
            The list of actions taken by the train.
        train_to_last_node : Dict[int, Tuple[Tuple[int, int], int]]
            A mapping from train handles to their last node positions and delays.
        port_blocked : List[bool]
            A list indicating whether the ports are blocked.

        Returns
        -------
        Tuple[float, float]
            the reward and the train's current delay
        """
        new_position = train.position
        new_direction = train.direction

        for action in train_actions:
            if action != RailEnvActions.STOP_MOVING:
                new_position, new_direction, _ = apply_action(
                    self.rail_env.rail, action, new_position, new_direction
                )

        curr_delay = compute_delay(self.rail_env, train, new_position, new_direction)

        _, last_delay = train_to_last_node[train.handle]

        reward = last_delay - curr_delay
        # a STOP is only penalised if at least one port was free
        if train_actions[0] == RailEnvActions.STOP_MOVING and not np.all(port_blocked):
            reward -= self.stop_penalty
        reward += self.extra_reward(train, train_actions, port_blocked, curr_delay)

        return reward, curr_delay

    def extra_reward(self, train, train_actions, port_blocked, curr_delay) -> float:
        """Hook for bonuses / additional penalties, added to the reward. Default: 0."""
        return 0.0
