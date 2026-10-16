from abc import ABC, abstractmethod
from typing import Any
from switchfl.observer import compute_delay
from flatland.envs.rail_env import RailEnvActions
import numpy as np


class _RewardFunction(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, observation: Any, action: Any) -> float:
        raise NotImplementedError


class StandardRewardFunction(_RewardFunction):
    def __init__(self, rail_env):
        super().__init__()
        self.rail_env = rail_env
        self.stop_penalty = 1300

    def __call__(self, train, train_actions, train_to_last_node, port_blocked) -> float:
        """
        Updates the rewards based on the current state of the environment

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
        """
        new_position = train.position
        new_direction = train.direction

        for action in train_actions:
            if action != RailEnvActions.STOP_MOVING:
                (
                    _,
                    (new_position,
                    new_direction),
                    _,
                    _,
                ) = self.rail_env.rail.check_action_on_agent(
                    action,
                    ((new_position),
                    new_direction)
                )

        curr_delay = compute_delay(self.rail_env, train, new_position, new_direction)

        _, last_delay = train_to_last_node[train.handle]
                
        delay_diff = last_delay - curr_delay

        if len(port_blocked) == 1:
            if port_blocked[0]:
                reward = delay_diff
            else:
                if train_actions[0] == RailEnvActions.STOP_MOVING:
                    reward = delay_diff - self.stop_penalty
                else:
                    reward = delay_diff
        else:
            if np.sum(port_blocked) == len(port_blocked):
                reward = delay_diff
            else:
                if train_actions[0] == RailEnvActions.STOP_MOVING:
                    reward = delay_diff - self.stop_penalty
                else:
                    reward = delay_diff
        
        return reward, curr_delay

