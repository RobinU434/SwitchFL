import logging
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
from flatland.envs.rail_env import RailEnvActions
from gymnasium import spaces
from gymnasium.spaces import MultiDiscrete, Space

from switchfl.utils.rail_graph import add_rail_actions
from switchfl.utils.switch_agent import build_rail_action_map


class TargetSpace(MultiDiscrete):
    def __init__(
        self, height: int, width: int, n_ports: int, dtype=np.int64, seed: int = None
    ):
        # Ranges: [-1, height -1] and [-1, width -1] for each (x, y) pair
        # Offsets: we shift everything up by 1 to fit MultiDiscrete which only supports lower bound = 0
        nvec = np.array(
            [[height + 1, width + 1]] * n_ports
        )  # +1 for including -1 and height/width-1
        super().__init__(
            nvec=nvec.flatten(), dtype=dtype, seed=seed
        )  # flatten to shape (6,)

        self.height = height
        self.width = width
        self.n_ports = n_ports

    def sample(self):
        # Shift down by 1 to account for offset applied in nvec
        raw_sample = super().sample().reshape(self.n_ports, 2)
        return raw_sample - 1  # Undo the -1 shift

    def contains(self, x):
        x = np.array(x)
        if len(x.shape) == 1:
            # flat layout is [row_0, col_0, row_1, col_1, ...]
            x = x.reshape(-1, 2)
        if x.shape != (self.n_ports, 2):
            return False
        return np.all(
            (-1 <= x) & (x[:, [0]] <= self.height) & (x[:, [1]] <= self.width)
        )

    @property
    def is_np_flattenable(self) -> bool:
        """Checks whether this space can be flattened to a :class:`gymnasium.spaces.Box`."""
        return True

    def human_format(self, sample: np.ndarray) -> Dict[str, np.ndarray]:
        assert self.contains(sample)
        return {"train_targets": sample}


class MultiDiscreteSwitchObsSpace(Space):
    def __init__(
        self,
        n_gaits: int,  # how many ports are there
        rail_grid_shape: Tuple[int, int],  # for target space
        n_delay_levels: int = 3,
        seed=None,
    ):
        n_dims = 2 + n_gaits + (n_gaits * 2) + n_gaits  # node id, semaphores, targets, delays
        shape = (n_dims,)
        dtype = int
        super().__init__(shape, dtype, seed)

        self.n_gaits = n_gaits  # how many rails are connected to a node -> Binary variable: is one occupied or not
        self.rail_grid_shape = rail_grid_shape
        self.n_delay_levels = n_delay_levels

        # position of the switch in the rail grid
        self.node_space = spaces.MultiDiscrete(
            np.array(self.rail_grid_shape), seed=seed, dtype=np.int64
        )
        self.station_space = TargetSpace(
            height=self.rail_grid_shape[0],
            width=self.rail_grid_shape[1],
            n_ports=self.n_gaits,
            seed=seed,
        )

        self.delay_space = spaces.MultiDiscrete(
            np.ones(self.n_gaits) * (self.n_delay_levels + 1),
            seed=seed,
            start=np.ones(self.n_gaits) * (-1),
            dtype=np.int64,
        )
        # of a rail is blocked or if a train can be send onto this rail
        self.semaphore_space = spaces.MultiDiscrete(
            np.ones(n_gaits) * 2, seed=seed, start=np.zeros(n_gaits), dtype=np.int64
        )

    def sample(self):
        sample = np.concatenate(
            [
                self.node_space.sample(),
                self.semaphore_space.sample(),
                self.station_space.sample().flatten(),
                self.delay_space.sample(),
            ]
        )
        return sample

    def contains(self, x):
        x = np.array(x)
        if x.shape != self._shape:
            logging.debug(
                f"shape is not correct: given={x.shape}, expected={self._shape}"
            )
            return False

        indices = [
            self.node_space.shape[0],
            self.semaphore_space.shape[0],
            self.station_space.shape[0],
            self.delay_space.shape[0],
        ]
        node_id, semaphore, stations, delay = np.split(x, np.cumsum(indices))[:-1]
        # break it up
        if not self.node_space.contains(node_id):
            logging.debug(f"node space does not contain: {node_id}")
            return False
        if not self.semaphore_space.contains(semaphore):
            logging.debug(f"semaphore space does not contain: {semaphore}")
            return False
        if not self.station_space.contains(stations):
            logging.debug(f"station_space space does not contain: {stations}")
            return False
        if not self.delay_space.contains(delay):
            logging.debug(f"delay space does not contain: {delay}")
            return False

        return True

    @property
    def is_np_flattenable(self) -> bool:
        """Checks whether this space can be flattened to a :class:`gymnasium.spaces.Box`."""
        return True

    def human_format(self, sample: np.ndarray) -> Dict[str, np.ndarray]:
        assert self.contains(sample)
        indices = [
            self.node_space.shape[0],
            self.semaphore_space.shape[0],
            self.station_space.shape[0],
            self.delay_space.shape[0],
        ]
        node_id, semaphore, stations, delay = np.split(sample, np.cumsum(indices))[:-1]

        return {
            "node_id": node_id,
            "semaphore": semaphore,
            "train_targets": stations.reshape(-1, 2),
            "delay": delay,
        }


def build_switch_to_rail_actions(
    switch_graph: nx.Graph,
) -> Tuple[List[Dict[Any, List[RailEnvActions]]], List[Tuple[Any, Any]]]:
    """returns a list of actions for each port.

    Args:
        switch_graph (nx.Graph): switch graph with ports and inter-connectivity's

    Returns:
        Tuple[List[Dict[Any, List[RailEnvActions]]], List[Any, Any]]: Each entry in the parent list corresponds to one action
            1. List[Dict[Any, List[RailEnvActions]]]: each entry contains commands for trains at each port of the switch
            2. List[Any, Any]: After executing, across which ports will the train transition the switch (source, target)
    """
    # add actions to switch_graph
    switch_graph = add_rail_actions(switch_graph)
    action_map, outcomes = build_rail_action_map(switch_graph)
    return action_map, outcomes
