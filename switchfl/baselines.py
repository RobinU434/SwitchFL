"""Baseline switch policies.

A policy is called once per (switch, train) decision with the env, the deciding switch's name, the train handle
and the action mask, and returns an allowed action index (the last index is STOP).
"""

from typing import Callable

import numpy as np

from switchfl.utils.naming import name2switch_id

Policy = Callable[[object, str, int, np.ndarray], int]


def random_policy(rng: np.random.Generator) -> Policy:
    """Uniformly random allowed action (STOP included)."""

    def act(env, switch_name: str, train: int, mask: np.ndarray) -> int:
        return int(rng.choice(np.flatnonzero(mask)))

    return act


def shortest_path_policy(env, switch_name: str, train: int, mask: np.ndarray) -> int:
    """Allowed rail action whose exit is closest to the train's target (Flatland distance map);
    STOP only if no rail action is allowed."""
    rail_network = env.rail_network
    switch = rail_network.get_switch_on_position(name2switch_id(switch_name))
    distance_map = env.rail_env.distance_map.get()
    best, best_distance = len(mask) - 1, np.inf
    for action in np.flatnonzero(mask[:-1]):
        out_port = switch.action_outcomes[action][1]
        row, col = rail_network.rail_graph.nodes[out_port]["rail_prev_node"]
        distance = distance_map[train, row, col, rail_network.map_direction(out_port)]
        if distance < best_distance:
            best, best_distance = int(action), distance
    return best
