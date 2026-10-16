import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Type

import networkx as nx
import numpy as np
from flatland.envs.agent_utils import EnvAgent as TrainAgent
from flatland.envs.rail_env import RailEnvActions
from gymnasium import Space, spaces

from flatland.envs.step_utils.states import TrainState


from switchfl import NodeId, PortId, TrainAgentHandle
from switchfl.utils.rail_graph import add_rail_actions
from switchfl.utils.switch_agent import build_rail_action_map


def build_switch_to_rail_actions(
    switch_graph: nx.Graph,
) -> Tuple[List[Dict[PortId, List[RailEnvActions]]], List[Tuple[PortId, PortId]]]:
    """returns a list of actions for each port.

    Args:
        switch_graph (nx.Graph): switch graph with ports and inter-connectivity's

    Returns:
        Tuple[List[Dict[PortId, List[RailEnvActions]]], List[Tuple[PortId, PortId]]]: Each entry in the parent list corresponds to one action
            1. List[Dict[PortId, List[RailEnvActions]]]: each entry contains commands for trains at each port of the switch
            2. List[Tuple[PortId, PortId]]: After executing, across which ports will the train transition the switch (source, target)
    """
    # add actions to switch_graph
    switch_graph = add_rail_actions(switch_graph)
    action_map, outcomes = build_rail_action_map(switch_graph)
    return action_map, outcomes


class _Switch(ABC):
    def __init__(
        self,
        id: NodeId,
        switch_graph: nx.Graph,
        port2neighbor: Dict[PortId, Tuple[NodeId, PortId]] = None,
    ):
        """Switch class for managing switch behavior and actions."""
        self.id = id
        """Unique identifier for the switch."""
        self.switch_graph = switch_graph  # local_switch_graph
        """Graph representation of the switch."""
        self.port2neighbor = port2neighbor
        """Mapping of own port IDs to neighboring switch nodes and their ports."""

        self.n_gaits = len(self.switch_graph.nodes)
        """Number of gait (port) nodes in the switch."""
        self.n_rails = len(self.switch_graph.edges)
        """Number of rail (edge) connections in the switch."""
        
        res = build_switch_to_rail_actions(self.switch_graph)
        self.actions = res[0]
        """List of actions per port: self.actions[z]: 2 actions per port"""
        self.action_outcomes = res[1]
        """List of port mappings. self.action_outcomes[z]: train from port x to port y"""
        self.n_actions = len(self.action_outcomes)
        
        self.semaphores: Dict[PortId, bool]
        """which ports are blocked: True, which are free: False"""
        self._port_nodes = OrderedDict(
            {int(str(node[0])[-1]): node for node in self.switch_graph.nodes}
        )
        """to have a ordered list of port nodes"""
        self._pos2port: Dict[Tuple[int, int], PortId] = {
            pos: port for port, pos in self.switch_graph.nodes.data("rail_prev_node")
        }
        """rail position before entering a node"""

        self.reset()

    def reset(self):
        self.semaphores = {port: False for port in self.switch_graph.nodes}

    def block_port(self, port: PortId):
        """indicate a given port is blocked because of an incoming train

        Args:
            port (PortId): which port is blocked
        """
        if port not in self.semaphores.keys():
            logging.error(f"{port=} is not part of switch:{self.id}")
            return
        self.semaphores[port] = True

    def free_port(self, port: PortId):
        """indicate a given port is freed because of an incoming train is already processed

        Args:
            port (PortId): which port is freed
        """
        if port not in self.semaphores.keys():
            logging.error(f"{port=} is not part of switch:{self.id}")
            return
        self.semaphores[port] = False

    def get_action_mask(self, port: PortId, semaphore) -> np.ndarray:
        """which actions are allowed wrt. incoming train semaphores

        Args:
            port (PortId): from which port the train is coming
            semaphore (List[bool]): which target ports are blocked

        Returns:
            np.ndarray: integer array. 1: action allowed, 0: action forbidden (n_actions, )
        """
        mask_0 = []
        for source, _ in self.action_outcomes:
            if source == port:
                mask_0.append(1)
            else:
                mask_0.append(0)

        mask_1 = []
        ports = self.get_port_nodes()
        semaphore_dict = dict(zip(ports, semaphore))

        for _, target in self.action_outcomes:
            mask_1.append(semaphore_dict[target])

        # stop action is always valid
        mask_0.append(1)
        mask_1.append(1)
        
        mask = (np.array(mask_0) & np.array(mask_1)).astype(np.int8)

        return mask

    def get_train_action(
        self, action: int, active_train: int, train_to_port: Dict[TrainAgentHandle, PortId]
    ) -> Tuple[int | None, List[RailEnvActions]]:
        """For the given trains which are about to enter this switch, return the actions sequences for each train

        Args:
            action (int): discrete action
            train_agents (List[TrainAgent]): all trains on the grid

        Returns:
            Tuple[int, List[RailEnvActions]]:
                - train agent which is moving / crossing the switch.
                    If all currently positioned trains have to wait -> return None.
                - For each train at the switch return actions to perform
        """

        result = []
        moving_train = None

        port_node = train_to_port.get(active_train)

        # Only the train at this switch port can get an action
        if action == len(self.actions):
            result = [RailEnvActions.STOP_MOVING]
        else:
            if port_node in self.actions[action]:
                actions = self.actions[action][port_node].copy()
                result = actions
            
                if actions[0] != RailEnvActions.STOP_MOVING:
                    moving_train = active_train

        return moving_train, result

    def get_port_nodes(self) -> List[PortId]:
        return list(self._port_nodes.values())

    def get_next_node(self, action: int) -> Tuple[NodeId, PortId]:
        """Given a discrete action return the node a train would transition to

        Args:
            action (int): discrete action

        Returns:
            Tuple[NodeId, PortId]: The next node and port the train would transition to
        """
        self.action_outcomes[action]

    @abstractmethod
    def get_action_space(self, seed: int = None) -> Space:
        raise NotImplementedError


# T or Y junction
class Switch1(_Switch):
    def __init__(self, id, switch_graph, port2neighbor=None):
        super().__init__(id, switch_graph, port2neighbor)

    def get_action_space(self, seed=None):
        # gaits: 0, 1, 2
        # switch gait: 3
        # 0  1  2
        # --------
        # g  w  w
        # w  g  w
        # w  w  g1
        # w  w  g2
        # can have a different permutation based on orientation
        return spaces.Discrete(5, seed=seed)


# Intersection
class Switch2(_Switch):
    def __init__(self, id, switch_graph, port2neighbor=None):
        super().__init__(id, switch_graph, port2neighbor)

    def get_action_space(self, seed=None):
        # gaits: 0, 1, 2, 3
        # switch gait: None
        # 0  1  2  3
        # ----------
        # g  w  g  w
        # w  g  w  g
        return spaces.Discrete(5, seed=seed)


# Intersection with one pass
class Switch3(_Switch):
    def __init__(self, id, switch_graph, port2neighbor=None):
        super().__init__(id, switch_graph, port2neighbor)

    def get_action_space(self, seed=None):
        # gaits: 0, 1, 2, 3
        # switch gait: 0, 3
        # 0  1  2  3
        # ----------
        # g1 w  w  w
        # g2 w  w  w
        # w  g  w  w
        # w  w  g  w
        # w  w  w  g1
        # w  w  w  g2
        return spaces.Discrete(7, seed=seed)


# Intersection with two passes
class Switch4(_Switch):
    def __init__(self, id, switch_graph, port2neighbor=None):
        super().__init__(id, switch_graph, port2neighbor)

    def get_action_space(self, seed=None):
        # gaits: 0, 1, 2, 3
        # switch gait: 0, 1, 2, 3
        # 0  1  2  3
        # ----------
        # g1 w  w  w
        # g2 w  w  w
        # w  g1 w  w
        # w  g2 w  w
        # w  w  g1 w
        # w  w  g2 w
        # w  w  w  g1
        # w  w  w  g2
        return spaces.Discrete(9, seed=seed)


SWITCH_AGENT_MAP = {
    (3, 4): Switch1,
    (4, 4): Switch2,
    (4, 6): Switch3,
    (4, 8): Switch4,
}


def get_switch_type(nodes, edges) -> Type[_Switch]:
    n_gaits = len(nodes)
    n_rails = len(edges)
    try:
        return SWITCH_AGENT_MAP[(n_gaits, n_rails)]
    except KeyError:
        raise ValueError(f"No Agent with {n_gaits=} and {n_rails=}")
