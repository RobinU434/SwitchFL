from typing import Dict, List, Tuple

import networkx as nx
from flatland.envs.rail_env import RailEnvActions
from switchfl import PortId


def build_rail_action_map(
    graph: nx.Graph,
) -> Tuple[List[Dict[PortId, List[RailEnvActions]]], List[Tuple[PortId, PortId]]]:
    """computes which discrete environment actions result in which environment actions

    Args:
        graph (nx.Graph): switch graph with actions annotation:
            Each action annotation is stored at the 'action' datafield as a dictionary:
                - key: where the action leads to
                - value: action sequence


    Returns:
        Tuple[List[Dict[PortId, List[RailEnvActions]]], List[Tuple[PortId, PortId]]]: Each entry in the parent list corresponds to one action
            1. List[Dict[PortId, List[RailEnvActions]]]: each entry contains commands for trains at each port of the switch
            2. List[Tuple[PortId, PortId]]: After executing, across which ports will the train transition the switch (source, target)
    """
    actions = []
    target_map = []
    for port, port_actions in graph.nodes.data("actions"):
        for target, a in port_actions.items():
            action = {port: a}
            target_map.append((port, target))

            # add default stop action for all other nodes
            for other_node in graph.nodes:
                if other_node == port:
                    continue
                action[other_node] = [
                    RailEnvActions.STOP_MOVING,
                    RailEnvActions.STOP_MOVING,
                ]

            actions.append(action)
    return actions, target_map
        