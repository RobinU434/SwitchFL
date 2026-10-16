import logging
from typing import Any, Dict, Iterator, List, Tuple
from gymnasium import Space
import networkx as nx
import numpy as np
import pandas as pd
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.agent_utils import EnvAgent as TrainAgent
from flatland.envs.step_utils.states import TrainState
from flatland.envs.agent_utils import Grid4TransitionsEnum
from switchfl import NodeId, PortId, TrainAgentHandle
from switchfl.switch_agents import _Switch, build_switch_to_rail_actions, get_switch_type
from switchfl.utils.logging import format_logger
from switchfl.utils.naming import get_node_id_on_port_id, switch_id2name
from switchfl.utils.rail_graph import (
    create_rail_graph,
    generate_local_switch_graphs,
    insert_switch_proximity_nodes,
    prune_non_switches,
)


def build_switch_network(rail_network: nx.Graph) -> nx.Graph:
    """Build a network of switches from the rail network.

    Args:
        rail_network (nx.Graph): The rail network graph.

    Returns:
        nx.Graph: The switch network graph.
    """
    switch_network = nx.Graph()
    df = pd.DataFrame(
        [attr for _, attr in rail_network.nodes.data()],
        index=[idx for idx, _ in rail_network.nodes.data()],
    )
    edges = set()
    for node, attr in df.groupby("switch_id"):
        switch_graph = rail_network.subgraph(attr.index)

        # find neighbors for edges
        port2neighbor = {}
        for port in switch_graph.nodes:
            neighbors = rail_network.neighbors(port)
            neighbor_port_df = df[df.index.isin(neighbors) & (df["switch_id"] != node)]
            edges.add((node, neighbor_port_df["switch_id"].item()))
            port2neighbor[port] = (
                neighbor_port_df["switch_id"].item(),
                neighbor_port_df.index.item(),
            )

        res = build_switch_to_rail_actions(switch_graph)

        switch = get_switch_type(switch_graph.nodes, res[1])(node, switch_graph, port2neighbor)

        switch_network.add_node(
            node,
            switch_cls=switch,
            node_color=attr["node_color"].iloc[0],
            position=(node[1], -node[0]),
        )

    switch_network.add_edges_from(edges)
    return switch_network


def build_rail_graph(rail_env: RailEnv) -> nx.Graph:
    """Build a graph representation of the rail environment.

    Args:
        rail_env (RailEnv): The rail environment instance.

    Returns:
        nx.Graph: The graph representation of the rail environment.
    """
    rail_env.reset()
    graph = create_rail_graph(rail_env)
    graph = insert_switch_proximity_nodes(graph)
    graph = prune_non_switches(graph)
    graph = generate_local_switch_graphs(graph)
    return graph


class RailNetwork:
    def __init__(self, rail_env: RailEnv):
        self.logger = logging.getLogger(type(self).__name__)
        self.logger = format_logger(self.logger)

        self.rail_env = rail_env
        self.switch_network: nx.Graph

        self.rail_graph = build_rail_graph(rail_env)
        self.switch_network = build_switch_network(self.rail_graph)

        # build lookup dictionaries
        self._dir2port_idx = {
            1.0: Grid4TransitionsEnum.WEST.value,
            2.0: Grid4TransitionsEnum.SOUTH.value,
            3.0: Grid4TransitionsEnum.EAST.value,
            4.0: Grid4TransitionsEnum.NORTH.value,
        }
        self._pos_dir2port: Dict[Tuple[Tuple[int, int], int], PortId] = {
            (pos, self._dir2port_idx[round(port[0] - int(port[0]), 1) * 10]): port
            for port, pos in self.rail_graph.nodes.data("rail_prev_node")
        }
        """rail position before entering a node"""
        self._pos2switch: Dict[Tuple[int, int], _Switch] = {
            switch_node: switch
            for switch_node, switch in self.switch_network.nodes.data("switch_cls")
        }
        """switch position to switch class"""

        self._train2next_port: Dict[int, PortId] = {
            train.handle: None for train in rail_env.agents
        }

        self._train2next_port_dist: Dict[int, int] = {
            train.handle: None for train in rail_env.agents
        }

        """get the next port the agent is going to enter. 
        This dictionary changes after a train transition got determined by an agent."""
        self._train_prev_port: Dict[int, PortId] = {
            train.handle: None for train in rail_env.agents
        }
        self._train_source_port: Dict[int, PortId] = {
            train.handle: None for train in rail_env.agents
        }
        
        """get the previous port the agent came from. 
        This dictionary changes after a train transition got determined by an agent."""

        self.semaphores: Dict[PortId, tuple] = {}

    def reset(self):
        # reset all switches
        for _, switch in self.switch_network.nodes.data("switch_cls"):
            switch: _Switch
            switch.reset()

        self._train2next_port: Dict[int, PortId] = {
            handle: None for handle in self._train2next_port.keys()
        }

        self._train2next_port_dist: Dict[int, int] = {
            handle: None for handle in self._train2next_port_dist.keys()
        }

        self.semaphores = {}

    def get_switch_on_position(self, switch: NodeId) -> _Switch | None:
        """get switch class for the corresponding position of the switch.
        This method only returns a switch if the trains is actually on the switch tile.
        Otherwise it returns None.

        Args:
            switch (NodeId): position in rail grid

        Returns:
            _Switch | None: If the given node does not belong to a switch return None.
                Otherwise the switch object.
        """
        if switch not in self.switch_network.nodes:
            return None
        return self.switch_network.nodes.data("switch_cls")[switch]

    def get_neighbor_switch(self, port: PortId) -> Tuple[_Switch, PortId]:
        """get the neighbor switch instance and its port assuming you are
        leaving the given port and end up at the next switch

        Args:p
            port (PortId): out port of a switch

        Returns:
            Tuple[_Switch, PortId]: _Switch object of the next switch node and the
                PortId of the port though you will ender the switch node.
        """
        node_id = get_node_id_on_port_id(port)
        switch = self.get_switch_on_position(node_id)
        neighbor_node_id, neighbor_port_id = switch.port2neighbor[port]
        neighbor_switch = self.get_switch_on_position(neighbor_node_id)
        # if neighbor_switch is None:
        #     neighbor_port_id = None
        return neighbor_switch, neighbor_port_id

    def get_switch_on_port(self, port: PortId) -> _Switch:
        """get the switch if the position is a node before entering a switch node

        Args:
            position (PortId): position in rail network

        Returns:
            _Switch | None: either a switch or None, if the given position does not belong to a switch
        """
        node_id = get_node_id_on_port_id(port)
        return self._pos2switch[node_id]

    def get_port_on_position(
        self, position: Tuple[int, int], direction: Grid4TransitionsEnum = None
    ) -> PortId | None:
        """get the port node if the position is a node before entering a switch node

        Args:
            position (Tuple[int, int]): position in rail network
            direction (Grid4TransitionsEnum, optional): which direction the train is going.
                Is need if there are multiple switches a train can enter from a
                single position (neighboring switches for instance). Defaults to None

        Returns:
            PortId | None: either a port or None, if the given position does not belong to a switch
        """
        if direction is not None:
            return self._pos_dir2port.get((position, direction))

        # only works if there is only one node neighbor to the current position
        ports = []
        for direction in self._dir2port_idx.values():
            port = self._pos_dir2port.get((position, direction))
            if port is not None:
                ports.append(port)

        if len(ports) > 1:
            raise ValueError(f"Multiple {ports=} reachable from given {position=}")
        elif len(ports) == 1:
            return ports[0]
        else:
            return None
        
    def extend_semaphores(self):
        """
        Extend semaphores for stopped trains
        """
        for train in self.rail_env.agents:
            for p, (tr, _, _, _, _) in self.semaphores.items():
                if tr == train.handle and (train.state == TrainState.STOPPED or train.state == TrainState.MALFUNCTION):
                    distance = self.semaphores[p][4] - self.semaphores[p][3]
                    self.semaphores[p][3] = self.rail_env._elapsed_steps
                    self.semaphores[p][4] = self.semaphores[p][3] + distance

            if train.state == TrainState.MALFUNCTION:
                port = self._train2next_port[train.handle]
                if port not in self.semaphores.keys():
                    self.semaphores[port] = [train.handle, 'in', self.map_direction(port), self.rail_env._elapsed_steps,
                                            self.rail_env._elapsed_steps + self._train2next_port_dist[train.handle]]

    def transition_train(
        self, train: TrainAgent, in_port: PortId, out_port: PortId
    ) -> Tuple[_Switch, PortId]:
        """
        update semaphores and information at which switch the train will arrive next

        Args:
            train (TrainAgent): train entering the switch
            in_port (PortId): the port a train entered the switch
            out_port (PortId): the port the train will leave the switch

        Returns:
            Tuple[_Switch, PortID]: instance of the next switch the train will arrive at after
                leaving on given out_port, and the port though which the train will enter the next switch
        """
        in_switch = np.array(in_port, dtype=int)
        out_switch = np.array(out_port, dtype=int)
        assert (
            in_switch == out_switch
        ).prod(), f"both given ports have to belong to the same switch: {in_switch} != {out_switch}"

        # find next switch with next port
        next_switch, target_port = self.get_neighbor_switch(out_port)
        # assign the semaphores
        self.transition_semaphore(in_port, out_port, target_port, train, next_switch)
        # the next port in self._train2next_port
        self._train_source_port[train.handle] = in_port
        self.set_trains_next_port(train, target_port)
        self.set_trains_prev_port(train, out_port)
        self.logger.debug(
            f"num rail segments to next port: {self.get_port_distance(out_port, target_port)} from {out_port} to {target_port}"
        )
        return next_switch, target_port

    def map_direction(self, port):
        """map port to flatland direction"""
        d = round((port[0] - int(port[0])) * 10)
        if d == 1:
            return 1
        elif d == 2:
            return 0
        elif d == 3:
            return 3
        else:
            return 2
        
    def map_inverse_direction(self, direction):
        """map flatland direction to port direction"""
        if direction == 1:
            return 3
        elif direction == 0:
            return 4
        elif direction == 3:
            return 1
        else:
            return 2
        
    def transition_semaphore(self, source: PortId, out_port: PortId, target: PortId, train: TrainAgent, next_switch: _Switch):
        """handle semaphore freeing and blocking if a train is moving from the given source port (source) and moving to the outgoing port.

        Args:
            source (PortId): the port a train entered a switch
            out_port (PortId): the port the train will leave the switch
            target (PortId): the port though which the train will enter the next switch
            train (TrainAgent): train entering the switch
            next_switch (_Switch): the next switch the train will arrive at
        """

        # free previous semaphores occupied by the train
        if train.state != TrainState.MALFUNCTION:
            for p in self.get_switch_on_port(self._train2next_port[train.handle]).get_port_nodes():
                if p in self.semaphores and self.semaphores[p][0] == train.handle:
                    del self.semaphores[p]

            if self._train_prev_port[train.handle] is not None:
                for p in self.get_switch_on_port(self._train_prev_port[train.handle]).get_port_nodes():
                    if p in self.semaphores and self.semaphores[p][0] == train.handle:
                        del self.semaphores[p]

        # set new semaphores occupied by the train
        if out_port not in self.semaphores.keys():
            self.semaphores[out_port] = [train.handle, 'out', self.map_direction(out_port), self.rail_env._elapsed_steps,
                                         self.rail_env._elapsed_steps + 3]
        else: # override if semaphore has same direction
            if self.semaphores[out_port][1] == 'out' or \
                self.semaphores[out_port][3] > self.rail_env._elapsed_steps:
                    self.semaphores[out_port][0] = train.handle
                    self.semaphores[out_port][3] = self.rail_env._elapsed_steps
                    self.semaphores[out_port][4] = self.rail_env._elapsed_steps + 3

        if target not in self.semaphores.keys():
            self.semaphores[target] = [train.handle, 'in', self.map_direction(target), self.rail_env._elapsed_steps,
                                       self.rail_env._elapsed_steps + self.get_port_distance(out_port, target) + 1]
        else:
            if self.semaphores[target][1] == 'in' or \
                self.semaphores[target][3] > self.rail_env._elapsed_steps:
                    self.semaphores[target][0] = train.handle
                    self.semaphores[target][3] = self.rail_env._elapsed_steps
                    self.semaphores[target][4] = self.rail_env._elapsed_steps + self.get_port_distance(out_port, target) + 1

        # find all edges related to target
        edge_list = list(self.rail_graph.edges(target))

        # find the current active edge
        for edge in edge_list:
            if edge[0] == out_port or edge[1] == out_port:
                moving_edge = edge
        edge_list.remove(moving_edge)

        # if there is only one port at the end of the edge, find the next edge and set the semaphores
        if len(edge_list) == 1:
            unique_port = edge_list[0][0] if edge_list[0][0] != target else edge_list[0][1]
            prox_list = list(self.rail_graph.edges(unique_port))
            edges_toremove = []
            for edge in prox_list:
                if get_node_id_on_port_id(edge[0]) == next_switch.id and get_node_id_on_port_id(edge[1]) == next_switch.id:
                    edges_toremove.append(edge)
            for edge in edges_toremove:
                prox_list.remove(edge)

            # Set the semaphores for the edges that have been identified
            out_edge = edge_list[0]
            for port in out_edge:
                if port != source and port != out_port and port != target:
                    distance = self.get_port_distance(out_port, target) + self.get_port_distance(target, port) + 1
                    if port not in self.semaphores.keys():
                        self.semaphores[port] = [train.handle, 'out', self.map_direction(port), self.rail_env._elapsed_steps,
                                                 self.rail_env._elapsed_steps + distance]
                    else:
                        if self.semaphores[port][1] == 'out' or \
                            self.semaphores[port][3] > self.rail_env._elapsed_steps:
                                self.semaphores[port] = [train.handle, 'out', self.map_direction(port), self.rail_env._elapsed_steps,
                                                        self.rail_env._elapsed_steps + distance]
                                
            distance = self.get_port_distance(out_port, target) + self.get_port_distance(target, unique_port)
            if unique_port not in self.semaphores.keys():
                self.semaphores[unique_port] = [train.handle, 'out', self.map_direction(unique_port), self.rail_env._elapsed_steps,
                                                self.rail_env._elapsed_steps + distance]
            else:
                if self.semaphores[unique_port] == 'out' or \
                    self.semaphores[unique_port][3] > self.rail_env._elapsed_steps:
                        self.semaphores[unique_port] = [train.handle, 'out', self.map_direction(unique_port), self.rail_env._elapsed_steps,
                                self.rail_env._elapsed_steps + distance]  

            prox_edge = prox_list[0]
            for port in prox_edge:
                if port != source and port != out_port and port != unique_port:
                    distance = self.get_port_distance(out_port, target) + self.get_port_distance(target, unique_port) + \
                        self.get_port_distance(unique_port, port) + 1        
                    if port not in self.semaphores.keys():
                        self.semaphores[port] = [train.handle, 'in', self.map_direction(port), self.rail_env._elapsed_steps,
                                                 self.rail_env._elapsed_steps + distance]
                    else:
                        if self.semaphores[port][1] == 'in' or \
                            self.semaphores[port][3] > self.rail_env._elapsed_steps:
                                self.semaphores[port] = [train.handle, 'in', self.map_direction(port), self.rail_env._elapsed_steps,
                                                            self.rail_env._elapsed_steps + distance]

        for port in moving_edge:
            if port != source and port != out_port:
                distance = self.get_port_distance(out_port, port) + 1
                if port not in self.semaphores.keys():
                    self.semaphores[port] = [train.handle, 'out', self.map_direction(port), self.rail_env._elapsed_steps,
                                             self.rail_env._elapsed_steps + distance]
                else:
                    if self.semaphores[port][1] == 'out' or \
                            self.semaphores[port][3] > self.rail_env._elapsed_steps:
                                self.semaphores[port] = [train.handle, 'out', self.map_direction(port), self.rail_env._elapsed_steps,
                                                        self.rail_env._elapsed_steps + distance]

        self.logger.debug(f"CURRENT SEMAPHORES: {self.semaphores}")


    def get_train_actions(
        self, node: NodeId, action: int, active_train: TrainAgent
    ) -> Tuple[int, List[RailEnvActions]]:
        """get the sequence of train actions from switch action and the train which is moving / transitioning over the switch

        Args:
            node (NodeId): node_id of node from where to get the actions from
            action (int): action index to get the train actions from
            train_agents (List[TrainAgent]): list of all train agents on grid

        Returns:
            Tuple[int, List[RailEnvActions]]:
                - train agent which is moving / crossing the switch.
                    If all currently positioned trains have to wait -> return None.
                - For each train at the switch return actions to perform
        """
        switch = self._pos2switch[node]
        return switch.get_train_action(action, active_train, self._train2next_port)

    def get_switch_names(self) -> List[str]:
        """Get the names of all switches in the network.

        Returns:
            List[str]: The names of all switches.
        """
        res = []
        for node in self.switch_network.nodes:
            res.append(switch_id2name(node))
        return res

    def get_switch_action_space(self, node: NodeId, seed: int = None) -> Space:
        """Get the action space for a switch.

        Args:
            node (NodeId): The ID of the switch to get the action space for.
            seed (int, optional): Random seed for action space generation. Defaults to None.

        Returns:
            Space: The action space for the switch.
        """
        switch = self.get_switch_on_position(node)
        return switch.get_action_space(seed=seed)

    def set_trains_next_port(self, train: TrainAgent, port: PortId):
        """Set the next port for a train.

        Args:
            train (TrainAgent): The train agent to set the next port for.
            port (PortId): The next port the train will move to.
        """
        self._train2next_port[train.handle] = port

    def get_trains_next_port(self, train: TrainAgent) -> PortId:
        """Get the next port for a train.

        Args:
            train (TrainAgent): The train agent to get the next port for.

        Returns:
            PortId: The next port the train will move to.
        """
        return self._train2next_port[train.handle]
    
    
    def set_trains_prev_port(self, train: TrainAgent, port: PortId):
        """Set the previous port for a train.

        Args:
            train (TrainAgent): The train agent to set the previous port for.
            port (PortId): The previous port the train came from.
        """
        self._train_prev_port[train.handle] = port

    def get_trains_prev_port(self, train: TrainAgent) -> PortId:
        """Get the previous port for a train.

        Args:
            train (TrainAgent): The train agent to get the previous port for.

        Returns:
            PortId: The previous port the train came from.
        """         
        return self._train_prev_port[train.handle]
    
    def get_rail_pieces_between_ports(self, source_node: NodeId, target_node: NodeId) -> List[Tuple[int, int]]:
        """Get the rail pieces between two switch nodes.

        Args:
            source_node (NodeId): The source switch node.
            target_node (NodeId): The target switch node.

        Returns:
            List[Tuple[int, int]]: A list of rail piece coordinates between the two nodes.
        """
        edge_data = self.rail_graph.get_edge_data(source_node, target_node)
        if edge_data is None:
            return []
        rail_nodes = edge_data.get("rail_nodes", [])
        return rail_nodes

    def get_switch_neighbor(
        self, switch_id: NodeId = None
    ) -> Dict[PortId, PortId] | Dict[NodeId, Dict[PortId, NodeId]]:
        """Get the neighboring ports for a switch.

        Args:
            switch_id (NodeId, optional): The ID of the switch to get neighbors for. Defaults to None.

        Returns:
            Dict[PortId, PortId] | Dict[NodeId, Dict[PortId, NodeId]]: A dictionary mapping port IDs to their neighboring port IDs, or a dictionary mapping switch IDs to their neighboring port IDs.
        """
        if switch_id is None:
            # get all neighbors based on port
            res = {}
            for _, switch in self.switches:
                res[switch.id] = self.get_switch_neighbor(switch.id)
            return res

        res = {}
        switch = self.get_switch_on_position(switch_id)
        for port in switch.get_port_nodes():
            res[port] = self.get_neighbor_switch(port)[1]
        return res

    def get_port_distance(self, port1: PortId, port2: PortId) -> int | None:
        """Get the number of rail segments between two ports.
        If there is no rail connecting those two ports directly return None

        Args:
            port1 (PortId): first port
            port2 (PortId): second port

        Returns:
            int | None: the number of rail segments connecting the two ports. If there is no direct edge -> return None
        """
        edge_data = self.rail_graph.get_edge_data(port1, port2)
        if edge_data is None:
            return None
        rail_nodes = edge_data.get("rail_nodes", None)
        return len(rail_nodes)

    def get_switch_transition_info(
        self, switch_id: NodeId, action: int
    ) -> Dict[str, Any]:
        """Get transition information for a specific node and action.

        Args:
            switch_id (NodeId): The ID of the node to get information for.
            action (int): The action to get information for.
        Returns:
            Dict[str, Any]: A dictionary containing transition information about the switch and its actions.
        """
        switch = self.get_switch_on_position(switch_id)

        transition_info = {
            "node": switch_id,
            "action": action,
        }

        in_port, out_port = switch.action_outcomes[action]
        transition_info["in_port"] = in_port
        transition_info["out_port"] = out_port
        next_node, next_port = switch.port2neighbor[out_port]
        transition_info["next_node"] = next_node
        transition_info["next_port"] = next_port
        transition_info["train_actions"] = switch.actions[action]
        return transition_info

    @property
    def switches(self) -> Iterator[Tuple[Tuple[int, int], _Switch]]:
        return iter(self.switch_network.nodes.data("switch_cls"))
