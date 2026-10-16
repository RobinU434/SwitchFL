from typing import Any, Dict, List, Tuple
from tqdm import tqdm
import numpy as np

import networkx as nx
from flatland.envs.rail_env import RailEnv, RailEnvActions
from itertools import product
import matplotlib.colors as mcolors
import matplotlib as mpl
from itertools import combinations


def create_rail_graph(env: RailEnv, cmap="tab20") -> nx.Graph:
    graph = nx.Graph()
    cmap = mpl.colormaps[cmap]

    get_txs = env.rail.get_transitions
    # for all the possible combinations of (row, col, dir)
    for row, col, dir in tqdm(
        product(range(env.height), range(env.width), range(4)),
        total=env.height * env.width * 4,
    ):
        # [North, East, South, West]
        if bin(env.rail.get_full_transitions(row, col)).count("1") == 0:
            continue
        for dir_idx, tx in enumerate(get_txs(((row, col), dir))): # get transition
            # tx = 0 if there is no transition, tx = 1 if there is a transition
            if tx == 0:
                continue
            # Find next cell coordinates based on the exit direction dx
            next_row = row
            next_col = col
            if dir_idx == 0:  # North
                next_row -= 1
            elif dir_idx == 1:  # East
                next_col += 1
            elif dir_idx == 2:  # South
                next_row += 1
            else:  # West
                next_col -= 1

            # Check if next cell is within the grid
            if (
                next_row >= 0
                and next_row < env.width
                and next_col >= 0
                and next_col < env.height
            ):
                # Use deterministic color based on position to avoid randomness
                color_index = (row * env.width + col) % 20
                node_color = mcolors.to_hex(
                    mcolors.to_rgba_array(cmap(color_index))
                )
                if not graph.has_node((row, col)):
                    pos = np.array(
                        [col, -row]
                    )  # convert image pixel space to plotting space
                    graph.add_node(
                        (row, col),
                        transition=bin(env.rail.get_full_transitions(row, col))[
                            2:
                        ].zfill(16),
                        node_color=node_color,
                        position=pos,
                        switch_id=(row, col),
                    )
                if not graph.has_node((next_row, next_col)):
                    pos = np.array(
                        [next_col, -next_row]
                    )  # covert image pixel space to plotting space
                    graph.add_node(
                        (next_row, next_col),
                        transition=bin(
                            env.rail.get_full_transitions(next_row, next_col)
                        )[2:].zfill(16),
                        node_color=node_color,
                        position=pos,
                        switch_id=(next_row, next_col),
                    )

                graph.add_edge(
                    (row, col),
                    (next_row, next_col),
                    rail_nodes=[],
                    # rail_node_to_switch={},
                )
    return graph


def insert_switch_proximity_nodes(graph: nx.Graph) -> nx.Graph:
    # get neighbors and order them based on relative movement
    direction_index = {
        (0, 1): 0.1,  # East if you condider +1 in y direction
        (-1, 0): 0.2,  # North if you condider -1 in x direction
        (0, -1): 0.3,  # West if you condider -1 in y direction
        (1, 0): 0.4,  # South if you condider +1 in x direction
    }

    for node in list(graph.nodes):
        node_degree = graph.degree(node)
        if node_degree == 2:  # node_degree > 2 is a switch
            # node is not a switch, it is just a transition node
            continue

        # add surrounding nodes
        for idx, neighbor in enumerate(list(graph.neighbors(node))):
            rel_pos = tuple(np.array(neighbor).astype(int) - np.array(node))
            name_suffix = direction_index[rel_pos]
            new_node = (int(node[0]) + name_suffix, int(node[1]) + name_suffix)

            pos = (
                graph.nodes[neighbor]["position"] + 2 * graph.nodes[node]["position"]
            ) / 3
            switch_color = graph.nodes[node]["node_color"]
            switch_id = graph.nodes[node]["switch_id"]
            graph.add_node(
                new_node,
                node_color=switch_color,
                position=pos,
                switch_id=switch_id,
                switch_position=switch_id,
                rail_prev_node=graph.nodes[neighbor]["switch_id"],
                approaching_trains=set(),
            )
            graph.add_edge(
                neighbor,
                new_node,
                rail_nodes=[],
            )
            graph.add_edge(
                node,
                new_node,
                rail_nodes=[],
            )
            graph.remove_edge(node, neighbor)
    return graph


def prune_non_switches(graph: nx.Graph) -> nx.Graph:
    assert (
        not graph.is_directed()
    ), "Only applicable for undirected graphs. But given graph is directed."
    for node in list(graph.nodes):
        node_degree = graph.degree(node)
        neighbors_degrees = set([graph.degree(n) for n in graph.neighbors(node)])
        if node_degree == 2 and neighbors_degrees == set([2]):
            prev_node, next_node = list(graph.neighbors(node))

            graph.add_edge(
                prev_node,
                next_node,
                rail_nodes=[
                    node,
                    *graph.edges[(prev_node, node)]["rail_nodes"],
                    *graph.edges[(node, next_node)]["rail_nodes"],
                ],
            )
            graph.remove_edge(prev_node, node)
            graph.remove_edge(node, next_node)
            graph.remove_node(node)
    return graph


def generate_local_switch_graphs(graph: nx.Graph) -> nx.Graph:
    rel_pos = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    directions = "NSEW"
    directions = dict(zip(rel_pos, directions))
    dir_combo = np.array(
        [
            "NN",
            "NE",
            "NS",
            "NW",
            "EN",
            "EE",
            "ES",
            "EW",
            "SN",
            "SE",
            "SS",
            "SW",
            "WN",
            "WE",
            "WS",
            "WW",
        ]
    )

    def find_degree_nodes(G: nx.Graph, k: int, rel=lambda x, y: x == 2) -> bool:
        res = [node for node in G.nodes if rel(G.degree(node), k)]
        return res

    visited = set()
    while True:
        nodes = find_degree_nodes(graph, 2, lambda x, y: x > y)  # switches
        if (
            set(graph.nodes) == set(nodes)
            or set(nodes) == visited
            or (set(nodes) - visited) == set()
        ):
            break
        node = (set(nodes) - visited).pop()
        visited.add(node)

        allowed_transitions = graph.nodes[node]["transition"]
        switch_graph = graph.subgraph(nx.ego_graph(graph, node))

        # get relative positions of the subgraphs
        for current_node, next_node in combinations(switch_graph.nodes, 2):
            if current_node == node or next_node == node:  # consider only combinations of switch proximity nodes
                continue
            visited.add(current_node)
            visited.add(next_node)
            node_pos = graph.nodes[node]["position"]
            current_node_pos = graph.nodes[current_node]["position"]
            next_node_pos = graph.nodes[next_node]["position"]

            train_facing = tuple(np.sign((node_pos - current_node_pos)).tolist())
            train_going = tuple(np.sign((next_node_pos - node_pos)).tolist())
            train_transition = directions[train_facing] + directions[train_going]
            trans_idx = np.where(dir_combo == train_transition)[0].item()
            if allowed_transitions[trans_idx] == "1":
                graph.add_edge(current_node, next_node)

            train_facing = directions[
                tuple(np.sign((node_pos - next_node_pos)).tolist())
            ]
            train_going = directions[
                tuple(np.sign((current_node_pos - node_pos)).tolist())
            ]
            trans_idx = np.where(dir_combo == (train_facing + train_going))[0].item()
            if allowed_transitions[trans_idx] == "1":
                graph.add_edge(
                    next_node, current_node, rail_nodes=[], rail_node_to_switch={}
                )
        graph.remove_node(node)
    return graph


def add_rail_actions(graph: nx.Graph) -> nx.Graph:
    """adds actions to each node.
    Each action annotation is stored at the 'action' datafield as a dictionary:
        - key: where the action leads to
        - value: action sequence

    Assume port-node naming within switch node x:

                N
            (x.1,x.1)
        W       |        E
    (x.2,x.2)---+----(x.0,x.0)
                |
            (x.3,x.3)
                S

    Example:
    >>> switch_graph = add_rail_actions(switch_graph)
    >>> switch_graph.nodes.data('actions')[<source-node>][<target-node>]

    Args:
        graph (nx.Graph): switch graph without actions  # LOCAL SWITCH GRAPH

    Returns:
        nx.Graph: switch graph with actions
    """
    actions = {}

    for i, incoming in enumerate(graph.nodes):
        incoming_decimal = round((incoming[0] - int(incoming[0])) * 10)
        # transform [1, 2, 3, 4] -> [0, 1, 2, 3] (modulo operation)
        incoming_decimal -= 1
        actions[incoming] = {"actions": {}}  # graph.nodes.data()[incoming]
        for j, target in enumerate(graph.nodes):
            if incoming == target or target not in graph.neighbors(incoming):
                continue  # Cannot go back to where you came from
            target_decimal = round((target[0] - int(target[0])) * 10)  # x.y -> y
            target_decimal -= 1

            action = [RailEnvActions.MOVE_FORWARD]  # enter switch
            if (incoming_decimal + 1) % 4 == target_decimal:  #
                action.append(RailEnvActions.MOVE_RIGHT)
            elif (incoming_decimal + 2) % 4 == target_decimal:
                action.append(RailEnvActions.MOVE_FORWARD)
            elif (incoming_decimal + 3) % 4 == target_decimal:
                action.append(RailEnvActions.MOVE_LEFT)
            else:
                raise ValueError(
                    f"No action possible to go from: {incoming=} to {target=}"
                )
            actions[incoming]["actions"][target] = action

    nx.set_node_attributes(G=graph, values=actions)
    return graph
