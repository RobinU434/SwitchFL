from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Tuple

from gymnasium import Space
from numpy import ndarray

from switchfl.rail_network import RailNetwork
from switchfl.spaces import MultiDiscreteSwitchObsSpace
from switchfl.switch_agents import _Switch
from switchfl.utils.logging import format_logger
from switchfl.utils.naming import name2switch_id, symmetric_string, get_node_id_on_port_id
from flatland.envs.rail_env import RailEnv
from flatland.envs.agent_utils import EnvAgent as TrainAgent
from flatland.envs.step_utils.states import TrainState
import numpy as np

def bool2int(x):
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y


def compute_delay(rail_env: RailEnv, train: TrainAgent, position, direction, earliest_departure=False) -> int:
    """
    Returns the delay of the given train

    Args:
        train_id (int): The id of the train

    Returns:
        int: The delay of the given train
    """

    row, col = position

    min_dist_to_target = rail_env.distance_map.get()[
        train.handle, row, col, direction
    ]
    
    if np.isinf(min_dist_to_target):
        raise ValueError("Infinite distance to target encountered.")

    
    if earliest_departure:
        delay = train.earliest_departure - train.latest_arrival + min_dist_to_target
    else:
        delay = rail_env._elapsed_steps - train.latest_arrival + min_dist_to_target
    # assume you are moving with max speed=1
    return delay

def reachable_action_mask(switch: _Switch, train: TrainAgent, rail_env: RailEnv, rail_network: RailNetwork) -> np.ndarray:
    """Masks rail actions after which the train can no longer reach its target.

    Leaving the switch through out-port p puts the train on the cell rail_prev_node[p],
    heading in map_direction(p). Without this mask a train can be routed onto a branch
    with infinite distance to its target and compute_delay raises.

    Args:
        switch (_Switch): the switch the train is in front of
        train (TrainAgent): the active train
        rail_env (RailEnv): the flatland environment
        rail_network (RailNetwork): the rail network

    Returns:
        np.ndarray: 1 if the action keeps the target reachable, else 0 (n_rail_actions, )
    """
    distance_map = rail_env.distance_map.get()
    mask = []
    for _, out_port in switch.action_outcomes:
        row, col = rail_network.rail_graph.nodes[out_port]["rail_prev_node"]
        direction = rail_network.map_direction(out_port)
        mask.append(int(np.isfinite(distance_map[train.handle, row, col, direction])))
    return np.array(mask, dtype=np.int8)

def check_route_blocked(out_port, train_agent_handle, rail_network: RailNetwork) -> bool:
    """Checks if leaving a switch through out_port is blocked anywhere on the forced route behind it, i.e.
    up to the next switch where the train has a choice (see RailNetwork.forced_route). Checking only the first
    hop lets trains drive head-on into a chain of crossings/merges and deadlock there."""
    return any(
        check_port_blocked(hop_in, hop_out, train_agent_handle, rail_network)
        for hop_out, hop_in in rail_network.forced_route(out_port)
    )


def check_port_blocked(next_port, out_port, train_agent_handle, rail_network: RailNetwork) -> bool:
    """ Checks if the segment between out_port (exit of the current switch) and next_port (in-port of the next
    switch) is blocked for the given train by another train's reservation.

    Opposing traffic always blocks: a train leaving the next switch through next_port ('out' at next_port) or
    approaching the current switch through out_port ('in' at out_port). A train going the same way ('in' at
    next_port, 'out' at out_port) only blocks while it is broken down.

        Args:   
            next_port (int): The id of the next port. None if the train stops (out_port is its in-port).
            out_port (int): The id of the out port.
            train_agent_handle (int): The handle of the train agent.
            rail_network (RailNetwork): The rail network."""
    now = rail_network.rail_env._elapsed_steps

    def blocked_by(port, opposing_kind):
        for handle, (_, kind, _, t_start, t_end) in rail_network.semaphores.get(port, {}).items():
            if handle == train_agent_handle or not t_start <= now <= t_end:
                continue
            if kind == opposing_kind or rail_network.rail_env.agents[handle].state == TrainState.MALFUNCTION:
                return True
        return False

    if next_port is None:
        # the stop action makes out_port == in_port
        return blocked_by(out_port, 'out')
    return blocked_by(next_port, 'out') or blocked_by(out_port, 'in')


class _Observer(ABC):
    def __init__(self):
        self.logger = logging.getLogger(type(self).__name__)
        self.logger = format_logger(self.logger)


    @abstractmethod
    def observe(
        self, agent: str, rail_env: RailEnv, rail_network: RailNetwork
    ) -> Tuple[ndarray, Dict[str, Any]]:
        """Observe the environment and return the observation for the given agent.

        Args:
            agent (str): The ID of the agent.
            rail_env (RailEnv): The rail environment.
            rail_network (RailNetwork): The rail network.

        Raises:
            NotImplementedError: If the method is not implemented.

        Returns:
            Tuple[ndarray, Dict[str, Any]]: The observation for the agent.
        """
        raise NotImplementedError

    @abstractmethod
    def get_observation_space(
        self, agent: str, rail_env: RailEnv, rail_network: RailNetwork, seed: int = None
    ) -> Space:
        """Get the observation space for the given agent.

        Args:
            agent (str): The ID of the agent.
            rail_env (RailEnv): The rail environment.
            rail_network (RailNetwork): The rail network.
            seed (int, optional): The random seed. Defaults to None.

        Raises:
            NotImplementedError: If the method is not implemented.

        Returns:
            Space: The observation space for the agent.
        """
        raise NotImplementedError

    def human_format(self, observation: Any) -> Dict[str, Any]:
        """Convert the raw observation into a human-readable format.

        Args:
            observation (Any): The raw observation.

        Returns:
            Dict[str, Any]: The human-readable observation.
        """
        if not hasattr(self, "obs_space"):
            self.logger.error(
                "No observation space initialized -> Not able to convert observation"
            )
            return
        if not hasattr(self.obs_space, "human_format"):
            self.logger.error(
                "Given observation space has not method to reinterpret the given observation -> Abort"
            )
            return
        return self.obs_space.human_format(observation)
    

class StandardObserver(_Observer):
    def __init__(self, delay_levels: int = 3, delay_threshold: int = 20):
        super().__init__()

        self.delay_levels = delay_levels

        self.delay_threshold = delay_threshold  # arbitrary value

    def _discretize_delay(self, train: TrainAgent, delay: int) -> int:
        """
        Discretizes the delay of the given train

        Args:
            train_id (TrainAgent): The id of the train
            delay (int): The delay of the train

        Returns:
            int: The discretized delay of the given train
        """
        available_time = train.latest_arrival - train.earliest_departure
        if delay <= 0:
            return 0
        if delay <= available_time * self.delay_threshold:
            return 1
        return 2

    def observe(self, agent, env, rail_network: RailNetwork) -> Tuple[ndarray, Dict[str, Any]]:
        """Observe the environment and return the observation for the given agent.
        Args:
            agent (str): The ID of the agent.
            env (RailEnv): The rail environment.
            rail_network (RailNetwork): The rail network."""
        
        self.logger.debug(symmetric_string(f"obs {agent}", frame="~"))
        node_id = name2switch_id(agent)
        switch: _Switch = rail_network.get_switch_on_position(node_id)

        semaphore = []
        target = []
        delay = []
        
        train_at_ports = rail_network._train2next_port

        self.logger.debug(f"next port for each train: {rail_network._train2next_port}")
        self.logger.debug(f"current agent: {agent} with position {switch.id}")
        train_counter = 0  # debugging
        # only the selected switch has a train in front of it. Every other switch (and every switch
        # outside agent_iter()) gets a "no train" observation: targets and delays -1, only STOP allowed
        has_train = env.active_train is not None and agent == env.agent_selection
        active_train_handle = env.active_train if has_train else -1
        train = env.rail_env.agents[active_train_handle] if has_train else None
        
        for port in switch.get_port_nodes():  # Get port IDs of this node
            self.logger.debug(f"port {port} coming from cell {switch.switch_graph.nodes.data('rail_prev_node')[port]}")

            _, next_port = rail_network.get_neighbor_switch(port)

            port_blocked = check_route_blocked(port, active_train_handle, rail_network)
            if port_blocked:
                semaphore.append(0)
            else:
                semaphore.append(1)

            self.logger.debug(f"Semaphore: {semaphore}")

            if has_train and train_at_ports[active_train_handle] == port:
                current_port = port
                self.logger.debug(f"train at port {port}")
                delay.append(
                    self._discretize_delay(train, compute_delay(env.rail_env, train, train.position, train.direction))
                )
                target.extend(train.target)
                train_counter += 1
            else:
                delay.append(-1)
                target.extend([-1, -1])  # 2D coordinates

        if has_train and train_counter == 0:
            self.logger.fatal(
                "Bug detected. No train detected at active switch. Check information in RailNetwork."
            )

        semaphore = np.array(semaphore).astype(int)
        target = np.array(target).astype(int)
        delay = np.array(delay).astype(int)
        observation = np.concatenate([node_id, semaphore, target, delay], dtype=np.int64)

        semaphore_int = [bool2int(semaphore)]
        reduced_obs = np.concatenate([node_id, semaphore_int, target[target != -1], delay[delay != -1], [active_train_handle], [env.rail_env._elapsed_steps]], dtype=np.int64)

        if has_train:
            action_mask = switch.get_action_mask(current_port, semaphore)
            action_mask[:-1] &= reachable_action_mask(switch, train, env.rail_env, rail_network)
            if env.stop_forced(switch.id, active_train_handle):
                # another train enters or occupies this switch cell, or the train is broken down
                action_mask[:-1] = 0
        else:
            action_mask = np.zeros(switch.n_actions + 1, dtype=np.int8)
            action_mask[-1] = 1

        info = {"action_mask": action_mask, "active_train": active_train_handle,
                "reduced_obs": reduced_obs}
        return observation, info
    
    def get_observation_space(self, agent, rail_env, rail_network, seed: int = None):
        """Get the observation space for the given agent.
        Args:
            agent (str): The ID of the agent.
            rail_env (RailEnv): The rail environment.
            rail_network (RailNetwork): The rail network.
            seed (int, optional): The random seed. Defaults to None."""
        node_id = name2switch_id(agent)
        switch = rail_network.get_switch_on_position(node_id)
        return MultiDiscreteSwitchObsSpace(
            n_gaits=switch.n_gaits,
            rail_grid_shape=(rail_env.rail.height, rail_env.rail.width),
            n_delay_levels=3,
            seed=seed,
        )
