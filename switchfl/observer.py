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

    min_dist_to_target = rail_env.distance_map.get(rail_env.agents)[
        train.handle, row, col, direction
    ]
    
    if np.isinf(min_dist_to_target):
        raise ValueError("Infinite distance to target encountered.")

    # shortest_path = rail_env.distance_map.get_shortest_paths(max_depth=None, agents = rail_env.agents, agent_handle=train.handle)[train.handle]
    # min_dist_to_target = len(shortest_path) - 1
    
    if earliest_departure:
        delay = train.earliest_departure - train.latest_arrival + min_dist_to_target
    else:
        delay = rail_env._elapsed_steps - train.latest_arrival + min_dist_to_target
    # assume you are moving with max speed=1
    return delay

def check_port_blocked(next_port, out_port, train_agent_handle, rail_network: RailNetwork) -> bool:
    """ Checks if the given port is blocked by a semaphore.
    
        Args:   
            next_port (int): The id of the next port.
            out_port (int): The id of the out port.
            train_agent_handle (int): The handle of the train agent.
            rail_network (RailNetwork): The rail network."""

    port_blocked = False
    if next_port is not None:
        if next_port in rail_network.semaphores:
            if rail_network.semaphores[next_port][0] != train_agent_handle \
                and rail_network.semaphores[next_port][1] == 'out' \
                    and rail_network.semaphores[next_port][2] == rail_network.map_direction(next_port) \
                        and rail_network.semaphores[next_port][3] <= rail_network.rail_env._elapsed_steps \
                            and rail_network.semaphores[next_port][4] >= rail_network.rail_env._elapsed_steps:
                                port_blocked = True

            elif rail_network.semaphores[next_port][0] != train_agent_handle \
                and rail_network.semaphores[next_port][1] == 'out' \
                    and rail_network.semaphores[next_port][2] != rail_network.map_direction(next_port) \
                        and rail_network.semaphores[next_port][3] <= rail_network.rail_env._elapsed_steps \
                            and rail_network.semaphores[next_port][4] >= rail_network.rail_env._elapsed_steps \
                                and rail_network.rail_env.agents[rail_network.semaphores[next_port][0]].state == TrainState.MALFUNCTION:
                                    port_blocked = True

            elif rail_network.semaphores[next_port][0] != train_agent_handle \
                and rail_network.semaphores[next_port][1] == 'in' \
                    and rail_network.semaphores[next_port][2] != rail_network.map_direction(next_port) \
                        and rail_network.semaphores[next_port][3] <= rail_network.rail_env._elapsed_steps \
                            and rail_network.semaphores[next_port][4] >= rail_network.rail_env._elapsed_steps:
                                port_blocked = True

            elif rail_network.semaphores[next_port][0] != train_agent_handle \
                and rail_network.semaphores[next_port][1] == 'in' \
                    and rail_network.semaphores[next_port][2] == rail_network.map_direction(next_port) \
                        and rail_network.semaphores[next_port][3] <= rail_network.rail_env._elapsed_steps \
                            and rail_network.semaphores[next_port][4] >= rail_network.rail_env._elapsed_steps and \
                                rail_network.rail_env.agents[rail_network.semaphores[next_port][0]].state == TrainState.MALFUNCTION:
                                    port_blocked = True
    
        if not port_blocked:
            if out_port in rail_network.semaphores:
                if rail_network.semaphores[out_port][0] != train_agent_handle \
                    and rail_network.semaphores[out_port][1] == 'out' \
                        and rail_network.semaphores[out_port][2] != rail_network.map_direction(out_port) \
                            and rail_network.semaphores[out_port][3] <= rail_network.rail_env._elapsed_steps \
                                and rail_network.semaphores[out_port][4] >= rail_network.rail_env._elapsed_steps:
                                    port_blocked = True

                elif rail_network.semaphores[out_port][0] != train_agent_handle \
                    and rail_network.semaphores[out_port][1] == 'out' \
                        and rail_network.semaphores[out_port][2] == rail_network.map_direction(out_port) \
                            and rail_network.semaphores[out_port][3] <= rail_network.rail_env._elapsed_steps \
                                and rail_network.semaphores[out_port][4] >= rail_network.rail_env._elapsed_steps \
                                    and rail_network.rail_env.agents[rail_network.semaphores[out_port][0]].state == TrainState.MALFUNCTION:
                                        port_blocked = True                

                elif rail_network.semaphores[out_port][0] != train_agent_handle \
                    and rail_network.semaphores[out_port][1] == 'in' \
                        and rail_network.semaphores[out_port][2] == rail_network.map_direction(out_port) \
                            and rail_network.semaphores[out_port][3] <= rail_network.rail_env._elapsed_steps \
                                and rail_network.semaphores[out_port][4] >= rail_network.rail_env._elapsed_steps:
                                    port_blocked = True

                elif rail_network.semaphores[out_port][0] != train_agent_handle \
                    and rail_network.semaphores[out_port][1] == 'in' \
                        and rail_network.semaphores[out_port][2] != rail_network.map_direction(out_port) \
                            and rail_network.semaphores[out_port][3] <= rail_network.rail_env._elapsed_steps \
                                and rail_network.semaphores[out_port][4] >= rail_network.rail_env._elapsed_steps \
                                    and rail_network.rail_env.agents[rail_network.semaphores[out_port][0]].state == TrainState.MALFUNCTION:
                                        port_blocked = True
                
    # handle the case in which the stop moving action makes the out_port == in_port
    else:
        if out_port in rail_network.semaphores:
            if rail_network.semaphores[out_port][0] != train_agent_handle \
                and rail_network.semaphores[out_port][1] == 'out' \
                    and rail_network.semaphores[out_port][2] == rail_network.map_direction(out_port) \
                        and rail_network.semaphores[out_port][3] <= rail_network.rail_env._elapsed_steps \
                            and rail_network.semaphores[out_port][4] >= rail_network.rail_env._elapsed_steps:
                                port_blocked = True

            elif rail_network.semaphores[out_port][0] != train_agent_handle \
                and rail_network.semaphores[out_port][1] == 'out' \
                    and rail_network.semaphores[out_port][2] != rail_network.map_direction(out_port) \
                        and rail_network.semaphores[out_port][3] <= rail_network.rail_env._elapsed_steps \
                            and rail_network.semaphores[out_port][4] >= rail_network.rail_env._elapsed_steps \
                                and rail_network.rail_env.agents[rail_network.semaphores[out_port][0]].state == TrainState.MALFUNCTION:
                                    port_blocked = True

            elif rail_network.semaphores[out_port][0] != train_agent_handle \
                and rail_network.semaphores[out_port][1] == 'in' \
                    and rail_network.semaphores[out_port][2] != rail_network.map_direction(out_port) \
                        and rail_network.semaphores[out_port][3] <= rail_network.rail_env._elapsed_steps \
                            and rail_network.semaphores[out_port][4] >= rail_network.rail_env._elapsed_steps:
                                port_blocked = True

            elif rail_network.semaphores[out_port][0] != train_agent_handle \
                and rail_network.semaphores[out_port][1] == 'in' \
                    and rail_network.semaphores[out_port][2] == rail_network.map_direction(out_port) \
                        and rail_network.semaphores[out_port][3] <= rail_network.rail_env._elapsed_steps \
                            and rail_network.semaphores[out_port][4] >= rail_network.rail_env._elapsed_steps \
                                and rail_network.rail_env.agents[rail_network.semaphores[out_port][0]].state == TrainState.MALFUNCTION:
                                    port_blocked = True

    return port_blocked

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
        active_train_handle = env.active_train
        train = env.rail_env.agents[active_train_handle]
        
        for port in switch.get_port_nodes():  # Get port IDs of this node
            self.logger.debug(f"port {port} coming from cell {switch.switch_graph.nodes.data('rail_prev_node')[port]}")

            _, next_port = rail_network.get_neighbor_switch(port)

            port_blocked = check_port_blocked(next_port, port, train.handle, rail_network)
            if port_blocked:
                semaphore.append(0)
            else:
                semaphore.append(1)

            self.logger.debug(f"Semaphore: {semaphore}")

            if train_at_ports[active_train_handle] == port:
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

        if (
            train_counter == 0
        ):  


            self.logger.fatal       (
                "Bug detected. No train detected at active switch. Check information in RailNetwork."
            )

        semaphore = np.array(semaphore).astype(int)
        target = np.array(target).astype(int)
        delay = np.array(delay).astype(int)
        observation = np.concatenate([node_id, semaphore, target, delay], dtype=np.int64)

        semaphore_int = [bool2int(semaphore)]
        reduced_obs = np.concatenate([node_id, semaphore_int, target[target != -1], delay[delay != -1], [active_train_handle], [env.rail_env._elapsed_steps]], dtype=np.int64)

        info = {"action_mask": switch.get_action_mask(current_port, semaphore), "active_train": active_train_handle, 
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
