import functools
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.agent_utils import EnvAgent as TrainAgent
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.step_utils.states import TrainState
from flatland.utils.rendertools import AgentRenderVariant
from pettingzoo import AECEnv
from switchfl import NodeId, TrainAgentHandle
from switchfl.observer import StandardObserver, _Observer, compute_delay, check_port_blocked
from switchfl.rail_network import RailNetwork
from switchfl.utils.logging import format_logger, set_seed
from switchfl.utils.naming import name2switch_id, switch_id2name, symmetric_string, get_node_id_on_port_id
from switchfl.reward_func import StandardRewardFunction

import time


class _SwitchEnv:
    metadata = {"render_mode": ["human", "rgb_array", None]}

    def __init__(
        self,
        rail_env: RailEnv,
        max_steps: int = None,
        render_mode: str = None,
        observer: _Observer = None,
        seed: int = None,
    ):
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)
        self.logger = format_logger(self.logger)

        self.rail_env = rail_env
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.seed = seed

        self.rail_network = RailNetwork(rail_env)
        self.reward_func = StandardRewardFunction(self.rail_env)

        self.observer = observer if observer is not None else StandardObserver()
        self.possible_agents = self.rail_network.get_switch_names()
        self.agents = self.possible_agents

        self.active_switch_agents = []
        self.active_trains = []

        self.terminated: bool
        self.truncated: bool
        self.train_action_plan: Dict[TrainAgentHandle, List[RailEnvActions]]

        self.rail_env_time: int
        self.train_done: Dict[TrainAgentHandle, bool]
        self.train_obs: Dict[TrainAgentHandle, Any]
        self.train_reward: Dict[TrainAgentHandle, float]
        self.train_info: Dict[TrainAgentHandle, Any]

        self.flatland_step_time = 0.
        self.step_time = 0.
        self.last_time = 0.
        self.action_selection_time = 0.
        self.update_time = 0.
        self.reset_time = 0.
        self.reset_total_time = 0.

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        obs_space = self.observer.get_observation_space(
            agent=agent,
            rail_env=self.rail_env,
            rail_network=self.rail_network,
            seed=self.seed,
        )
        return obs_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        node_id = name2switch_id(agent)
        action_space = self.rail_network.get_switch_action_space(
            node_id, seed=self.seed
        )
        return action_space

    def reset(self, seed=None, options=None):
        # Set all possible random seeds if provided - need to do this before rail_env.reset()
        # as Flatland may use various random number generators internally during reset
        if seed is not None:
            set_seed(seed)
        start_reset_time = time.time()
        self.rail_env.reset(random_seed=seed)
        self.reset_time += time.time() - start_reset_time
        
        # NOTE: Force deterministic agent ordering and handle assignment
        # This works around Flatland's non-deterministic train assignment
        if seed is not None and hasattr(self.rail_env, 'agents') and len(self.rail_env.agents) > 1:
            # Sort agents by initial position and then by direction (as secondary key)
            # This ensures trains at same position maintain consistent direction ordering
            def sort_key(agent):
                pos = agent.initial_position or (0, 0)
                dir_val = agent.initial_direction.value if hasattr(agent.initial_direction, 'value') else int(agent.initial_direction)
                return pos + (dir_val,)
            
            sorted_agents = sorted(self.rail_env.agents, key=sort_key)
            
            # Reassign handles deterministically based on sorted order
            for new_handle, agent in enumerate(sorted_agents):
                agent.handle = new_handle
            
            # Update the agents list to be in sorted order
            self.rail_env.agents = sorted_agents
        
        
        self.rail_network.reset()

        self.terminated = False
        self.truncated = False

        self.terminations = {switch_name: False for switch_name in self.agents}
        self.truncations = {switch_name: False for switch_name in self.agents}
        self.rewards = {switch_name: 0 for switch_name in self.agents}
        self._cumulative_rewards = {switch_name: {train.handle : 0 for train in self.rail_env.agents} for switch_name in self.agents}
        self.infos = {switch_name: {} for switch_name in self.agents}
        self.step_counter = {switch_name: 0 for switch_name in self.agents}

        self.train_action_plan = {train.handle: [] for train in self.rail_env.agents}

        self.rail_env_time = 0
        self.train_done = {train.handle: False for train in self.rail_env.agents}
        self.train_obs = {train.handle: None for train in self.rail_env.agents}
        self.train_reward = {train.handle: 0.0 for train in self.rail_env.agents}
        self.train_info = {train.handle: None for train in self.rail_env.agents}
        self.malfunctions = []
        self.num_malfunctions = 0

        self.active_switch_agents = []
        self.active_trains = []
        
        self.agent_selection = None
        self.active_train = None

        self.prev_actions = {train.handle: None for train in self.rail_env.agents}
        self.train_to_last_node = {train.handle: (None, compute_delay(
            self.rail_env, train, train.initial_position, train.initial_direction, earliest_departure=True)) for train in self.rail_env.agents}

        self._init_ports()
        self._move_trains_to_switch()
        

        self.reset_total_time += time.time() - start_reset_time

    def render(self):
        if self.render_mode is None:
            return

        rgb = self.rail_env.render(
            agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS, show_debug=True
        )
        if self.render_mode == "rgb_array":
            return rgb
        elif self.render_mode == "human":
            fig, (ax1, ax2) = plt.subplots(ncols=2)
            ax1.imshow(rgb)
            nx.draw(
                self.rail_network.rail_graph.to_undirected(),
                self.rail_network.rail_graph.nodes.data("position"),
                with_labels=True,
                node_color=dict(
                    self.rail_network.rail_graph.nodes.data(data="node_color")
                ).values(),
                edge_color="gray",
                node_size=3,
                font_size=5,
                ax=ax2,
            )
            ax1.axis("off")
            ax2.axis("off")
            plt.show()

    def obs2human(self, agent: str, observation: Any) -> Dict[str, Any]:
        obs_space = self.observation_space(agent)
        human_format = getattr(obs_space, "human_format", None)
        if not callable(human_format):
            self.logger.error(
                f"Observation space: {obs_space} has not method to convert observation into a human format."
            )
            return
        return human_format(observation)

    def obs2json(self, agent: str, observation: Any) -> Dict[str, Any]:
        obs = self.obs2human(agent, observation)
        obs = {k: v.tolist() for k, v in obs.items()}
        return obs

    def _apply_action(self, agent_selection: str, action: int) -> NodeId:
        """get train actions and update semaphores at the corresponding switches for a SINGLE train transitioning a switch

        Args:
            agent_selection (str): which switch agent is performing the action
            action (int): which actions is performed

        Returns:
            NodeId: to which switch the action is sending the train
        """
        assert self.action_space(agent_selection).contains(
            action
        ), f"Invalid action performed. Allowed action space: {self.action_space(agent_selection)}"

        self.logger.debug(symmetric_string(agent_selection))
        self.logger.debug(f"rail time: {self.rail_env_time}")

        node_id = name2switch_id(agent_selection)

        current_switch = self.rail_network.get_switch_on_position(node_id)

        # update train_action_plan such that move_trains step can work it down
        moving_train, next_train_actions = self.rail_network.get_train_actions(
            node_id, action, self.active_train
        )

        train_agent_handle = self.active_train

        # if the action is stop moving take the next port corresponding to next action
        if next_train_actions[0] == RailEnvActions.STOP_MOVING:
            # in and out port of the traversing switch
            in_port = self.rail_network._train2next_port[train_agent_handle]
            out_port = in_port
        else:
            in_port, out_port = current_switch.action_outcomes[action]

        # # transition a train if there is actually a train moving
        if moving_train is not None:
            moving_train = self.rail_env.agents[moving_train]

            next_switch, next_port = self.rail_network.transition_train(
                moving_train, in_port, out_port
            )
        else:
            next_switch = current_switch
            next_port = None
        
        self.logger.debug(f"existing plan: {self.train_action_plan}")
        self.logger.debug(f"new actions: {next_train_actions}")

        # adapt next train actions
        # NOTE: if two switches are direct neighbors, the first given train action for the moving train
        # has to be replaced by the last action of the train_action plan because otherwise the environment
        # will send the train forward which messes up the scheduling and planning          
        if (
            moving_train is not None
            and len(self.train_action_plan[train_agent_handle]) > 0
        ):
            # cut away the MOVE_FORWARD action of the new plan, because
            # moving into the new switch is done by the action from the previous switch
            next_train_actions.pop(0)
            if len(self.train_action_plan[train_agent_handle]) > 1:
                self.train_action_plan[train_agent_handle] = self.train_action_plan[train_agent_handle][:1] 
            self.train_action_plan[train_agent_handle].extend(next_train_actions)
        elif (
            moving_train is None
        ):
            self.train_action_plan[train_agent_handle].insert(0, RailEnvActions.STOP_MOVING)
        else:
            self.train_action_plan[train_agent_handle].extend(next_train_actions)

        if moving_train is not None:
            port_blocked = [check_port_blocked(next_port, out_port, train_agent_handle, self.rail_network)]
        else:
            port_blocked = []
            for p in current_switch.action_outcomes:
                if p[0] == in_port:
                    out_port = p[1]
                    _, next_port = self.rail_network.get_neighbor_switch(out_port)
                    port_blocked.append(check_port_blocked(next_port, out_port, train_agent_handle, self.rail_network))

        reward, curr_delay = self.reward_func(self.rail_env.agents[train_agent_handle],
                                                self.train_action_plan[train_agent_handle],
                                                self.train_to_last_node,
                                                port_blocked)

        self._cumulative_rewards[switch_id2name(next_switch.id)][self.active_train] = reward
        
        self.train_to_last_node[train_agent_handle] = (node_id, curr_delay)

        self.logger.debug(f"updated_plan: {self.train_action_plan}")
        return next_switch.id

    def _move_trains(self):
        """Move all train agents in the RailEnv by one step
        Also take into account if there is a next action, pre-computed for a train to transit the switch
        """
        # NOTE: the time when the train departures is already taken into account
        train_actions = {}
        train_new_positions = {}

        for train in self.rail_env.agents:
            handle = train.handle
            if self.train_done[handle]:
                # no action for already done trains
                continue

            if len(self.train_action_plan[handle]) == 0:
                # use base action
                train_actions[handle] = RailEnvActions.MOVE_FORWARD
            else:
                # use predetermined action
                self.prev_actions[handle] = self.train_action_plan[handle][0]
                train_actions[handle] = self.train_action_plan[handle].pop(0)

            if train.position is not None:
                (
                    _,
                    (new_position,
                    _),
                    transition_valid,
                    _,
                ) = self.rail_env.rail.check_action_on_agent(
                    train_actions[handle],
                    ((train.position),
                    train.direction)
                )
                if transition_valid:
                    train_new_positions[handle] = (new_position, True)
                    self.logger.debug(
                        f"Train {handle} has valid transition with action {train_actions[handle]}"
                        f" from position {train.position} and direction {train.direction} to position {new_position}")
                else:
                    self.logger.debug(
                        f"Train {handle} has INVALID transition with action {train_actions[handle]}"
                        f" from position {train.position} and direction {train.direction} to position {new_position}")
                    train_new_positions[handle] = (train.position, False)

        start_time = time.time()
        # do rail env step
        (
            self.train_obs,
            self.train_reward,
            self.train_done,
            self.train_info,
        ) = self.rail_env.step(train_actions)

        self.flatland_step_time += time.time() - start_time

        # Correct if flatland has stopped some trains that simultaneously try to enter the same cell
        for train in self.rail_env.agents:
            if train.handle in train_new_positions:
                expected_pos = train_new_positions[train.handle][0]
                transition_valid = train_new_positions[train.handle][1]
                actual_pos = train.position
                if expected_pos != actual_pos and transition_valid:

                    if train_actions[train.handle] != RailEnvActions.STOP_MOVING:
                        self.logger.debug(
                            f"Train {train.handle} deviated from planned path! Expected pos: {expected_pos}, Actual pos: {actual_pos}"
                        )
                        self.train_action_plan[train.handle].insert(0, train_actions[train.handle])
                        if self.rail_network.get_switch_on_position(expected_pos) is not None:
                            source_port = self.rail_network._train_source_port[train.handle]
                            self.rail_network.set_trains_next_port(train, source_port)
            
            # Reset semaphores of trains that have arrived
            if self.train_done[train.handle]:
                todel=[]
                for p, (tr, _, _, _, _) in self.rail_network.semaphores.items():
                    if tr == train.handle:
                        todel.append(p)
                for p in todel:
                    del self.rail_network.semaphores[p]

        # Set semamphores of trains that are about to depart
        for train in self.rail_env.agents:
            if self.rail_env._elapsed_steps == train.earliest_departure - 2:
                port = self.rail_network._train2next_port[train.handle]
                self.rail_network.semaphores[port] = [train.handle, 'in', self.rail_network.map_direction(port),
                                                        train.earliest_departure - 2,
                                                        train.earliest_departure + self.rail_network._train2next_port_dist[train.handle]]
                
        # Extend semaphores of stopped trains
        self.rail_network.extend_semaphores()

        self.rail_env_time += 1
        self._check_action_execution()

        # check for all trains being done
        if self.train_done["__all__"]:
            self.terminations = {
                switch_agent: True for switch_agent in self.terminations.keys()
            }
            self.terminated = True

        new_trains_malfunctions = np.nonzero([v for v in self.train_info['malfunction'].values()])[0]
        self.num_malfunctions += len(set(new_trains_malfunctions).difference(set(self.malfunctions)))
        self.malfunctions = new_trains_malfunctions

    def _move_trains_to_switch(self):
        self.logger.debug(symmetric_string("move trains", 80, "-"))
        while len(self.active_switch_agents) == 0 and not self.terminated:
            # NOTE: after resetting the environment all trains are in a standstill
            # -> they have to be moved first after the reset
            self._move_trains()
            self._check_active_switch()

        # remove duplicates in agents but maintaining order
        train_positions = {
            t.handle: (t.position, Grid4TransitionsEnum(t.direction).name)
            for t in self.rail_env.agents
        }
        self.logger.debug(f"Train pos: {train_positions}")

        sorting_order = np.argsort(self.active_trains)
        self.active_trains = sorted(self.active_trains)

        self.active_switch_agents = np.array(self.active_switch_agents)[np.array(sorting_order)].tolist()

        self.logger.debug(f"active_switches: {self.active_switch_agents}")
        self.logger.debug(f"active_trains: {self.active_trains}")


    def _check_active_switch(self):
        """do simulation step and see if a train enters a switch node
        -> then add the switch to active agents
        """

        for train in self.rail_env.agents:
            # train is not one the grid yet or if it waiting don't execute something on it.
            if train.position is None or train.state == TrainState.WAITING:
                continue

            # get first next action
            if len(self.train_action_plan[train.handle]) > 0:
                next_action = self.train_action_plan[train.handle][0]
            else:
                next_action = RailEnvActions.MOVE_FORWARD

            # do a simulation step an get the next switch if needed
            (
                new_cell_valid,
                (new_position,
                new_direction),
                transition_valid,
                preprocessed_action,
            ) = self.rail_env.rail.check_action_on_agent(
                next_action,
                ((train.position),
                train.direction)
            )

            # if with the next action the train has entered a switch add the switch to the active switches
            next_switch = self.rail_network.get_switch_on_position(new_position)
            self.logger.debug(
                f"Next switch for train {train.handle} ({train.state.name, train.position}): {next_switch.id if next_switch is not None else None}"
            ) 

            if next_switch is not None:
                if (
                train.state == TrainState.READY_TO_DEPART
                or train.state == TrainState.MOVING
                ):
                    # use new pos because the switch coordinates are its node_id
                    switch_id = switch_id2name(new_position)
                    self.active_switch_agents.append(switch_id)
                    self.active_trains.append(train.handle)
                elif (
                    (train.state == TrainState.STOPPED or train.state == TrainState.MALFUNCTION)
                    and self.prev_actions[train.handle] == RailEnvActions.STOP_MOVING
                ):
                    switch_id = switch_id2name(new_position)
                    self.active_switch_agents.append(switch_id)
                    self.active_trains.append(train.handle)
                # handle the case in which multiple trains want to enter the same cell
                elif (
                    (train.state == TrainState.STOPPED or train.state == TrainState.MALFUNCTION)
                    and self.prev_actions[train.handle] != RailEnvActions.STOP_MOVING
                ):
                    switch_id = switch_id2name(get_node_id_on_port_id(self.rail_network._train2next_port[train.handle]))
                    self.active_switch_agents.append(switch_id)
                    self.active_trains.append(train.handle)


    def _check_action_execution(self):
        """checks if the action has been executed successfully

        This function will log an error if a train deviates from its planned path.
        """
        for train in self.rail_env.agents:
            next_port = self.rail_network.get_trains_next_port(train)
            prev_port = self.rail_network.get_trains_prev_port(train)
            if next_port is None or prev_port is None:
                # train is not at a switch or has no next port planned
                continue
            source_node = self.rail_network.get_switch_on_port(prev_port).id
            target_node = self.rail_network.get_switch_on_port(next_port).id
            rail_pieces = self.rail_network.get_rail_pieces_between_ports(
                source_node, target_node
            )
            # if train.position not in [*rail_pieces, source_node, target_node]:
            #     self.logger.error(f"Train {train.handle} deviated from planned path!")

    def _init_ports(self):
        """executed in reset()

        finds all train positions and setups all semaphores for all switches.

        Assume:
        - all trains are already on the grid
        - at least one train is in front of a switch
        """
        for train in self.rail_env.agents:
            # simulate steps of a train until they arrive at a switch
            # NOTE: yes this is expensive, but only executed once in reset()
            current_position = train.position
            current_direction = train.direction
            last_pos = train.old_position
            last_dir = train.old_direction
            if current_position is None or current_direction is None:
                current_position = train.initial_position
                current_direction = train.initial_direction

            distance = 0
            while True:
                if (            
                    self.rail_network.get_switch_on_position(current_position)
                    is not None
                ):
                    # train on switch
                    switch = self.rail_network.get_switch_on_position(current_position)
                    break
                last_pos = current_position
                last_dir = current_direction
                next_actions = self.rail_env.rail.get_valid_move_actions_(last_dir, last_pos)
                (
                    new_cell_valid,
                    (current_position,
                    current_direction),
                    transition_valid,
                    preprocessed_action,
                ) = self.rail_env.rail.check_action_on_agent(
                    next_actions[0].action,
                    (current_position,
                    current_direction),
                )
                distance += 1

            # last pos corresponds to rail_prev_node
            for p in switch.get_port_nodes():
                port_prev_node = self.rail_network.rail_graph.nodes.data("rail_prev_node")[p]
                if port_prev_node == last_pos:
                    port = p
                    break

            self.logger.debug(f"port: {port}")
            self.rail_network.set_trains_next_port(train, port)
            self.rail_network._train2next_port_dist[train.handle] = distance

        # Set semaphores of trains
        for train in self.rail_env.agents:
            port = self.rail_network._train2next_port[train.handle]
            self.rail_network.semaphores[port] = [train.handle, 'in', self.rail_network.map_direction(port),
                                                    train.earliest_departure - 2,
                                                    train.earliest_departure + self.rail_network._train2next_port_dist[train.handle]]

    @property
    def n_steps(self) -> int:
        """accumulated steps from each agents

        Returns:
            int: _description_
        """
        return sum(self.step_counter.values())

    def get_env_plan(self, path: Path = None):
        """Extract information about each switch as a csv from the environment:
        This csv should contain:
            - switch_id
            - action (int)
            - rail env actions (move forward, move_right, move_left, stop) with their corresponding ports
            - from where to where an action is sending a train
        """
        env_plan = []
        for _, switch in self.rail_network.switches:
            for action_idx in range(switch.n_actions):
                env_plan.append(
                    self.rail_network.get_switch_transition_info(
                        switch.id, action=action_idx
                    )
                )

        if path is not None:
            df = pd.DataFrame(env_plan)
            df.to_csv(path, index=False, sep=";")
            self.logger.info(f"Saved environment plan to {path}")
            return
        else:
            return env_plan


class ASyncSwitchEnv(_SwitchEnv, AECEnv):
    def __init__(
        self,
        rail_env,
        max_steps=None,
        render_mode=None,
        observer=None,
        seed=None,
    ):
        super().__init__(rail_env, max_steps, render_mode, observer, seed)

    def agent_iter(self, max_iter=2**63):
        while not (self.terminated or self.truncated):
            self.agent_selection = self.active_switch_agents.pop(0)
            self.active_train = self.active_trains.pop(0)
            # if len(self.active_switch_agents) == 0:
            #     print('No active switch agents left!')
            yield self.agent_selection
        if self.terminated:
            self.logger.debug(
                "termination."
            )
        elif self.truncated:
            self.logger.debug(
                "truncation."
            )

    def step(self, action) -> Dict[str, Any]:

        start_time = time.time()

        # check if current agent is still operating
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
            or action is None
        ):
            # the agent is done
            return {}

        next_switch = self._apply_action(self.agent_selection, action)

        # no switches with non processed trains left -> time to move trains
        if len(self.active_switch_agents) == 0:
            self._move_trains_to_switch()

        # check for done episode
        self.step_counter[self.agent_selection] += 1
        if self.max_steps is not None:
            if self.n_steps > self.max_steps:
                self.truncations = {
                    switch_agent: True for switch_agent in self.truncations.keys()
                }
                self.truncated = True

        arrived_trains = [train.handle for train in self.rail_env.agents \
                          if train.position == None and train.arrival_time != None]

        # prepare info dict
        post_step_info = {"next_switch": next_switch, "arrived_trains" : arrived_trains}

        self.step_time += time.time() - start_time
        return post_step_info

    def observe(self, agent) -> np.ndarray:
        obs, info = self.observer.observe(
            agent=agent,
            env=self,
            rail_network=self.rail_network,
        )
        self.infos[agent].update(info)
        return obs

    def close(self):
        return super().close()
