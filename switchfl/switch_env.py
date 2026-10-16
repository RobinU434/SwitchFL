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
from flatland.envs.timetable_utils import Line
from flatland.utils.rendertools import AgentRenderVariant
from pettingzoo import AECEnv
from switchfl import NodeId, TrainAgentHandle
from switchfl.observer import StandardObserver, _Observer, compute_delay, check_route_blocked
from switchfl.rail_network import RailNetwork
from switchfl.utils.flatland_compat import apply_action, first_valid_move, path_exists
from switchfl.utils.logging import format_logger, set_seed
from switchfl.utils.naming import name2switch_id, switch_id2name, symmetric_string, get_node_id_on_port_id
from switchfl.reward_func import StandardRewardFunction

import time


class _FrozenRailGenerator:
    """Wraps a Flatland rail generator and returns the first generated rail on every reset.

    RailNetwork is built once from the rail, so the rail must never change afterwards.
    Flatland's reset(regenerate_rail=False) can not be used instead because then the
    rail generator hints (city positions) needed by sparse_line_generator are missing.
    """

    def __init__(self, rail_generator):
        self.rail_generator = rail_generator
        self.product = None

    def generate(self, width, height, num_agents, num_resets=0, np_random=None):
        if self.product is None:
            if hasattr(self.rail_generator, "generate"):
                self.product = self.rail_generator.generate(width, height, num_agents, num_resets, np_random)
            else:
                self.product = self.rail_generator(width, height, num_agents, num_resets, np_random)
        return self.product


class _FeasibleLineGenerator:
    """Wraps a Flatland line generator and replaces trains that can not reach their target.

    sparse_line_generator picks random city pairs. If no start orientation leads to the target,
    Flatland silently starts the train facing North (SparseLineGen.decide_orientation), often on
    a cell without a northern exit. On large maps with many cities the rail generator leaves some
    city pairs unconnected, these trains have an infinite distance to their target and
    compute_delay raises. Replacements are the next feasible trains drawn with the same np_random,
    so a line without infeasible trains is returned unchanged and seeding stays deterministic.
    """

    def __init__(self, line_generator, max_tries: int = 100):
        self.line_generator = line_generator
        self.max_tries = max_tries

    @staticmethod
    def feasible(rail, waypoints) -> bool:
        # waypoints: [[start], [intermediate stops]..., [target]], every leg must be drivable
        for leg_start, leg_end in zip(waypoints, waypoints[1:]):
            start, end = leg_start[0], leg_end[0]
            if not path_exists(rail, start.position, start.direction, end.position):
                return False
        return True

    def __call__(self, rail, num_agents, hints=None, num_resets=0, np_random=None):
        agent_waypoints, agent_speeds = [], []
        for _ in range(self.max_tries):
            line = self.line_generator(rail, num_agents, hints, num_resets, np_random)
            feasible = [h for h, waypoints in sorted(line.agent_waypoints.items()) if self.feasible(rail, waypoints)]
            if not agent_waypoints and len(feasible) == len(line.agent_waypoints):
                return line
            for handle in feasible[:num_agents - len(agent_waypoints)]:
                agent_waypoints.append(line.agent_waypoints[handle])
                agent_speeds.append(line.agent_speeds[handle])
            if len(agent_waypoints) == num_agents:
                return Line(agent_waypoints=dict(enumerate(agent_waypoints)), agent_speeds=agent_speeds)
        raise RuntimeError(
            f"could not place {num_agents} trains with a reachable target in {self.max_tries} tries, "
            f"only {len(agent_waypoints)} found; the rail network is probably badly connected")


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

        # the switch graph below is built once, so pin the rail for all future resets
        if not isinstance(rail_env.rail_generator, _FrozenRailGenerator):
            rail_env.rail_generator = _FrozenRailGenerator(rail_env.rail_generator)
        if not isinstance(rail_env.line_generator, _FeasibleLineGenerator):
            rail_env.line_generator = _FeasibleLineGenerator(rail_env.line_generator)
        self.rail_network = RailNetwork(rail_env)
        self._rail_grid = rail_env.rail.grid.copy()
        self.reward_func = StandardRewardFunction(self.rail_env)

        self.observer = observer if observer is not None else StandardObserver()
        self.possible_agents = self.rail_network.get_switch_names()
        self.agents = self.possible_agents

        self.active_switch_agents = []
        self.active_trains = []
        self.switch_entries: Dict[NodeId, TrainAgentHandle] = {}
        """switch cell -> train sent into it in the current tick (enters with the next Flatland step)"""

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
        assert np.array_equal(self.rail_env.rail.grid, self._rail_grid), \
            "rail changed after the switch graph was built"

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
        self.switch_entries = {}
        
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
            assert not self.stop_forced(node_id, train_agent_handle), \
                f"train {train_agent_handle} can not enter {agent_selection} now, only STOP is allowed"
            self.switch_entries[node_id] = train_agent_handle
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
            port_blocked = [check_route_blocked(out_port, train_agent_handle, self.rail_network)]
        else:
            # every exit counts as blocked if the train could not enter the switch anyway (STOP was forced)
            switch_taken = self.stop_forced(node_id, train_agent_handle)
            port_blocked = []
            for p in current_switch.action_outcomes:
                if p[0] == in_port:
                    out_port = p[1]
                    _, next_port = self.rail_network.get_neighbor_switch(out_port)
                    port_blocked.append(
                        switch_taken or check_route_blocked(out_port, train_agent_handle, self.rail_network)
                    )

        reward, curr_delay = self.reward_func(self.rail_env.agents[train_agent_handle],
                                                self.train_action_plan[train_agent_handle],
                                                self.train_to_last_node,
                                                port_blocked)

        self._cumulative_rewards[switch_id2name(next_switch.id)][self.active_train] = reward
        
        self.train_to_last_node[train_agent_handle] = (node_id, curr_delay)

        self.logger.debug(f"updated_plan: {self.train_action_plan}")
        return next_switch.id

    def switch_cell_taken(self, node_id: NodeId, train_handle: TrainAgentHandle) -> bool:
        """True if another train was sent into this switch cell in the current tick or stands on it.

        Flatland moves only one train into a cell per step and never into a cell a train stays on, so the
        action mask allows only STOP in both cases. A train standing on the cell may still leave it in the
        same step; waiting one tick then is the price for never having a switch decision overridden.
        """
        if self.switch_entries.get(node_id, train_handle) != train_handle:
            return True
        return any(train.position == node_id and train.handle != train_handle for train in self.rail_env.agents)

    def stop_forced(self, node_id: NodeId, train_handle: TrainAgentHandle) -> bool:
        """True if the train can not enter the switch in the next Flatland step, so only STOP is allowed:
        the switch cell is taken (see switch_cell_taken) or the train is broken down (malfunction).

        A malfunction that starts during the next step can not be foreseen; in that case Flatland keeps the
        train in place and the switch's transition is undone (RailNetwork.rollback_transition).
        """
        if self.rail_env.agents[train_handle].state == TrainState.MALFUNCTION:
            return True
        return self.switch_cell_taken(node_id, train_handle)

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
                new_position, _, transition_valid = apply_action(
                    self.rail_env.rail, train_actions[handle], train.position, train.direction
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
                            f"Train {train.handle} was stopped by Flatland (cell conflict). Expected pos: {expected_pos}, Actual pos: {actual_pos}"
                        )
                        self.train_action_plan[train.handle].insert(0, train_actions[train.handle])
                        if self.rail_network.get_switch_on_position(expected_pos) is not None:
                            # the train is still in front of the switch: undo the switch's transition
                            # (ports and semaphores), the switch is asked again in _check_active_switch()
                            if not self.rail_network.rollback_transition(train):
                                source_port = self.rail_network._train_source_port[train.handle]
                                self.rail_network.set_trains_next_port(train, source_port)
                elif transition_valid and self.rail_network.get_switch_on_position(actual_pos) is not None:
                    self.rail_network.commit_transition(train)
            
            # Reset semaphores of trains that have arrived
            if self.train_done[train.handle]:
                self.rail_network.release_train(train.handle)

        # Set semamphores of trains that are about to depart
        for train in self.rail_env.agents:
            if self.rail_env._elapsed_steps == train.earliest_departure - 2:
                port = self.rail_network._train2next_port[train.handle]
                self.rail_network.reserve(port, 'in', train.handle, train.earliest_departure - 2,
                                          train.earliest_departure + self.rail_network._train2next_port_dist[train.handle])
                
        # Extend semaphores of stopped trains
        self.rail_network.extend_semaphores()

        self.rail_env_time += 1
        # the trains sent into switches in this tick have now entered (or were stopped by Flatland)
        self.switch_entries = {}
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
            new_position, _, _ = apply_action(
                self.rail_env.rail, next_action, train.position, train.direction
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
            if train.position is None:
                # not departed yet or already arrived
                continue
            next_port = self.rail_network.get_trains_next_port(train)
            prev_port = self.rail_network.get_trains_prev_port(train)
            if next_port is None or prev_port is None:
                # train has not passed a switch yet or has no next port planned
                continue
            source_node = self.rail_network.get_switch_on_port(prev_port).id
            target_node = self.rail_network.get_switch_on_port(next_port).id
            rail_pieces = self.rail_network.get_rail_pieces_between_ports(prev_port, next_port)
            if train.position not in [*rail_pieces, source_node, target_node]:
                self.logger.error(
                    f"Train {train.handle} deviated from planned path! Position {train.position} is not between "
                    f"port {prev_port} and port {next_port}"
                )

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
                current_position, current_direction, _ = apply_action(
                    self.rail_env.rail,
                    first_valid_move(self.rail_env.rail, last_pos, last_dir),
                    current_position,
                    current_direction,
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
            self.rail_network.reserve(port, 'in', train.handle, train.earliest_departure - 2,
                                      train.earliest_departure + self.rail_network._train2next_port_dist[train.handle])

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
