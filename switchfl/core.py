"""Decision-event interface on top of ASyncSwitchEnv.

SwitchFL is a semi-MDP: a switch decides for one train, and the consequence of that decision is
only known once the train reaches its next decision point (or arrives), tau Flatland steps later.
SwitchCore hands out one Decision at a time (in the same order as ASyncSwitchEnv.agent_iter())
and returns every Transition that got closed by an action.

Rewards are credited to the switch that acted. Two reward functions are available (`reward=`):

"delay" (default): delay differences of the train between its two decision points

    reward      = D_k(t) - D_k(t') - stop_penalty * [STOP chosen although a rail action was allowed]
    team_reward = sum_j D_j(t) - sum_j D_j(t')

with D_k(t) = max(t, earliest_departure if not departed) - latest_arrival + distance to target,
or arrival_time - latest_arrival once the train is done. Summed over a train's transitions,
reward + stop penalties telescopes to D_k(first decision) - D_k(end).

"arrival": a vanilla sparse reward, the train reached its target

    reward      = 1 if the train arrived at its target before its next decision point, else 0
    team_reward = number of trains that arrived between t and t'
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.envs.step_utils.states import TrainState

from switchfl.switch_env import ASyncSwitchEnv


@dataclass
class Decision:
    """One (switch, train) decision point."""

    switch: str
    train: int
    obs: np.ndarray
    mask: np.ndarray
    t: int
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Transition:
    """Closed once the train reaches its next decision point, arrives, or the episode ends."""

    switch: str
    train: int
    obs: np.ndarray
    action: int
    reward: float
    team_reward: float
    tau: int
    next_switch: Optional[str]
    next_obs: Optional[np.ndarray]
    next_mask: Optional[np.ndarray]
    terminal: bool
    """the train arrived at its target"""
    truncated: bool
    """the episode ended before the train arrived"""


@dataclass
class _Pending:
    decision: Decision
    action: int
    stop_penalty: float
    train_delay: float
    total_delay: float
    n_arrived: int


def train_delay(rail_env: RailEnv, train) -> float:
    """Current delay estimate of a train, assuming it travels at full speed from now on."""
    if train.state == TrainState.DONE:
        return train.arrival_time - train.latest_arrival
    if train.position is None:
        position, direction = train.initial_position, train.initial_direction
        t = max(rail_env._elapsed_steps, train.earliest_departure)
    else:
        position, direction = train.position, train.direction
        t = rail_env._elapsed_steps
    distance = rail_env.distance_map.get()[train.handle, position[0], position[1], direction]
    return t - train.latest_arrival + distance


def total_delay(rail_env: RailEnv) -> float:
    return sum(train_delay(rail_env, train) for train in rail_env.agents)


def n_arrived(rail_env: RailEnv) -> int:
    return sum(train.state == TrainState.DONE for train in rail_env.agents)


class SwitchCore:
    REWARDS = ("delay", "arrival")

    def __init__(self, env: ASyncSwitchEnv, stop_penalty: float = 1300.0, reward: str = "delay"):
        """
        Args:
            env (ASyncSwitchEnv): the environment to wrap
            stop_penalty (float): "delay" reward only: penalty for STOP although a rail action was allowed
            reward (str): "delay" (dense, delay differences) or "arrival" (sparse, 1 when the train arrives)
        """
        assert reward in self.REWARDS, f"reward must be one of {self.REWARDS}, got {reward!r}"
        self.env = env
        self.stop_penalty = stop_penalty
        self.reward = reward
        self._agent_iter = None
        self._decision: Optional[Decision] = None
        self._pending: Dict[int, _Pending] = {}

    @property
    def done(self) -> bool:
        return self._decision is None

    def reset(self, seed: int = None) -> Optional[Decision]:
        """Resets the environment and returns the first decision (None if the episode is already over)."""
        self.env.reset(seed=seed)
        self._agent_iter = self.env.agent_iter()
        self._pending = {}
        self._decision = self._next_decision()
        return self._decision

    def act(self, action: int) -> Tuple[Optional[Decision], List[Transition]]:
        """Applies the action of the current decision.

        Returns:
            Tuple[Optional[Decision], List[Transition]]: the next decision (None once the episode is over)
                and all transitions closed in the meantime
        """
        assert self._decision is not None, "episode is over, call reset()"
        decision = self._decision
        assert decision.mask[action], f"action {action} is masked for {decision.switch}"

        stopped_while_free = (
            self.reward == "delay" and action == len(decision.mask) - 1 and decision.mask[:-1].any()
        )
        self._pending[decision.train] = _Pending(
            decision=decision,
            action=int(action),
            stop_penalty=self.stop_penalty if stopped_while_free else 0.0,
            train_delay=train_delay(self.env.rail_env, self.env.rail_env.agents[decision.train]),
            total_delay=total_delay(self.env.rail_env),
            n_arrived=n_arrived(self.env.rail_env),
        )
        self.env.step(action)

        transitions = []
        self._decision = self._next_decision()
        if self._decision is not None:
            transitions.append(self._close(self._decision.train, self._decision))
        # trains that arrived, or every remaining one if the episode is over
        for handle in list(self._pending):
            train = self.env.rail_env.agents[handle]
            if self._decision is None or train.state == TrainState.DONE:
                transitions.append(self._close(handle, None))
        return self._decision, [t for t in transitions if t is not None]

    def _next_decision(self) -> Optional[Decision]:
        agent = next(self._agent_iter, None)
        if agent is None:
            return None
        obs = self.env.observe(agent)
        info = self.env.infos[agent]
        return Decision(
            switch=agent,
            train=self.env.active_train,
            obs=obs,
            mask=np.asarray(info["action_mask"]).copy(),
            t=self.env.rail_env._elapsed_steps,
            info=dict(info),
        )

    def _close(self, handle: int, next_decision: Optional[Decision]) -> Optional[Transition]:
        pending = self._pending.pop(handle, None)
        if pending is None:
            return None
        rail_env = self.env.rail_env
        train = rail_env.agents[handle]
        arrived = train.state == TrainState.DONE
        if self.reward == "arrival":
            reward = float(arrived)
            team_reward = float(n_arrived(rail_env) - pending.n_arrived)
        else:
            reward = pending.train_delay - train_delay(rail_env, train) - pending.stop_penalty
            team_reward = pending.total_delay - total_delay(rail_env)
        return Transition(
            switch=pending.decision.switch,
            train=handle,
            obs=pending.decision.obs,
            action=pending.action,
            reward=reward,
            team_reward=team_reward,
            tau=(train.arrival_time if arrived else rail_env._elapsed_steps) - pending.decision.t,
            next_switch=next_decision.switch if next_decision is not None else None,
            next_obs=next_decision.obs if next_decision is not None else None,
            next_mask=next_decision.mask if next_decision is not None else None,
            terminal=arrived,
            truncated=not arrived and next_decision is None,
        )
