"""Regression tests for full episodes on stock Flatland 4.2.2.

Covers the KeyError at max_episode_steps (stale distance map), trains routed onto branches
with infinite distance to their target, trains stopped by Flatland in front of a switch
(prevented by the action mask), determinism of repeated resets and the rail
staying fixed after the switch graph is built.
"""

import logging

import numpy as np
import pytest
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator

from switchfl.switch_env import ASyncSwitchEnv
from switchfl.utils.build_env import build_standard_async_env

logging.disable(logging.CRITICAL)


def _build(seed):
    return build_standard_async_env(height=30, width=30, max_num_cities=3, num_trains=3, seed=seed)


def _run_episode(env, seed, p_stop=0.0, max_decisions=None):
    """Random masked policy. Returns the list of (switch, train, action) decisions."""
    rng = np.random.default_rng(seed)
    decisions = []
    for agent in env.agent_iter():
        env.observe(agent)
        mask = env.infos[agent]["action_mask"]
        if rng.random() < p_stop:
            action = len(mask) - 1
        else:
            action = int(rng.choice(np.flatnonzero(mask)))
        decisions.append((agent, env.active_train, action))
        env.step(action)
        if max_decisions is not None and len(decisions) >= max_decisions:
            break
    return decisions


@pytest.mark.parametrize("seed", range(5))
def test_random_episode_completes(seed):
    env = _build(seed)
    env.reset(seed=seed)
    _run_episode(env, seed)
    assert env.terminated


@pytest.mark.parametrize("seed", range(3))
def test_episode_reaches_max_episode_steps(seed):
    env = _build(seed)
    env.reset(seed=seed)
    _run_episode(env, seed, p_stop=0.9)
    assert env.terminated
    assert env.rail_env._elapsed_steps == env.rail_env._max_episode_steps


def test_distance_map_matches_handles():
    env = _build(7)
    env.reset(seed=7)
    distance_map = env.rail_env.distance_map.get()
    for handle, train in enumerate(env.rail_env.agents):
        assert train.handle == handle
        row, col = train.target
        assert np.all(distance_map[handle, row, col] == 0)


@pytest.mark.parametrize("seed", range(4))
def test_no_deviation_from_planned_path(seed):
    # 4 trains on a 30x30 map: trains regularly arrive at the same switch in the same tick or at a switch
    # another train stands on; without the mask Flatland would stop them and the env would undo the decisions
    env = build_standard_async_env(height=30, width=30, max_num_cities=3, num_trains=4, seed=seed)
    errors, rollbacks = [], []
    env.logger.error = lambda msg, *args, **kwargs: errors.append(msg)
    rollback_transition = env.rail_network.rollback_transition
    env.rail_network.rollback_transition = lambda train: rollbacks.append(rollback_transition(train)) or rollbacks[-1]
    for episode_seed in (seed, seed + 100):  # the second reset must not keep ports from the first episode
        env.reset(seed=episode_seed)
        _run_episode(env, seed)
    assert not [e for e in errors if "deviated from planned path" in e]
    # without malfunctions the action mask prevents every cell conflict at switches, so nothing is undone
    assert rollbacks == []


def test_same_seed_same_episode():
    seed = 3
    env1, env2 = _build(seed), _build(seed)
    env1.reset(seed=seed)
    env2.reset(seed=seed)
    trains1 = [(t.handle, t.initial_position, t.target, t.earliest_departure) for t in env1.rail_env.agents]
    trains2 = [(t.handle, t.initial_position, t.target, t.earliest_departure) for t in env2.rail_env.agents]
    assert trains1 == trains2
    assert np.array_equal(env1.rail_env.distance_map.get(), env2.rail_env.distance_map.get())
    assert _run_episode(env1, seed, max_decisions=50) == _run_episode(env2, seed, max_decisions=50)


def test_rail_stays_fixed_across_resets():
    # unseeded rail generator: without pinning, every reset would draw a new map
    rail_env = RailEnv(
        width=30,
        height=30,
        rail_generator=sparse_rail_generator(
            max_num_cities=3, grid_mode=True, max_rails_between_cities=1, max_rail_pairs_in_city=1, seed=None
        ),
        line_generator=sparse_line_generator(seed=0),
        number_of_agents=3,
    )
    env = ASyncSwitchEnv(rail_env)
    grid = env.rail_env.rail.grid.copy()
    for seed in [1, 2, 3]:
        env.reset(seed=seed)
        assert np.array_equal(env.rail_env.rail.grid, grid)
