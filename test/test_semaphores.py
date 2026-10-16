"""Semaphores (port reservations) along single-track lines.

A train entering a single-track line that runs through switches without a choice in either direction
(e.g. crossings) has to reserve the whole line: two trains meeting head-on on it can never pass each other.
At a merge an oncoming train has a choice (e.g. the other track of a passing loop), so the line ends there.
Several trains going the same way can hold the same port; each one only releases its own reservation.
"""

import logging

import numpy as np
import pytest

from switchfl.baselines import random_policy, shortest_path_policy
from switchfl.core import SwitchCore
from switchfl.observer import check_port_blocked
from switchfl.utils.build_env import build_standard_async_env

from test.test_learning_readiness import _run_core

logging.disable(logging.CRITICAL)


def _build(seed):
    return build_standard_async_env(height=30, width=30, max_num_cities=3, num_trains=6, seed=seed)


def test_forced_route_runs_through_crossings():
    # seed 8: switch 20-8 is a crossing and 20-9 offers no choice either, so leaving 20-5 eastwards commits
    # the train to the whole line up to 5-27
    rail_network = _build(8).rail_network
    assert rail_network.forced_route((20.1, 5.1)) == [
        ((20.1, 5.1), (20.3, 8.3)),
        ((20.1, 8.1), (20.3, 9.3)),
        ((20.1, 9.1), (5.1, 27.1)),
    ]


def test_forced_route_ends_at_passing_loop():
    # seed 3: 1-9 and 3-12 are joined by two parallel tracks; an oncoming train can use the other track,
    # so the route ends at the merge and a train may wait inside the loop
    rail_network = _build(3).rail_network
    assert rail_network.forced_route((1.4, 9.4)) == [((1.4, 9.4), (3.3, 12.3))]
    assert rail_network.forced_route((1.1, 9.1)) == [((1.1, 9.1), (3.2, 12.2))]


def test_whole_line_reserved_and_blocks_oncoming_train():
    env = _build(8)
    env.reset(seed=8)
    rail_network = env.rail_network
    rail_network.semaphores = {}
    route = rail_network.forced_route((20.1, 5.1))
    now = env.rail_env._elapsed_steps
    for hop_out, hop_in in route:
        rail_network.reserve(hop_out, 'out', 0, now, now + 50)
        rail_network.reserve(hop_in, 'in', 0, now, now + 50)
    # a train at 5-27 heading west onto the far end of the line is blocked
    assert check_port_blocked((20.1, 9.1), (5.1, 27.1), 1, rail_network)
    # a train following train 0 eastwards is not
    assert not check_port_blocked((20.3, 8.3), (20.1, 5.1), 1, rail_network)


def test_following_trains_keep_their_reservations():
    env = _build(8)
    env.reset(seed=8)
    rail_network = env.rail_network
    rail_network.semaphores = {}
    now = env.rail_env._elapsed_steps
    # trains 0 and 1 both head east through the segment (20.1, 5.1) -> (20.3, 8.3)
    for handle in (0, 1):
        rail_network.reserve((20.1, 5.1), 'out', handle, now, now + 50)
        rail_network.reserve((20.3, 8.3), 'in', handle, now, now + 50)
    rail_network.release_train(1)  # e.g. train 1 arrived
    assert set(rail_network.semaphores[(20.3, 8.3)]) == {0}
    # the oncoming train 2 is still blocked by train 0
    assert check_port_blocked((20.1, 5.1), (20.3, 8.3), 2, rail_network)


def _line_hops(rail_network, train):
    """Hops of the single-track line the train is committed to, from its current segment on."""
    prev_port = rail_network._train_prev_port[train.handle]
    if train.position is None or prev_port is None or rail_network._train2next_port[train.handle] is None:
        return set()
    return set(rail_network.forced_route(prev_port))


@pytest.mark.parametrize("policy_name", ["random", "shortest_path"])
@pytest.mark.parametrize("seed", range(6))
def test_no_head_on_trains_on_a_single_track_line(seed, policy_name):
    core = SwitchCore(_build(seed))
    env, rail_network = core.env, core.env.rail_network
    policy = random_policy(np.random.default_rng(seed)) if policy_name == "random" else shortest_path_policy
    violations = []

    move_trains = env._move_trains

    def checked_move_trains():
        move_trains()
        hops = {train.handle: _line_hops(rail_network, train) for train in env.rail_env.agents}
        for a in hops:
            for b in hops:
                if a < b and any((hop_in, hop_out) in hops[b] for hop_out, hop_in in hops[a]):
                    violations.append((env.rail_env._elapsed_steps, a, b))

    env._move_trains = checked_move_trains
    _run_core(core, seed, policy)
    assert violations == []
