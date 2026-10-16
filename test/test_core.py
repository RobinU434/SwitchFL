"""Tests for the decision-event interface in switchfl.core."""

import logging
from collections import Counter, defaultdict

import numpy as np
import pytest

from switchfl.core import SwitchCore, train_delay
from switchfl.utils.build_env import build_standard_async_env

logging.disable(logging.CRITICAL)


def _run(seed, p_stop=0.0):
    """Random masked policy. Returns the core, the (decision, action) pairs, all transitions and
    each train's delay at its first decision."""
    core = SwitchCore(build_standard_async_env(height=30, width=30, max_num_cities=3, num_trains=3, seed=seed))
    rng = np.random.default_rng(seed)
    decision = core.reset(seed=seed)
    first_delay = {}
    decisions, transitions = [], []
    while decision is not None:
        rail_env = core.env.rail_env
        first_delay.setdefault(decision.train, train_delay(rail_env, rail_env.agents[decision.train]))
        if rng.random() < p_stop:
            action = len(decision.mask) - 1
        else:
            action = int(rng.choice(np.flatnonzero(decision.mask)))
        decisions.append((decision, action))
        decision, closed = core.act(action)
        transitions.extend(closed)
    return core, decisions, transitions, first_delay


RUNS = [(0, 0.0), (1, 0.0), (2, 0.3), (3, 0.9)]


@pytest.mark.parametrize("seed,p_stop", RUNS)
def test_every_decision_closes_exactly_once(seed, p_stop):
    core, decisions, transitions, _ = _run(seed, p_stop)
    assert core.done
    assert len(transitions) == len(decisions)
    assert Counter((d.switch, d.train, a) for d, a in decisions) == Counter(
        (t.switch, t.train, t.action) for t in transitions
    )


@pytest.mark.parametrize("seed,p_stop", RUNS)
def test_transition_fields(seed, p_stop):
    _, _, transitions, _ = _run(seed, p_stop)
    for transition in transitions:
        assert transition.tau >= 1
        assert not (transition.terminal and transition.truncated)
        if transition.next_switch is None:
            assert transition.terminal or transition.truncated
        else:
            assert not transition.terminal and not transition.truncated
            assert transition.next_obs is not None and transition.next_mask[-1] == 1


@pytest.mark.parametrize("seed,p_stop", RUNS)
def test_rewards_telescope_per_train(seed, p_stop):
    core, decisions, transitions, first_delay = _run(seed, p_stop)
    penalties = defaultdict(float)
    for decision, action in decisions:
        if action == len(decision.mask) - 1 and decision.mask[:-1].any():
            penalties[decision.train] += core.stop_penalty
    rewards = defaultdict(float)
    for transition in transitions:
        rewards[transition.train] += transition.reward
    rail_env = core.env.rail_env
    for handle, delay in first_delay.items():
        final = train_delay(rail_env, rail_env.agents[handle])
        assert rewards[handle] + penalties[handle] == pytest.approx(delay - final)


def test_episode_reaches_max_steps_and_truncates():
    core, _, transitions, _ = _run(4, p_stop=0.95)
    rail_env = core.env.rail_env
    assert rail_env._elapsed_steps == rail_env._max_episode_steps
    assert any(t.truncated for t in transitions)
