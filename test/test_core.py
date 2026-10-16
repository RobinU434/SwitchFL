"""Tests for the decision-event interface in switchfl.core."""

import logging
from collections import Counter, defaultdict

import numpy as np
import pytest

from flatland.envs.step_utils.states import TrainState

from switchfl.core import SwitchCore, arrival_bonus, train_delay, truncation_penalty
from switchfl.utils.build_env import build_standard_async_env

logging.disable(logging.CRITICAL)


def _run(seed, p_stop=0.0, reward="delay", reward_terms=()):
    """Random masked policy. Returns the core, the (decision, action) pairs, all transitions and
    each train's delay at its first decision."""
    core = SwitchCore(
        build_standard_async_env(height=30, width=30, max_num_cities=3, num_trains=3, seed=seed),
        reward=reward,
        reward_terms=reward_terms,
    )
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


@pytest.mark.parametrize("seed,p_stop", RUNS)
def test_arrival_reward(seed, p_stop):
    core, decisions, transitions, _ = _run(seed, p_stop, reward="arrival")
    rail_env = core.env.rail_env
    arrived = {train.handle for train in rail_env.agents if train.state == TrainState.DONE}
    # 1 exactly for the decision after which the train arrived, 0 for every other decision
    for transition in transitions:
        assert transition.reward == float(transition.terminal)
    assert {t.train for t in transitions if t.reward == 1.0} == arrived & {d.train for d, _ in decisions}
    # no STOP penalty and team rewards count arrivals
    assert sum(t.team_reward for t in transitions) >= len({t.train for t in transitions if t.terminal})
    assert all(t.team_reward >= 0 for t in transitions)


def test_unknown_reward_is_rejected():
    env = build_standard_async_env(height=30, width=30, max_num_cities=3, num_trains=3, seed=0)
    with pytest.raises(AssertionError):
        SwitchCore(env, reward="speed")


@pytest.mark.parametrize("seed,p_stop", RUNS)
def test_reward_terms_add_bonuses_and_penalties(seed, p_stop):
    _, _, base, _ = _run(seed, p_stop)
    contexts = []
    terms = [arrival_bonus(1000), truncation_penalty(500), lambda ctx: contexts.append(ctx) or -1.0]
    _, _, shaped, _ = _run(seed, p_stop, reward_terms=terms)
    # the terms only change `reward`, the episode itself is the same
    assert [(t.switch, t.train, t.action, t.tau, t.team_reward) for t in base] == [
        (t.switch, t.train, t.action, t.tau, t.team_reward) for t in shaped
    ]
    for b, t in zip(base, shaped):
        assert t.reward == pytest.approx(b.reward + 1000 * t.terminal - 500 * t.truncated - 1.0)
    assert len(contexts) == len(shaped)
    for ctx, t in zip(contexts, shaped):
        assert (ctx.decision.switch, ctx.train.handle, ctx.action) == (t.switch, t.train, t.action)
        assert (ctx.tau, ctx.terminal, ctx.truncated) == (t.tau, t.terminal, t.truncated)


def test_arrival_bonus_on_time_only():
    contexts = []
    _, _, transitions, _ = _run(0, reward_terms=[lambda ctx: contexts.append(ctx) or 0.0])
    on_time = arrival_bonus(1.0, on_time_only=True)
    for ctx in contexts:
        expected = ctx.terminal and ctx.train.arrival_time <= ctx.train.latest_arrival
        assert on_time(ctx) == float(expected)
