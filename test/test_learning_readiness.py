"""Properties a learner relies on, checked over full episodes with several policies.

- every observation lies in the agent's observation space, masks have the right shape and allow STOP
- rewards are finite
- episodes end, and the number of decisions per episode is bounded
- reusing one env for many episodes gives the same results as a fresh env (no state leaks between episodes)
- a decision has the consequence the mask promised: after a rail action the train's next decision is at the
  switch behind the chosen exit, after STOP it is at the same switch
- the same also holds with Flatland malfunctions enabled (a broken-down train may only STOP); the only exception
  is a malfunction that starts in the step right after the decision
- a shortest-path policy delivers clearly more trains than a random one (the reward signal is learnable)
"""

import logging

import numpy as np
import pytest
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.step_utils.states import TrainState

from switchfl.baselines import random_policy, shortest_path_policy
from switchfl.core import SwitchCore
from switchfl.utils.build_env import build_standard_async_env
from switchfl.utils.naming import name2switch_id, switch_id2name

logging.disable(logging.CRITICAL)

SEEDS = range(4)


def _build(seed, num_trains=4, size=30, cities=3, malfunctions=False):
    malfunction_generator = None
    if malfunctions:
        malfunction_generator = ParamMalfunctionGen(
            MalfunctionParameters(malfunction_rate=1 / 30, min_duration=2, max_duration=6)
        )
    return build_standard_async_env(
        height=size, width=size, max_num_cities=cities, num_trains=num_trains, seed=seed,
        malfunction_generator=malfunction_generator,
    )


def _run_core(core, seed, policy):
    """Runs one episode on a SwitchCore. Returns the (decision, action) pairs and all transitions."""
    decision = core.reset(seed=seed)
    decisions, transitions = [], []
    while decision is not None:
        action = policy(core.env, decision.switch, decision.train, decision.mask)
        decisions.append((decision, action))
        decision, closed = core.act(action)
        transitions.extend(closed)
    return decisions, transitions


def _signature(decisions, transitions):
    return (
        [(d.switch, d.train, d.t, a, d.obs.tolist()) for d, a in decisions],
        [(t.switch, t.train, t.action, t.reward, t.tau, t.next_switch, t.terminal, t.truncated) for t in transitions],
    )


@pytest.mark.parametrize("malfunctions", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_observations_masks_and_rewards(seed, malfunctions):
    env = _build(seed, malfunctions=malfunctions)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    n_decisions = 0
    for agent in env.agent_iter():
        obs = env.observe(agent)
        mask = np.asarray(env.infos[agent]["action_mask"])
        assert env.observation_space(agent).contains(obs), f"{agent}: {obs}"
        assert mask.shape == (env.action_space(agent).n,)
        assert mask[-1] == 1, "STOP must always be allowed"
        assert set(np.unique(mask)) <= {0, 1}
        action = int(rng.choice(np.flatnonzero(mask)))
        assert env.action_space(agent).contains(action)
        env.step(action)
        n_decisions += 1
        assert all(np.isfinite(r) for rewards in env._cumulative_rewards.values() for r in rewards.values())
    assert env.terminated
    # one decision per train and Flatland step is the upper bound (a stopped train is asked every step)
    assert n_decisions <= env.rail_env._max_episode_steps * len(env.rail_env.agents)


def test_observation_space_samples_are_contained():
    env = _build(0)
    env.reset(seed=0)
    for agent in env.agents:
        space = env.observation_space(agent)
        for _ in range(5):
            assert space.contains(space.sample())


@pytest.mark.parametrize("malfunctions", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_core_rewards_finite_and_episode_ends(seed, malfunctions):
    core = SwitchCore(_build(seed, malfunctions=malfunctions))
    decisions, transitions = _run_core(core, seed, random_policy(np.random.default_rng(seed)))
    assert core.done
    assert len(transitions) == len(decisions)
    for transition in transitions:
        assert np.isfinite(transition.reward) and np.isfinite(transition.team_reward)
    if malfunctions:
        assert core.env.num_malfunctions > 0, "malfunction rate too low to test anything"


@pytest.mark.parametrize("malfunctions", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_decision_leads_to_the_promised_switch(seed, malfunctions):
    core = SwitchCore(_build(seed, malfunctions=malfunctions))
    _, transitions = _run_core(core, seed, random_policy(np.random.default_rng(seed)))
    rail_network = core.env.rail_network
    checked = unforeseen_malfunctions = 0
    for transition in transitions:
        if transition.next_switch is None:
            continue
        switch = rail_network.get_switch_on_position(name2switch_id(transition.switch))
        if transition.action == switch.n_actions:  # STOP
            expected = transition.switch
        else:
            out_port = switch.action_outcomes[transition.action][1]
            expected = switch_id2name(rail_network.get_neighbor_switch(out_port)[0].id)
        if transition.next_switch != expected:
            # only allowed if the train broke down in the step right after the decision (can not be masked):
            # Flatland keeps it in front of the switch, the transition is undone and the same switch is asked again
            assert malfunctions and transition.next_switch == transition.switch and transition.tau == 1, transition
            unforeseen_malfunctions += 1
        checked += 1
    assert checked > 0
    assert unforeseen_malfunctions <= 0.05 * checked


def test_reused_env_matches_fresh_env():
    # a learner resets one env many times: earlier episodes must not influence later ones
    policy = random_policy
    reused = SwitchCore(_build(0))
    for seed in (11, 12, 13):
        _run_core(reused, seed, policy(np.random.default_rng(seed)))
    result_reused = _signature(*_run_core(reused, 14, policy(np.random.default_rng(14))))
    result_fresh = _signature(*_run_core(SwitchCore(_build(0)), 14, policy(np.random.default_rng(14))))
    assert result_reused == result_fresh


def test_many_trains_episode():
    core = SwitchCore(_build(1, num_trains=10, size=40, cities=4))
    decisions, transitions = _run_core(core, 1, random_policy(np.random.default_rng(1)))
    assert core.done and len(transitions) == len(decisions)


def test_shortest_path_beats_random():
    arrived = {"random": 0, "shortest_path": 0}
    for seed in range(6):
        for name in arrived:
            core = SwitchCore(_build(seed))
            policy = random_policy(np.random.default_rng(seed)) if name == "random" else shortest_path_policy
            _run_core(core, seed, policy)
            arrived[name] += sum(t.state == TrainState.DONE for t in core.env.rail_env.agents)
    assert arrived["shortest_path"] >= 2 * max(arrived["random"], 1), arrived
