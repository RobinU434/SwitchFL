"""The env's own reward (StandardRewardFunction) can be configured and extended (issue #1)."""

import logging

import numpy as np
import pytest

from switchfl.reward_func import StandardRewardFunction
from switchfl.switch_env import ASyncSwitchEnv
from switchfl.utils.build_env import build_standard_async_env
from switchfl.utils.naming import switch_id2name

logging.disable(logging.CRITICAL)


class _ConstantBonus(StandardRewardFunction):
    def extra_reward(self, train, train_actions, port_blocked, curr_delay) -> float:
        return 7.0


def _rewards(seed, reward_func=None, p_stop=0.3):
    """Random masked policy. Returns the env reward of every decision, with and without a stop."""
    env = build_standard_async_env(height=30, width=30, max_num_cities=3, num_trains=3, seed=seed)
    if reward_func is not None:
        env = ASyncSwitchEnv(env.rail_env, reward_func=reward_func(env.rail_env))
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    rewards = []
    for agent in env.agent_iter():
        env.observe(agent)
        mask = env.infos[agent]["action_mask"]
        action = len(mask) - 1 if rng.random() < p_stop else int(rng.choice(np.flatnonzero(mask)))
        stopped_while_free = action == len(mask) - 1 and bool(np.any(mask[:-1]))
        train = env.active_train
        info = env.step(action)
        rewards.append((env._cumulative_rewards[switch_id2name(info["next_switch"])][train], stopped_while_free))
    return rewards


@pytest.mark.parametrize("seed", range(3))
def test_extra_reward_and_stop_penalty(seed):
    base = _rewards(seed)
    assert any(stopped for _, stopped in base)

    bonus = _rewards(seed, reward_func=_ConstantBonus)
    assert [r + 7.0 for r, _ in base] == pytest.approx([r for r, _ in bonus])

    no_penalty = _rewards(seed, reward_func=lambda rail_env: StandardRewardFunction(rail_env, stop_penalty=0))
    # the env penalises a STOP when at least one port is free; the mask can be stricter (e.g. an occupied
    # switch cell), so the penalty may also hit stops that were forced by the mask, never more than that
    for (r, _), (r0, _) in zip(base, no_penalty):
        assert r0 - r in (pytest.approx(0.0), pytest.approx(1300.0))
    assert any(r0 - r == pytest.approx(1300.0) for (r, _), (r0, _) in zip(base, no_penalty))
