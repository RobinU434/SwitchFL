# TODO

Read `CLAUDE.md` first, especially the Flatland section. Setup:

```bash
uv sync --all-extras
uv run pytest test/ -q -W ignore   # expected: 151 passed
```

The working tree contains uncommitted changes. Ask the user before committing or pushing.

Done: multi-holder semaphores reserving whole single-track lines (`forced_route`), the action mask forbidding entry into a switch cell another train enters or occupies (and rail actions for broken-down trains), the observation space covering the node id, the formerly skipped tests, `test/test_learning_readiness.py`, the `flatland_patch/` removal and the agent re-sort removal, the restored path-deviation check with a complete rollback of transitions cancelled by Flatland cell conflicts, the reachability action mask, the pinned rail (`_FrozenRailGenerator`), `switchfl/utils/flatland_compat.py`, the "no train" observation, the test-suite repair and the `SwitchCore` decision-event interface.

## 1. Adapters (only when needed)

`switchfl/core.py` (`SwitchCore`) is done. Still open: an RLlib `MultiAgentEnv` adapter on top of it (use sub-rounds when one switch has two trains in the same tick), and making `ASyncSwitchEnv` pass more of `pettingzoo.test.api_test`.

## 2. Open decisions for the user

- Remaining deadlocks (shortest-path policy, 20 seeds x 6 trains: ~13/120 trains in wait-for cycles): trains facing each other at a diverging switch whose other branch is blocked. Needs global route planning; decide whether the env should do anything here or leave it to the learners (e.g. a deadlock penalty / early termination).
  On large, dense maps it dominates: `uv run python examples/demo.py --cities 60 --trains 200 --size 120` (seed 0, shortest path) delivers only 25/200 trains in 631 Flatland steps, and 21362 of 25201 decisions are forced STOPs (no rail action allowed). Not yet analysed: how much of this is real wait-for cycles vs. trains waiting on long line reservations (`forced_route`) or occupied switch cells (`stop_forced`).

## 3. Later

- In-tick ordering: pairs in one tick are processed in train-handle order, and earlier pairs reserve semaphores first.
- Q-routing spec in `docs/q_routing.md`. Implementation only after the env is stable.
- One env per process on the same map. Threads are unsafe because `set_seed` uses the global RNGs.
- Port to Flatland 4.3+ after the next release (`flatland_compat.py`, then the train attributes).

## 4. Small cleanups

- Remove the dead per-switch `_Switch.semaphores` / `block_port` / `free_port` in `switch_agents.py` if nothing uses them any more (check with grep first).
- `switchfl/README.md` predates the STOP action and the global semaphore redesign. Update the observation and action descriptions.
