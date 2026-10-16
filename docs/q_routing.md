# Distributed Q-learning for SwitchFL (Q-routing) — implementation spec

Status: specification only. Implement after the environment is declared stable. Before writing code, read the learner in the PoliMi fork `gvlos/SwitchFL` and reuse what fits.

## Idea

This is Q-routing (Boyan & Littman, 1994, *Packet routing in dynamically changing networks*): switches play the role of routers and trains the role of packets. Each switch learns a value for every exit, namely how much delay a train will still accumulate if it leaves through that exit. It bootstraps from the value estimate of the switch the train reaches next. Because decisions are spaced by a variable number of Flatland steps τ, the problem is a semi-MDP, and discounting uses γ^τ.

## Interface used

`switchfl.core.SwitchCore` (see `CLAUDE.md`):

```python
decision = core.reset(seed)                    # Decision(switch, train, obs, mask, t, info)
decision, transitions = core.act(action)       # Transition(switch, train, obs, action, reward, team_reward,
                                               #            tau, next_switch, next_obs, next_mask, terminal, truncated)
```

Rewards already belong to the switch that acted, and pending transitions are tracked per train inside the core. The learner only consumes `Transition`s.

## Learners

One Q-function per switch, `Q_i(o, a)`, with `a ∈ {rail actions…, STOP}`. Actions are always taken under `decision.mask`.

| Variant | Input | Parameters |
|---|---|---|
| **A: tabular** (start here) | `info["reduced_obs"]` without handle and time: node id, packed port flags, target, delay bucket | one dict `obs_key -> np.ndarray[n_actions]` per switch |
| **B: deep, shared per switch type** | full `obs`, plus a one-hot or embedding of the switch id | one MLP per switch class in `SWITCH_AGENT_MAP` (same action space), shared across switches of that type |

Initialise Q with 0, which is optimistic, because rewards are mostly negative delay increases.

## Update rule

For every `Transition` `tr`:

```
if tr.terminal:      y = r
elif tr.truncated:   skip the update        # time limit, no next state; treating it as terminal would bias Q upward
else:                y = r + γ^τ · max_{a' : tr.next_mask[a']} Q_{tr.next_switch}(tr.next_obs, a')
Q_{tr.switch}(tr.obs, tr.action) += α · (y − Q_{tr.switch}(tr.obs, tr.action))
```

- `r = tr.reward` gives independent learners (local objective); `r = tr.team_reward` gives the cooperative objective. Run both and compare.
- `SwitchCore(env, reward="arrival")` gives the vanilla sparse objective (`r = 1` when the train arrives, else 0). Bootstrapping propagates it back along the route; start with it as a sanity check, then compare it with the delay reward.
- After STOP, the train stays in front of the same switch: `next_switch == switch` and τ is small (usually 1).
- `stop_penalty` defaults to 1300 (the same as the env). It dominates the delay scale, so sweep it (for example 0, 10, 100, 1300).
- Hyperparameters to start with: γ = 0.99, α = 0.1 (tabular), ε-greedy over the mask, decaying from 1.0 to 0.05.

## Decentralisation

The only data that crosses switch boundaries is `max_{a'} Q_{next_switch}(next_obs, a')`, i.e. one scalar message from the next switch along the train's route. This keeps the algorithm distributed by construction:

1. **Phase 1:** a single process, `dict[switch_name] -> learner`. Messages are plain function calls.
2. **Phase 2 (optional):** one actor per switch or per region (for example Ray). The bootstrap value becomes an asynchronous message, possibly stale.

## Deep variant (B) details

- Replay buffer per switch type storing `Transition`s. Store τ and use γ^τ in the target.
- Target network and Double DQN: pick `a'` with the online net of `next_switch`'s type and evaluate it with the target net.
- Mask invalid actions with `-inf` before the max and the argmax.

## Baselines and metrics

- **Random-masked:** a uniformly random allowed action.
- **Shortest-path greedy:** pick the allowed rail action whose exit configuration has the smallest distance-map entry; STOP only if no rail action is allowed. Exit configuration: `rail_graph.nodes[out_port]["rail_prev_node"]` and `map_direction(out_port)`, as in `observer.reachable_action_mask`.
- **Metrics per episode:** total and mean delay at the end (`Σ core.train_delay`), arrival rate, number of trains stuck at the end (deadlocks), share of STOPs while a rail action was allowed, and decisions per episode.
- Evaluate on fixed seeds and also on unseen schedules on the same map (the rail is pinned per env).

## Open questions

- With several trains at one switch in the same tick, the order of processing follows train handles. Decide whether to randomise it before comparing algorithms (`TODO.md`).
- The observation shows only the active train. Decide whether neighbouring trains' positions should be added for the cooperative variant.
