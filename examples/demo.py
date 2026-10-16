"""SwitchFL demo: switches as agents in a multi-agent RL environment.

Every railway switch is an agent. Trains drive on their own; whenever a train is about to enter a switch, that
switch decides where the train goes (or holds it with STOP). Decisions happen asynchronously, one
(switch, train) pair at a time, through the PettingZoo AEC loop:

    for agent in env.agent_iter():         # the next switch that has a train in front of it
        obs = env.observe(agent)           # what this switch sees
        mask = env.infos[agent]["action_mask"]
        env.step(policy(...))              # route the train; Flatland only advances once every switch decided

Usage:
    uv run python examples/demo.py                                 # shortest-path routing, seed 0
    uv run python examples/demo.py --policy random --seed 3
    uv run python examples/demo.py --policy compare --episodes 5   # random vs. shortest path
    uv run python examples/demo.py --gif demo.gif                  # also save the episode as an animation
"""

import argparse
import logging
import time
from collections import Counter

import numpy as np
from flatland.envs.step_utils.states import TrainState

from switchfl.baselines import random_policy, shortest_path_policy
from switchfl.core import total_delay
from switchfl.utils.build_env import build_standard_async_env


def make_policy(name: str, seed: int):
    if name == "random":
        return random_policy(np.random.default_rng(seed))
    return shortest_path_policy


def describe_decision(env, agent, obs, mask, action):
    """One line per decision: which switch decided for which train, what it saw and what it did."""
    switch = env.rail_network.get_switch_on_position(tuple(obs[:2]))
    human = env.obs2human(agent, obs)
    if action == len(mask) - 1:
        what = "STOP"
    else:
        in_port, out_port = switch.action_outcomes[action]
        what = f"route {in_port} -> {out_port}"
    blocked = [int(not free) for free in human["semaphore"]]
    allowed = [i for i in np.flatnonzero(mask)]
    return (
        f"t={env.rail_env._elapsed_steps:4d}  {agent:<14} train {env.active_train}  "
        f"ports blocked {blocked}  allowed {allowed}  -> {what}"
    )


def run_episode(env, policy, seed, show_decisions=0, frames=None):
    if frames is not None:
        # one env.step() can advance Flatland by several steps, so record a frame after every Flatland step
        move_trains = env._move_trains

        def move_trains_and_record():
            move_trains()
            frames.append(env.render())

        env._move_trains = move_trains_and_record

    env.reset(seed=seed)
    stats = Counter()
    acting_switches = set()
    start = time.time()

    for agent in env.agent_iter():
        obs = env.observe(agent)
        mask = env.infos[agent]["action_mask"]
        action = policy(env, agent, env.active_train, mask)

        if stats["decisions"] < show_decisions:
            print("  " + describe_decision(env, agent, obs, mask, action))
        stats["decisions"] += 1
        stats["stop"] += action == len(mask) - 1
        stats["forced_stop"] += not mask[:-1].any()
        acting_switches.add(agent)

        env.step(action)

    if frames is not None:
        del env._move_trains  # back to the class method

    trains = env.rail_env.agents
    stats["arrived"] = sum(train.state == TrainState.DONE for train in trains)
    stats["trains"] = len(trains)
    stats["flatland_steps"] = env.rail_env._elapsed_steps
    stats["switches_acting"] = len(acting_switches)
    stats["total_delay"] = total_delay(env.rail_env)
    stats["seconds"] = time.time() - start
    return stats


def print_stats(title, stats):
    print(f"\n{title}")
    print(f"  trains delivered      {stats['arrived']}/{stats['trains']}")
    print(f"  total delay           {stats['total_delay']:.0f} steps (negative = early)")
    print(f"  Flatland steps        {stats['flatland_steps']}")
    print(f"  switch decisions      {stats['decisions']} by {stats['switches_acting']} different switches")
    print(f"  STOP decisions        {stats['stop']} ({stats['forced_stop']} forced: no rail action allowed)")
    print(f"  wall time             {stats['seconds']:.2f}s")


class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Keeps the module docstring's layout and shows every option's default."""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=_HelpFormatter)
    parser.add_argument("--policy", choices=["shortest_path", "random", "compare"], default="shortest_path",
                        help="switch policy; compare runs random and shortest_path on the same seeds")
    parser.add_argument("--seed", type=int, default=0, help="seed of the map and of the first episode")
    parser.add_argument("--episodes", type=int, default=1, help="episodes per policy (seeds seed, seed+1, ...)")
    parser.add_argument("--trains", type=int, default=6, help="number of trains")
    parser.add_argument("--size", type=int, default=30, help="grid height and width")
    parser.add_argument("--cities", type=int, default=3, help="maximum number of cities")
    parser.add_argument("--show", type=int, default=15, help="print the first N decisions of the first episode")
    parser.add_argument("--gif", type=str, default=None, help="save the first episode as an animated GIF")
    parser.add_argument("--gif-width", type=int, default=None, help="scale GIF frames to this width in pixels")
    parser.add_argument("--gif-fps", type=float, default=8, help="GIF frames (Flatland steps) per second")
    args = parser.parse_args()

    logging.disable(logging.WARNING)  # the env logs a lot on INFO/DEBUG

    env = build_standard_async_env(
        height=args.size, width=args.size, max_num_cities=args.cities, num_trains=args.trains,
        seed=args.seed, render_mode="rgb_array" if args.gif else None,
    )
    print(f"{args.size}x{args.size} map, {args.cities} cities, {args.trains} trains, "
          f"{len(env.possible_agents)} switch agents")
    agent = env.possible_agents[0]
    print(f"e.g. {agent}: action space {env.action_space(agent)} (last action = STOP), "
          f"observation = [node id, port blocked flags, train target per port, delay level per port]")

    policies = ["random", "shortest_path"] if args.policy == "compare" else [args.policy]
    for name in policies:
        totals = Counter()
        for episode in range(args.episodes):
            seed = args.seed + episode
            first = episode == 0
            frames = [] if (first and args.gif and name == policies[-1]) else None
            if first and args.show:
                print(f"\nfirst decisions with {name} routing (seed {seed}):")
            stats = run_episode(env, make_policy(name, seed), seed, args.show if first else 0, frames)
            if args.episodes == 1:
                print_stats(f"{name} routing, seed {seed}:", stats)
            totals.update(stats)
            if frames:
                save_gif(frames, args.gif, args.gif_width, args.gif_fps)
        if args.episodes > 1:
            print(f"\n{name} routing over {args.episodes} episodes (seeds {args.seed}..{args.seed + args.episodes - 1}):")
            print(f"  trains delivered      {totals['arrived']}/{totals['trains']}")
            print(f"  mean total delay      {totals['total_delay'] / args.episodes:.0f} steps")
            print(f"  switch decisions      {totals['decisions']} ({totals['stop']} STOP)")


def save_gif(frames, path, width=None, fps=8):
    from PIL import Image

    images = [Image.fromarray(frame) for frame in frames]
    if width is not None:
        height = round(images[0].height * width / images[0].width)
        images = [image.resize((width, height), Image.LANCZOS) for image in images]
    # one shared palette keeps the file small and the colours stable between frames
    palette = images[0].quantize(colors=128)
    images = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    images[0].save(path, save_all=True, append_images=images[1:], duration=round(1000 / fps), loop=0, optimize=True)
    print(f"\nsaved {len(images)} frames to {path}")


if __name__ == "__main__":
    main()
