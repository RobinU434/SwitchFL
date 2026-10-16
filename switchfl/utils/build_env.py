from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from switchfl.switch_env import ASyncSwitchEnv
from switchfl.utils.logging import set_seed


def build_standard_async_env(
    height: int,
    width: int,
    max_num_cities: int,
    num_trains: int,
    max_rails_between_cities: int = 1,
    max_rail_pairs_in_city: int = 1,
    grid_mode: bool = True,
    render_mode: str = None,
    seed: int = None,
    malfunction_generator=None,
):
    # Apply comprehensive seeding before creating any generators
    if seed is not None:
        set_seed(seed)
    
    rail_env = RailEnv(
        width=width,
        height=height,
        rail_generator=sparse_rail_generator(
            max_num_cities=max_num_cities,
            grid_mode=grid_mode,
            max_rails_between_cities=max_rails_between_cities,
            max_rail_pairs_in_city=max_rail_pairs_in_city,
            seed=seed,  # Seed controls rail layout
        ),
        line_generator=sparse_line_generator(seed=seed),  # Seed controls train placement
        number_of_agents=num_trains,
        malfunction_generator=malfunction_generator,  # None: no malfunctions
    )

    env = ASyncSwitchEnv(rail_env, render_mode=render_mode)
    return env
