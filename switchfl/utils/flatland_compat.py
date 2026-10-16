"""All calls into Flatland's rail transition API go through this module.

Implemented against flatland-rl 4.2.2. Known changes in later versions:
- 4.2.4+: `check_action_on_agent` became `_check_action_on_agent` (returns 5 values,
  public replacement: `apply_action_independent(action, config) -> Optional[(new_config, straight)]`)
  and `get_valid_move_actions_` was removed (replacement: `get_successor_configurations(config)`).
- 4.2.6+: `EnvAgent.position`, `direction`, `target`, `initial_*` and `old_*` are replaced by
  `current_configuration`, `initial_configuration`, `old_configuration` and `target_configuration`.
  These attributes are still read directly throughout switchfl and need a separate pass when porting.
- 4.3.0: the distance map moved to `flatland/core/distance_map.py` / `flatland/envs/grid/distance_map.py`.
"""

from typing import Tuple

from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap

Position = Tuple[int, int]


def apply_action(
    rail: RailGridTransitionMap, action: RailEnvActions, position: Position, direction: int
) -> Tuple[Position, int, bool]:
    """Simulates one action of a train standing at (position, direction).

    Args:
        rail (RailGridTransitionMap): the rail of the flatland environment
        action (RailEnvActions): action to simulate
        position (Position): current cell of the train
        direction (int): current flatland direction of the train

    Returns:
        Tuple[Position, int, bool]: new position, new direction and whether the transition is valid.
            New position and direction are returned even if the transition is invalid.
    """
    _, (new_position, new_direction), transition_valid, _ = rail.check_action_on_agent(
        action, (position, direction)
    )
    return new_position, new_direction, transition_valid


def first_valid_move(rail: RailGridTransitionMap, position: Position, direction: int) -> RailEnvActions:
    """Returns the first valid move action of a train standing at (position, direction).

    Args:
        rail (RailGridTransitionMap): the rail of the flatland environment
        position (Position): current cell of the train
        direction (int): current flatland direction of the train

    Returns:
        RailEnvActions: first valid move action
    """
    return rail.get_valid_move_actions_(direction, position)[0].action


def get_transitions(rail: RailGridTransitionMap, position: Position, direction: int) -> Tuple[int, int, int, int]:
    """Returns the possible exit directions [North, East, South, West] when entering a cell facing direction."""
    return rail.get_transitions((position, direction))


def get_full_transitions(rail: RailGridTransitionMap, row: int, col: int) -> int:
    """Returns the 16 bit transition code of a cell."""
    return rail.get_full_transitions(row, col)
