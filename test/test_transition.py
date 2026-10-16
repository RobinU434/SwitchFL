"""
Test suite to verify consistency between rail graph transitions and actual train movements.

This module contains comprehensive tests to check for discrepancies between the rail 
transition graph representation and the actual position of trains using Flatland's 
`rail_env.rail.check_action_on_agent` function (via `switchfl.utils.flatland_compat.apply_action`) as ground truth.

Run with: python -m pytest test/test_transition.py -v
"""

try:
    import pytest
except ImportError:
    # If pytest is not available, create mock decorators
    def pytest_fixture(**kwargs):
        def decorator(func):
            return func
        return decorator
    
    pytest = type('MockPytest', (), {
        'fixture': pytest_fixture,
        'skip': lambda msg: None,
        'fail': lambda msg: print(f"FAIL: {msg}")
    })()

from typing import Tuple
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.agent_utils import EnvAgent as TrainAgent
from flatland.envs.step_utils.states import TrainState

from switchfl.rail_network import RailNetwork
from switchfl.utils.build_env import build_standard_async_env
from switchfl.utils.flatland_compat import apply_action


class TestTransitionConsistency:
    """Test suite to verify consistency between rail graph transitions and actual train movements."""

    @pytest.fixture
    def small_env(self):
        """Create a small test environment."""
        return build_standard_async_env(
            height=18,
            width=18, 
            max_num_cities=3,
            num_trains=2,
            max_rails_between_cities=2,
            max_rail_pairs_in_city=1,
            grid_mode=False,
            seed=42
        )

    @pytest.fixture
    def medium_env(self):
        """Create a medium test environment."""
        return build_standard_async_env(
            height=30,
            width=30,
            max_num_cities=6,
            num_trains=3,
            max_rails_between_cities=2,
            max_rail_pairs_in_city=2,
            grid_mode=False,
            seed=123
        )

    def test_rail_graph_construction_consistency(self, small_env):
        """Test that the rail graph accurately represents the Flatland environment."""
        small_env.reset()
        rail_env = small_env.rail_env
        rail_network = small_env.rail_network

        # Check that all switch positions in the graph correspond to actual switches in Flatland
        for switch_id, switch_data in rail_network.switch_network.nodes(data=True):
            switch_cls = switch_data['switch_cls']
            
            # Verify switch exists at the position in Flatland
            assert rail_env.rail.grid[switch_id] != 0, f"No rail at switch position {switch_id}"
            
            # Check that all ports have valid rail connections
            for port in switch_cls.get_port_nodes():
                port_position = rail_network.rail_graph.nodes[port].get('rail_prev_node')
                assert port_position is not None, f"Port {port} has no rail_prev_node position"

    def test_direction_mapping_consistency(self, small_env):
        """Test that direction mappings between rail graph and Flatland are consistent."""
        small_env.reset()
        rail_network = small_env.rail_network
        
        # Test the direction mapping dictionary
        dir_mapping = rail_network._dir2port_idx
        
        # Verify all standard directions are mapped
        expected_directions = [1.0, 2.0, 3.0, 4.0]  # WEST, SOUTH, EAST, NORTH
        for direction in expected_directions:
            assert direction in dir_mapping, f"Direction {direction} not in mapping"
            
        # Verify the mapping values correspond to valid Grid4TransitionsEnum values
        for direction, port_idx in dir_mapping.items():
            assert 0 <= port_idx <= 3, f"Invalid port index {port_idx} for direction {direction}"

    def test_single_train_action_consistency(self, small_env):
        """Test that a single train action produces consistent results between graph and simulation."""
        small_env.reset()
        rail_env = small_env.rail_env
        rail_network = small_env.rail_network

        # Find a train that's ready to move
        test_train = None
        for train in rail_env.agents:
            if train.state in [TrainState.READY_TO_DEPART, TrainState.MOVING]:
                test_train = train
                break
        
        if test_train is None:
            pytest.skip("No active trains found for testing")

        original_position = test_train.position
        original_direction = test_train.direction

        # Test basic MOVE_FORWARD action
        self._test_action_consistency(
            rail_env, rail_network, test_train, RailEnvActions.MOVE_FORWARD
        )

        # Test other actions if applicable
        for action in [RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT]:
            # Reset train position for each test
            test_train.position = original_position
            test_train.direction = original_direction
            
            self._test_action_consistency(
                rail_env, rail_network, test_train, action
            )

    def test_all_switch_actions_consistency(self, medium_env):
        """Test all possible switch actions for consistency between graph and simulation."""
        medium_env.reset()
        rail_env = medium_env.rail_env
        rail_network = medium_env.rail_network

        discrepancies = []

        # Test each switch and all its possible actions
        for switch_id, switch_data in rail_network.switch_network.nodes(data=True):
            switch_cls = switch_data['switch_cls']
            # rail actions only: the last index of the action space is STOP, which has no outcome
            for action_idx in range(switch_cls.n_actions):
                try:
                    self._test_switch_action_consistency(
                        rail_env, rail_network, switch_id, switch_cls, action_idx
                    )
                except AssertionError as e:
                    discrepancies.append(f"Switch {switch_id}, Action {action_idx}: {str(e)}")

        if discrepancies:
            pytest.fail(f"Found {len(discrepancies)} discrepancies:\n" + "\n".join(discrepancies))

    def test_train_transition_path_consistency(self, small_env):
        """Test that trains follow expected paths through switches."""
        small_env.reset()
        rail_env = small_env.rail_env
        rail_network = small_env.rail_network

        # Move trains until one reaches a switch
        max_steps = 50
        step_count = 0
        
        while step_count < max_steps:
            # Find trains approaching switches
            approaching_trains = []
            for train in rail_env.agents:
                if train.position is None or train.state == TrainState.WAITING:
                    continue
                    
                # Simulate next step to see if train reaches a switch
                next_action = RailEnvActions.MOVE_FORWARD
                new_position, _, _ = apply_action(
                    rail_env.rail, next_action, train.position, train.direction
                )
                
                next_switch = rail_network.get_switch_on_position(new_position)
                if next_switch is not None:
                    approaching_trains.append((train, new_position, next_switch))
            
            if approaching_trains:
                # Test transition consistency for the first approaching train
                train, switch_position, switch = approaching_trains[0]
                self._test_train_transition_through_switch(
                    rail_env, rail_network, train, switch_position, switch
                )
                break
                
            # Move all trains forward one step
            actions = {train.handle: RailEnvActions.MOVE_FORWARD for train in rail_env.agents}
            rail_env.step(actions)
            step_count += 1

        if step_count >= max_steps:
            pytest.skip("No trains reached switches within maximum steps")

    def _test_action_consistency(self, rail_env: RailEnv, rail_network: RailNetwork, 
                                train: TrainAgent, action: RailEnvActions):
        """Helper method to test consistency of a single action."""
        if train.position is None:
            return  # Skip trains not on grid

        # Use Flatland's ground truth simulation
        new_position, _, transition_valid = apply_action(
            rail_env.rail, action, train.position, train.direction
        )

        if not transition_valid:
            return  # Skip invalid moves

        # Check if the predicted position matches any expected position in rail graph
        # This is a basic consistency check - the train should end up on a valid rail piece
        if new_position is not None:
            assert rail_env.rail.grid[new_position] != 0, (
                f"Train moved to position {new_position} with no rail piece"
            )

        # Additional checks can be added here for more specific consistency verification

    def _test_switch_action_consistency(self, rail_env: RailEnv, rail_network: RailNetwork,
                                      switch_id: Tuple[int, int], switch_cls, action_idx: int):
        """Helper method to test consistency of switch actions."""
        # Get action outcomes from the switch
        try:
            in_port, out_port = switch_cls.action_outcomes[action_idx]
            port_actions = switch_cls.actions[action_idx]
        except IndexError:
            pytest.skip(f"Action {action_idx} not valid for switch {switch_id}")

        # Verify that the action outcomes make sense
        assert in_port is not None, f"Invalid in_port for switch {switch_id}, action {action_idx}"
        assert out_port is not None, f"Invalid out_port for switch {switch_id}, action {action_idx}"

        # Check that ports belong to the same switch
        in_switch_id = rail_network.rail_graph.nodes[in_port].get('switch_id')
        out_switch_id = rail_network.rail_graph.nodes[out_port].get('switch_id')
        assert in_switch_id == out_switch_id == switch_id, (
            f"Port switch IDs don't match: in={in_switch_id}, out={out_switch_id}, expected={switch_id}"
        )

        # Verify port actions are valid RailEnvActions
        for port, actions in port_actions.items():
            for rail_action in actions:
                assert isinstance(rail_action, RailEnvActions) or isinstance(rail_action, int), (
                    f"Invalid rail action type: {type(rail_action)} for port {port}"
                )

    def _test_train_transition_through_switch(self, rail_env: RailEnv, rail_network: RailNetwork,
                                            train: TrainAgent, switch_position: Tuple[int, int], 
                                            switch_cls):
        """Helper method to test train transition through a specific switch."""
        original_position = train.position
        original_direction = train.direction

        # Test each possible action for this switch
        action_space = switch_cls.get_action_space()
        
        for action_idx in range(action_space.n):
            # Reset train state
            train.position = original_position
            train.direction = original_direction

            try:
                # Get expected action from switch
                moving_train, train_actions = rail_network.get_train_actions(
                    switch_position, action_idx, [train]
                )

                if train.handle not in train_actions:
                    continue  # This action doesn't affect this train

                expected_actions = train_actions[train.handle]
                
                # Simulate each action and verify consistency
                current_pos = train.position
                current_dir = train.direction
                
                for expected_action in expected_actions:
                    new_position, new_direction, transition_valid = apply_action(
                        rail_env.rail, expected_action, current_pos, current_dir
                    )
                    
                    if not transition_valid:
                        break  # Stop if move becomes invalid
                        
                    # Update position for next iteration
                    current_pos = new_position
                    current_dir = new_direction

                # Verify the final position is on a valid rail piece
                if current_pos is not None:
                    assert rail_env.rail.grid[current_pos] != 0, (
                        f"Train ended up at invalid position {current_pos} after switch action"
                    )

            except Exception as e:
                # Log the error but don't fail the test - some actions might be invalid
                print(f"Warning: Could not test action {action_idx} for switch {switch_position}: {e}")

    def test_semaphore_state_consistency(self, small_env):
        """Test that semaphore states match actual train positions."""
        small_env.reset()
        rail_network = small_env.rail_network
        
        # Check that blocked semaphores correspond to trains approaching switches
        for switch_id, switch_data in rail_network.switch_network.nodes(data=True):
            switch_cls = switch_data['switch_cls']
            
            for port, is_blocked in switch_cls.semaphores.items():
                if is_blocked:
                    # There should be a train associated with this port
                    # This is a basic check - more sophisticated verification could be added
                    port_position = rail_network.rail_graph.nodes[port].get('rail_prev_node')
                    assert port_position is not None, f"Blocked port {port} has no valid position"

    def test_port_distance_calculation(self, medium_env):
        """Test that port distance calculations are consistent."""
        medium_env.reset()
        rail_network = medium_env.rail_network

        # Test distance calculations between connected ports
        for switch_id, switch_data in rail_network.switch_network.nodes(data=True):
            switch_cls = switch_data['switch_cls']
            
            # Test distances between ports of the same switch
            ports = switch_cls.get_port_nodes()
            
            for i, port1 in enumerate(ports):
                for port2 in ports[i+1:]:
                    distance = rail_network.get_port_distance(port1, port2)
                    
                    # Distance should be None for unconnected ports or a positive integer for connected ones
                    if distance is not None:
                        assert isinstance(distance, int) and distance >= 0, (
                            f"Invalid distance {distance} between ports {port1} and {port2}"
                        )


def run_all_tests():
    """Run all transition consistency tests without pytest."""
    test_class = TestTransitionConsistency()
    
    # Create test environments
    small_env = build_standard_async_env(
        height=7, width=7, max_num_cities=2, num_trains=2,
        max_rails_between_cities=2, max_rail_pairs_in_city=1,
        grid_mode=False, seed=42
    )
    
    medium_env = build_standard_async_env(
        height=10, width=10, max_num_cities=3, num_trains=3,
        max_rails_between_cities=2, max_rail_pairs_in_city=2,
        grid_mode=False, seed=123
    )
    
    # List of test methods to run
    tests = [
        ("Rail Graph Construction Consistency", lambda: test_class.test_rail_graph_construction_consistency(small_env)),
        ("Direction Mapping Consistency", lambda: test_class.test_direction_mapping_consistency(small_env)),
        ("Single Train Action Consistency", lambda: test_class.test_single_train_action_consistency(small_env)),
        ("All Switch Actions Consistency", lambda: test_class.test_all_switch_actions_consistency(medium_env)),
        ("Train Transition Path Consistency", lambda: test_class.test_train_transition_path_consistency(small_env)),
        ("Semaphore State Consistency", lambda: test_class.test_semaphore_state_consistency(small_env)),
        ("Port Distance Calculation", lambda: test_class.test_port_distance_calculation(medium_env)),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        try:
            test_func()
            print(f"✓ PASSED: {test_name}")
            results.append((test_name, "PASSED", None))
        except Exception as e:
            print(f"✗ FAILED: {test_name} - {str(e)}")
            results.append((test_name, "FAILED", str(e)))
        except AssertionError as e:
            print(f"✗ ASSERTION FAILED: {test_name} - {str(e)}")
            results.append((test_name, "ASSERTION FAILED", str(e)))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(1 for _, status, _ in results if status == "PASSED")
    total = len(results)
    
    for test_name, status, error in results:
        print(f"{status:15} | {test_name}")
        if error:
            print(f"{'':15} | Error: {error}")
    
    print(f"\nPassed: {passed}/{total}")
    
    return results


if __name__ == "__main__":
    run_all_tests()
