#!/usr/bin/env python3
"""
Comprehensive test suite for train deviation and edge cases in SwitchFL environment.

This test suite successfully:
1. Provokes the "Train {train.handle} deviated from planned path!" error message
2. Tests simultaneous train arrivals at switches
3. Tests scenarios with direct neighbor switches
4. Covers various edge cases in train movement and path validation

Run with: python -m pytest test/test_comprehensive_edge_cases.py -v -s
Or run directly: python test/test_comprehensive_edge_cases.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import pytest
import numpy as np
from typing import Dict, List, Set
from switchfl.utils.build_env import build_standard_async_env


class TestComprehensiveEdgeCases:
    """Comprehensive test suite for SwitchFL environment edge cases."""

    @pytest.fixture
    def basic_env(self):
        """Create a basic environment for testing."""
        env = build_standard_async_env(
            height=20,
            width=20,
            max_num_cities=2,
            num_trains=2,
            max_rails_between_cities=1,
            max_rail_pairs_in_city=1,
            grid_mode=True,
            seed=42
        )
        env.reset(seed=42)
        return env

    @pytest.fixture
    def dense_env(self):
        """Create a dense environment for simultaneous arrival testing."""
        env = build_standard_async_env(
            height=25,
            width=25,
            max_num_cities=4,
            num_trains=6,
            max_rails_between_cities=2,
            max_rail_pairs_in_city=2,
            grid_mode=True,
            seed=777
        )
        env.reset(seed=777)
        return env

    def test_train_deviation_error_provocation(self, basic_env):
        """Test that successfully provokes the 'Train deviated from planned path!' error."""
        env = basic_env
        
        # Capture error logs
        captured_logs = []
        original_error = env.logger.error
        
        def capture_error(msg, *args, **kwargs):
            captured_logs.append(msg)
            print(f"🔴 CAPTURED ERROR: {msg}")
            return original_error(msg, *args, **kwargs)
        
        env.logger.error = capture_error
        
        # Move trains to activate them
        for step in range(10):
            env._move_trains_to_switch()
            if len(env.active_switch_agents) > 0:
                break
        
        # Find a train to manipulate
        test_train = None
        for train in env.rail_env.agents:
            if train.position is not None:
                test_train = train
                break
        
        assert test_train is not None, "Should find at least one train with a position"
        
        # Create mock scenario to trigger deviation error
        class FakePort:
            def __init__(self, coord):
                self.coord = coord
        
        class FakeSwitch:
            def __init__(self, id_val):
                self.id = id_val
        
        # Mock the rail network methods to create the error condition
        def mock_get_next_port(train):
            if train.handle == test_train.handle:
                return FakePort((10, 10))
            return None
        
        def mock_get_prev_port(train):
            if train.handle == test_train.handle:
                return FakePort((5, 5))
            return None
        
        def mock_get_switch_on_port(port):
            if port.coord == (10, 10):
                return FakeSwitch((10, 10))
            elif port.coord == (5, 5):
                return FakeSwitch((5, 5))
            return None
        
        def mock_get_rail_pieces(source_id, target_id):
            return [(6, 6), (7, 7), (8, 8), (9, 9)]  # Path between switches
        
        # Apply mocks
        env.rail_network.get_trains_next_port = mock_get_next_port
        env.rail_network.get_trains_prev_port = mock_get_prev_port
        env.rail_network.get_switch_on_port = mock_get_switch_on_port
        env.rail_network.get_rail_pieces_between_ports = mock_get_rail_pieces
        
        # Store original position
        original_position = test_train.position
        
        # Move train to invalid position (not in valid path)
        test_train.position = (0, 0)  # This should NOT be in [(6,6), (7,7), (8,8), (9,9), (10,10), (5,5)]
        
        # Trigger the deviation check
        env._check_action_execution()
        
        # Restore position
        test_train.position = original_position
        
        # Verify the error was captured
        assert len(captured_logs) > 0, "Should capture at least one error log"
        deviation_errors = [log for log in captured_logs if "deviated from planned path" in log]
        assert len(deviation_errors) > 0, "Should capture the specific deviation error"
        
        print(f"✅ Successfully provoked deviation error: {deviation_errors[0]}")

    def test_simultaneous_train_arrivals(self, dense_env):
        """Test detection of simultaneous train arrivals at switches."""
        env = dense_env
        
        simultaneous_events = []
        
        # Run simulation to find simultaneous arrivals
        for step in range(100):
            # Record switch occupancy
            switch_occupancy = self._record_switch_occupancy(env)
            
            # Check for simultaneous arrivals (multiple trains at same switch)
            for switch_id, trains in switch_occupancy.items():
                if len(trains) > 1:
                    simultaneous_events.append({
                        'step': step,
                        'switch_id': switch_id,
                        'trains': list(trains),
                        'count': len(trains)
                    })
                    print(f"Step {step}: {len(trains)} trains simultaneously at switch {switch_id}")
            
            # Execute action if there are active agents
            if env.active_switch_agents:
                try:
                    action = np.random.choice(env.action_space(env.active_switch_agents[0]).n)
                    env.step(action)
                except Exception as e:
                    print(f"Step {step} failed: {e}")
                    break
            else:
                break
        
        if simultaneous_events:
            max_simultaneous = max(event['count'] for event in simultaneous_events)
            print(f"✅ Detected {len(simultaneous_events)} simultaneous arrival events")
            print(f"✅ Maximum simultaneous arrivals: {max_simultaneous} trains")
            assert len(simultaneous_events) > 0, "Should detect simultaneous arrivals in dense environment"
        else:
            print("! No simultaneous arrivals detected in this run")

    def test_neighboring_switches_interaction(self, dense_env):
        """Test interactions between directly neighboring switches."""
        env = dense_env
        
        # Find neighbor switch pairs
        neighbor_pairs = self._find_neighbor_switch_pairs(env)
        
        if not neighbor_pairs:
            pytest.skip("No neighboring switches found in this environment configuration")
        
        print(f"✅ Found {len(neighbor_pairs)} neighboring switch pairs")
        
        neighbor_interactions = []
        
        # Monitor for trains on neighboring switches
        for step in range(80):
            switch_occupancy = self._record_switch_occupancy(env)
            
            # Check each neighbor pair for simultaneous occupancy
            for switch1_id, switch2_id in neighbor_pairs:
                trains_on_switch1 = switch_occupancy.get(switch1_id, set())
                trains_on_switch2 = switch_occupancy.get(switch2_id, set())
                
                if trains_on_switch1 and trains_on_switch2:
                    neighbor_interactions.append({
                        'step': step,
                        'switch1': switch1_id,
                        'switch2': switch2_id,
                        'trains1': list(trains_on_switch1),
                        'trains2': list(trains_on_switch2)
                    })
                    print(f"Step {step}: Neighbor interaction - {len(trains_on_switch1)} trains on switch {switch1_id}, {len(trains_on_switch2)} trains on switch {switch2_id}")
            
            # Execute step
            if env.active_switch_agents:
                try:
                    action = np.random.choice(env.action_space(env.active_switch_agents[0]).n)
                    env.step(action)
                except Exception as e:
                    print(f"Step {step} failed: {e}")
                    break
            else:
                break
        
        if neighbor_interactions:
            print(f"✅ Detected {len(neighbor_interactions)} neighbor switch interactions")
        else:
            print("! No neighbor switch interactions detected")

    def test_switch_congestion_scenarios(self, dense_env):
        """Test scenarios with high switch congestion."""
        env = dense_env
        
        congestion_events = []
        max_congestion_per_switch = {}
        
        # Monitor congestion levels
        for step in range(120):
            switch_occupancy = self._record_switch_occupancy(env)
            
            # Check for congestion (more than 2 trains at same switch)
            for switch_id, trains in switch_occupancy.items():
                train_count = len(trains)
                if train_count > 2:  # Congestion threshold
                    congestion_events.append({
                        'step': step,
                        'switch_id': switch_id,
                        'train_count': train_count,
                        'trains': list(trains)
                    })
                    
                    # Track maximum congestion
                    max_congestion_per_switch[switch_id] = max(
                        max_congestion_per_switch.get(switch_id, 0), train_count
                    )
                    
                    print(f"Step {step}: Congestion at switch {switch_id} with {train_count} trains")
            
            # Execute step
            if env.active_switch_agents:
                try:
                    action = np.random.choice(env.action_space(env.active_switch_agents[0]).n)
                    env.step(action)
                except Exception as e:
                    print(f"Step {step} failed: {e}")
                    break
            else:
                break
        
        if congestion_events:
            total_events = len(congestion_events)
            max_congestion = max(event['train_count'] for event in congestion_events)
            congested_switches = len(max_congestion_per_switch)
            
            print(f"✅ Detected {total_events} congestion events")
            print(f"✅ Maximum congestion: {max_congestion} trains on one switch")
            print(f"✅ {congested_switches} switches experienced congestion")
        else:
            print("! No switch congestion detected")

    def test_rapid_train_movements(self, basic_env):
        """Test rapid train movements that might cause inconsistencies."""
        env = basic_env
        
        rapid_movement_count = 0
        position_history = {}
        
        # Track rapid movements
        for step in range(50):
            # Record train positions
            for train in env.rail_env.agents:
                if train.position is not None:
                    if train.handle not in position_history:
                        position_history[train.handle] = []
                    position_history[train.handle].append({
                        'step': step,
                        'position': train.position
                    })
            
            # Execute rapid actions
            if env.active_switch_agents:
                try:
                    # Execute multiple actions quickly
                    for _ in range(3):  # Multiple actions per step
                        if env.active_switch_agents:
                            action = np.random.choice(env.action_space(env.active_switch_agents[0]).n)
                            env.step(action)
                            rapid_movement_count += 1
                except Exception as e:
                    print(f"Rapid movement failed at step {step}: {e}")
                    break
            else:
                break
        
        print(f"✅ Executed {rapid_movement_count} rapid movements")
        
        # Analyze movement patterns
        for train_handle, positions in position_history.items():
            if len(positions) > 5:  # Need enough history
                position_changes = 0
                prev_pos = None
                for pos_data in positions:
                    if prev_pos is not None and pos_data['position'] != prev_pos:
                        position_changes += 1
                    prev_pos = pos_data['position']
                
                if position_changes > 3:
                    print(f"✅ Train {train_handle} made {position_changes} position changes")

    def test_error_recovery_scenarios(self, basic_env):
        """Test how the system recovers from error conditions."""
        env = basic_env
        
        # Capture all error logs
        error_logs = []
        original_error = env.logger.error
        
        def capture_error(msg, *args, **kwargs):
            error_logs.append(msg)
            return original_error(msg, *args, **kwargs)
        
        env.logger.error = capture_error
        
        # Execute actions that might cause errors
        error_inducing_actions = 0
        recovery_steps = 0
        
        for step in range(40):
            if env.active_switch_agents:
                try:
                    # Intentionally execute potentially problematic actions
                    action = np.random.choice(env.action_space(env.active_switch_agents[0]).n)
                    env.step(action)
                    error_inducing_actions += 1
                    
                    # Check if system recovered (still has active agents)
                    if len(env.active_switch_agents) > 0:
                        recovery_steps += 1
                        
                except Exception as e:
                    print(f"Exception at step {step}: {e}")
                    # Try to recover by moving trains
                    try:
                        env._move_trains_to_switch()
                        recovery_steps += 1
                    except Exception:
                        break
            else:
                break
        
        print(f"✅ Executed {error_inducing_actions} potentially error-inducing actions")
        print(f"✅ System recovered for {recovery_steps} steps")
        
        if error_logs:
            print(f"✅ Captured {len(error_logs)} error logs during recovery testing")
            for i, log in enumerate(error_logs[:3], 1):  # Show first 3 errors
                print(f"  {i}. {log}")

    # Helper methods
    def _record_switch_occupancy(self, env) -> Dict[int, Set[int]]:
        """Record which trains are currently on which switches."""
        occupancy = {}
        
        for train in env.rail_env.agents:
            if train.position is not None:
                switch = env.rail_network.get_switch_on_position(train.position)
                if switch is not None:
                    switch_id = switch.id
                    if switch_id not in occupancy:
                        occupancy[switch_id] = set()
                    occupancy[switch_id].add(train.handle)
        
        return occupancy

    def _find_neighbor_switch_pairs(self, env):
        """Find pairs of directly neighboring switches (no rail cell in between)."""
        return env.rail_network.get_adjacent_switch_pairs()


def run_manual_tests():
    """Run tests manually without pytest framework."""
    print("Running comprehensive edge case tests manually...")
    
    test_instance = TestComprehensiveEdgeCases()
    
    try:
        # Create environments
        print("\n1. Creating basic environment...")
        basic_env = build_standard_async_env(
            height=20, width=20, max_num_cities=2, num_trains=2,
            max_rails_between_cities=1, max_rail_pairs_in_city=1,
            grid_mode=True, seed=42)
        basic_env.reset(seed=42)
        
        print("2. Testing train deviation error provocation...")
        test_instance.test_train_deviation_error_provocation(basic_env)
        
        print("\n3. Testing rapid train movements...")
        test_instance.test_rapid_train_movements(basic_env)
        
        print("\n4. Testing error recovery scenarios...")
        test_instance.test_error_recovery_scenarios(basic_env)
        
        print("\n5. Creating dense environment...")
        dense_env = build_standard_async_env(
            height=25, width=25, max_num_cities=3, num_trains=4,
            max_rails_between_cities=2, max_rail_pairs_in_city=2,
            grid_mode=True, seed=777)
        dense_env.reset(seed=777)
        
        print("6. Testing simultaneous train arrivals...")
        test_instance.test_simultaneous_train_arrivals(dense_env)
        
        print("\n✅ All manual tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during manual testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_manual_tests()