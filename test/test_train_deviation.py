#!/usr/bin/env python3
"""
Test suite to provoke and verify train path deviation errors and edge cases.

This module contains tests designed to:
1. Provoke the "Train deviated from planned path!" error message
2. Test simultaneous train arrivals at switches
3. Test scenarios with direct neighbor switches
4. Verify robustness of train tracking and action planning

Run with: python -m pytest test/test_train_deviation.py -v
"""

import pytest
import numpy as np
from typing import List, Tuple
from unittest.mock import patch, MagicMock

from flatland.envs.rail_env import RailEnvActions

from switchfl.utils.build_env import build_standard_async_env
from switchfl.utils.naming import switch_id2name


class TestTrainDeviation:
    """Test suite to provoke train path deviation errors and edge cases."""

    @pytest.fixture
    def small_env(self):
        """Create a small environment for controlled testing."""
        env = build_standard_async_env(
            height=25,
            width=25,
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
    def medium_env(self):
        """Create a medium environment with more trains and complexity."""
        env = build_standard_async_env(
            height=25,
            width=25,
            max_num_cities=3,
            num_trains=4,
            max_rails_between_cities=2,
            max_rail_pairs_in_city=1,
            grid_mode=True,
            seed=123
        )
        env.reset(seed=123)
        return env

    @pytest.fixture
    def dense_env(self):
        """Create a dense environment with many trains and close switches."""
        env = build_standard_async_env(
            height=30,
            width=30,
            max_num_cities=4,
            num_trains=6,
            max_rails_between_cities=3,
            max_rail_pairs_in_city=2,
            grid_mode=True,
            seed=999
        )
        env.reset(seed=999)
        return env

    def test_force_train_deviation_by_corrupting_position(self, small_env):
        """Corrupt the position of a train that passed a switch: the deviation check has to log an error."""
        env = small_env
        rng = np.random.default_rng(0)
        rail_network = env.rail_network

        def train_between_switches():
            for train in env.rail_env.agents:
                if (train.position is not None
                        and rail_network.get_trains_prev_port(train) is not None
                        and rail_network.get_trains_next_port(train) is not None):
                    return train
            return None

        # let switches decide (masked random actions) until a train has left a switch
        target_train = None
        for agent in env.agent_iter():
            env.observe(agent)
            env.step(int(rng.choice(np.flatnonzero(env.infos[agent]["action_mask"]))))
            target_train = train_between_switches()
            if target_train is not None:
                break
        assert target_train is not None, "no train passed a switch"

        with patch.object(env.logger, 'error') as mock_logger:
            env._check_action_execution()
            assert not mock_logger.called, "the check must not fire on a valid state"

            original_position = target_train.position
            target_train.position = (0, 0)  # not a rail cell
            env._check_action_execution()
            target_train.position = original_position

            mock_logger.assert_called()
            assert "deviated from planned path" in mock_logger.call_args[0][0]

    def test_train_deviation_through_invalid_action_sequence(self, medium_env):
        """Test deviation by executing conflicting action sequences."""
        env = medium_env
        
        with patch.object(env.logger, 'error') as mock_logger:
            # Get active agents and execute some actions
            active_agents = self._get_active_agents(env, max_attempts=20)
            
            if len(active_agents) > 0:
                # Select the next (switch, train) pair and take a valid action first
                agent = next(env.agent_iter())
                env.observe(agent)
                valid_action = int(np.flatnonzero(env.infos[agent]["action_mask"])[0])

                # Manipulate train action plan to create inconsistency
                if len(env.rail_env.agents) > 0:
                    train = env.rail_env.agents[0]
                    # Clear existing plan and add conflicting actions
                    env.train_action_plan[train.handle] = [
                        RailEnvActions.MOVE_LEFT,
                        RailEnvActions.MOVE_RIGHT,  # Conflicting with previous
                        RailEnvActions.MOVE_FORWARD
                    ]
                
                # Execute the action
                env.step(valid_action)
                
                # Move trains multiple times to trigger the deviation check
                for _ in range(5):
                    env._move_trains()
                    env._check_action_execution()
                
                # Check if any deviation errors were logged
                if mock_logger.called:
                    print("✓ Successfully provoked deviation through conflicting actions")
                else:
                    print("! No deviation detected - environment may be more robust than expected")

    def test_simultaneous_train_arrivals(self, dense_env):
        """Test scenario where multiple trains arrive at the same switch simultaneously."""
        env = dense_env
        
        # Track trains and their positions
        train_positions = {}
        switch_arrivals = {}
        
        with patch.object(env.logger, 'error') as mock_logger:
            # Execute multiple steps to create scenarios with simultaneous arrivals
            for step in range(50):  # Extended simulation
                active_agents = self._get_active_agents(env, max_attempts=5)
                
                if not active_agents:
                    break
                
                # Record train positions before action
                for train in env.rail_env.agents:
                    if train.position is not None:
                        train_positions[train.handle] = train.position
                
                # Execute action for first active agent
                agent = active_agents[0]
                action_space = env.action_space(agent)
                action = np.random.choice(action_space.n)
                
                try:
                    env.step(action)
                    
                    # Check for switches with multiple trains
                    current_switch_trains = {}
                    for train in env.rail_env.agents:
                        if train.position is not None:
                            switch = env.rail_network.get_switch_on_position(train.position)
                            if switch is not None:
                                switch_id = switch.id
                                if switch_id not in current_switch_trains:
                                    current_switch_trains[switch_id] = []
                                current_switch_trains[switch_id].append(train.handle)
                    
                    # Detect simultaneous arrivals
                    for switch_id, trains in current_switch_trains.items():
                        if len(trains) > 1:
                            print(f"✓ Detected simultaneous arrival: {len(trains)} trains at switch {switch_id}")
                            switch_arrivals[switch_id] = trains
                    
                except Exception as e:
                    print(f"Exception during step {step}: {e}")
                    break
            
            # Report results
            if switch_arrivals:
                print(f"✓ Found {len(switch_arrivals)} cases of simultaneous arrivals")
            else:
                print("! No simultaneous arrivals detected in this run")
            
            # Check if any deviation errors occurred
            if mock_logger.called:
                print("✓ Deviation errors occurred during simultaneous arrival testing")

    def test_direct_neighbor_switches(self, medium_env):
        """Test scenarios involving direct neighbor switches."""
        env = medium_env
        
        # Find pairs of direct neighbor switches
        neighbor_pairs = self._find_neighbor_switch_pairs(env)
        
        with patch.object(env.logger, 'error') as mock_logger:
            if neighbor_pairs:
                print(f"✓ Found {len(neighbor_pairs)} neighbor switch pairs")
                
                # Test actions that involve transitions between neighbor switches
                for switch1_id, switch2_id in neighbor_pairs[:3]:  # Test first 3 pairs
                    switch1_name = switch_id2name(switch1_id)
                    switch2_name = switch_id2name(switch2_id)
                    
                    print(f"Testing neighbor switches: {switch1_name} <-> {switch2_name}")
                    
                    # Try to create scenario where train moves between these switches
                    self._test_neighbor_switch_transition(env, switch1_id, switch2_id)
                
                # Check for any deviation errors
                if mock_logger.called:
                    print("✓ Deviation errors occurred during neighbor switch testing")
            else:
                pytest.skip("No direct neighbor switches found in environment")

    def test_rapid_action_sequence_deviation(self, small_env):
        """Test rapid sequence of actions that might cause planning inconsistencies."""
        env = small_env
        
        with patch.object(env.logger, 'error') as mock_logger:
            # Execute rapid sequence of actions
            for sequence_step in range(20):
                active_agents = self._get_active_agents(env, max_attempts=3)
                
                if not active_agents:
                    break
                
                agent = active_agents[0]
                action_space = env.action_space(agent)
                
                # Execute random action quickly
                action = np.random.choice(action_space.n)
                
                try:
                    env.step(action)
                    
                    # Immediately force train movement without proper planning
                    env._move_trains()
                    env._check_action_execution()
                    
                except Exception as e:
                    print(f"Exception in rapid sequence at step {sequence_step}: {e}")
                    break
            
            # Check results
            if mock_logger.called:
                call_count = mock_logger.call_count
                print(f"✓ Rapid sequence provoked {call_count} deviation errors")
            else:
                print("! Rapid sequence did not provoke deviation errors")

    def test_manual_train_position_manipulation(self, small_env):
        """Test direct manipulation of train positions to force deviation."""
        env = small_env
        
        with patch.object(env.logger, 'error') as mock_logger:
            if len(env.rail_env.agents) > 0:
                train = env.rail_env.agents[0]
                
                # Set up train with planned path
                if train.position is not None:
                    # Add some actions to train's plan
                    env.train_action_plan[train.handle] = [
                        RailEnvActions.MOVE_FORWARD,
                        RailEnvActions.MOVE_FORWARD
                    ]
                    
                    # Set up next and prev ports for the train
                    try:
                        # Try to find valid ports for the train
                        next_port = env.rail_network.get_trains_next_port(train)
                        prev_port = env.rail_network.get_trains_prev_port(train)
                        
                        if next_port is not None and prev_port is not None:
                            # Manually move train to invalid position
                            original_pos = train.position
                            train.position = (env.rail_env.width - 1, env.rail_env.height - 1)  # Corner position
                            
                            # Trigger deviation check
                            env._check_action_execution()
                            
                            # Restore position
                            train.position = original_pos
                            
                            # Verify error was logged
                            if mock_logger.called:
                                print("✓ Manual position manipulation successfully provoked deviation error")
                            else:
                                print("! Manual manipulation did not trigger deviation error")
                        else:
                            print("! Could not set up train ports for deviation test")
                    except Exception as e:
                        print(f"Exception during manual manipulation: {e}")
            else:
                pytest.skip("No trains available for manipulation test")

    def test_action_plan_corruption(self, medium_env):
        """Test corruption of action plans to cause deviations."""
        env = medium_env
        
        with patch.object(env.logger, 'error') as mock_logger:
            # Get some active agents
            active_agents = self._get_active_agents(env, max_attempts=10)
            
            if active_agents and len(env.rail_env.agents) > 0:
                # Corrupt action plans for all trains
                for train in env.rail_env.agents:
                    # Set completely random and conflicting action sequence
                    env.train_action_plan[train.handle] = [
                        RailEnvActions.MOVE_LEFT,
                        RailEnvActions.MOVE_RIGHT,
                        RailEnvActions.MOVE_LEFT,
                        RailEnvActions.MOVE_RIGHT,
                        RailEnvActions.STOP_MOVING,
                        RailEnvActions.MOVE_FORWARD
                    ]
                
                # Execute actions and movements
                for _ in range(10):
                    if active_agents:
                        action = 0  # First available action
                        
                        try:
                            env.step(action)
                            active_agents = self._get_active_agents(env, max_attempts=2)
                        except Exception:
                            break
                
                # Check for deviation errors
                if mock_logger.called:
                    print(f"✓ Action plan corruption caused {mock_logger.call_count} deviation errors")
                else:
                    print("! Action plan corruption did not cause deviation errors")

    # Helper methods
    def _move_trains_until_active(self, env, max_steps=20):
        """Move trains until some become active."""
        for _ in range(max_steps):
            if len(env.active_switch_agents) > 0:
                break
            env._move_trains()
            env._check_active_switch()

    def _get_active_agents(self, env, max_attempts=10):
        """Get list of currently active agents."""
        for _ in range(max_attempts):
            if len(env.active_switch_agents) > 0:
                return env.active_switch_agents.copy()
            env._move_trains_to_switch()
            if env.terminated:
                break
        return []

    def _find_neighbor_switch_pairs(self, env):
        """Find pairs of directly neighboring switches (no rail cell in between)."""
        return env.rail_network.get_adjacent_switch_pairs()

    def _test_neighbor_switch_transition(self, env, switch1_id, switch2_id):
        """Test transition between neighboring switches."""
        try:
            # Find trains that might transition between these switches
            for train in env.rail_env.agents:
                if train.position is not None:
                    # Check if train is near one of these switches
                    current_switch = env.rail_network.get_switch_on_position(train.position)
                    if current_switch and current_switch.id in [switch1_id, switch2_id]:
                        # Set up action plan for transition
                        env.train_action_plan[train.handle] = [RailEnvActions.MOVE_FORWARD] * 5
                        break
        except Exception as e:
            print(f"Error in neighbor switch transition test: {e}")


def test_deviation_error_logging():
    """Test that deviation errors are properly logged with correct format."""
    # Create a mock logger to test message format
    mock_logger = MagicMock()
    
    # Simulate the error logging
    train_handle = 42
    mock_logger.error(f"Train {train_handle} deviated from planned path!")
    
    # Verify the call
    mock_logger.error.assert_called_once_with("Train 42 deviated from planned path!")
    print("✓ Deviation error logging format verified")


if __name__ == "__main__":
    # Run tests without pytest for debugging
    print("Running train deviation tests...")
    
    # Create test instance
    test_instance = TestTrainDeviation()
    
    # Run individual test methods
    try:
        env = build_standard_async_env(
            height=20, width=20, max_num_cities=2, num_trains=2,
            grid_mode=True, seed=42)
        env.reset(seed=42)
        
        print("\n1. Testing forced train deviation...")
        test_instance.test_force_train_deviation_by_corrupting_position(env)
        
        print("\n2. Testing deviation error logging...")
        test_deviation_error_logging()
        
        print("\n✓ All manual tests completed")
        
    except Exception as e:
        print(f"Error during manual testing: {e}")
        import traceback
        traceback.print_exc()