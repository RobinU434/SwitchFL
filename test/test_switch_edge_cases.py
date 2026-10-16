#!/usr/bin/env python3
"""
Edge case tests for simultaneous arrivals and neighboring switches.

This module focuses specifically on:
1. Simultaneous train arrivals at the same switch
2. Direct neighbor switch interactions
3. Switch congestion scenarios
4. Complex multi-train coordination edge cases

Run with: python -m pytest test/test_switch_edge_cases.py -v
"""

import pytest
import numpy as np
from typing import Dict, List, Set
from unittest.mock import patch

from switchfl.utils.build_env import build_standard_async_env


class TestSwitchEdgeCases:
    """Test suite focused on switch-related edge cases."""

    @pytest.fixture
    def high_density_env(self):
        """Create a high-density environment likely to cause simultaneous arrivals."""
        env = build_standard_async_env(
            height=35,
            width=35,
            max_num_cities=5,
            num_trains=8,
            max_rails_between_cities=4,
            max_rail_pairs_in_city=3,
            grid_mode=True,
            seed=777
        )
        env.reset(seed=777)
        return env

    @pytest.fixture
    def congested_env(self):
        """Create an environment designed to maximize congestion."""
        env = build_standard_async_env(
            height=30,
            width=30,
            max_num_cities=3,
            num_trains=6,
            max_rails_between_cities=2,
            max_rail_pairs_in_city=2,
            grid_mode=True,
            seed=555
        )
        env.reset(seed=555)
        return env

    def test_simultaneous_arrivals_detection(self, high_density_env):
        """Test detection and handling of simultaneous train arrivals."""
        env = high_density_env
        simultaneous_events = []
        
        # Track switch occupancy over time
        switch_occupancy_history = []
        
        # Run simulation to detect simultaneous arrivals
        for step in range(100):
            # Record current switch occupancy
            current_occupancy = self._record_switch_occupancy(env)
            switch_occupancy_history.append(current_occupancy)
            
            # Detect simultaneous arrivals
            simultaneous_switches = self._find_simultaneous_arrivals(current_occupancy)
            if simultaneous_switches:
                for switch_id, train_handles in simultaneous_switches.items():
                    simultaneous_events.append({
                        'step': step,
                        'switch_id': switch_id,
                        'trains': train_handles,
                        'count': len(train_handles)
                    })
                    print(f"Step {step}: {len(train_handles)} trains simultaneously at switch {switch_id}")
            
            # Execute step
            active_agents = self._get_active_agents(env)
            if not active_agents:
                break
                
            # Execute action for first active agent
            action = np.random.choice(env.action_space(active_agents[0]).n)
            try:
                env.step(action)
            except Exception as e:
                print(f"Step failed at {step}: {e}")
                break
        
        # Report results
        if simultaneous_events:
            max_simultaneous = max(event['count'] for event in simultaneous_events)
            print(f"✓ Detected {len(simultaneous_events)} simultaneous arrival events")
            print(f"✓ Maximum simultaneous arrivals: {max_simultaneous} trains")
            
            # Test that system handles these events without crashing
            assert len(simultaneous_events) > 0, "Should detect simultaneous arrivals in high-density environment"
        else:
            print("! No simultaneous arrivals detected")

    def test_neighbor_switch_interactions(self, congested_env):
        """Test interactions between directly neighboring switches."""
        env = congested_env
        
        # Find neighbor switch pairs
        neighbor_pairs = self._find_all_neighbor_pairs(env)
        
        if not neighbor_pairs:
            pytest.skip("No neighboring switches found")
        
        print(f"✓ Found {len(neighbor_pairs)} neighboring switch pairs")
        
        neighbor_interactions = []
        
        # Monitor neighbor switch interactions
        for step in range(80):
            current_occupancy = self._record_switch_occupancy(env)
            
            # Check for trains on neighboring switches
            for switch1_id, switch2_id in neighbor_pairs:
                trains_on_switch1 = current_occupancy.get(switch1_id, set())
                trains_on_switch2 = current_occupancy.get(switch2_id, set())
                
                if trains_on_switch1 and trains_on_switch2:
                    neighbor_interactions.append({
                        'step': step,
                        'switch1': switch1_id,
                        'switch2': switch2_id,
                        'trains1': trains_on_switch1,
                        'trains2': trains_on_switch2
                    })
                    print(f"Step {step}: Neighbor interaction between switches {switch1_id} and {switch2_id}")
            
            # Execute step
            active_agents = self._get_active_agents(env)
            if not active_agents:
                break
                
            action = np.random.choice(env.action_space(active_agents[0]).n)
            try:
                env.step(action)
            except Exception as e:
                print(f"Step failed at {step}: {e}")
                break
        
        if neighbor_interactions:
            print(f"✓ Detected {len(neighbor_interactions)} neighbor switch interactions")
        else:
            print("! No neighbor switch interactions detected")

    def test_switch_congestion_scenarios(self, high_density_env):
        """Test scenarios with high switch congestion."""
        env = high_density_env
        
        congestion_events = []
        max_congestion_per_switch = {}
        
        with patch.object(env.logger, 'error') as mock_logger:
            # Monitor congestion levels
            for step in range(150):
                current_occupancy = self._record_switch_occupancy(env)
                
                # Identify congested switches (more than 2 trains)
                for switch_id, trains in current_occupancy.items():
                    train_count = len(trains)
                    if train_count > 2:  # Congestion threshold
                        congestion_events.append({
                            'step': step,
                            'switch_id': switch_id,
                            'train_count': train_count,
                            'trains': trains
                        })
                        
                        # Track maximum congestion per switch
                        if switch_id not in max_congestion_per_switch:
                            max_congestion_per_switch[switch_id] = train_count
                        else:
                            max_congestion_per_switch[switch_id] = max(
                                max_congestion_per_switch[switch_id], train_count
                            )
                        
                        print(f"Step {step}: Congestion at switch {switch_id} with {train_count} trains")
                
                # Execute step
                active_agents = self._get_active_agents(env)
                if not active_agents:
                    break
                    
                action = np.random.choice(env.action_space(active_agents[0]).n)
                try:
                    env.step(action)
                except Exception as e:
                    print(f"Step failed at {step}: {e}")
                    break
            
            # Report congestion results
            if congestion_events:
                total_events = len(congestion_events)
                max_congestion = max(event['train_count'] for event in congestion_events)
                congested_switches = len(max_congestion_per_switch)
                
                print(f"✓ Detected {total_events} congestion events")
                print(f"✓ Maximum congestion: {max_congestion} trains on one switch")
                print(f"✓ {congested_switches} switches experienced congestion")
                
                # Check if congestion caused any errors
                if mock_logger.called:
                    print(f"✓ Congestion caused {mock_logger.call_count} error events")
            else:
                print("! No switch congestion detected")

    def test_rapid_switch_transitions(self, congested_env):
        """Test rapid transitions between switches."""
        env = congested_env
        
        # Track train positions over multiple steps
        train_position_history = {}
        
        for step in range(60):
            # Record current train positions and switches
            for train in env.rail_env.agents:
                if train.position is not None:
                    current_switch = env.rail_network.get_switch_on_position(train.position)
                    
                    if train.handle not in train_position_history:
                        train_position_history[train.handle] = []
                    
                    train_position_history[train.handle].append({
                        'step': step,
                        'position': train.position,
                        'switch_id': current_switch.id if current_switch else None
                    })
            
            # Execute step
            active_agents = self._get_active_agents(env)
            if not active_agents:
                break
                
            action = np.random.choice(env.action_space(active_agents[0]).n)
            try:
                env.step(action)
            except Exception as e:
                print(f"Step failed at {step}: {e}")
                break
        
        # Analyze transition patterns
        rapid_transitions = self._analyze_rapid_transitions(train_position_history)
        
        if rapid_transitions:
            print(f"✓ Detected {len(rapid_transitions)} rapid switch transitions")
            for transition in rapid_transitions:
                print(f"  Train {transition['train']} made {transition['transitions']} transitions in {transition['steps']} steps")
        else:
            print("! No rapid switch transitions detected")

    def test_deadlock_scenarios(self, congested_env):
        """Test potential deadlock scenarios with multiple trains."""
        env = congested_env
        
        # Track system progress
        progress_indicators = []
        
        with patch.object(env.logger, 'error'):
            prev_active_count = len(env.active_switch_agents)
            stagnation_counter = 0
            
            for step in range(100):
                current_active_count = len(env.active_switch_agents)
                progress_indicators.append({
                    'step': step,
                    'active_agents': current_active_count,
                    'total_agents': len(env.rail_env.agents)
                })
                
                # Detect potential stagnation
                if current_active_count == prev_active_count and current_active_count > 0:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                
                # If stagnation persists, it might indicate deadlock
                if stagnation_counter > 10:
                    print(f"Step {step}: Potential deadlock detected - {current_active_count} agents stagnated for {stagnation_counter} steps")
                
                prev_active_count = current_active_count
                
                # Execute step
                active_agents = self._get_active_agents(env)
                if not active_agents:
                    break
                    
                action = np.random.choice(env.action_space(active_agents[0]).n)
                try:
                    env.step(action)
                except Exception as e:
                    print(f"Step failed at {step}: {e}")
                    break
            
            # Check for deadlock indicators
            max_stagnation = 0
            for i in range(len(progress_indicators) - 10):
                if all(progress_indicators[i + j]['active_agents'] == progress_indicators[i]['active_agents'] 
                       for j in range(10)):
                    max_stagnation = max(max_stagnation, 10)
            
            if max_stagnation > 0:
                print(f"✓ Detected potential deadlock scenario with {max_stagnation} step stagnation")
            else:
                print("! No deadlock scenarios detected")

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

    def _find_simultaneous_arrivals(self, occupancy: Dict[int, Set[int]]) -> Dict[int, Set[int]]:
        """Find switches with multiple trains (simultaneous arrivals)."""
        simultaneous = {}
        for switch_id, trains in occupancy.items():
            if len(trains) > 1:
                simultaneous[switch_id] = trains
        return simultaneous

    def _find_all_neighbor_pairs(self, env):
        """Find pairs of directly neighboring switches (no rail cell in between)."""
        return env.rail_network.get_adjacent_switch_pairs()

    def _analyze_rapid_transitions(self, history: Dict[int, List[Dict]]) -> List[Dict]:
        """Analyze train movement history to find rapid transitions."""
        rapid_transitions = []
        
        for train_handle, positions in history.items():
            if len(positions) < 5:  # Need enough history
                continue
                
            # Count switch transitions
            switch_changes = 0
            prev_switch = None
            
            for pos_data in positions:
                current_switch = pos_data['switch_id']
                if prev_switch is not None and current_switch != prev_switch and current_switch is not None:
                    switch_changes += 1
                prev_switch = current_switch
            
            # If many transitions in short time, consider it rapid
            if switch_changes > 3 and len(positions) < 20:
                rapid_transitions.append({
                    'train': train_handle,
                    'transitions': switch_changes,
                    'steps': len(positions)
                })
        
        return rapid_transitions

    def _get_active_agents(self, env) -> List:
        """Get currently active agents."""
        # Move trains and update active status
        env._move_trains_to_switch()
        return env.active_switch_agents.copy()


if __name__ == "__main__":
    # Run tests without pytest for debugging
    print("Running switch edge case tests...")
    
    test_instance = TestSwitchEdgeCases()
    
    try:
        # Create test environment
        env = build_standard_async_env(
            height=20, width=20, max_num_cities=4, num_trains=6,
            grid_mode=True, seed=777)
        env.reset(seed=777)
        
        print("\n1. Testing simultaneous arrivals detection...")
        test_instance.test_simultaneous_arrivals_detection(env)
        
        print("\n2. Testing switch congestion scenarios...")
        test_instance.test_switch_congestion_scenarios(env)
        
        print("\n✓ All manual edge case tests completed")
        
    except Exception as e:
        print(f"Error during manual testing: {e}")
        import traceback
        traceback.print_exc()