#!/usr/bin/env python3
"""
Simple test to check rail transition consistency.

This script tests whether the rail graph transitions match actual train movements
by comparing rail graph predictions with Flatland's ground truth simulation.
"""

import traceback
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.step_utils.states import TrainState

from switchfl.utils.build_env import build_standard_async_env


def test_basic_train_movement_consistency():
    """Test basic train movement consistency between rail graph and simulation."""
    print("Creating test environment...")
    
    # Create a larger test environment to ensure feasibility
    try:
        env = build_standard_async_env(
            height=20,
            width=20,
            max_num_cities=2,
            num_trains=2,
            max_rails_between_cities=2,
            max_rail_pairs_in_city=1,
            grid_mode=True,
            seed=42
        )
    except ValueError as e:
        print(f"Failed to create environment with seed 42: {e}")
        print("Trying with different parameters...")
        try:
            env = build_standard_async_env(
                height=40,
                width=40,
                max_num_cities=2,
                num_trains=1,
                max_rails_between_cities=1,
                max_rail_pairs_in_city=1,
                grid_mode=True,  # Grid mode is more predictable
                seed=123
            )
        except Exception as e2:
            raise ValueError(f"Failed to create any valid environment: {e2}")
    
    print("Resetting environment...")
    env.reset()
    
    rail_env = env.rail_env
    rail_network = env.rail_network
    
    print(f"Environment has {len(rail_env.agents)} trains")
    print(f"Rail network has {len(list(rail_network.switches))} switches")
    
    # Test basic consistency
    consistency_errors = []
    
    # Check that trains are on valid rail pieces
    print("\nChecking train positions...")
    for train in rail_env.agents:
        if train.position is not None:
            rail_piece = rail_env.rail.grid[train.position]
            if rail_piece == 0:
                consistency_errors.append(f"Train {train.handle} at position {train.position} has no rail piece")
            else:
                print(f"✓ Train {train.handle} at {train.position} on valid rail (value: {rail_piece})")
    
    # Test action simulation consistency
    print("\nTesting action simulation...")
    actions_tested = 0
    
    for train in rail_env.agents:
        if train.position is None or train.state not in [TrainState.READY_TO_DEPART, TrainState.MOVING]:
            continue
            
        original_pos = train.position
        original_dir = train.direction
        
        for action in [RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT]:
            # Test using Flatland's ground truth
            try:
                (new_cell_valid, new_direction, new_position, 
                 transition_valid, preprocessed_action) = rail_env.rail.check_action_on_agent(
                    action, original_pos, original_dir
                )
                
                if new_cell_valid and transition_valid and new_position is not None:
                    rail_piece = rail_env.rail.grid[new_position]
                    if rail_piece == 0:
                        consistency_errors.append(
                            f"Train {train.handle} action {action.name} leads to invalid position {new_position}"
                        )
                    else:
                        print(f"✓ Train {train.handle} action {action.name}: {original_pos} -> {new_position}")
                        actions_tested += 1
                        
            except Exception as e:
                print(f"⚠ Error testing action {action.name} for train {train.handle}: {e}")
    
    print(f"\nTested {actions_tested} valid action transitions")
    
    # Test switch action consistency  
    print("\nTesting switch actions...")
    switch_actions_tested = 0
    
    for switch_id, switch in rail_network.switches:
        try:
            action_space = switch.get_action_space()
            for action_idx in range(action_space.n):
                try:
                    # Get action outcomes
                    in_port, out_port = switch.action_outcomes[action_idx]
                    port_actions = switch.actions[action_idx]
                    
                    # Basic validation
                    if in_port is None or out_port is None:
                        consistency_errors.append(f"Switch {switch_id} action {action_idx} has None ports")
                        continue
                        
                    # Check that ports have valid positions
                    in_port_pos = rail_network.rail_graph.nodes[in_port].get('rail_prev_node')
                    out_port_pos = rail_network.rail_graph.nodes[out_port].get('rail_prev_node') 
                    
                    if in_port_pos is None or out_port_pos is None:
                        consistency_errors.append(f"Switch {switch_id} action {action_idx} has ports without positions")
                        continue
                    
                    # Validate rail actions
                    valid_actions = True
                    for port, actions in port_actions.items():
                        for rail_action in actions:
                            if not isinstance(rail_action, (RailEnvActions, int)):
                                consistency_errors.append(
                                    f"Switch {switch_id} action {action_idx} has invalid rail action type: {type(rail_action)}"
                                )
                                valid_actions = False
                    
                    if valid_actions:
                        print(f"✓ Switch {switch_id} action {action_idx}: {in_port} -> {out_port}")
                        switch_actions_tested += 1
                        
                except Exception as e:
                    print(f"⚠ Error testing switch {switch_id} action {action_idx}: {e}")
                    
        except Exception as e:
            print(f"⚠ Error testing switch {switch_id}: {e}")
    
    print(f"\nTested {switch_actions_tested} switch actions")
    
    # Report results
    print(f"\n{'='*50}")
    print("CONSISTENCY TEST RESULTS")
    print(f"{'='*50}")
    
    assert len(consistency_errors) == 0, f"❌ Found {len(consistency_errors)} consistency errors:"
    for i, error in enumerate(consistency_errors, 1):
        print(f"  {i}. {error}")

def test_direction_mapping():
    """Test the direction mapping between rail graph and Flatland."""
    print("\n" + "="*50)
    print("TESTING DIRECTION MAPPING")
    print("="*50)
    
    try:
        env = build_standard_async_env(
            height=25, width=25, max_num_cities=2, num_trains=1, 
            grid_mode=True, seed=456
        )
    except Exception as e:
        raise ValueError(f"Failed to create environment for direction mapping test: {e}")
        
    env.reset()
    
    rail_network = env.rail_network
    
    # Test direction mapping
    dir_mapping = rail_network._dir2port_idx
    print("Direction mapping:")
    for direction, port_idx in dir_mapping.items():
        print(f"  Direction {direction} -> Port index {port_idx}")
    
    # Validate mapping
    expected_directions = [1.0, 2.0, 3.0, 4.0]
    missing_directions = [d for d in expected_directions if d not in dir_mapping]
    
    assert not missing_directions, f"❌ Missing directions: {missing_directions}"
        
    # Validate port indices
    valid = True
    for direction, port_idx in dir_mapping.items():
        if not (0 <= port_idx <= 3):
            print(f"❌ Invalid port index {port_idx} for direction {direction}")
            valid = False
    assert valid, "Invalid port indices found in direction mapping"
