#!/usr/bin/env python3
"""
Example demonstrating base rotation offset functionality for xarm7-ik solver.

This example shows how to use the IK solver when the robotic arm base is rotated
from its canonical orientation (e.g., rotated 90 degrees left or right due to 
space constraints).
"""

import numpy as np
from xarm7_ik.solver import InverseKinematicsSolver


def test_base_rotation_offset():
    """Test IK solver with different base rotation offsets."""
    
    # Target gripper position and orientation (same for all tests)
    target_gripper_pos = np.array([0.4, 0.2, 0.3])  # 40cm forward, 20cm to the right, 30cm up
    target_gripper_rot = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion (no rotation)
    
    # Initial joint configuration
    initial_configuration = np.zeros(7)
    
    print("=== Base Rotation Offset IK Solver Example ===\n")
    
    # Test cases with different base rotation offsets
    test_cases = [
        ("Canonical orientation (no offset)", 0.0),
        ("90° counter-clockwise (π/2)", np.pi/2),
        ("90° clockwise (-π/2)", -np.pi/2),
        ("180° rotation (π)", np.pi),
        ("45° counter-clockwise (π/4)", np.pi/4),
    ]
    
    for description, base_rotation_offset in test_cases:
        print(f"--- {description} ---")
        print(f"Base rotation offset: {base_rotation_offset:.4f} rad ({np.degrees(base_rotation_offset):.1f}°)")
        
        # Create IK solver with base rotation offset
        ik_solver = InverseKinematicsSolver(
            use_linear_motor=False,
            rotation_repr="quaternion",
            base_rotation_offset=base_rotation_offset
        )
        
        # Solve IK
        result_joints = ik_solver.inverse_kinematics(
            initial_configuration,
            target_gripper_pos,
            target_gripper_rot
        )
        
        # Verify the result using forward kinematics
        achieved_pos, achieved_rot = ik_solver.forward_kinematics(result_joints)
        
        # Calculate errors
        pos_error = np.linalg.norm(achieved_pos - target_gripper_pos)
        
        # For orientation error, we need to account for the quaternion offset that the solver applies
        # The solver applies a quat_offset of [0.0, 1.0, 0.0, 0.0] to transform coordinate frames
        # So we need to apply the same offset to our target for comparison
        from xarm7_ik.utils import quaternion_multiply, normalize_quaternion
        quat_offset = np.array([0.0, 1.0, 0.0, 0.0])
        expected_target_quat = quaternion_multiply(quat_offset, target_gripper_rot)
        expected_target_quat = normalize_quaternion(expected_target_quat)
        
        # Use a proper quaternion distance metric
        from xarm7_ik.utils import quaternion_displacement_based_distance
        rot_error = quaternion_displacement_based_distance(achieved_rot, expected_target_quat)
        
        print(f"Target position:    [{target_gripper_pos[0]:.3f}, {target_gripper_pos[1]:.3f}, {target_gripper_pos[2]:.3f}]")
        print(f"Achieved position:  [{achieved_pos[0]:.3f}, {achieved_pos[1]:.3f}, {achieved_pos[2]:.3f}]")
        print(f"Position error:     {pos_error:.6f} m")
        print(f"Target orientation: [{target_gripper_rot[0]:.3f}, {target_gripper_rot[1]:.3f}, {target_gripper_rot[2]:.3f}, {target_gripper_rot[3]:.3f}]")
        print(f"Expected (with offset): [{expected_target_quat[0]:.3f}, {expected_target_quat[1]:.3f}, {expected_target_quat[2]:.3f}, {expected_target_quat[3]:.3f}]")
        print(f"Achieved orientation: [{achieved_rot[0]:.3f}, {achieved_rot[1]:.3f}, {achieved_rot[2]:.3f}, {achieved_rot[3]:.3f}]")
        print(f"Orientation error:  {rot_error:.6f}")
        print(f"Joint angles (rad): {result_joints}")
        print(f"Joint angles (deg): {np.degrees(result_joints)}")
        
        # Success criteria
        success = pos_error < 1e-3 and rot_error < 1e-2
        print(f"IK Success: {'✓' if success else '✗'}")
        print()


def demonstrate_workspace_difference():
    """Demonstrate how base rotation affects the workspace."""
    
    print("=== Workspace Demonstration ===\n")
    
    # Test reaching the same global position with different base rotations
    global_target_pos = np.array([0.4, 0.3, 0.2])  # Fixed global position
    target_gripper_rot = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    initial_configuration = np.zeros(7)
    
    base_rotations = [0.0, np.pi/2, np.pi, -np.pi/2]
    rotation_names = ["0°", "90°", "180°", "-90°"]
    
    print(f"Trying to reach global position: [{global_target_pos[0]:.3f}, {global_target_pos[1]:.3f}, {global_target_pos[2]:.3f}]")
    print("with different base orientations:\n")
    
    for i, (base_rot, name) in enumerate(zip(base_rotations, rotation_names)):
        print(f"Base rotation {name} ({base_rot:.3f} rad):")
        
        # Create solver with base rotation
        ik_solver = InverseKinematicsSolver(
            use_linear_motor=False,
            rotation_repr="quaternion", 
            base_rotation_offset=base_rot
        )
        
        # Solve IK for the global target position
        result_joints = ik_solver.inverse_kinematics(
            initial_configuration,
            global_target_pos,
            target_gripper_rot
        )
        
        # Verify with forward kinematics
        achieved_pos, achieved_rot = ik_solver.forward_kinematics(result_joints)
        
        pos_error = np.linalg.norm(achieved_pos - global_target_pos)
        
        # Check orientation error with proper quaternion handling
        from xarm7_ik.utils import quaternion_multiply, normalize_quaternion, quaternion_displacement_based_distance
        quat_offset = np.array([0.0, 1.0, 0.0, 0.0])
        expected_target_quat = quaternion_multiply(quat_offset, target_gripper_rot)
        expected_target_quat = normalize_quaternion(expected_target_quat)
        rot_error = quaternion_displacement_based_distance(achieved_rot, expected_target_quat)
        
        print(f"  Joint solution: {np.degrees(result_joints)}")
        print(f"  Achieved position: [{achieved_pos[0]:.3f}, {achieved_pos[1]:.3f}, {achieved_pos[2]:.3f}]")
        print(f"  Position error: {pos_error:.6f} m")
        print(f"  Orientation error: {rot_error:.6f}")
        print(f"  Success: {'✓' if pos_error < 1e-3 and rot_error < 1e-2 else '✗'}")
        print()


def main():
    """Run the base rotation offset examples."""
    try:
        test_base_rotation_offset()
        demonstrate_workspace_difference()
        
        print("=== Summary ===")
        print("The base rotation offset functionality allows the IK solver to work correctly")
        print("even when the robotic arm base is rotated from its canonical orientation.")
        print("This is useful in scenarios where space constraints require the arm to be")
        print("mounted in a non-standard orientation (e.g., rotated 90° left or right).")
        print("\nThe solver automatically:")
        print("1. Applies the base rotation at the beginning of the kinematic chain")
        print("2. Maintains the target pose in the global coordinate frame")
        print("3. Ensures the end-effector reaches the desired global pose")
        print("4. Works with all rotation representations (quaternion, euler, axis-angle)")
        print("\nNote: The solver applies an internal coordinate frame transformation")
        print("(quat_offset) to align with the robot's natural end-effector orientation.")
        
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
