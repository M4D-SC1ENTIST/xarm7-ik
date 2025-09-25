import numpy as np
from xarm7_ik.kinematics import LookAtInverseKinematicsSolver

def main():
    # Initial joint configuration (7 DOF, no linear motor)
    initial_configuration = np.zeros(7)

    # Target gripper position (in meters)
    target_gripper_pos = np.array([0.4, 0.2, 0.3])

    # Position that the gripper should look at (e.g., an object on a table)
    lookat_pos = np.array([0.0, 0.0, 0.1])

    # Create Look-At IK solver instance (no linear motor)
    # The lookat_offset defines the direction relative to the gripper frame that should point toward the target
    # Default is [0, 0, -0.15] which means 15cm below the gripper should point toward the lookat position
    lookat_solver = LookAtInverseKinematicsSolver(
        use_linear_motor=False,
        rotation_repr="quaternion",
        lookat_offset=np.array([0.0, 0.0, -0.15])  # Look direction relative to gripper
    )

    print("=== Look-At Inverse Kinematics Example ===")
    print(f"Target gripper position: {target_gripper_pos}")
    print(f"Look-at position: {lookat_pos}")
    print(f"Look-at offset: {lookat_solver.lookat_offset}")

    # Solve Look-At IK and measure frequency
    import time
    iter = 10  # Fewer iterations for testing
    start_time = time.time()
    
    for i in range(iter):
        result = lookat_solver.inverse_kinematics(
            initial_configuration,
            target_gripper_pos,
            lookat_pos
        )
        
        if i == 0:  # Print detailed results for first iteration
            print(f"\nLook-At IK result (joint angles): {result}")
            
            # Verify the result by computing forward kinematics
            pos, quat = lookat_solver.forward_kinematics(result)
            print(f"Achieved gripper position: {pos}")
            print(f"Achieved gripper orientation (quat): {quat}")
            
            # Calculate position error
            pos_error = np.linalg.norm(pos - target_gripper_pos)
            print(f"Position error: {pos_error:.6f} meters")
            
            # Calculate look-at direction
            from xarm7_ik.utils import calculate_look_at_error
            lookat_error = calculate_look_at_error(pos, lookat_pos, quat, lookat_solver.lookat_offset)
            print(f"Look-at error: {lookat_error:.6f}")

    end_time = time.time()
    elapsed = end_time - start_time
    frequency = iter / elapsed if elapsed > 0 else float('inf')
    print(f"\nLook-At IK solver frequency: {frequency:.2f} Hz over {iter} iterations")

    # Test different scenarios
    print("\n=== Testing Different Scenarios ===")
    
    # Scenario 1: Look at a point to the left
    lookat_left = np.array([-0.3, 0.5, 0.2])
    result_left = lookat_solver.inverse_kinematics(
        initial_configuration,
        target_gripper_pos,
        lookat_left
    )
    pos_left, _ = lookat_solver.forward_kinematics(result_left)
    pos_error_left = np.linalg.norm(pos_left - target_gripper_pos)
    print(f"Scenario 1 - Look left at {lookat_left}: Position error = {pos_error_left:.6f}m")

    # Scenario 2: Look at a point to the right  
    lookat_right = np.array([0.3, -0.5, 0.2])
    result_right = lookat_solver.inverse_kinematics(
        initial_configuration,
        target_gripper_pos,
        lookat_right
    )
    pos_right, _ = lookat_solver.forward_kinematics(result_right)
    pos_error_right = np.linalg.norm(pos_right - target_gripper_pos)
    print(f"Scenario 2 - Look right at {lookat_right}: Position error = {pos_error_right:.6f}m")

    # Scenario 3: Look at a point above
    lookat_up = np.array([0.4, 0.2, 0.8])
    result_up = lookat_solver.inverse_kinematics(
        initial_configuration,
        target_gripper_pos,
        lookat_up
    )
    pos_up, _ = lookat_solver.forward_kinematics(result_up)
    pos_error_up = np.linalg.norm(pos_up - target_gripper_pos)
    print(f"Scenario 3 - Look up at {lookat_up}: Position error = {pos_error_up:.6f}m")

if __name__ == "__main__":
    main()
