import numpy as np
from xarm7_ik.solver import LookAtInverseKinematicsSolver

def main():
    # Initial joint configuration (8 DOF with linear motor)
    # [linear_position, joint1, joint2, joint3, joint4, joint5, joint6, joint7]
    initial_configuration = np.zeros(8)
    initial_configuration[0] = 0.3  # Start linear rail at 0.3m position

    # Target gripper position (in meters) - further away to utilize the rail
    target_gripper_pos = np.array([0.8, 0.3, 0.4])

    # Position that the gripper should look at (e.g., an object on a conveyor belt)
    lookat_pos = np.array([0.2, -0.2, 0.15])

    # Create Look-At IK solver instance WITH linear motor
    # The linear_motor_x_offset defines the X offset of the linear rail from the base
    lookat_solver = LookAtInverseKinematicsSolver(
        use_linear_motor=True,
        linear_motor_x_offset=0.0,  # Rail aligned with robot base
        rotation_repr="quaternion",
        lookat_offset=np.array([0.0, 0.0, -0.15])  # Look direction relative to gripper
    )

    print("=== Look-At Inverse Kinematics with Linear Rail Example ===")
    print(f"Target gripper position: {target_gripper_pos}")
    print(f"Look-at position: {lookat_pos}")
    print(f"Look-at offset: {lookat_solver.lookat_offset}")
    print(f"Using linear motor: {lookat_solver.use_linear_motor}")
    print(f"Linear motor X offset: {lookat_solver.linear_motor_x_offset}")

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
            print(f"\nLook-At IK result:")
            print(f"  Linear rail position: {result[0]:.4f} m")
            print(f"  Joint angles: {result[1:]}")
            
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

    # Test different scenarios that benefit from the linear rail
    print("\n=== Testing Different Scenarios with Linear Rail ===")
    
    # Scenario 1: Far target position (utilizing rail extension)
    target_far = np.array([1.2, 0.0, 0.3])
    lookat_far = np.array([0.5, 0.5, 0.1])
    result_far = lookat_solver.inverse_kinematics(
        initial_configuration,
        target_far,
        lookat_far
    )
    pos_far, _ = lookat_solver.forward_kinematics(result_far)
    pos_error_far = np.linalg.norm(pos_far - target_far)
    print(f"Scenario 1 - Far target {target_far}:")
    print(f"  Linear rail position: {result_far[0]:.4f} m")
    print(f"  Position error: {pos_error_far:.6f} m")

    # Scenario 2: Moving along a line (simulating conveyor belt tracking)
    print("\nScenario 2 - Conveyor belt tracking simulation:")
    conveyor_positions = [
        np.array([0.6, -0.3, 0.25]),
        np.array([0.8, -0.1, 0.25]),
        np.array([1.0, 0.1, 0.25]),
        np.array([1.2, 0.3, 0.25])
    ]
    
    for i, conv_pos in enumerate(conveyor_positions):
        # Object to look at moves along with the conveyor
        conv_lookat = conv_pos + np.array([-0.2, -0.2, -0.1])
        
        result_conv = lookat_solver.inverse_kinematics(
            initial_configuration,
            conv_pos,
            conv_lookat
        )
        pos_conv, _ = lookat_solver.forward_kinematics(result_conv)
        pos_error_conv = np.linalg.norm(pos_conv - conv_pos)
        
        print(f"  Position {i+1}: target={conv_pos}, rail={result_conv[0]:.3f}m, error={pos_error_conv:.6f}m")

    # Scenario 3: Different rail starting positions
    print("\nScenario 3 - Different rail starting positions:")
    rail_starts = [0.0, 0.2, 0.4, 0.6]
    target_test = np.array([0.9, 0.2, 0.35])
    lookat_test = np.array([0.3, 0.0, 0.2])
    
    for rail_start in rail_starts:
        init_config = np.zeros(8)
        init_config[0] = rail_start
        
        result_rail = lookat_solver.inverse_kinematics(
            init_config,
            target_test,
            lookat_test
        )
        pos_rail, _ = lookat_solver.forward_kinematics(result_rail)
        pos_error_rail = np.linalg.norm(pos_rail - target_test)
        rail_movement = abs(result_rail[0] - rail_start)
        
        print(f"  Start rail={rail_start:.1f}m -> Final rail={result_rail[0]:.3f}m (moved {rail_movement:.3f}m), error={pos_error_rail:.6f}m")

    print("\n=== Linear Rail Workspace Demonstration ===")
    # Show how the linear rail extends the workspace
    print("Testing workspace extension with linear rail...")
    
    # Test positions that would be unreachable without the rail
    extreme_positions = [
        np.array([1.3, 0.0, 0.3]),  # Very far in X
        np.array([0.9, 0.6, 0.2]),  # Far in Y
        np.array([1.1, -0.5, 0.4])  # Far negative Y
    ]
    
    for i, ext_pos in enumerate(extreme_positions):
        ext_lookat = np.array([0.0, 0.0, 0.1])  # Look at origin
        
        result_ext = lookat_solver.inverse_kinematics(
            initial_configuration,
            ext_pos,
            ext_lookat
        )
        pos_ext, _ = lookat_solver.forward_kinematics(result_ext)
        pos_error_ext = np.linalg.norm(pos_ext - ext_pos)
        
        print(f"  Extreme position {i+1}: {ext_pos}")
        print(f"    Rail position: {result_ext[0]:.3f}m")
        print(f"    Position error: {pos_error_ext:.6f}m")
        print(f"    {'SUCCESS' if pos_error_ext < 0.01 else 'CHALLENGING'}")

if __name__ == "__main__":
    main()
