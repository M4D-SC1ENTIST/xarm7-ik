
import numpy as np
from xarm7_ik.solver import InverseKinematicsSolver, RotationRepresentation

def main():
    # Initial joint configuration (7 DOF, no linear motor)
    initial_configuration = np.zeros(7)

    # Target gripper position (in meters)
    target_gripper_pos = np.array([0.4, 0.0, 0.3])

    # Target gripper orientation as quaternion (w, x, y, z)
    # Here, identity quaternion (no rotation)
    target_gripper_rot = np.array([1.0, 0.0, 0.0, 0.0])

    # Create IK solver instance (no linear motor)
    ik_solver = InverseKinematicsSolver(
        use_linear_motor=False,
        rotation_repr="quaternion"
    )

    # Solve IK and measure frequency
    import time
    iter = 100
    start_time = time.time()
    for i in range(iter):
        result = ik_solver.inverse_kinematics(
            initial_configuration,
            target_gripper_pos,
            target_gripper_rot,
        )
        print("IK result (joint angles):", result)
    end_time = time.time()
    elapsed = end_time - start_time
    frequency = iter / elapsed if elapsed > 0 else float('inf')
    print(f"IK solver frequency: {frequency:.2f} Hz over {iter} iterations")

if __name__ == "__main__":
    main()
