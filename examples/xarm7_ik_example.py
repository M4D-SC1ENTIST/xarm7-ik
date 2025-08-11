
import numpy as np
from solver import InverseKinematicsSolver, RotationRepresentation

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
        rotation_repr=RotationRepresentation.QUATERNION
    )

    # Solve IK
    result = ik_solver.inverse_kinematics(
        initial_configuration,
        target_gripper_pos,
        target_gripper_rot,
        rot_repr=RotationRepresentation.QUATERNION
    )

    print("IK result (joint angles):", result)

if __name__ == "__main__":
    main()
