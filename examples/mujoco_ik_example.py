import numpy as np
from xarm7_ik.solver import InverseKinematicsSolver, RotationRepresentation
import mujoco
import mujoco.viewer
import os

# Path to MJCF model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "xarm7_env/mjcf/xarm7.xml")

# Target gripper position and orientation
TARGET_POS = np.array([0.0, -0.5, 0.0])
TARGET_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

# Initial joint configuration (7 DOF)
initial_configuration = np.zeros(7)

# Create IK solver instance
ik_solver = InverseKinematicsSolver(
    use_linear_motor=False,
    rotation_repr="quaternion"
)

# Solve IK
ik_result = ik_solver.inverse_kinematics(
    initial_configuration,
    TARGET_POS,
    TARGET_QUAT,
)
print("IK result (joint angles):", ik_result)

# Load Mujoco model
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Set joint positions from IK result
for i in range(7):
    data.qpos[i] = ik_result[i]

# Step simulation to update geometry
mujoco.mj_forward(model, data)

# Launch Mujoco viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Press ESC to exit viewer.")
    while viewer.is_running():
        viewer.sync()
