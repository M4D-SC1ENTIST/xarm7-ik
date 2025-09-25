import numpy as np
from xarm7_ik.solver import InverseKinematicsSolver, RotationRepresentation
import mujoco
import mujoco.viewer
import os

# Path to MJCF model with linear rail
MODEL_PATH = os.path.join(os.path.dirname(__file__), "xarm7_env/mjcf/xarm7_with_linear_rail.xml")

# Target gripper position and orientation
TARGET_POS = np.array([0.2, -0.0, 0.3])  
#TARGET_QUAT = np.array([0.707, 0.0, 0.0, 0.707])  
TARGET_QUAT = np.array([1, 0, 0.0, 0.0])  

# Initial joint configuration (8 DOF: rail + 7 joints)
initial_configuration = np.zeros(8)
initial_configuration[0] += 0.37

# Create IK solver instance (with rail)
ik_solver = InverseKinematicsSolver(
    use_linear_motor=True,
    rotation_repr="quaternion"
)

# Solve IK
ik_result = ik_solver.inverse_kinematics(
    initial_configuration,
    TARGET_POS,
    TARGET_QUAT,
)
print("IK result (rail + joint angles):", ik_result)

# Load Mujoco model
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Set rail and joint positions from IK result
ik_result[0] -= 0.37
for i in range(8):
    data.qpos[i] = ik_result[i]

# Step simulation to update geometry
mujoco.mj_forward(model, data)

# Launch Mujoco viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Press ESC to exit viewer.")
    while viewer.is_running():
        viewer.sync()
