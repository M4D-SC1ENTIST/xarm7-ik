# Fast Inverse Kinematics Solver for xArm7

This package provides a fast inverse kinematics (IK) solver for the xArm7 robotic arm, capable of running at approximately 150 Hz on a laptop with an i7-11800H CPU. It supports both standard 7-DOF xArm7 and configurations with a linear rail (8-DOF).


## Features
- Fast inverse kinematics for xArm7 (7-DOF) and xArm7 with linear rail (8-DOF)
- Supports quaternion, Euler angle, and axis-angle rotation representations
- Lightweight and easy to integrate

## Installation

**Requirements:**
- Python >= 3.10
- [nlopt](https://github.com/stevengj/nlopt), [numba](https://numba.pydata.org/), [scipy](https://scipy.org/)
- (Optional for running examples) [mujoco](https://mujoco.org/)

Install via pip:
```bash
pip install xarm7-ik
```
Or clone this repo and:
```bash
pip install .
# or, for example dependencies (Mujoco):
pip install .[examples]
```


## Usage

To use the IK solver in your own code:

1. **Import the solver and rotation representation:**
   ```python
   from xarm7_ik.solver import InverseKinematicsSolver
   ```

2. **Create an instance of the solver:**
   - For a standard xArm7 (7-DOF):
     ```python
     ik_solver = InverseKinematicsSolver(use_linear_motor=False, rotation_repr="quaternion")
     ```
   - For xArm7 with linear rail (8-DOF):
     ```python
     ik_solver = InverseKinematicsSolver(use_linear_motor=True, rotation_repr="quaternion")
     ```

3. **Call the inverse kinematics method:**
   ```python
   result = ik_solver.inverse_kinematics(
       initial_configuration,         # np.ndarray of joint angles (7 or 8 elements)
       target_gripper_pos,            # np.ndarray, shape (3,)
       target_gripper_rot,            # np.ndarray, shape depends on the rotation representation
       
       # Quaternions are in the W-X-Y-Z order
       # Euler angles are X-Y-Z
       # Axis-angle representations are (axis, angle)
   )
   # result: np.ndarray of joint angles
   ```

For more complete usage and simulation integration, see the scripts in the `examples/` folder.
\

## Citation

This project was originally developed for our paper that uses xArm7. If you find this project useful, please consider citing our paper:

```bibtex
@article{sun2025dynamic,
  title={Dynamic Rank Adjustment in Diffusion Policies for Efficient and Flexible Training},
  author={Sun, Xiatao and Yang, Shuo and Chen, Yinxing and Fan, Francis and Liang, Yiyan and Rakita, Daniel},
  journal={arXiv preprint arXiv:2502.03822},
  year={2025}
}
```