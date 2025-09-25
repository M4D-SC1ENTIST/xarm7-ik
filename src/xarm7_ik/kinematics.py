import numpy as np
from numba import jit
from .utils import compute_transform_error, matrix_to_quaternion, apply_base_rotation_to_transform, calculate_look_at_error, calculate_side_axis_error
from typing import Tuple

@jit(nopython=True)
def forward_kinematics(configuration: np.ndarray, 
                       use_linear_motor: bool = False,
                       linear_motor_x_offset=0.0,
                       base_rotation_offset=0.0) -> Tuple[np.ndarray, np.ndarray]:
    if use_linear_motor and configuration.shape[0] != 8:
        raise ValueError("Configuration must have 8 elements when using linear motor.")
    if not use_linear_motor and configuration.shape[0] != 7:
        raise ValueError("Configuration must have 7 elements when not using linear motor.")

    if use_linear_motor:
        linear_position = configuration[0]
        linear_motor_pos = np.array([linear_motor_x_offset, linear_position, 0.0], dtype=np.float64)
        
        joint_angles = configuration[1:]
    else:
        joint_angles = configuration

    
    # DH parameters: https://help.ufactory.cc/en/articles/4330809-kinematic-and-dynamic-parameters-of-ufactory-xarm-series
    dh_params = np.array([
        [0.0, 0.267, -np.pi / 2, joint_angles[0]],
        [0.0, 0.0, np.pi / 2, joint_angles[1]],
        [0.0525, 0.293, np.pi / 2, joint_angles[2]],
        [0.0775, 0.0, np.pi / 2, joint_angles[3]],
        [0.0, 0.3425, np.pi / 2, joint_angles[4]],
        [0.076, 0.0, -np.pi / 2, joint_angles[5]],
        [0.0, 0.097, 0.0, joint_angles[6]]
    ], dtype=np.float64)

    # Apply base rotation offset at the beginning of the kinematic chain
    T = np.eye(4, dtype=np.float64)
    if abs(base_rotation_offset) > 1e-6:
        # Create base rotation matrix (rotation around Z-axis)
        cos_rot = np.cos(base_rotation_offset)
        sin_rot = np.sin(base_rotation_offset)
        T_base = np.array([
            [cos_rot, -sin_rot, 0.0, 0.0],
            [sin_rot, cos_rot, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        T = T @ T_base
    
    for i in range(dh_params.shape[0]):
        a, d, alpha, theta = dh_params[i]
        T_i = np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0.0, np.sin(alpha), np.cos(alpha), d],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        T = T @ T_i

    if use_linear_motor:
        T_linear = np.eye(4, dtype=np.float64)
        T_linear[:3, 3] = linear_motor_pos
        T = T_linear @ T

    position = T[:3, 3]
    orientation_matrix = T[:3, :3]
    orientation_quat = matrix_to_quaternion(orientation_matrix)

    return position, orientation_quat


@jit(nopython=True)
def ik_objective_function(configuration, target_pos, target_quat, use_linear_motor, linear_motor_x_offset=0.0, base_rotation_offset=0.0):
    curr_pos, curr_quat = forward_kinematics(configuration, use_linear_motor, linear_motor_x_offset, base_rotation_offset)
    error = compute_transform_error(curr_pos, curr_quat, target_pos, target_quat)
    return error

@jit(nopython=True)
def lookat_ik_objective_function(configuration, target_pos, 
                                  lookat_pos, lookat_offset=np.array([0.0, 0.0, -0.15]),
                                  use_linear_motor=False, linear_motor_x_offset=0.0, base_rotation_offset=0.0):
    curr_pos, curr_quat = forward_kinematics(configuration, use_linear_motor, linear_motor_x_offset, base_rotation_offset)
    pos_error = np.linalg.norm(curr_pos - target_pos)
    lookat_error = calculate_look_at_error(curr_pos, lookat_pos, curr_quat, lookat_offset)
    side_axis_error = calculate_side_axis_error(curr_quat)
    return pos_error + lookat_error + side_axis_error

@jit(nopython=True)
def compute_ik_obj_func_grad_finite_diff(configuration, target_pos, target_quat, use_linear_motor, linear_motor_x_offset=0.0, base_rotation_offset=0.0, perturbation=1e-6):
    configuration = np.copy(configuration)
    grad = np.zeros_like(configuration)

    for i in range(len(configuration)):
        original_value = configuration[i]
        configuration[i] += perturbation
        error_plus = ik_objective_function(configuration, target_pos, target_quat, use_linear_motor, linear_motor_x_offset, base_rotation_offset)

        configuration[i] = original_value - perturbation
        error_minus = ik_objective_function(configuration, target_pos, target_quat, use_linear_motor, linear_motor_x_offset, base_rotation_offset)

        grad[i] = (error_plus - error_minus) / (2 * perturbation)
        configuration[i] = original_value

    return grad

@jit(nopython=True)
def compute_lookat_ik_obj_func_grad_finite_diff(configuration, target_pos, 
                                                lookat_pos, lookat_offset, 
                                                use_linear_motor, linear_motor_x_offset=0.0, base_rotation_offset=0.0, perturbation=1e-6):
    configuration = np.copy(configuration)
    grad = np.zeros_like(configuration)

    for i in range(len(configuration)):
        original_value = configuration[i]
        configuration[i] += perturbation
        error_plus = lookat_ik_objective_function(configuration, target_pos, lookat_pos, lookat_offset, use_linear_motor, linear_motor_x_offset, base_rotation_offset)
        configuration[i] = original_value - perturbation
        error_minus = lookat_ik_objective_function(configuration, target_pos, lookat_pos, lookat_offset, use_linear_motor, linear_motor_x_offset, base_rotation_offset)
        grad[i] = (error_plus - error_minus) / (2 * perturbation)
        configuration[i] = original_value
    return grad


def ik_objective_function_nlopt(configuration, grad, target_pos, target_quat, use_linear_motor=False, linear_motor_x_offset=0.0, base_rotation_offset=0.0):
    try:
        if grad.shape[0] != configuration.shape[0]:
            grad = np.zeros_like(configuration)
        error = ik_objective_function(configuration, target_pos, target_quat, use_linear_motor, linear_motor_x_offset, base_rotation_offset)
        grad[:] = compute_ik_obj_func_grad_finite_diff(configuration, target_pos, target_quat, use_linear_motor, linear_motor_x_offset, base_rotation_offset)
        return error, grad
    except Exception as e:
        print("Error in objective function:", str(e))
        return np.inf, np.zeros_like(grad)


def lookat_ik_objective_function_nlopt(configuration, grad, target_pos, 
                                       lookat_pos, lookat_offset=np.array([0.0, 0.0, -0.15]),
                                       use_linear_motor=False, linear_motor_x_offset=0.0, base_rotation_offset=0.0):
    try:
        if grad.shape[0] != configuration.shape[0]:
            grad = np.zeros_like(configuration)
        error = lookat_ik_objective_function(configuration, target_pos, lookat_pos, lookat_offset, use_linear_motor, linear_motor_x_offset, base_rotation_offset)
        grad[:] = compute_lookat_ik_obj_func_grad_finite_diff(configuration, target_pos, lookat_pos, lookat_offset, use_linear_motor, linear_motor_x_offset, base_rotation_offset)
        return error, grad
    except Exception as e:
        print("Error in objective function:", str(e))
        return np.inf, np.zeros_like(grad)


class LookAtInverseKinematicsSolver():
    def __init__(self, 
                 use_linear_motor=False,
                 linear_motor_x_offset=0.0,
                 rotation_repr="quaternion",
                 opt_solver=None,
                 base_rotation_offset=0.0,
                 lookat_offset=np.array([0.0, 0.0, -0.15])):
        import nlopt
        
        self.use_linear_motor = use_linear_motor
        self.linear_motor_x_offset = linear_motor_x_offset
        self.base_rotation_offset = base_rotation_offset
        self.lookat_offset = lookat_offset

        if rotation_repr == "quaternion":
            from .solver import RotationRepresentation
            self.rotation_repr = RotationRepresentation.QUATERNION
        elif rotation_repr == "euler":
            from .solver import RotationRepresentation
            self.rotation_repr = RotationRepresentation.EULER
        elif rotation_repr == "axis-angle":
            from .solver import RotationRepresentation
            self.rotation_repr = RotationRepresentation.AXIS_ANGLE
        else:
            raise ValueError("Invalid rotation representation. Choose from 'quaternion', 'euler', 'axis-angle'.")

        if self.use_linear_motor:
            lower_bounds = [0.0, -6.28319, -2.05900, -6.28319, -1.91980, -6.28319, -1.69297, -6.28319]
            upper_bounds = [0.74, 6.28319, 2.09440, 6.28319, 3.92700, 6.28319, 3.14159, 6.28319]
        else:
            lower_bounds = [-6.28319, -2.05900, -6.28319, -1.91980, -6.28319, -1.69297, -6.28319]
            upper_bounds = [6.28319, 2.09440, 6.28319, 3.92700, 6.28319, 3.14159, 6.28319]

        self.rotation_repr = rotation_repr

        self.lower_bounds = np.array(lower_bounds, dtype=np.float64)
        self.upper_bounds = np.array(upper_bounds, dtype=np.float64)
        self.bounds = list(zip(lower_bounds, upper_bounds))

        # Set default optimizer if none provided
        if opt_solver is None:
            opt_solver = nlopt.LD_SLSQP

        if self.use_linear_motor:
            self.opt = nlopt.opt(opt_solver, 8)
            self.opt.set_lower_bounds([bound[0] for bound in self.bounds[:8]])
            self.opt.set_upper_bounds([bound[1] for bound in self.bounds[:8]])
        else:
            self.opt = nlopt.opt(opt_solver, 7)
            self.opt.set_lower_bounds([bound[0] for bound in self.bounds[:7]])
            self.opt.set_upper_bounds([bound[1] for bound in self.bounds[:7]])
        self.opt.set_xtol_rel(1e-6)
        self.opt.set_maxtime(0.5)

        # Dry run to warm up the JIT compiler
        self.init_dry_run()
    
    def init_dry_run(self):
        if self.use_linear_motor:
            temp_state = np.zeros(8, dtype=np.float64)
        else:
            temp_state = np.zeros(7, dtype=np.float64)
        temp_target_pos = np.array([0.4, 0.0, 0.3], dtype=np.float64)
        temp_lookat_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        self.opt.set_min_objective(
            lambda x, grad: lookat_ik_objective_function_nlopt(x, 
                                                              grad, 
                                                              temp_target_pos, 
                                                              temp_lookat_pos,
                                                              self.lookat_offset,
                                                              self.use_linear_motor, 
                                                              self.linear_motor_x_offset,
                                                              self.base_rotation_offset)[0])
        self.opt.optimize(temp_state)

    def inverse_kinematics(self, 
                           initial_configuration, 
                           target_gripper_pos, 
                           lookat_pos):
        """
        Solve inverse kinematics with look-at constraint.
        
        Args:
            initial_configuration: Initial joint configuration
            target_gripper_pos: Target position for the gripper
            lookat_pos: Position that the gripper should look at
            
        Returns:
            Joint configuration that achieves the target position while looking at lookat_pos
        """
        try:
            self.opt.set_min_objective(
                lambda x, grad: lookat_ik_objective_function_nlopt(x, 
                                                                  grad, 
                                                                  target_gripper_pos, 
                                                                  lookat_pos,
                                                                  self.lookat_offset,
                                                                  self.use_linear_motor, 
                                                                  self.linear_motor_x_offset,
                                                                  self.base_rotation_offset)[0])

            if self.use_linear_motor:
                result = self.opt.optimize(initial_configuration[:8])
            else:
                result = self.opt.optimize(initial_configuration[:7])
            
            return result
        except Exception as e:
            print("Look-at inverse kinematics failed with error:", str(e))

            if self.use_linear_motor:
                result = initial_configuration[:8]
            else:
                result = initial_configuration[:7]

            return result

    def forward_kinematics(self, configuration):
        """
        Compute forward kinematics with the solver's configuration.
        
        Args:
            configuration: Joint configuration array
            
        Returns:
            tuple: (position, quaternion) of the end-effector
        """
        return forward_kinematics(
            configuration, 
            use_linear_motor=self.use_linear_motor, 
            linear_motor_x_offset=self.linear_motor_x_offset,
            base_rotation_offset=self.base_rotation_offset
        )
