import numpy as np
import nlopt
import enum
from numba import jit
from scipy.spatial.transform import Rotation as R
from .utils import normalize_quaternion, quaternion_multiply, axis_angle_to_quaternion, inverse_base_rotation_to_transform
from .kinematics import ik_objective_function_nlopt

class RotationRepresentation(enum.Enum):
    QUATERNION = "quaternion"
    EULER = "euler"
    AXIS_ANGLE = "axis-angle"

class InverseKinematicsSolver():
    def __init__(self, 
                 use_linear_motor=False,
                 linear_motor_x_offset=0.0,
                 rotation_repr="quaternion",
                 opt_solver=nlopt.LD_SLSQP,
                 base_rotation_offset=0.0):
        self.use_linear_motor = use_linear_motor
        self.linear_motor_x_offset = linear_motor_x_offset
        self.base_rotation_offset = base_rotation_offset

        if rotation_repr == "quaternion":
            self.rotation_repr = RotationRepresentation.QUATERNION
        elif rotation_repr == "euler":
            self.rotation_repr = RotationRepresentation.EULER
        elif rotation_repr == "axis-angle":
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


        self.quat_offset = np.array([0.0, 1.0, 0.0, 0.0])

        if self.use_linear_motor:
            self.opt = nlopt.opt(nlopt.LD_SLSQP, 8)
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
        temp_target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.opt.set_min_objective(
            lambda x, grad: ik_objective_function_nlopt(x, 
                                                        grad, 
                                                        temp_target_pos, 
                                                        temp_target_quat, 
                                                        self.use_linear_motor, 
                                                        self.linear_motor_x_offset,
                                                        self.base_rotation_offset)[0])
        self.opt.optimize(temp_state)

    def inverse_kinematics(self, 
                           initial_configuration, 
                           target_gripper_pos, 
                           target_gripper_rot):
        if self.rotation_repr == RotationRepresentation.QUATERNION:
            pass
        elif self.rotation_repr == RotationRepresentation.EULER:
            target_gripper_rot = R.from_euler('xyz', target_gripper_rot).as_quat()

            # from scipy quaternion (X-Y-Z-W) to mujoco quaternion (W-X-Y-Z)
            # Refer to: https://github.com/clemense/quaternion-conventions
            target_gripper_rot = np.array([target_gripper_rot[3], target_gripper_rot[0], target_gripper_rot[1], target_gripper_rot[2]])
        elif self.rotation_repr == RotationRepresentation.AXIS_ANGLE:
            target_gripper_axis = target_gripper_rot[:3]
            target_gripper_angle = target_gripper_rot[3]


            target_gripper_rot = axis_angle_to_quaternion(target_gripper_axis, target_gripper_angle)
        else:
            ValueError("Invalid rotation representation. Choose from 'quat', 'euler', 'axis-angle'.")

        target_gripper_rot = normalize_quaternion(target_gripper_rot)
        target_gripper_rot = quaternion_multiply(self.quat_offset, target_gripper_rot)
        target_gripper_rot = normalize_quaternion(target_gripper_rot)

        # Note: No need to transform target pose since base rotation is now applied 
        # directly in forward kinematics at the base of the kinematic chain

        try:
            self.opt.set_min_objective(
                lambda x, grad: ik_objective_function_nlopt(x, 
                                                            grad, 
                                                            target_gripper_pos, 
                                                            target_gripper_rot, 
                                                            self.use_linear_motor, 
                                                            self.linear_motor_x_offset,
                                                            self.base_rotation_offset)[0])

            if self.use_linear_motor:
                result = self.opt.optimize(initial_configuration[:8])
            else:
                result = self.opt.optimize(initial_configuration[:7])
            
            return result
        except (nlopt.RoundoffLimited, nlopt.ForcedStop, Exception) as e:
            print("Inverse kinematics failed with error:", str(e))

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
        from .kinematics import forward_kinematics
        return forward_kinematics(
            configuration, 
            use_linear_motor=self.use_linear_motor, 
            linear_motor_x_offset=self.linear_motor_x_offset,
            base_rotation_offset=self.base_rotation_offset
        )


class LookAtInverseKinematicsSolver():
    def __init__(self, 
                 use_linear_motor=False,
                 linear_motor_x_offset=0.0,
                 rotation_repr="quaternion",
                 opt_solver=nlopt.LD_SLSQP,
                 base_rotation_offset=0.0,
                 lookat_offset=np.array([0.0, 0.0, -0.15])):
        
        self.use_linear_motor = use_linear_motor
        self.linear_motor_x_offset = linear_motor_x_offset
        self.base_rotation_offset = base_rotation_offset
        self.lookat_offset = lookat_offset

        if rotation_repr == "quaternion":
            self.rotation_repr = RotationRepresentation.QUATERNION
        elif rotation_repr == "euler":
            self.rotation_repr = RotationRepresentation.EULER
        elif rotation_repr == "axis-angle":
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
        
        from .kinematics import lookat_ik_objective_function_nlopt
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
                           target_pos, 
                           lookat_pos):
        """
        Solve inverse kinematics with look-at constraint.
        
        Args:
            initial_configuration: Initial joint configuration
            target_pos: Target position for the end-effector
            lookat_pos: Position that the end-effector should look at
            
        Returns:
            Joint configuration that achieves the target position while looking at lookat_pos
        """
        try:
            from .kinematics import lookat_ik_objective_function_nlopt
            self.opt.set_min_objective(
                lambda x, grad: lookat_ik_objective_function_nlopt(x, 
                                                                  grad, 
                                                                  target_pos, 
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
        from .kinematics import forward_kinematics
        return forward_kinematics(
            configuration, 
            use_linear_motor=self.use_linear_motor, 
            linear_motor_x_offset=self.linear_motor_x_offset,
            base_rotation_offset=self.base_rotation_offset
        )