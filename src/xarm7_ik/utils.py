import numpy as np
from numba import jit
import math

@jit(nopython=True)
def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return v
    return v / norm


@jit(nopython=True)
def axis_angle_to_quaternion(axis, angle):
    """Convert axis-angle representation to a unit quaternion (w, x, y, z)."""
    axis = normalize_vector(axis)
    half_angle = angle / 2.0
    sin_half_angle = np.sin(half_angle)
    cos_half_angle = np.cos(half_angle)

    w = cos_half_angle
    x = axis[0] * sin_half_angle
    y = axis[1] * sin_half_angle
    z = axis[2] * sin_half_angle

    return np.array([w, x, y, z])


@jit(nopython=True)
def quaternion_to_axis_angle(quaternion):
    """Convert a unit quaternion (w, x, y, z) to axis-angle representation."""
    w, x, y, z = quaternion
    angle = 2.0 * np.arccos(w)
    sin_half_angle = np.sqrt(1.0 - w * w)

    if sin_half_angle < 1e-6:
        axis = np.array([1.0, 0.0, 0.0])  # Default axis when angle is zero
    else:
        axis = np.array([x, y, z]) / sin_half_angle
        axis = normalize_vector(axis)

    # Adjust angle to be within [0, pi] and reverse axis if necessary
    if angle > np.pi:
        angle = 2 * np.pi - angle
        axis = -axis

    return axis, angle


@jit(nopython=True)
def rotation_matrix_to_axis_angle(matrix):
    """Convert a rotation matrix to axis-angle representation."""
    # Ensure the matrix is of type float
    matrix = matrix.astype(np.float64)

    angle = np.arccos((np.trace(matrix) - 1) / 2.0)

    if angle < 1e-6:
        # If the angle is very small, return a default axis
        axis = np.array([1.0, 0.0, 0.0])
        angle = 0.0
    else:
        x = matrix[2, 1] - matrix[1, 2]
        y = matrix[0, 2] - matrix[2, 0]
        z = matrix[1, 0] - matrix[0, 1]
        axis = np.array([x, y, z])
        axis = normalize_vector(axis)

    # Adjust angle to be within [0, pi] and reverse axis if necessary
    if angle > np.pi:
        angle = 2 * np.pi - angle
        axis = -axis

    return axis, angle

@jit(nopython=True)
def normalize_quaternion(q):
    norm = math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    return np.array([q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm])

@jit(nopython=True)
def quaternion_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

@jit(nopython=True)
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


@jit(nopython=True)
def H1_wxyz_quat_logarithm(q):
    w, x, y, z = q
    theta = math.acos(w)
    sin_theta = math.sin(theta)
    if sin_theta == 0:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        scalar = theta / sin_theta
        return np.array([0, scalar * x, scalar * y, scalar * z], dtype=np.float64)

@jit(nopython=True)
def quaternion_displacement_based_distance(q1, q2):
    # Compute the displacement-based distance between two quaternions based on Danny's notes
    # Make sure they are unit quaternions
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)

    # Quaternion conjugate is the same as the inverse for unit quaternions
    disp1 = H1_wxyz_quat_logarithm(quaternion_multiply(quaternion_conjugate(q1), q2))
    disp2 = H1_wxyz_quat_logarithm(quaternion_multiply(quaternion_conjugate(q1), -q2))

    # Calculate norm on the vector part because of the vee operator
    error1 = np.linalg.norm(disp1[1:])
    error2 = np.linalg.norm(disp2[1:])
    return min(error1, error2)


@jit(nopython=True)
def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    trace = np.trace(matrix)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    elif (matrix[0, 0] > matrix[1, 1]) and (matrix[0, 0] > matrix[2, 2]):
        s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s

    normalized_quat = normalize_quaternion(np.array([w, x, y, z], dtype=np.float64))

    return normalized_quat


@jit(nopython=True)
def compute_transform_error(curr_pos, curr_quat, target_pos, target_quat):
    pos_error = np.linalg.norm(curr_pos - target_pos)
    quat_error = quaternion_displacement_based_distance(curr_quat, target_quat)

    return pos_error + quat_error

@jit(nopython=True)
def proj_scalar(v, u):
    return np.dot(v, u) / np.dot(u, u)


@jit(nopython=True)
def proj_clamped(v, u):
    scalar = proj_scalar(v, u)
    scalar = min(max(scalar, 0), 1)
    return scalar * u


@jit(nopython=True)
def proj_onto_line_segment(p, a, b):
    return proj_clamped(p - a, b - a) + a


@jit(nopython=True)
def quaternion_rotate(q, v):
    qv = np.array([0.0, v[0], v[1], v[2]])
    normalized_q = normalize_quaternion(q)
    return quaternion_multiply(quaternion_multiply(normalized_q, qv), quaternion_conjugate(normalized_q))[1:]

@jit(nopython=True)
def calculate_look_at_error(eef_pos, lookat_pos, eef_quat, lookat_offset=np.array([0.0, 0.0, -0.15])):
    v = quaternion_rotate(eef_quat, np.array([0.0, 0.0, 1.0]))

    a = eef_pos
    b = eef_pos + 999 * v

    offset = lookat_offset
    eef_pos += offset

    proj_t_v = proj_onto_line_segment(lookat_pos, a, b)

    error = np.sum((proj_t_v - lookat_pos) ** 2)

    return error


@jit(nopython=True)
def calculate_side_axis_error(eef_quat):
    s = quaternion_rotate(eef_quat, np.array([0.0, 1.0, 0.0]))
    up = np.array([0.0, 0.0, 1.0])
    error = np.dot(s, up) ** 2
    return error


@jit(nopython=True)
def euler_z_to_quaternion(angle):
    """Convert a Z-axis Euler angle to quaternion (w, x, y, z)."""
    half_angle = angle / 2.0
    cos_half = np.cos(half_angle)
    sin_half = np.sin(half_angle)
    return np.array([cos_half, 0.0, 0.0, sin_half])


@jit(nopython=True)
def apply_base_rotation_to_transform(position, quaternion, base_rotation_offset):
    """Apply base rotation offset to a transform (position, quaternion)."""
    if abs(base_rotation_offset) < 1e-6:
        return position, quaternion
    
    # Create base rotation quaternion (rotation around Z-axis)
    base_quat = euler_z_to_quaternion(base_rotation_offset)
    
    # Create rotation matrix from base quaternion for position transformation
    base_rot_matrix = quaternion_to_rotation_matrix(base_quat)
    
    # Apply base rotation to position
    rotated_position = base_rot_matrix @ position
    
    # Apply base rotation to orientation
    rotated_quaternion = quaternion_multiply(base_quat, quaternion)
    rotated_quaternion = normalize_quaternion(rotated_quaternion)
    
    return rotated_position, rotated_quaternion


@jit(nopython=True)
def inverse_base_rotation_to_transform(position, quaternion, base_rotation_offset):
    """Apply inverse base rotation offset to a transform (position, quaternion)."""
    if abs(base_rotation_offset) < 1e-6:
        return position, quaternion
    
    # Create inverse base rotation quaternion (rotation around -Z-axis)
    base_quat = euler_z_to_quaternion(-base_rotation_offset)
    
    # Create rotation matrix from base quaternion for position transformation
    base_rot_matrix = quaternion_to_rotation_matrix(base_quat)
    
    # Apply inverse base rotation to position
    rotated_position = base_rot_matrix @ position
    
    # Apply inverse base rotation to orientation
    rotated_quaternion = quaternion_multiply(base_quat, quaternion)
    rotated_quaternion = normalize_quaternion(rotated_quaternion)
    
    return rotated_position, rotated_quaternion


@jit(nopython=True)
def quaternion_to_rotation_matrix(q):
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    w, x, y, z = q
    
    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Create rotation matrix
    R = np.zeros((3, 3), dtype=np.float64)
    R[0, 0] = 1 - 2*(y*y + z*z)
    R[0, 1] = 2*(x*y - w*z)
    R[0, 2] = 2*(x*z + w*y)
    R[1, 0] = 2*(x*y + w*z)
    R[1, 1] = 1 - 2*(x*x + z*z)
    R[1, 2] = 2*(y*z - w*x)
    R[2, 0] = 2*(x*z - w*y)
    R[2, 1] = 2*(y*z + w*x)
    R[2, 2] = 1 - 2*(x*x + y*y)
    
    return R
