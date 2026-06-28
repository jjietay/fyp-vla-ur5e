import rtde_control
import rtde_receive
import math
import numpy as np


def pose_to_T(pose):
    ''' Convert current pose in a list to a 4x4 Transform, T '''
    
    # Extract rotation angle and rotation axis from pose
    trans_vector = np.array(pose[:3])
    rot_vector = np.array(pose[3:6])
    rot_angle = np.linalg.norm(rot_vector)

    # Handle edge case of rot_angle == 0
    if rot_angle < 1e-6:
        rot_axis = rot_vector
    else:
        rot_axis = rot_vector / rot_angle

    kx, ky, kz = rot_axis

    # Create helper constants
    identity_3 = np.eye(3)

    # Computation of 3x3 Rotation Matrix, R
    K = np.array([
        [0, -kz, ky],
        [kz, 0, -kx],
        [-ky, kx, 0]
        ])
    
    R = identity_3 + np.dot(math.sin(rot_angle), K) + np.dot((1-math.cos(rot_angle)), (K @ K))

    # Computation of 4x4 Transform Matrix, T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = trans_vector
    
    return T

def T_to_pose(T):
    '''Convert a 4x4 transform T back into a pose [x,y,z,rx,ry,rz].'''
    R = T[:3, :3]
    p = T[:3, 3]

    # get rotational angle
    cos_theta = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = math.acos(cos_theta)

    if theta < 1e-6:
        # no rotation
        rot_vector = np.zeros(3)

    elif abs(theta - math.pi) < 1e-6:
        '''
        This is useful if theta=pi, sin(theta)=0, division by 0'''
        col  = (R + np.eye(3))[:, int(np.argmax(np.diag(R)))]
        axis = col / np.linalg.norm(col)
        rot_vector = math.pi * axis

    else:
        # general case: axis from the off-diagonal differences (lower minus upper)
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]) / (2 * math.sin(theta))
        rot_vector = theta * axis

    return [*p, *rot_vector]



def pose_trans(pose_a, pose_b):
    '''pose_b expressed in pose_a's frame
    In other words,
    ==> T_new = T_old @ T_offset'''

    return T_to_pose(pose_to_T(pose_a) @ pose_to_T(pose_b))



# Setup UR-RTDE
URSIM_IP = "127.0.0.1"
rtde_c = rtde_control.RTDEControlInterface(URSIM_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(URSIM_IP)

# Initial pose in terms of metres
initial_pose = rtde_r.getActualTCPPose()
initial_transform = pose_to_T(initial_pose)

# Side length of the square
side_length = 0.100

p1 = initial_pose
p2 = pose_trans(p1, [side_length, 0, 0, 0, 0, 0])
p3 = pose_trans(p1, [side_length, side_length, 0, 0, 0, 0])
p4 = pose_trans(p1, [0, side_length, 0, 0, 0, 0])

# Movement parameters
speed = 0.25
acceleration = 0.5

print("Moving to starting Position.")
rtde_c.moveL(p1, speed, acceleration)
rtde_c.moveL(p2, speed, acceleration)
rtde_c.moveL(p3, speed, acceleration)
rtde_c.moveL(p4, speed, acceleration)
rtde_c.moveL(p1, speed, acceleration)
rtde_c.moveL(p2, speed, acceleration)
print("Square drawn. Exiting...")

# Clear exit
rtde_c.stopScript()