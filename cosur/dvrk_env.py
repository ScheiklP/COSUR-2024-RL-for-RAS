import time
import numpy as np
import pybullet as p
import pybullet_data

PSM_XYZ = [0.0, 0.0, 0.0]
PSM_RPY = [0.0, 0.0, 0.0]

ECM_XYZ = [0.5, -0.5, 0.0]
ECM_RPY = [0.0, 0.0, np.pi / 2]

# Connect to PyBullet
physicsClient = p.connect(p.GUI)

# Optional: Set additional paths to find the robot URDF files
p.setAdditionalSearchPath(pybullet_data.getDataPath())

cam_target_pos = [-0.21, -0.15, -0.02]
cam_dist = 1.2
cam_yaw = 125.2
cam_pitch = -25.0

p.resetDebugVisualizerCamera(
    cameraDistance=cam_dist,
    cameraYaw=cam_yaw,
    cameraPitch=cam_pitch,
    cameraTargetPosition=cam_target_pos,
)


def joint_values_for_pose(pose, frame_id, robot_id):
    return p.calculateInverseKinematics(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=frame_id,
        targetPosition=pose[:3],
        targetOrientation=pose[3:],
        maxNumIterations=1000,
    )


# Load the plane and robot
psm_id = p.loadURDF(
    "dvrk.urdf",
    useFixedBase=True,
    flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
    basePosition=PSM_XYZ,
    baseOrientation=p.getQuaternionFromEuler(PSM_RPY),
)

ecm_id = p.loadURDF(
    "ecm.urdf",
    useFixedBase=True,
    flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
    basePosition=ECM_XYZ,
    baseOrientation=p.getQuaternionFromEuler(ECM_RPY),
)


num_joints = p.getNumJoints(psm_id)
joint_values = [0.0] * num_joints
# Set all joints to zero
index = 0
for j in range(p.getNumJoints(psm_id)):
    p.changeDynamics(psm_id, j, linearDamping=0, angularDamping=0)
    info = p.getJointInfo(psm_id, j)
    joint_type = info[2]
    if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
        p.resetJointState(psm_id, j, joint_values[index])
        index = index + 1
    # Add visual text to the links
    text_xy = np.random.rand(2) / 10
    p.addUserDebugText(
        text=str(j),
        textPosition=[text_xy[0], text_xy[1], 0],
        textColorRGB=[0, 0, 1],
        textSize=1.5,
        parentObjectUniqueId=psm_id,
        parentLinkIndex=j,
    )

    # Add coordinate frames to the links
    p.addUserDebugLine(
        lineFromXYZ=[0, 0, 0],
        lineToXYZ=[0.1, 0, 0],
        lineColorRGB=[1, 0, 0],
        parentObjectUniqueId=psm_id,
        parentLinkIndex=j,
    )


num_ecm_joint = p.getNumJoints(ecm_id)
joint_values = [0.0] * num_ecm_joint
# Set all joints to zero
index = 0
for j in range(p.getNumJoints(ecm_id)):
    p.changeDynamics(ecm_id, j, linearDamping=0, angularDamping=0)
    info = p.getJointInfo(ecm_id, j)
    joint_type = info[2]
    if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
        p.resetJointState(ecm_id, j, joint_values[index])
        index = index + 1

    # Add visual text to the links
    p.addUserDebugText(
        text=str(j),
        textPosition=[0.1, 0, 0],
        textColorRGB=[0, 1, 0],
        textSize=1,
        parentObjectUniqueId=ecm_id,
        parentLinkIndex=j,
    )


target_position = np.array([0.5, 0.5, 0.5])
target_orientation = np.array([0.0, 0.0, 0.0, 1.0])
target_pose = target_position.tolist() + target_orientation.tolist()
planning_frame = 10

# print(">>> Number of joints in the robot:", num_joints)
joint_trajectory = np.linspace(0, np.pi / 8, 10000).tolist()
# joint_trajectory = np.zeros(10000).tolist()
while True:
    joints = [2, 3, 4, 11, 12]
    directions = [1, -1, 1, -1, 1]

    for val in joint_trajectory:
        for j, direction in zip(joints, directions):
            info = p.getJointInfo(psm_id, j)
            joint_type = info[2]
            if joint_type == p.JOINT_REVOLUTE:
                set_val = val * direction
                p.resetJointState(psm_id, j, set_val)
            elif joint_type == p.JOINT_PRISMATIC:
                val = val / 10 + 0.2
                p.resetJointState(psm_id, j, val)

        p.stepSimulation()
        time.sleep(1.0)
    joint_trajectory = joint_trajectory[::-1]

    # control_joints = list(range(p.getNumJoints(ecm_id)))
    # for j in control_joints:
    #     for _ in range(2):
    #         for val in joint_trajectory:
    #             info = p.getJointInfo(ecm_id, j)
    #             joint_type = info[2]
    #             if joint_type == p.JOINT_REVOLUTE:
    #                 p.resetJointState(ecm_id, j, val)
    #             elif joint_type == p.JOINT_PRISMATIC:
    #                 val = val / 10 + 0.2
    #                 p.resetJointState(ecm_id, j, val)

    #             p.stepSimulation()
    #         joint_trajectory = joint_trajectory[::-1]


# Disconnect
p.disconnect()
