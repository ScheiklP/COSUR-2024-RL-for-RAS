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

# print(">>> Number of joints in the robot:", num_joints)
joint_trajectory = np.linspace(0, np.pi / 8, 10000).tolist()
# joint_trajectory = np.zeros(10000).tolist()
while True:
    control_joints = list(range(p.getNumJoints(psm_id)))
    for j in control_joints:
        print(">>> Joint:", j)
        for _ in range(2):
            for val in joint_trajectory:
                info = p.getJointInfo(psm_id, j)
                joint_type = info[2]
                if joint_type == p.JOINT_REVOLUTE:
                    p.resetJointState(psm_id, j, val)
                elif joint_type == p.JOINT_PRISMATIC:
                    val = val / 10 + 0.2
                    p.resetJointState(psm_id, j, val)

                p.stepSimulation()
            joint_trajectory = joint_trajectory[::-1]

    control_joints = list(range(p.getNumJoints(ecm_id)))
    for j in control_joints:
        for _ in range(2):
            for val in joint_trajectory:
                info = p.getJointInfo(ecm_id, j)
                joint_type = info[2]
                if joint_type == p.JOINT_REVOLUTE:
                    p.resetJointState(ecm_id, j, val)
                elif joint_type == p.JOINT_PRISMATIC:
                    val = val / 10 + 0.2
                    p.resetJointState(ecm_id, j, val)

                p.stepSimulation()
            joint_trajectory = joint_trajectory[::-1]


# Disconnect
p.disconnect()
