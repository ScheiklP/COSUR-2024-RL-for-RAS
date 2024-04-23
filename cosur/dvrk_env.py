import numpy as np
import pybullet as p
import pybullet_data

# Connect to PyBullet
physicsClient = p.connect(p.GUI)

# Optional: Set additional paths to find the robot URDF files
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the plane and robot
robotId = p.loadURDF(
    "dvrk.urdf",
    useFixedBase=True,
    flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
)

ecmId = p.loadURDF(
    "ecm.urdf",
    useFixedBase=True,
    flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
)


num_joints = p.getNumJoints(robotId)
joint_values = [0.0] * num_joints
# Set all joints to zero
index = 0
for j in range(p.getNumJoints(robotId)):
    p.changeDynamics(robotId, j, linearDamping=0, angularDamping=0)
    info = p.getJointInfo(robotId, j)
    joint_type = info[2]
    if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
        p.resetJointState(robotId, j, joint_values[index])
        index = index + 1


num_ecm_joint = p.getNumJoints(ecmId)
joint_values = [0.0] * num_ecm_joint
# Set all joints to zero
index = 0
for j in range(p.getNumJoints(ecmId)):
    p.changeDynamics(ecmId, j, linearDamping=0, angularDamping=0)
    info = p.getJointInfo(ecmId, j)
    joint_type = info[2]
    if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
        p.resetJointState(ecmId, j, joint_values[index])
        index = index + 1

print(">>> Number of joints in the robot:", num_joints)
joint_trajectory = np.linspace(0, np.pi / 8, 10000).tolist()
while True:
    for val in joint_trajectory:
        index = 0
        for j in range(p.getNumJoints(robotId)):
            info = p.getJointInfo(robotId, j)
            joint_type = info[2]
            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                p.resetJointState(robotId, j, val)
                index = index + 1
        p.stepSimulation()
    joint_trajectory = joint_trajectory[::-1]


# Disconnect
p.disconnect()
