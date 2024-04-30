import pybullet as p
import time
import numpy as np


def create_constraint(parent_index, child_index, ratio):
    c = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=parent_index,
        childBodyUniqueId=robot_id,
        childLinkIndex=child_index,
        jointType=p.JOINT_GEAR,
        jointAxis=[0, 0, 1],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
    )
    p.changeConstraint(
        c,
        gearRatio=ratio,
        erp=5.5,
        relativePositionTarget=0,
        maxForce=10000000,
    )


def get_mimic_joint_error(joint_id, mimic_joint_id, ratio):
    joint_state = p.getJointState(robot_id, joint_id)
    mimic_joint_state = p.getJointState(robot_id, mimic_joint_id)
    joint_error = joint_state[0] + mimic_joint_state[0] * ratio
    return joint_error


if __name__ == "__main__":
    simulation_hz = 500
    # Setup the simulation environment
    p.connect(p.GUI)
    p.setRealTimeSimulation(0)
    p.setPhysicsEngineParameter(fixedTimeStep=1 / simulation_hz)

    # Load the URDF file (specify the path to your URDF file)
    robot_id = p.loadURDF("dvrk.urdf", useFixedBase=True)

    for i in range(p.getNumJoints(robot_id)):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetVelocity=0, force=0)

    constraints = []
    constraint_pitch_back_pitch_bottom = [1, 2, 1.0]
    constraints.append(constraint_pitch_back_pitch_bottom)
    constraint_pitch_bottom_pitch_end = [2, 3, 1.0]
    constraints.append(constraint_pitch_bottom_pitch_end)
    constraint_pitch_end_pitch_top = [3, 10, 1.0]
    constraints.append(constraint_pitch_end_pitch_top)
    constraint_pitch_top_pitch_front = [10, 11, 1.0]
    constraints.append(constraint_pitch_top_pitch_front)

    for constraint in constraints:
        create_constraint(
            parent_index=constraint[0],
            child_index=constraint[1],
            ratio=constraint[2],
        )

    joint_id = 1
    motion_time = 2.0
    joint_values = np.linspace(0.0, np.pi / 8, int(motion_time * simulation_hz))
    while True:
        for val in joint_values:
            p.resetJointState(robot_id, 4, 0.1)
            p.resetJointState(robot_id, joint_id, val)
            p.stepSimulation()
            error_string = ">>> "
            for i, constraint in enumerate(constraints):
                joint_error = get_mimic_joint_error(constraint[0], constraint[1], constraint[2])
                error_string += f"Constraint {i}: {abs(joint_error):.5f} "
            print(error_string)
            time.sleep(1 / simulation_hz)
        joint_values = joint_values[::-1]
