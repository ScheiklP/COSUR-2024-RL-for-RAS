import pybullet as p
import numpy as np
import time


def create_constraint(robot_id: int, parent_index: int, child_index: int, ratio: float, erp: float = 10.0, max_force: float = 1e7):
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
        erp=erp,
        maxForce=max_force,
    )


def get_mimic_joint_error(robot_id: int, joint_id: int, mimic_joint_id: int, ratio: float):
    joint_state = p.getJointState(robot_id, joint_id)
    mimic_joint_state = p.getJointState(robot_id, mimic_joint_id)
    joint_error = joint_state[0] + mimic_joint_state[0] * ratio
    return joint_error


def setup_psm_with_constraints(urdf_path: str) -> tuple:
    # Load the URDF file (specify the path to your URDF file)
    robot_id = p.loadURDF(urdf_path, useFixedBase=True)

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
            robot_id,
            parent_index=constraint[0],
            child_index=constraint[1],
            ratio=constraint[2],
        )

    return robot_id, constraints


def move_robot_a_bit(robot_id: int, joint_id: int, constraints: None | list, num_motions: int, motion_time: float, simulation_hz: int):
    joint_values = np.linspace(0.0, np.pi / 8, int(motion_time * simulation_hz))
    for _ in range(num_motions):
        for val in joint_values:
            p.resetJointState(robot_id, joint_id, val)
            p.stepSimulation()
            if constraints is not None:
                error_string = ">>> "
                for i, constraint in enumerate(constraints):
                    joint_error = get_mimic_joint_error(robot_id, constraint[0], constraint[1], constraint[2])
                    error_string += f"Constraint {i}: {abs(joint_error):.5f} "
                print(error_string)
            time.sleep(1 / simulation_hz)
        joint_values = joint_values[::-1]
