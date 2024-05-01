import pybullet as p
import time
import numpy as np


def add_coordinate_frame(body_id, frame_id, size=0.1, line_width=1.0):
    p.addUserDebugLine(
        lineFromXYZ=[0, 0, 0],
        lineToXYZ=[size, 0, 0],
        lineColorRGB=[1, 0, 0],
        parentObjectUniqueId=body_id,
        parentLinkIndex=frame_id,
        lineWidth=line_width,
    )
    p.addUserDebugLine(
        lineFromXYZ=[0, 0, 0],
        lineToXYZ=[0, size, 0],
        lineColorRGB=[0, 1, 0],
        parentObjectUniqueId=body_id,
        parentLinkIndex=frame_id,
        lineWidth=line_width,
    )
    p.addUserDebugLine(
        lineFromXYZ=[0, 0, 0],
        lineToXYZ=[0, 0, size],
        lineColorRGB=[0, 0, 1],
        parentObjectUniqueId=body_id,
        parentLinkIndex=frame_id,
        lineWidth=line_width,
    )


def create_constraint(robot_id, parent_index, child_index, ratio, erp=10.0, max_force=1e7):
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


def get_mimic_joint_error(joint_id, mimic_joint_id, ratio):
    joint_state = p.getJointState(robot_id, joint_id)
    mimic_joint_state = p.getJointState(robot_id, mimic_joint_id)
    joint_error = joint_state[0] + mimic_joint_state[0] * ratio
    return joint_error


def setup_psm_with_constraints(urdf_path):
    # Load the URDF file (specify the path to your URDF file)
    robot_id = p.loadURDF(urdf_path, useFixedBase=True)

    # for i in range(p.getNumJoints(robot_id)):
    #     p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetVelocity=0, force=0)

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


def move_robot_a_bit(robot_id, constraints, num_motions):
    joint_id = 1
    motion_time = 2.0
    joint_values = np.linspace(0.0, np.pi / 8, int(motion_time * simulation_hz))
    for _ in range(num_motions):
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


if __name__ == "__main__":
    simulation_hz = 500
    # Setup the simulation environment
    p.connect(p.GUI)
    p.setRealTimeSimulation(0)
    p.setPhysicsEngineParameter(fixedTimeStep=1 / simulation_hz)
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

    robot_id, constraints = setup_psm_with_constraints("dvrk.urdf")

    target_position = np.array([0.0, 0.4, -0.2])
    target_orientation = p.getQuaternionFromEuler([-np.pi / 2, 0, 0])
    planning_frame = 9

    # Add coordinate frames to the planning frame
    add_coordinate_frame(robot_id, planning_frame, size=0.1, line_width=2)

    # Add sphere and coordinate frame to the target pose
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.01,
        rgbaColor=[1, 0, 0, 1],
    )

    object_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=target_position,
        baseOrientation=target_orientation,
    )

    # Add coordinate frames to the sphere
    add_coordinate_frame(object_id, -1, size=0.1, line_width=2)

    move_robot_a_bit(robot_id, constraints, num_motions=1)

    joint_values = p.calculateInverseKinematics(
        robot_id,
        planning_frame,
        target_position,
        target_orientation,
        maxNumIterations=int(1e5),
    )
    joint_values = np.array(joint_values)

    current_joint_values = np.array([p.getJointState(robot_id, i)[0] for i in range(p.getNumJoints(robot_id))])[:12]

    print(">>> Current joint values:", current_joint_values)
    print(">>> Target joint values:", joint_values)

    joint_trajectory = np.linspace(current_joint_values, joint_values, int(2.0 * simulation_hz))

    # Print joint names
    joint_names = [p.getJointInfo(robot_id, i)[1].decode("utf-8") for i in range(p.getNumJoints(robot_id))]
    for j, name in enumerate(joint_names):
        try:
            target = joint_values[j]
        except IndexError:
            target = None
        print(f"Joint {j}: {name} | Target: {target}")

    num_motions = 5
    for _ in range(num_motions):
        for val in joint_trajectory:
            for j, v in enumerate(val):
                p.resetJointState(robot_id, j, v)
            p.stepSimulation()
            error_string = ">>> "
            for i, constraint in enumerate(constraints):
                joint_error = get_mimic_joint_error(constraint[0], constraint[1], constraint[2])
                error_string += f"Constraint {i}: {abs(joint_error):.5f} "
            print(error_string)
            time.sleep(1 / simulation_hz)
        joint_trajectory = joint_trajectory[::-1]

    p.disconnect()
