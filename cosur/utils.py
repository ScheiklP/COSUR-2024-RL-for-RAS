import pybullet as p
import time
import numpy as np


def add_coordinate_frame(body_id: int, frame_id: int, size: float = 0.1, line_width: float = 1.0) -> None:

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


class PSM:
    def __init__(self, urdf_path: str, simulation_hz: int = 500, show_frames: bool = False) -> None:
        self.urdf_path = urdf_path
        self.robot_id = p.loadURDF(urdf_path, useFixedBase=True)
        self.simulation_hz = simulation_hz

        # 0 -> yaw
        # 1 -> pitch
        # 4 -> main insertion
        # 5 -> tool roll
        # 6 -> tool pitch
        # 7 -> tool yaw
        # 9 -> gripper opening -> manual mimic to 8
        self.joint_ids = [0, 1, 4, 5, 6, 7, 9]
        self.joint_limits = [
            [-1.605, 1.5994],
            [-0.93556, 0.94249],
            [0.0, 0.24],
            [-3.14159, 3.14159],
            [-1.5708, 1.5708],
            [-1.5708, 1.5708],
            [0.0, 1.0],
        ]

        if show_frames:
            for i in range(7):
                add_coordinate_frame(self.robot_id, i, size=0.1, line_width=1.0)

    def clip_joint_position(self, joint: int, position: float):
        lower_limit, upper_limit = self.joint_limits[joint]
        return max(lower_limit, min(upper_limit, position))

    def set_joint_position(self, joint: int, position: float):

        joint_id = self.joint_ids[joint]
        clipped_position = self.clip_joint_position(joint, position)

        if position != clipped_position:
            print(f"Joint {joint} position clipped from {position} to {clipped_position}.")

        p.resetJointState(self.robot_id, joint_id, clipped_position)

        # Gripper jaw mimic joint
        if joint_id == 9:
            p.resetJointState(self.robot_id, 8, -clipped_position)

        # Pitch mimic joints
        if joint_id == 1:
            p.resetJointState(self.robot_id, 2, -clipped_position)
            p.resetJointState(self.robot_id, 3, clipped_position)
            p.resetJointState(self.robot_id, 10, -clipped_position)
            p.resetJointState(self.robot_id, 11, clipped_position)

    def set_joint_positions(self, positions: list):
        if not len(positions) == 7:
            raise ValueError(f"The number of joint positions should be 7. Got {len(positions)} instead.")

        for i, position in enumerate(positions):
            self.set_joint_position(i, position)

    def demo_motion(self):
        for i in range(7):
            joint_values = []
            joint_values += np.linspace(0.0, self.joint_limits[i][0], int(3.0 * self.simulation_hz)).tolist()
            joint_values += np.linspace(self.joint_limits[i][0], self.joint_limits[i][1], int(3.0 * self.simulation_hz)).tolist()
            joint_values += np.linspace(self.joint_limits[i][1], 0.0, int(3.0 * self.simulation_hz)).tolist()

            for val in joint_values:
                joint_states = np.zeros(7).tolist()
                joint_states[i] = val
                self.set_joint_positions(joint_states)
                p.stepSimulation()
                time.sleep(1 / self.simulation_hz)
