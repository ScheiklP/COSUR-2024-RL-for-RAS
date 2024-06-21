import time
import numpy as np


def add_dummy_sphere(bullet_client, radius: float = 0.1, position: list = [0, 0, 0], orientation: list = [0, 0, 0], color: list = [1, 0, 0, 1], with_frame: bool = True) -> int:
    sphere_id = bullet_client.createVisualShape(
        shapeType=bullet_client.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color,
    )

    sphere_body_id = bullet_client.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=sphere_id,
        basePosition=position,
        baseOrientation=bullet_client.getQuaternionFromEuler(orientation),
    )

    if with_frame:
        add_coordinate_system(bullet_client, position, bullet_client.getQuaternionFromEuler(orientation))

    return sphere_body_id


def add_dummy_box(bullet_client, half_extents: list = [0.1, 0.1, 0.1], position: list = [0, 0, 0], orientation: list = [0, 0, 0], color: list = [1, 0, 0, 1]) -> int:
    box_id = bullet_client.createVisualShape(
        shapeType=bullet_client.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=color,
    )

    box_body_id = bullet_client.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=box_id,
        basePosition=position,
        baseOrientation=bullet_client.getQuaternionFromEuler(orientation),
    )

    add_coordinate_system(bullet_client, position, bullet_client.getQuaternionFromEuler(orientation))

    return box_body_id


def add_coordinate_system(bullet_client, position: list, orientation: list, scale: float = 0.1):
    # apply quaternion orientation to x, y, z vectors
    x = np.array([scale, 0, 0])
    y = np.array([0, scale, 0])
    z = np.array([0, 0, scale])
    rotation_matrix = np.array(bullet_client.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    x = np.dot(rotation_matrix, x)
    y = np.dot(rotation_matrix, y)
    z = np.dot(rotation_matrix, z)
    # draw the orientation
    bullet_client.addUserDebugLine(position, position + x, [1, 0, 0])
    bullet_client.addUserDebugLine(position, position + y, [0, 1, 0])
    bullet_client.addUserDebugLine(position, position + z, [0, 0, 1])


def add_coordinate_frame(bullet_client, body_id: int, frame_id: int, size: float = 0.1, line_width: float = 1.0) -> None:
    bullet_client.addUserDebugLine(
        lineFromXYZ=[0, 0, 0],
        lineToXYZ=[size, 0, 0],
        lineColorRGB=[1, 0, 0],
        parentObjectUniqueId=body_id,
        parentLinkIndex=frame_id,
        lineWidth=line_width,
    )
    bullet_client.addUserDebugLine(
        lineFromXYZ=[0, 0, 0],
        lineToXYZ=[0, size, 0],
        lineColorRGB=[0, 1, 0],
        parentObjectUniqueId=body_id,
        parentLinkIndex=frame_id,
        lineWidth=line_width,
    )
    bullet_client.addUserDebugLine(
        lineFromXYZ=[0, 0, 0],
        lineToXYZ=[0, 0, size],
        lineColorRGB=[0, 0, 1],
        parentObjectUniqueId=body_id,
        parentLinkIndex=frame_id,
        lineWidth=line_width,
    )


class PSM:
    def __init__(
        self,
        bullet_client,
        urdf_path: str,
        show_frames: bool = False,
        base_position: list = [0, 0, 0],
        base_orientation: list = [0, 0, 0],
        max_motor_force: float = 1000.0,
        mimic_joint_force_factor: float = 100.0,
    ) -> None:
        self.bullet_client = bullet_client

        self.urdf_path = urdf_path
        self.robot_id = self.bullet_client.loadURDF(
            urdf_path,
            useFixedBase=True,
            basePosition=base_position,
            baseOrientation=self.bullet_client.getQuaternionFromEuler(base_orientation),
        )

        for j in range(self.bullet_client.getNumJoints(self.robot_id)):
            self.bullet_client.changeDynamics(self.robot_id, j, linearDamping=0, angularDamping=0)

        self.ee_link_index = 7

        # 0 -> yaw
        # 1 -> pitch
        # 4 -> main insertion
        # 5 -> tool roll
        # 6 -> tool pitch
        # 7 -> tool yaw
        # 9 -> gripper opening -> manual mimic to 8
        self.joint_ids = [0, 1, 4, 5, 6, 7, 9]
        self.joint_names = ["yaw", "pitch", "main_insertion", "tool_roll", "tool_pitch", "tool_yaw", "gripper_opening"]
        self.joint_limits = [
            [-1.605, 1.5994],
            [-0.93556, 0.94249],
            [0.0, 0.24],
            [-3.14159, 3.14159],
            [-1.5708, 1.5708],
            [-1.5708, 1.5708],
            [0.0, 1.0],
        ]

        self.max_motor_force = max_motor_force
        self.mimic_joint_force_factor = mimic_joint_force_factor

        if show_frames:
            for i in range(self.bullet_client.getNumJoints(self.robot_id)):
                add_coordinate_frame(self.bullet_client, self.robot_id, i, size=0.1, line_width=1.0)

    def clip_joint_position(self, joint: int, position: float):
        lower_limit, upper_limit = self.joint_limits[joint]
        return max(lower_limit, min(upper_limit, position))

    def set_joint_position(self, joint: int, position: float):
        joint_id = self.joint_ids[joint]
        clipped_position = self.clip_joint_position(joint, position)

        if position != clipped_position:
            print(f"Joint {joint} position clipped from {position} to {clipped_position}.")

        self.bullet_client.setJointMotorControl2(
            self.robot_id,
            joint_id,
            self.bullet_client.POSITION_CONTROL,
            targetPosition=clipped_position,
            force=self.max_motor_force,
        )

        # Gripper jaw mimic joint
        if joint_id == 9:
            self.bullet_client.setJointMotorControl2(
                self.robot_id,
                8,
                self.bullet_client.POSITION_CONTROL,
                targetPosition=-clipped_position,
                force=self.mimic_joint_force_factor * self.max_motor_force,
            )

        # Pitch mimic joints
        if joint_id == 1:
            for mimic_joint_id, direction in zip((2, 3, 11, 12), (-1, 1, -1, 1)):
                # for mimic_joint_id, direction in zip((2, 3, 10, 11), (-1, 1, -1, 1)):
                self.bullet_client.setJointMotorControl2(
                    self.robot_id,
                    mimic_joint_id,
                    self.bullet_client.POSITION_CONTROL,
                    targetPosition=direction * clipped_position,
                    force=self.mimic_joint_force_factor * self.max_motor_force,
                )

    def set_joint_positions(self, positions: list | np.ndarray):
        if not len(positions) == 7:
            raise ValueError(f"The number of joint positions should be 7. Got {len(positions)} instead.")

        for i, position in enumerate(positions):
            self.set_joint_position(i, position)

    def get_joint_positions(self) -> np.ndarray:
        joint_positions = []
        for joint_id in self.joint_ids:
            joint_positions.append(self.bullet_client.getJointState(self.robot_id, joint_id)[0])

        return np.array(joint_positions)

    def demo_motion(self, simulation_hz: int = 500):
        for i in range(7):
            joint_values = []
            joint_values += np.linspace(0.0, self.joint_limits[i][0], int(3.0 * simulation_hz)).tolist()
            joint_values += np.linspace(self.joint_limits[i][0], self.joint_limits[i][1], int(3.0 * simulation_hz)).tolist()
            joint_values += np.linspace(self.joint_limits[i][1], 0.0, int(3.0 * simulation_hz)).tolist()

            for val in joint_values:
                joint_states = np.zeros(7).tolist()
                joint_states[i] = val
                self.set_joint_positions(joint_states)
                self.bullet_client.stepSimulation()
                time.sleep(1 / simulation_hz)
