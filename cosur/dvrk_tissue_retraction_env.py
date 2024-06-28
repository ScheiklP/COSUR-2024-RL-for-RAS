import time
import numpy as np
import gymnasium as gym
import cv2

from pathlib import Path

from utils import add_dummy_sphere

from dvrk_point_reach_env import DVRKEnv, ActionType

HERE = Path(__file__).parent


# TODO: task versions
# 1. Occluded point becomes visible
# 2. Cloth corner is at a target position
# 3. First go to a target position, then grasp the cloth, then go to another target position


class DVRKEnvTR(DVRKEnv):
    def __init__(
        self,
        simulation_hz: int = 500,
        render_mode: str | None = "human",
        frame_skip: int = 1,
        action_type: ActionType = ActionType.ABSOLUTE_POSITION,
        randomize_initial_joint_values: bool = True,
        egl_rendering: bool = False,
        image_shape: tuple[int, int] = (420, 420),
        reward_feature_weights: dict = {},
        coarse_cloth: bool = True,
        fixed_visual_target_index: int | None = None,
    ):
        self.cloth_position = [-0.05, 0.45]
        self.cloth_rotation = np.deg2rad(0.0)
        self.fixed_visual_target_index = fixed_visual_target_index

        self.coarse_cloth = coarse_cloth
        if coarse_cloth:
            self.fixed_indices = [0, 1, 5, 6]
            self.squares = [
                [2, 3, 7, 8],
                [3, 4, 8, 9],
                [7, 8, 12, 13],
                [8, 9, 13, 14],
                [12, 13, 17, 18],
                [13, 14, 18, 19],
                [17, 18, 22, 23],
                [18, 19, 23, 24],
                [16, 17, 21, 22],
                [15, 16, 20, 21],
                [10, 11, 15, 16],
            ]
            self.visual_target_indices = list(range(len(self.squares)))

            num_vertices = 25
            num_indices_for_obs = 5
        else:
            self.fixed_indices = [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33]
            num_vertices = 100
            num_indices_for_obs = 10

        non_fixed_cloth_indices = [i for i in range(num_vertices) if i not in self.fixed_indices]
        self.observation_indices = np.linspace(0, len(non_fixed_cloth_indices) - 1, num_indices_for_obs).astype(int)

        super().__init__(
            simulation_hz=simulation_hz,
            render_mode=render_mode,
            frame_skip=frame_skip,
            action_type=action_type,
            randomize_initial_joint_values=randomize_initial_joint_values,
            egl_rendering=egl_rendering,
            image_shape=image_shape,
            reward_feature_weights=reward_feature_weights,
        )

        # Update observation space
        obs_len = 0
        # Cloth vertex positions
        obs_len += num_indices_for_obs * 3
        # Joint positions
        obs_len += 7
        # End-effector position
        obs_len += 3
        # End-effector orientation
        obs_len += 4
        # Grasped (yes/no)
        obs_len += 1

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)

    def setup_simulation_scene(self):
        super().setup_simulation_scene()

        # Initialize the cloth
        self.add_cloth(self.cloth_position, self.cloth_rotation)
        self.grasp_constraint = None

        # Make the plane transparent
        self.bullet_client.changeVisualShape(
            self.plane,
            -1,
            rgbaColor=[0.4, 0.4, 0.4, 0.3],
        )

        # Add a very flat cylinder that is hidden beneath the cloth
        radius = 0.003
        length = 0.001
        visual_target_index = self.fixed_visual_target_index if self.fixed_visual_target_index is not None else self.np_random.choice(self.visual_target_indices)
        square_vertices = self.squares[visual_target_index]
        square_vertex_positions = np.array(self.bullet_client.getMeshData(self.cloth_id, -1, flags=self.bullet_client.MESH_DATA_SIMULATION_MESH)[1])[square_vertices]
        random_weights = self.np_random.uniform(0, 1, len(square_vertices))
        random_weights /= np.sum(random_weights)
        visual_target_position = np.sum([square_vertex_positions[i] * random_weights[i] for i in range(len(square_vertices))], axis=0)
        # visual_target_position[2] = -(length + 0.0005)
        visual_target_position[2] = -(length + 0.0002)

        visual_id = self.bullet_client.createVisualShape(
            shapeType=self.bullet_client.GEOM_CYLINDER,
            radius=radius,
            length=length,
            rgbaColor=[1.0, 0, 1.0, 1],
        )
        self.visual_target_body_id = self.bullet_client.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_id,
            basePosition=visual_target_position,
            baseOrientation=[0, 0, 0, 1],
            useMaximalCoordinates=True,
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one time step in the environment.

        1. Apply the action to the robot and simulate the environment for the duration of the frame skip. And check if the gripper is in contact with the cloth.
        2. Compute the reward based on the new state of the environment.
        3. Check if the episode is done.
        4. Return the new observation, the reward, whether the episode is done, and additional information.
        """

        if self.needs_reset:
            raise ValueError("Cannot call step() before calling reset()")

        start_time = time.time()

        # Apply the action n times for the duration of the frame skip
        if self.action_type == ActionType.ABSOLUTE_POSITION:
            new_joint_target_positions = self.scale_action(action)
        elif self.action_type == ActionType.RELATIVE_POSITION:
            new_joint_target_positions = self.joint_target_positions + self.scale_action(action)
        else:
            raise ValueError(f"Invalid action type: {self.action_type}")

        self.joint_target_positions = new_joint_target_positions
        for _ in range(self.frame_skip):
            self.psm.set_joint_positions(new_joint_target_positions)
            self.check_grasp()
            self.bullet_client.stepSimulation()

        # Compute the reward and additional information
        reward_features = self.calculate_reward_features()
        reward = 0.0
        info = {}
        for feature, weight in self.reward_feature_weights.items():
            reward += weight * reward_features[feature]
            info[f"reward_{feature}"] = weight * reward_features[feature]
        self.previous_reward_features = reward_features

        # info["success"] = reward_features["done"]
        # Check if the episode is done
        # terminated = reward_features["done"]
        terminated = False
        truncated = False

        # Get the new observation
        observation = self.get_observation()

        end_time = time.time()
        self.dt_queue.append(end_time - start_time)
        self.current_fps = 1.0 / np.mean(self.dt_queue)

        return observation, reward, terminated, truncated, info

    def calculate_reward_features(self) -> dict:
        reward_features = {}

        num_collision_with_floor = len(self.bullet_client.getContactPoints(bodyA=self.psm.robot_id, bodyB=self.plane))
        reward_features["num_collision_with_floor"] = num_collision_with_floor

        joint_velocities = self.psm.get_joint_velocities()
        simulation_unstable = np.linalg.norm(joint_velocities) > self.joint_velocities_unstable_threshold
        reward_features["simulation_unstable"] = simulation_unstable

        grasp_in_previous_step = bool(self.previous_reward_features.get("grasp_constraint", False))
        grasp_in_current_step = bool(self.grasp_constraint)
        established_new_grasp = not grasp_in_previous_step and grasp_in_current_step
        reward_features["established_new_grasp"] = established_new_grasp
        reward_features["cloth_is_grasped"] = grasp_in_current_step

        return reward_features

    def get_observation(self) -> np.ndarray:
        observation_features = {}

        # Joint positions
        observation_features["joint_positions"] = self.psm.get_joint_positions()

        # End-effector pose
        ee_position, ee_quaternion = self.psm.get_ee_pose()
        observation_features["ee_position"] = ee_position
        observation_features["ee_orientation"] = ee_quaternion

        # Grasped (yes/no)
        observation_features["grasped"] = np.array([1.0]) if self.grasp_constraint else np.array([0.0])

        # Cloth vertex positions
        _, cloth_vertex_positions = self.bullet_client.getMeshData(self.cloth_id, -1, flags=self.bullet_client.MESH_DATA_SIMULATION_MESH)
        observation_features["cloth_vertex_positions"] = np.array(cloth_vertex_positions)[self.observation_indices].flatten()

        return np.concatenate(list(observation_features.values()), dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        # Parent class reset initializes the RNG
        super().reset(seed=seed, options=options)

        # Reset the simulation
        self.bullet_client.resetSimulation(self.bullet_client.RESET_USE_DEFORMABLE_WORLD)
        self.bullet_client.setGravity(0, 0, -9.81)  # Set gravity
        self.setup_simulation_scene()

        # Reset the robot
        if self.randomize_initial_joint_values:
            valid_reset = False
            while not valid_reset:
                # Reset the PSM to joint values within 80% of the joint limits
                low, high = [np.array(limits) for limits in zip(*self.psm.joint_limits)]
                initial_joint_positions = low + 0.2 * (high - low) + 0.6 * self.np_random.random(7) * (high - low)
                self.joint_target_positions = initial_joint_positions

                self.psm.reset_joint_positions(initial_joint_positions)
                self.bullet_client.stepSimulation()

                robot_in_collision = False
                contact_points = self.bullet_client.getContactPoints(bodyA=self.psm.robot_id, bodyB=self.plane)
                contact_points += self.bullet_client.getContactPoints(bodyA=self.psm.robot_id, bodyB=self.cloth_id)
                robot_in_collision = len(contact_points) > 0

                if not robot_in_collision:
                    valid_reset = True
        else:
            self.joint_target_positions = self.inital_joint_positions.copy()
            self.psm.reset_joint_positions(self.inital_joint_positions)
            self.bullet_client.stepSimulation()

        self.needs_reset = False

        observation = self.get_observation()
        reset_info = {}

        return observation, reset_info

    def check_grasp(self):
        gripper_opening_angle = self.psm.get_joint_position(6)
        grasp_threshold = np.deg2rad(10.0)

        # If the gripper is closed enough and the cloth is in contact with the gripper, create a grasp constraint
        if self.grasp_constraint is None and gripper_opening_angle <= grasp_threshold:
            gripper_collides_with_cloth, closest_vertex = self.check_gripper_contacts()
            if gripper_collides_with_cloth:
                assert closest_vertex is not None
                self.grasp_constraint = env.create_grasp_constraint(closest_vertex)

        # If the gripper is too open, remove the grasp constraint
        elif self.grasp_constraint is not None and gripper_opening_angle > grasp_threshold:
            self.bullet_client.removeConstraint(self.grasp_constraint)
            self.grasp_constraint = None

    def check_gripper_contacts(self) -> tuple[bool, int | None]:
        """Check if the gripper is in contact with the cloth.

        Returns:
            bool: True if the gripper is in contact with the cloth, False otherwise.
            int | None: The index of the closest vertex of the cloth to the gripper if there is a collision, None otherwise.
        """

        contact_points = self.bullet_client.getContactPoints(bodyA=self.psm.robot_id, bodyB=self.cloth_id)

        # Check if the gripper is in contact with the cloth
        collision_with_gripper = False
        contact_position = None
        if len(contact_points) > 0:
            for contact in contact_points:
                if contact[3] in [8, 9]:
                    collision_with_gripper = True
                    contact_position = contact[6]
                    break

        # Find the closest vertex of the cloth to the gripper if there is a collision
        closest_vertex = None
        if collision_with_gripper:
            _, cloth_vertex_positions = self.bullet_client.getMeshData(self.cloth_id, -1, flags=self.bullet_client.MESH_DATA_SIMULATION_MESH)
            closest_vertex = int(np.argmin(np.linalg.norm(np.array(cloth_vertex_positions) - np.array(contact_position), axis=1)))

        return collision_with_gripper, closest_vertex

    def create_grasp_constraint(self, cloth_vertex_id: int) -> int:
        """Create a constraint between the gripper and the cloth.

        Uses a spherical joint to attach the gripper to the cloth.
        The spherical joint is attached to the yaw link of the gripper.
        """
        # Create a constraint between the gripper and the cloth
        constraint_id = self.bullet_client.createSoftBodyAnchor(
            softBodyBodyUniqueId=self.cloth_id,
            nodeIndex=cloth_vertex_id,
            bodyUniqueId=self.psm.robot_id,
            linkIndex=10,  # -> attached to a link with a spherical joint on the yaw link (7)
        )

        return constraint_id

    def add_cloth(self, cloth_position: list[float], cloth_rotation: float):
        cloth_scale = 0.05
        collision_margin = 0.001
        position = [cloth_position[0], cloth_position[1], collision_margin / 2]

        # Create the cloth
        self.cloth_id = self.bullet_client.loadSoftBody(
            fileName=f"{HERE}/meshes/coarse_cloth.obj" if self.coarse_cloth else f"{HERE}/meshes/cloth.obj",
            basePosition=position,
            baseOrientation=self.bullet_client.getQuaternionFromEuler([np.pi / 2, 0, cloth_rotation]),
            collisionMargin=collision_margin,
            scale=cloth_scale,
            mass=0.03,
            useNeoHookean=0,
            useBendingSprings=1,
            useMassSpring=1,
            springElasticStiffness=10,
            springDampingStiffness=0.1,
            springDampingAllDirections=0,
            useSelfCollision=1,
            frictionCoeff=1.0,
            useFaceContact=1,
        )

        # Change the color of the cloth to yellow
        self.bullet_client.changeVisualShape(
            self.cloth_id,
            -1,
            flags=self.bullet_client.VISUAL_SHAPE_DOUBLE_SIDED,
            rgbaColor=[1, 1, 0, 1],
        )

        # Attach the cloth to the world
        self.cloth_anchors = []
        for i in self.fixed_indices:
            id = self.bullet_client.createSoftBodyAnchor(
                softBodyBodyUniqueId=self.cloth_id,
                nodeIndex=i,
                bodyUniqueId=-1,
                linkIndex=-1,
            )
            self.cloth_anchors.append(id)

        # Visualize the attachment points
        self.anchor_spheres = []
        for i in self.fixed_indices:
            _, vertex_positions = self.bullet_client.getMeshData(self.cloth_id, -1, flags=self.bullet_client.MESH_DATA_SIMULATION_MESH)
            id = add_dummy_sphere(self.bullet_client, position=vertex_positions[i], radius=0.002, color=[0, 0, 1, 0.6], with_frame=False)
            self.anchor_spheres.append(id)

        # Add text to each cloth vertex with the index
        # for i in range(len(vertex_positions)):
        #     self.bullet_client.addUserDebugText(text=str(i), textPosition=vertex_positions[i], textColorRGB=[1, 1, 1], textSize=5.0)


if __name__ == "__main__":
    target_dt = 0.02
    simulation_hz = 250
    frame_skip = int(round(target_dt * simulation_hz))
    env = DVRKEnvTR(
        render_mode="rgb_array",
        action_type=ActionType.RELATIVE_POSITION,
        simulation_hz=simulation_hz,
        frame_skip=frame_skip,
        randomize_initial_joint_values=False,
        egl_rendering=True,
    )

    color_map = {
        -1: [0, 0, 0],
        0: [255, 0, 0],
        1: [0, 255, 0],
        2: [0, 0, 255],
        3: [255, 255, 0],
        4: [255, 0, 255],
        5: [0, 255, 255],
        6: [255, 255, 255],
    }
    img_type = "rgb"

    # Disable scientific notation for numpy
    np.set_printoptions(suppress=True)

    env.reset()
    counter = 0
    while True:
        before = time.time()
        random_action = np.zeros(7)
        if not env.grasp_constraint:
            random_action[2] = 1.0
        else:
            if env.psm.get_joint_position(2) < 0.05:
                random_action[2] = 0.0
            else:
                random_action[2] = -1.0
        obs, reward, terminated, truncated, info = env.step(random_action)
        counter += 1
        img = env.render(view="tissue", img_type=img_type)
        if img_type == "segmentation":
            display_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            for i, color in color_map.items():
                display_img[img == i] = color
        elif img_type == "depth":
            display_img = img
        elif img_type == "rgb":
            display_img = img[..., ::-1]

        cv2.imshow("Image", display_img)
        cv2.waitKey(1)
        after = time.time()
        print(f"Framerate: {1.0 / (after - before):.2f} Hz")
        if counter > 200:
            env.reset()
            counter = 0
