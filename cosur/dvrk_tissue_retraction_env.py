import time
import numpy as np
import gymnasium as gym

from pathlib import Path

from utils import add_dummy_sphere

from dvrk_point_reach_env import DVRKEnv, ActionType, CAMERA_VIEW

HERE = Path(__file__).parent


# TODO: task versions
# 1. Occluded point becomes visible
# 2. Cloth corner is at a target position
# 3. First go to a target position, then grasp the cloth, then go to another target position

# TODO: restrict tool yaw, pith, and gripper opening, when the tool is still in the insertion shaft


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
    ):
        self.fixed_indices = [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33]
        num_indices_for_obs = 10
        non_fixed_cloth_indices = [i for i in range(100) if i not in self.fixed_indices]
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
        # Cloth vertex positions, joint positions, end-effector pose
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_indices_for_obs * 3 + 7 + 7,), dtype=np.float32)

    def setup_simulation_scene(self):
        super().setup_simulation_scene()

        # Initialize the cloth
        self.add_cloth()
        self.grasp_constraint = None

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
        # for feature, weight in self.reward_feature_weights.items():
        #     reward += weight * reward_features[feature]
        #     info[f"reward_{feature}"] = weight * reward_features[feature]
        # info["success"] = reward_features["done"]
        self.previous_reward_features = reward_features

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

    def add_cloth(self):
        cloth_scale = 0.05
        collision_margin = 0.001
        position = [-0.05, 0.45, collision_margin / 2]

        # Create the cloth
        self.cloth_id = self.bullet_client.loadSoftBody(
            fileName=f"{HERE}/meshes/cloth.obj",
            basePosition=position,
            baseOrientation=self.bullet_client.getQuaternionFromEuler([np.pi / 2, 0, 0]),
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


if __name__ == "__main__":
    target_dt = 0.02
    simulation_hz = 250
    frame_skip = int(round(target_dt * simulation_hz))
    env = DVRKEnvTR(
        render_mode="human",
        action_type=ActionType.RELATIVE_POSITION,
        # action_type=ActionType.ABSOLUTE_POSITION,
        simulation_hz=simulation_hz,
        frame_skip=frame_skip,
        randomize_initial_joint_values=False,
    )

    # Disable scientific notation for numpy
    np.set_printoptions(suppress=True)

    env.reset()
    while True:
        random_action = np.zeros(7)
        if not env.grasp_constraint:
            random_action[2] = 1.0
        else:
            if env.psm.get_joint_position(2) < 0.05:
                random_action[2] = 0.0
            else:
                random_action[2] = -1.0
        obs, reward, terminated, truncated, info = env.step(random_action)
        print(f"Current FPS: {env.current_fps:.2f}")

    # insertion_values = np.linspace(0.05, 0.145, 1000)
    # current_joint_values = env.psm.get_joint_positions()
    # grasped = False
    #
    # while True:
    #     for insertion_value in insertion_values:
    #         current_joint_values[2] = insertion_value
    #         env.psm.set_joint_positions(current_joint_values)
    #         env.bullet_client.stepSimulation()
    #
    #         # positions, velocities, forces, torques = env.bullet_client.getJointStates(env.psm.robot_id, env.psm.joint_ids)
    #         positions = [val[0] for val in env.bullet_client.getJointStates(env.psm.robot_id, env.psm.joint_ids)]
    #         control_error = np.rad2deg(np.array(current_joint_values) - np.array(positions))
    #         # print(control_error)
    #
    #         if not grasped:
    #             contact, vertex = env.check_gripper_contacts()
    #             if contact:
    #                 assert vertex is not None
    #                 env.create_grasp_constraint(vertex)
    #         time.sleep(1.0 / env.simulation_hz)
    #
    #     for insertion_value in insertion_values[::-1]:
    #         current_joint_values[2] = insertion_value
    #         env.psm.set_joint_positions(current_joint_values)
    #         env.bullet_client.stepSimulation()
    #         positions = [val[0] for val in env.bullet_client.getJointStates(env.psm.robot_id, env.psm.joint_ids)]
    #         control_error = np.rad2deg(np.array(current_joint_values) - np.array(positions))
    #         # print(control_error)
    #         # contact, vertex = env.check_gripper_contacts()
    #         time.sleep(1.0 / env.simulation_hz)
