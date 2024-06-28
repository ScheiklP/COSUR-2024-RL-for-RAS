import time
import numpy as np
import gymnasium as gym

from collections import deque
from enum import Enum
from pathlib import Path

import pybullet
import pkgutil
import pybullet_data
from pybullet_utils import bullet_client

from utils import PSM, add_coordinate_frame, add_dummy_sphere

HERE = Path(__file__).parent

CAMERA_VIEW = {
    "overview": {
        "cameraTargetPosition": (-0.21, -0.15, -0.02),
        "cameraDistance": 1.2,
        "cameraPitch": -25.0,
        "cameraYaw": 125.2,
    },
    "closeup": {
        "cameraTargetPosition": (-0.27, -0.33, -0.4),
        "cameraDistance": 1.06,
        "cameraPitch": -23.0,
        "cameraYaw": 160.8,
    },
    "workspace": {
        "cameraTargetPosition": (-0.41, -0.05, -0.27),
        "cameraDistance": 1.2,
        "cameraPitch": -25.80,
        "cameraYaw": 144.0,
    },
    "tissue": {
        "cameraTargetPosition": (-0.47, -0.23, -0.39),
        "cameraDistance": 1.1,
        "cameraPitch": -27.40,
        "cameraYaw": 147.6,
    },
}

STANDARDIZATION_VALUES = {
    "obs_mean": np.array(
        [
            -0.020213507127663222,
            -0.01869795543403532,
            0.12069798633646923,
            -0.05316467279775374,
            -0.10771351336254666,
            -0.010344871223157033,
            0.6706101237722515,
            -0.0011738641262115756,
            0.4680406437091224,
            0.062311319006895884,
            -0.20996672867213306,
            0.013811142829082733,
            0.0013659369797257384,
            0.415733201744104,
            -0.0018221878649365257,
            0.48359944559982804,
            0.04945635386727688,
            -0.668002120156091,
            0.0010590768690189665,
            -0.00023844815994337765,
            0.6298654749956958,
        ],
        dtype=np.float32,
    ),
    "obs_var": np.array(
        [
            0.24568176207513684,
            0.23733063697108495,
            0.0018577462178327498,
            0.5035955394799235,
            0.4841230335953069,
            1.5987642935177806,
            0.1960225680385471,
            0.004329091390798756,
            0.020790695571665165,
            0.0033878893450064508,
            0.4074766216099463,
            0.07556956054717207,
            0.07026786106897699,
            0.22957324026535278,
            0.0029446992495264688,
            0.0029780398682169944,
            0.0006766550724938759,
            0.027047582114347223,
            0.04977482749820537,
            0.05164073802620834,
            0.028578321566117827,
        ],
        dtype=np.float32,
    ),
}

STANDARDIZATION_VALUES["obs_std"] = np.sqrt(STANDARDIZATION_VALUES["obs_var"])


class ActionType(Enum):
    ABSOLUTE_POSITION = 1
    RELATIVE_POSITION = 2


class DVRKEnv(gym.Env):
    def __init__(
        self,
        simulation_hz: int = 500,
        render_mode: str | None = "human",
        frame_skip: int = 1,
        action_type: ActionType = ActionType.ABSOLUTE_POSITION,
        randomize_initial_joint_values: bool = True,
        target_position_threshold: float = 0.005,  # in meters
        target_orientation_threshold: float = 5.0,  # in degrees
        reward_feature_weights: dict = {
            "num_collision_with_floor": -1.0,
            "simulation_unstable": -1.0,
            "position_difference": -10.0,
            "orientation_difference": -1.0,
            "done": 100.0,
        },
        egl_rendering: bool = False,
        image_shape: tuple[int, int] = (420, 420),
        fixed_target_position: list[float] | None = None,
        fixed_target_orientation: list[float] | None = None,
        manually_standardize_obs: bool = False,
    ):
        # Controlling the robot at a super high frequency
        # can become pretty hard, because there are no real changes between frames.
        # We can artificially increase the time between observations (and actions) by skipping frames.
        self.simulation_hz = simulation_hz
        self.frame_skip = frame_skip
        self.effective_dt = 1.0 / simulation_hz * frame_skip
        # Set metadata
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": 1.0 / self.effective_dt,
        }

        self.image_shape = image_shape
        self.egl_rendering = egl_rendering
        self.render_mode = render_mode
        self.physics_client_id = None

        # Setup the simulation scene
        self.randomize_initial_joint_values = randomize_initial_joint_values
        self.joint_velocities_unstable_threshold = 1.0

        # Action and observation spaces
        self.action_type = action_type

        self.psm_joint_limits = [
            [-1.605, 1.5994],
            [-0.93556, 0.94249],
            [0.0, 0.24],
            [-3.14159, 3.14159],
            [-1.5708, 1.5708],
            [-1.5708, 1.5708],
            [0.0, 1.0],
        ]

        # We will use action spaces that are bound between [-1, 1] to make our life easier.
        # Depending on the action type, we will scale these values to the appropriate range.
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        low, high = [np.array(limits) for limits in zip(*self.psm_joint_limits)]
        if action_type == ActionType.ABSOLUTE_POSITION:
            self.scale_action = lambda action: low + (action + 1) * 0.5 * (high - low)
        elif action_type == ActionType.RELATIVE_POSITION:
            # Joint velocities in rad/2 and m/s
            max_joint_velocities_rev = np.deg2rad(45)
            max_joint_velocities_pris = 0.08

            scaling_values = np.array([max_joint_velocities_rev] * 2 + [max_joint_velocities_pris] + [max_joint_velocities_rev] * 4)

            # Action of 0 means no movement, 1 means full speed in one direction, -1 means full speed in the other direction.
            self.scale_action = lambda action: action * scaling_values * self.effective_dt
        else:
            raise ValueError(f"Invalid action type: {action_type}")

        # Target position and orientation
        self.fixed_target_position = fixed_target_position
        self.fixed_target_orientation = fixed_target_orientation
        self.target_position = None
        self.target_orientation = None

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)
        self.manually_standardize_obs = manually_standardize_obs

        self.reward_feature_weights = reward_feature_weights
        self.previous_reward_features = {}
        self.target_position_threshold = target_position_threshold
        self.target_orientation_threshold = target_orientation_threshold
        self.dt_queue = deque(maxlen=100)
        self.visual_target = None

        self.needs_reset = True

    def setup_simulation_scene(self):
        self.bullet_client.setTimeStep(1 / self.simulation_hz)  # Set the simulation frequency
        self.bullet_client.setGravity(0, 0, -9.81)  # Set gravity
        if self.render_mode == "human":
            # Set camera pose of GUI
            camera_config = CAMERA_VIEW["workspace"]
            self.bullet_client.resetDebugVisualizerCamera(**camera_config)

        # Add a plane
        self.bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())  # Add the pybullet data path to the search path for plane.urdf
        self.plane = self.bullet_client.loadURDF("plane.urdf", [0, 0, 0])

        # Initialize the PSM
        self.psm = PSM(
            bullet_client=self.bullet_client,
            urdf_path="dvrk.urdf",
            show_frames=False,
            base_position=[0.0, 0.0, 0.15],
            base_orientation=[0.0, 0.0, 0.0],
            max_motor_force=1_000.0,
            mimic_joint_force_factor=1.0,
        )
        self.psm.joint_limits = self.psm_joint_limits

        if self.render_mode == "human":
            self.psm.show_ee_frame()

        self.joint_target_positions = self.psm.get_joint_positions()
        self.inital_joint_positions = self.joint_target_positions.copy()

        # The effective workspace of the robot is a cone with the RCM point at the tip.
        # The side of the cone is equal to the main insertion link length (minus a bit, because it can fully retract into the shaft).
        self.rcm_position = self.psm.get_rcm_position()
        instrument_shaft_offset = 0.05
        instrument_length = self.psm_joint_limits[2][1]

        z_min = 0.01
        cone_side_length = instrument_length - instrument_shaft_offset
        cone_height = self.rcm_position[2] - instrument_shaft_offset - z_min
        cone_radius = np.sqrt(cone_side_length**2 - cone_height**2)

        self.z_min = z_min
        self.z_max = self.z_min + cone_height
        self.sample_radius = cone_radius

        # Euler angles
        self.target_orientation_limits = [
            [-np.pi / 4 - np.pi / 2, -np.pi / 4, -np.pi / 4],
            [np.pi / 4 - np.pi / 2, np.pi / 4, np.pi / 4],
        ]

    def visualize_workspace(self):
        # Draw the cone by adding lines from the RCM point to the circle around the RCM point
        num_lines = 100
        num_points = 200
        rcm_position = self.psm.get_rcm_position()
        tip = rcm_position
        tip[-1] = self.z_max

        for i in range(num_lines):
            angle = 2 * np.pi * i / num_lines
            x = self.sample_radius * np.cos(angle)
            y = self.sample_radius * np.sin(angle)
            z = -self.z_max
            self.bullet_client.addUserDebugLine(
                lineFromXYZ=tip,
                lineToXYZ=tip + np.array([x, y, z]),
                lineColorRGB=[1, 0, 0],
                lineWidth=1.0,
            )

        # Randomly sample points within the workspace and visualize them
        for _ in range(num_points):
            target_position = self.sample_target_position()
            add_dummy_sphere(bullet_client=self.bullet_client, position=target_position, radius=0.001, color=[0, 1, 0, 1], with_frame=False)

    def sample_target_position(self) -> np.ndarray:
        """Sample a target position within the conical workspace of the robot."""

        # Sample uniformly in terms of z coordinate
        height = self.z_max - self.z_min
        radius = self.sample_radius
        z = self.np_random.uniform(0, height)

        # Calculate the maximum radius at each height
        max_radius_at_z = radius * (1 - z / height)

        # Sample uniformly within the circle at each height
        r = max_radius_at_z * np.sqrt(self.np_random.uniform(0, 1))
        theta = 2 * np.pi * self.np_random.uniform(0, 1)

        # Convert from polar to Cartesian coordinates
        x = r * np.cos(theta) + self.rcm_position[0]
        y = r * np.sin(theta) + self.rcm_position[1]
        z = z + self.z_min

        return np.array([x, y, z])

    def sample_target_orientation(self) -> np.ndarray:
        """Sample a target orientation within the limits of the robot as a quaternion."""
        euler_angles = self.np_random.uniform(self.target_orientation_limits[0], self.target_orientation_limits[1])
        return np.array(self.bullet_client.getQuaternionFromEuler(euler_angles))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one time step in the environment.

        1. Apply the action to the robot and simulate the environment for the duration of the frame skip.
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
            self.bullet_client.stepSimulation()

        # Compute the reward and additional information
        reward_features, reward_info = self.calculate_reward_features()
        reward = 0.0
        info = {}
        for feature, weight in self.reward_feature_weights.items():
            reward += weight * reward_features[feature]
            info[f"reward_{feature}"] = weight * reward_features[feature]

        self.previous_reward_features = reward_features
        info["distance_to_target_position"] = reward_features["position_difference"]
        info["distance_to_target_orientation"] = reward_features["orientation_difference"]

        info["success"] = reward_features["done"]
        info = info | reward_info

        # Check if the episode is done
        terminated = reward_features["done"]
        truncated = False

        # Get the new observation
        observation = self.get_observation()

        end_time = time.time()
        self.dt_queue.append(end_time - start_time)
        self.current_fps = 1.0 / np.mean(self.dt_queue)

        return observation, reward, terminated, truncated, info

    def calculate_reward_features(self) -> tuple[dict, dict]:
        reward_features = {}
        info = {}

        num_collision_with_floor = len(self.bullet_client.getContactPoints(bodyA=self.psm.robot_id, bodyB=self.plane))
        reward_features["num_collision_with_floor"] = num_collision_with_floor

        joint_velocities = self.psm.get_joint_velocities()
        simulation_unstable = np.linalg.norm(joint_velocities) > self.joint_velocities_unstable_threshold
        reward_features["simulation_unstable"] = simulation_unstable

        current_position, current_quaternion = self.psm.get_ee_pose()
        position_difference = np.linalg.norm(current_position - self.target_position)
        reward_features["position_difference"] = position_difference
        reward_features["reached_target_position"] = position_difference < self.target_position_threshold

        quaternion_difference = current_quaternion - self.target_orientation
        quaternion_sum = current_quaternion + self.target_orientation
        # Paragraph 3.3 in https://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf
        # Huynh, D. Q. (2009). Metrics for 3D rotations: Comparison and analysis. Journal of Mathematical Imaging and Vision, 35, 155-164.
        # Loss is in range [0, sqrt(2)]
        # Difference and sum to account for quaternion double coverage
        orientation_difference = min(np.linalg.norm(quaternion_difference), np.linalg.norm(quaternion_sum))
        reward_features["orientation_difference"] = orientation_difference

        ee_euler = self.bullet_client.getEulerFromQuaternion(current_quaternion)
        target_euler = self.bullet_client.getEulerFromQuaternion(self.target_orientation)
        euler_error = np.rad2deg(np.arccos(np.cos(np.array(ee_euler) - np.array(target_euler))))

        info["euler_error_x"] = euler_error[0]
        info["euler_error_y"] = euler_error[1]
        info["euler_error_z"] = euler_error[2]

        reward_features["reached_target_orientation"] = np.all(euler_error < self.target_orientation_threshold)

        reward_features["done"] = reward_features["reached_target_position"] and reward_features["reached_target_orientation"]

        return reward_features, info

    def get_observation(self) -> np.ndarray:
        observation_features = {}

        # Joint positions
        observation_features["joint_positions"] = self.psm.get_joint_positions()

        # End-effector pose
        ee_position, ee_quaternion = self.psm.get_ee_pose()
        observation_features["ee_position"] = ee_position
        observation_features["ee_orientation"] = ee_quaternion

        # Target position and orientation
        observation_features["target_position"] = self.target_position
        observation_features["target_orientation"] = self.target_orientation

        obs = np.concatenate(list(observation_features.values()), dtype=np.float32)

        if self.manually_standardize_obs:
            obs = (obs - STANDARDIZATION_VALUES["obs_mean"]) / STANDARDIZATION_VALUES["obs_std"]

        return obs

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        # Parent class reset initializes the RNG
        super().reset(seed=seed, options=options)

        # Initial setup of the simulation scene
        if self.physics_client_id is None:
            if self.render_mode == "human":
                self.bullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            else:
                self.bullet_client = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)

            self.bullet_client.resetSimulation()
            self.bullet_client.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

            if self.egl_rendering and not self.render_mode == "human":
                egl = pkgutil.get_loader("eglRenderer")
                self.bullet_client.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

            self.setup_simulation_scene()
            self.physics_client_id = self.bullet_client._client

        # Set fixed target position and orientation if provided
        if options is not None:
            if "fixed_target_position" in options:
                self.fixed_target_position = options["fixed_target_position"]
            if "fixed_target_orientation" in options:
                self.fixed_target_orientation = options["fixed_target_orientation"]

        # Sample a new target position and orientation
        self.target_position = np.array(self.fixed_target_position) if self.fixed_target_position is not None else self.sample_target_position()
        self.target_orientation = np.array(self.fixed_target_orientation) if self.fixed_target_orientation is not None else self.sample_target_orientation()

        # Visualize the target position and orientation
        if self.visual_target is None:
            self.visual_target, _ = add_dummy_sphere(
                bullet_client=self.bullet_client,
                position=self.target_position,
                orientation=self.target_orientation,
                radius=0.005,
                color=[0, 0, 1, 1],
                with_frame=False,
            )
            if self.render_mode == "human":
                add_coordinate_frame(bullet_client=self.bullet_client, body_id=self.visual_target, size=0.05, frame_id=-1)
        else:
            self.bullet_client.resetBasePositionAndOrientation(self.visual_target, self.target_position, self.target_orientation)

        # Reset the robot
        if self.randomize_initial_joint_values:
            valid_reset = False
            while not valid_reset:
                # Reset the PSM to joint values within 80% of the joint limits
                low, high = [np.array(limits) for limits in zip(*self.psm_joint_limits)]
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
            self.joint_target_positions = [0.0] * 7
            self.psm.reset_joint_positions(self.inital_joint_positions)
            self.bullet_client.stepSimulation()

        self.needs_reset = False

        observation = self.get_observation()
        reset_info = {}

        return observation, reset_info

    def render(self, mode: str = "human", view: str = "workspace", img_type: str = "rgb") -> np.ndarray | None:
        view_matrix = self.bullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=CAMERA_VIEW[view]["cameraTargetPosition"],
            distance=CAMERA_VIEW[view]["cameraDistance"],
            yaw=CAMERA_VIEW[view]["cameraYaw"],
            pitch=CAMERA_VIEW[view]["cameraPitch"],
            roll=0,
            upAxisIndex=2,
        )
        far = 10.0
        near = 0.1
        projection_matrix = self.bullet_client.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=near,
            farVal=far,
        )
        extra_flags = {}
        if img_type != "segmentation":
            extra_flags["flags"] = self.bullet_client.ER_NO_SEGMENTATION_MASK
        img = self.bullet_client.getCameraImage(
            width=self.image_shape[1],
            height=self.image_shape[0],
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            **extra_flags,
        )
        if img_type == "rgb":
            return np.array(img[2])[:, :, :3]
        elif img_type == "depth":
            depth_image = np.array(img[3])
            return far * near / (far - (far - near) * depth_image)
        elif img_type == "segmentation":
            return np.array(img[4])
        else:
            raise ValueError(f"Invalid image type: {img_type}")

    def close(self):
        self.bullet_client.disconnect()


if __name__ == "__main__":
    import cv2

    target_dt = 0.1
    simulation_hz = 50
    reset_time_s = 25.0
    frame_skip = int(round(target_dt * simulation_hz))
    env = DVRKEnv(
        render_mode="rgb_array",
        action_type=ActionType.RELATIVE_POSITION,
        simulation_hz=simulation_hz,
        frame_skip=frame_skip,
        randomize_initial_joint_values=True,
        egl_rendering=True,
        # fixed_target_position=[-0.2, -0.2, 0.1],
        # fixed_target_orientation=[0.0, 0.0, 0.0, 1.0],
    )

    # Disable scientific notation for numpy
    np.set_printoptions(suppress=True)

    def normalize_ypi(yaw, pitch, insertion):
        # normalizing the values with the joint limits to the interval of [-1, 1]
        yaw = 2 * (yaw - env.psm_joint_limits[0][0]) / (env.psm_joint_limits[0][1] - env.psm_joint_limits[0][0]) - 1
        pitch = 2 * (pitch - env.psm_joint_limits[1][0]) / (env.psm_joint_limits[1][1] - env.psm_joint_limits[1][0]) - 1
        insertion = 2 * (insertion - env.psm_joint_limits[2][0]) / (env.psm_joint_limits[2][1] - env.psm_joint_limits[2][0]) - 1
        return yaw, pitch, insertion

    env.reset()
    counter = 0
    # random_action = np.zeros(7)
    # yaw, pitch, insertion = env.psm.tool_position_ik(env.target_position)
    # yaw, pitch, insertion = normalize_ypi(yaw, pitch, insertion)
    # random_action[0] = yaw
    # random_action[1] = pitch
    # random_action[2] = insertion

    while True:
        start = time.time()
        random_action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(random_action)

        if counter == 50:
            env.reset()
            counter = 0
            yaw, pitch, insertion = env.psm.tool_position_ik(env.target_position)
            yaw, pitch, insertion = normalize_ypi(yaw, pitch, insertion)
            # random_action[0] = yaw
            # random_action[1] = pitch
            # random_action[2] = insertion

        counter += 1
        img = env.render()
        cv2.imshow("DVRK", img[:, :, ::-1])
        cv2.waitKey(1)
        end = time.time()
        time.sleep(max(0, target_dt - (end - start)))
