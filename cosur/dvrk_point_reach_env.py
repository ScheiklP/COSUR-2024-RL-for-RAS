import time
import numpy as np
import gymnasium as gym

from collections import deque
from enum import Enum
from pathlib import Path

from utils import PSM, add_dummy_sphere

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
}


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
        target_position_threshold: float = 0.005,
        target_orientation_threshold: float = 0.01,
        reward_feature_weights: dict = {
            "num_collision_with_floor": -1.0,
            "simulation_unstable": -1.0,
            "position_difference": -5.0,
            "orientation_difference": -1.0,
            "done": 100.0,
        },
        show_workspace: bool = False,
    ):
        # Import pybullet only when the environment is created.
        # This is to avoid problems when running multiple environments in parallel.
        # Multiprocessing is hard when the libraries do a lot of stuff on import.
        import pybullet as p
        import pybullet_data

        self.bullet_client = p

        # Set the render mode
        self.render_mode = render_mode
        self.bullet_client.connect(self.bullet_client.GUI if render_mode == "human" else self.bullet_client.DIRECT)

        # Controlling the robot at a super high frequency
        # can become pretty hard, because there are no real changes between frames.
        # We can artificially increase the time between observations (and actions) by skipping frames.
        self.simulation_hz = simulation_hz
        self.frame_skip = frame_skip
        self.effective_dt = 1.0 / (self.simulation_hz * self.frame_skip)
        # Set metadata
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": 1.0 / self.effective_dt,
        }

        # Simulation behavior
        self.bullet_client.setRealTimeSimulation(0)  # Disable real-time simulation
        self.bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())  # Add the pybullet data path to the search path for plane.urdf
        self.bullet_client.setPhysicsEngineParameter(fixedTimeStep=1 / self.simulation_hz)  # Set the simulation frequency
        self.bullet_client.setGravity(0, 0, -9.81)  # Set gravity
        if self.render_mode == "human":
            self.bullet_client.resetDebugVisualizerCamera(**CAMERA_VIEW["closeup"])

        # Setup the simulation scene
        self.randomize_initial_joint_values = randomize_initial_joint_values
        self.joint_velocities_unstable_threshold = 1.0
        self.setup_simulation_scene()

        if show_workspace:
            self.visualize_workspace()

        # Action and observation spaces
        self.action_type = action_type

        # We will use action spaces that are bound between [-1, 1] to make our life easier.
        # Depending on the action type, we will scale these values to the appropriate range.
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        low, high = [np.array(limits) for limits in zip(*self.psm.joint_limits)]
        if action_type == ActionType.ABSOLUTE_POSITION:
            self.scale_action = lambda action: low + (action + 1) * 0.5 * (high - low)
        elif action_type == ActionType.RELATIVE_POSITION:
            # We set the maximum speed of the robot to the range of motion divided by the time it takes to reach the end of the range.
            # We will use this to scale the action space.
            time_for_full_rom = 1.0  # in seconds
            max_speed = (high - low) / (time_for_full_rom * self.simulation_hz)
            # Action of 0 means no movement, 1 means full speed in one direction, -1 means full speed in the other direction.
            self.scale_action = lambda action: action * max_speed
        else:
            raise ValueError(f"Invalid action type: {action_type}")

        self.reward_feature_weights = reward_feature_weights
        self.previous_reward_features = {}
        self.target_position_threshold = target_position_threshold
        self.target_orientation_threshold = target_orientation_threshold
        self.dt_queue = deque(maxlen=100)

        self.needs_reset = True

    def setup_simulation_scene(self):
        # Add a plane
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
        self.psm.show_ee_frame()
        self.joint_target_positions = self.psm.get_joint_positions()
        self.inital_joint_positions = self.joint_target_positions.copy()

        # The effective workspace of the robot is a cone with the RCM point at the tip.
        # The side of the cone is equal to the main insertion link length (minus a bit, because it can fully retract into the shaft).
        self.rcm_position = self.psm.get_rcm_position()
        self.z_min = 0.01
        self.z_max = self.rcm_position[2] - 0.04  # Part of the insertion shaft is inside the patient -> cut off a bit
        cone_side_length = 0.233
        cone_height = self.rcm_position[2] - self.z_min
        cone_radius = np.sqrt(cone_side_length**2 - cone_height**2)
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
        for i in range(num_lines):
            angle = 2 * np.pi * i / num_lines
            x = self.sample_radius * np.cos(angle)
            y = self.sample_radius * np.sin(angle)
            z = -self.rcm_position[2] + self.z_min
            self.bullet_client.addUserDebugLine(
                lineFromXYZ=self.rcm_position,
                lineToXYZ=self.rcm_position + np.array([x, y, z]),
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
        height = self.rcm_position[2]
        radius = self.sample_radius
        z = self.np_random.uniform(0, height)

        # Calculate the maximum radius at each height
        max_radius_at_z = radius * (1 - z / height)

        # Sample uniformly within the circle at each height
        r = max_radius_at_z * np.sqrt(self.np_random.uniform(0, 1))
        theta = 2 * np.pi * self.np_random.uniform(0, 1)

        # Convert from polar to Cartesian coordinates and clip the z coordinate
        x = r * np.cos(theta) + self.rcm_position[0]
        y = r * np.sin(theta) + self.rcm_position[1]
        z = np.clip(z, self.z_min, self.z_max)

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

        for _ in range(self.frame_skip):
            self.psm.set_joint_positions(new_joint_target_positions)
            self.joint_target_positions = new_joint_target_positions
            self.bullet_client.stepSimulation()

        # Compute the reward and additional information
        reward_features = self.calculate_reward_features()
        reward = 0.0
        info = {}
        for feature, weight in self.reward_feature_weights.items():
            reward += weight * reward_features[feature]
            info[f"reward_{feature}"] = weight * reward_features[feature]
        info["distance_to_target_position"] = reward_features["position_difference"]
        info["distance_to_target_orientation"] = reward_features["orientation_difference"]
        self.previous_reward_features = reward_features

        # Check if the episode is done
        terminated = reward_features["done"]
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
        reward_features["reached_target_orientation"] = orientation_difference < self.target_orientation_threshold

        reward_features["done"] = reward_features["reached_target_position"] and reward_features["reached_target_orientation"]

        return reward_features

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

        return np.concatenate(list(observation_features.values()))

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        # Parent class reset initializes the RNG
        super().reset(seed=seed, options=options)

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

        self.target_position = self.sample_target_position()
        self.target_orientation = self.sample_target_orientation()

        # Visualize the target position and orientation
        add_dummy_sphere(
            bullet_client=self.bullet_client,
            position=self.target_position,
            orientation=self.target_orientation,
            radius=0.005,
            color=[0, 0, 1, 1],
            with_frame=True,
        )

        self.needs_reset = False

        observation = self.get_observation()
        reset_info = {}

        return observation, reset_info

    def render(self, mode: str = "human") -> np.ndarray | None:
        pass

    def close(self):
        self.bullet_client.disconnect()


if __name__ == "__main__":
    target_dt = 0.1
    simulation_hz = 200
    frame_skip = int(round(target_dt * simulation_hz))
    env = DVRKEnv(
        render_mode="human",
        action_type=ActionType.RELATIVE_POSITION,
        simulation_hz=simulation_hz,
        frame_skip=frame_skip,
        randomize_initial_joint_values=False,
    )

    # Disable scientific notation for numpy
    np.set_printoptions(suppress=True)

    env.reset()
    while True:
        random_action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(random_action)
