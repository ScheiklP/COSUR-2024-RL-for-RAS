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

# TASK OVERVIEW:
# 1. In the step function, implement the logic for the different action types.
# 2. Define the action space in the __init__ function.
# 3. Implement the get_observation function.
# 4. Define the observation space in the __init__ function.
# 5. Implement the calculate_reward_features function.
# 6. In the step function, define when the episode is done.
# 7. Define the reward_feature_weights.
# 8. In the step function, add information that you want to monitor during training to the info dictionary.
# 9. Implement the sample_target_position function. Why is it important to randomly sample target positions?
# 10. Implement the sample_target_orientation function. Why is it important to randomly sample target orientations?
# 11. In the reset function, implement the randomization of the initial joint values. Why is it important to randomize the initial joint values?
# 12. Find appropriate values for the STANDARDIZATION_VALUES. Why is it important to standardize the observations?


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
}

# TODO: How would you define these values that can be used to standardize the observations?
STANDARDIZATION_VALUES = {"obs_mean": -1, "obs_var": -1}
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
        target_position_threshold: float = 0.005,
        target_orientation_threshold: float = 5.0,
        # TODO: How do you decide how to weight the different reward features?
        reward_feature_weights: dict = {},
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
        low, high = [np.array(limits) for limits in zip(*self.psm_joint_limits)]

        # We will use action spaces that are bound between [-1, 1] to make our life easier.
        # Depending on the action type, we will scale these values to the appropriate range.
        # TODO: Define the action space. What should be included in the action space? See below in the step function.
        # self.action_space =

        # TODO: A place to precompute / define some behavior for the different action spaces. See below in the step function.

        # Target position and orientation
        self.fixed_target_position = fixed_target_position
        self.fixed_target_orientation = fixed_target_orientation
        self.target_position = None
        self.target_orientation = None

        # TODO: Define the observation space. What should be included in the observation? See below in the get_observation function.
        # self.observation_space =

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
            urdf_path=str(HERE / "dvrk.urdf"),
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

        # TODO: Define values that can be used to sample new target positions and orientations. See below in the sample_target_position function.
        # Can we just sample any point in space? What about the orientation?

    def visualize_workspace(self, num_points: int = 500) -> None:
        # Randomly sample points within the workspace and visualize them
        for _ in range(num_points):
            target_position = self.sample_target_position()
            add_dummy_sphere(bullet_client=self.bullet_client, position=target_position, radius=0.001, color=[0, 1, 0, 1], with_frame=False)

    def sample_target_position(self) -> np.ndarray:
        """Sample a target position within the workspace of the robot"""

        # TODO: Sample a target position within the workspace of the robot. What should be the limits of the workspace?
        pass

    def sample_target_orientation(self) -> np.ndarray:
        """Sample a target orientation within the limits of the robot as a quaternion."""
        # TODO: Sample a target orientation within the limits of the robot as a quaternion.
        pass

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

        if self.action_type == ActionType.ABSOLUTE_POSITION:
            # TODO: How do we chose the new joint target positions based on the action?
            # new_joint_target_positions =
            new_joint_target_positions = self.psm.get_joint_positions()  # Placeholder
        elif self.action_type == ActionType.RELATIVE_POSITION:
            # TODO: How do we chose the new joint target positions based on the action?
            # new_joint_target_positions =
            new_joint_target_positions = self.psm.get_joint_positions()  # Placeholder
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

        # If we need features from the previous time step, we can access them here
        self.previous_reward_features = reward_features

        # TODO: What would be interesting to include in the info dictionary?
        # info[....] =
        info = info | reward_info

        # TODO: Check if the episode is done
        terminated = False  # Placeholder
        truncated = False  # Rarely used in practice except for time limits

        # Get the new observation
        observation = self.get_observation()

        end_time = time.time()
        self.dt_queue.append(end_time - start_time)
        self.current_fps = 1.0 / np.mean(self.dt_queue)

        return observation, reward, terminated, truncated, info

    def calculate_reward_features(self) -> tuple[dict, dict]:
        reward_features = {}
        info = {}

        # TODO: Define the reward features. What should be included in the reward features?
        # How can we describe how good the current state of the environment is?

        return reward_features, info

    def get_observation(self) -> np.ndarray:
        observation_features = {}
        observation_features["random_placeholder"] = np.random.rand(10)

        # TODO: Define the observation features. What should be included in the observation?:

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
            # TODO: Implement randomization of the initial joint values. Can we just sample any value?
            pass  # Placeholder
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
    # TODO: experiment with these values. What happens if the simulation_hz is too high or too low?
    # What does it mean for the MDP, if the target_dt is too high or too low?
    target_dt = 0.1
    simulation_hz = 50
    frame_skip = int(round(target_dt * simulation_hz))
    env = DVRKEnv(
        render_mode="human",
        action_type=ActionType.RELATIVE_POSITION,
        simulation_hz=simulation_hz,
        frame_skip=frame_skip,
        randomize_initial_joint_values=False,
        fixed_target_position=[0.05, 0.5, 0.05],
        fixed_target_orientation=[1.0, 0.0, 0.0, 0.0],
    )

    env.reset()
    counter = 0

    while True:
        start = time.time()
        random_action = None  # Placeholder
        obs, reward, terminated, truncated, info = env.step(random_action)

        if counter == 5.0 / target_dt:  # 5 seconds
            env.reset()
            counter = 0

        counter += 1
        end = time.time()
        time.sleep(max(0, target_dt - (end - start)))
