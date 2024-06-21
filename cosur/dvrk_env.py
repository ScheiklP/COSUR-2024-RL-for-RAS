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


# TODO: task versions
# 1. Occluded point becomes visible
# 2. Cloth corner is at a target position
# 3. First go to a target position, then grasp the cloth, then go to another target position

# TODO: restrict tool yaw, pith, and gripper opening, when the tool is still in the insertion shaft


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
        self.bullet_client.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)  # Enable FEM simulation of deformable objects
        self.bullet_client.setGravity(0, 0, -9.81)  # Set gravity
        if self.render_mode == "human":
            self.bullet_client.resetDebugVisualizerCamera(**CAMERA_VIEW["closeup"])

        self.setup_simulation_scene()

        # Action and observation spaces
        self.action_type = action_type

        # We will use action spaces that are bound between [-1, 1] to make our life easier.
        # Depending on the action type, we will scale these values to the appropriate range.
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        low, high = [np.array(limits) for limits in zip(*self.psm.joint_limits)]
        if action_type == ActionType.ABSOLUTE_POSITION:
            self.scale_action = lambda action: low + (action + 1) * 0.5 * (high - low)
        elif action_type == ActionType.RELATIVE_POSITION:
            # The maximum speed of the robot is the range of motion divided by the time it takes to reach the end of the range.
            # We will use this to scale the action space.
            time_for_full_rom = 1.0  # in seconds
            max_speed = (high - low) / (time_for_full_rom * self.simulation_hz)
            # Action of 0 means no movement, 1 means full speed in one direction, -1 means full speed in the other direction.
            self.scale_action = lambda action: action * max_speed
        else:
            raise ValueError(f"Invalid action type: {action_type}")

        # TODO: Observation space based on feature names?

        # TODO: Reset the environment. Randomize the initial joint values of the PSM.

        # TODO: Reward features

        self.reward_feature_weights = {}
        self.dt_queue = deque(maxlen=100)

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
        self.joint_target_positions = self.psm.get_joint_positions()

        # Initialize the cloth
        self.add_cloth()

        # visual_sphere = self.bullet_client.createVisualShape(
        #     shapeType=self.bullet_client.GEOM_SPHERE,
        #     radius=0.005,
        #     rgbaColor=[1, 0, 0, 1],
        # )
        #
        # ee_link_position = self.bullet_client.getLinkState(self.psm.robot_id, self.psm.ee_link_index)[0]
        # ee_link_orientation = self.bullet_client.getLinkState(self.psm.robot_id, self.psm.ee_link_index)[1]
        # self.grasp_sphere = self.bullet_client.createMultiBody(
        #     baseMass=0.01,
        #     baseVisualShapeIndex=visual_sphere,
        #     basePosition=ee_link_position,
        #     baseOrientation=ee_link_orientation,
        #     # useMaximalCoordinates=True,
        #     useMaximalCoordinates=False,
        # )
        # # Create a constraint between the gripper and the sphere
        # self.bullet_client.createConstraint(
        #     parentBodyUniqueId=self.psm.robot_id,
        #     parentLinkIndex=self.psm.ee_link_index,
        #     childBodyUniqueId=self.grasp_sphere,
        #     childLinkIndex=-1,
        #     jointType=self.bullet_client.JOINT_POINT2POINT,
        #     # jointType=self.bullet_client.JOINT_FIXED,
        #     jointAxis=[1, 1, 1],
        #     parentFramePosition=[0, 0, 0],
        #     childFramePosition=[0, 0, 0],
        # )
        # self.bullet_client.changeConstraint(self.psm.robot_id, self.grasp_sphere, maxForce=10000.0)
        #
        # # Ignore collisions between the gripper and the sphere
        # self.bullet_client.setCollisionFilterPair(self.psm.robot_id, self.grasp_sphere, 6, -1, 0)
        # self.bullet_client.setCollisionFilterPair(self.psm.robot_id, self.grasp_sphere, 7, -1, 0)
        # self.bullet_client.setCollisionFilterPair(self.psm.robot_id, self.grasp_sphere, 8, -1, 0)
        # self.bullet_client.setCollisionFilterPair(self.psm.robot_id, self.grasp_sphere, 9, -1, 0)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one time step in the environment.

        1. Apply the action to the robot and simulate the environment for the duration of the frame skip.
        2. Compute the reward based on the new state of the environment.
        3. Check if the episode is done.
        4. Return the new observation, the reward, whether the episode is done, and additional information.
        """

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

        # Compute the reward
        reward_features = self.calculate_reward_features()
        reward = 0.0
        for feature, weight in self.reward_feature_weights.items():
            reward += weight * reward_features[feature]

        # Check if the episode is done
        terminated = self.check_episode_done()
        truncated = False

        # Get the new observation
        observation = self.get_observation()

        # Additional information
        info = {}

        end_time = time.time()
        self.dt_queue.append(end_time - start_time)
        self.current_fps = 1.0 / np.mean(self.dt_queue)

        return observation, reward, terminated, truncated, info

    def calculate_reward_features(self) -> dict:
        reward_features = {}

        num_collision_with_floor = len(self.bullet_client.getContactPoints(bodyA=self.psm.robot_id, bodyB=self.plane))
        reward_features["num_collision_with_floor"] = num_collision_with_floor

        return reward_features

    def check_episode_done(self) -> bool:
        pass

    def get_observation(self) -> np.ndarray:
        pass

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        # Parent class reset initializes the RNG
        super().reset(seed=seed, options=options)

        # Reset the simulation
        self.bullet_client.resetSimulation(self.bullet_client.RESET_USE_DEFORMABLE_WORLD)
        self.setup_simulation_scene()

        max_reset_sim_time = 2.0
        max_sim_steps = int(max_reset_sim_time * self.simulation_hz)
        valid_reset = False
        while not valid_reset:
            # Reset the PSM to joint values within 80% of the joint limits
            low, high = [np.array(limits) for limits in zip(*self.psm.joint_limits)]
            initial_joint_positions = low + 0.2 * (high - low) + 0.6 * self.np_random.random(7) * (high - low)
            self.joint_target_positions = initial_joint_positions

            num_sim_steps = 0
            reached_target_joint_positions = False
            while not reached_target_joint_positions:
                self.psm.set_joint_positions(self.joint_target_positions)
                self.bullet_client.stepSimulation()
                actual_joint_values = self.psm.get_joint_positions()
                control_error = self.joint_target_positions - actual_joint_values
                if np.all(np.abs(control_error) < 0.1):
                    reached_target_joint_positions = True

                robot_in_collision = False
                for object_id in [self.plane, self.cloth_id]:
                    contact_points = self.bullet_client.getContactPoints(bodyA=self.psm.robot_id, bodyB=object_id)
                    if len(contact_points) > 0:
                        robot_in_collision = True
                        break

                if robot_in_collision:
                    break

                num_sim_steps += 1
                if num_sim_steps >= max_sim_steps:
                    break

            if not robot_in_collision and reached_target_joint_positions:
                valid_reset = True

        # TODO return reset observation and additional information
        pass

    def render(self, mode: str = "human") -> np.ndarray | None:
        pass

    def close(self):
        self.bullet_client.disconnect()

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
            # bodyUniqueId=self.grasp_sphere,
            # linkIndex=-1,  # -> attached to a link with a spherical joint on the yaw link (7)
            # bodyFramePosition=[0, 0, 0],
        )

        return constraint_id

    def add_cloth(self):
        fixed_indices = [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33]

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
        for i in fixed_indices:
            id = self.bullet_client.createSoftBodyAnchor(
                softBodyBodyUniqueId=self.cloth_id,
                nodeIndex=i,
                bodyUniqueId=-1,
                linkIndex=-1,
            )
            self.cloth_anchors.append(id)

        # Visualize the attachment points
        self.anchor_spheres = []
        for i in fixed_indices:
            _, vertex_positions = self.bullet_client.getMeshData(self.cloth_id, -1, flags=self.bullet_client.MESH_DATA_SIMULATION_MESH)
            id = add_dummy_sphere(self.bullet_client, position=vertex_positions[i], radius=0.002, color=[0, 0, 1, 0.6], with_frame=False)
            self.anchor_spheres.append(id)

    def render_cloth_indices(self):
        _, vertex_positions = self.bullet_client.getMeshData(self.cloth_id, -1, flags=self.bullet_client.MESH_DATA_SIMULATION_MESH)

        text_object_ids = []
        for i, pos in enumerate(vertex_positions):
            id = self.bullet_client.addUserDebugText(str(i), pos, textColorRGB=[1, 1, 1])
            text_object_ids.append(id)

        return text_object_ids

    def print_contact_data_verbose(self, contact_points: list):
        for contact in contact_points:
            for val, name in zip(contact, ["contactFlag", "bodyUniqueIdA", "bodyUniqueIdB", "linkIndexA", "linkIndexB", "positionOnA", "positionOnB", "contactNormalOnB", "contactDistance", "normalForce", "lateralFriction1", "lateralFrictionDir1", "lateralFriction2", "lateralFrictionDir2"]):
                print(f"{name}: {val}")

    def print_link_names(self):
        for i in range(self.bullet_client.getNumJoints(self.psm.robot_id)):
            name = self.bullet_client.getJointInfo(self.psm.robot_id, i)[1]
            print(f"Link {i}: {name}")

    def print_joint_info(self):
        for i in range(self.bullet_client.getNumJoints(self.psm.robot_id)):
            full_info = self.bullet_client.getJointInfo(self.psm.robot_id, i)

            for val, name in zip(
                full_info,
                ["jointIndex", "jointName", "jointType", "qIndex", "uIndex", "flags", "jointDamping", "jointFriction", "jointLowerLimit", "jointUpperLimit", "jointMaxForce", "jointMaxVelocity", "linkName", "jointAxis", "parentFramePos", "parentFrameOrn", "parentIndex"],
            ):
                if name == "jointType":
                    choices = {
                        self.bullet_client.JOINT_REVOLUTE: "REVOLUTE",
                        self.bullet_client.JOINT_PRISMATIC: "PRISMATIC",
                        self.bullet_client.JOINT_SPHERICAL: "SPHERICAL",
                        self.bullet_client.JOINT_PLANAR: "PLANAR",
                        self.bullet_client.JOINT_FIXED: "FIXED",
                        self.bullet_client.JOINT_POINT2POINT: "POINT2POINT",
                        self.bullet_client.JOINT_GEAR: "GEAR",
                    }
                    val = choices[val]
                print(f"{name}: {val}")
            print()


if __name__ == "__main__":
    target_dt = 0.1
    simulation_hz = 200
    frame_skip = int(round(target_dt * simulation_hz))
    env = DVRKEnv(
        render_mode="human",
        action_type=ActionType.RELATIVE_POSITION,
        # action_type=ActionType.ABSOLUTE_POSITION,
        simulation_hz=simulation_hz,
        frame_skip=frame_skip,
    )

    # Disable scientific notation for numpy
    np.set_printoptions(suppress=True)

    j = 2
    while True:
        env.reset()
        # range_val = np.linspace(-1, 1, 100)
        for i in range(100):
            random_action = np.zeros(7)
            # random_action[j] = range_val[i]
            random_action[j] = 1.0
            # random_action[2] = 1.0
            obs, reward, terminated, truncated, info = env.step(random_action)
            # print(f"Current FPS: {env.current_fps:.2f}")

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
