import time
import numpy as np
import gymnasium as gym

from pathlib import Path

from utils import PSM, add_dummy_sphere

HERE = Path(__file__).parent


class DVRKEnv(gym.Env):
    def __init__(
        self,
        simulation_hz: int = 500,
        render_mode: str | None = "human",
    ):
        import pybullet as p
        import pybullet_data

        self.bullet_client = p
        # Set the render mode
        self.render_mode = render_mode
        self.bullet_client.connect(self.bullet_client.GUI if render_mode == "human" else self.bullet_client.DIRECT)

        self.simulation_hz = simulation_hz
        self.bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.bullet_client.setRealTimeSimulation(0)
        self.bullet_client.setPhysicsEngineParameter(fixedTimeStep=1 / self.simulation_hz)
        self.bullet_client.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        self.bullet_client.setGravity(0, 0, -9.81)

        # Add a plane
        self.bullet_client.loadURDF("plane.urdf", [0, 0, 0])

        if self.render_mode == "human":
            # Add a camera
            # cam_target_pos = [-0.21, -0.15, -0.02]
            # cam_dist = 1.2
            # cam_yaw = 125.2
            # cam_pitch = -25.0

            cam_target_pos = [-0.27, -0.33, -0.4]
            cam_dist = 1.06
            cam_pitch = -23.0
            cam_yaw = 160.8

            self.bullet_client.resetDebugVisualizerCamera(
                cameraDistance=cam_dist,
                cameraYaw=cam_yaw,
                cameraPitch=cam_pitch,
                cameraTargetPosition=cam_target_pos,
            )

        self.psm = PSM(
            bullet_client=self.bullet_client,
            urdf_path="dvrk.urdf",
            show_frames=True,
            base_position=[0.0, 0.0, 0.15],
            base_orientation=[0.0, 0.0, 0.0],
            max_motor_force=10_000.0,
            mimic_joint_force_factor=100.0,
        )

        self.add_cloth()

        # grasp_sphere_col = self.bullet_client.createCollisionShape(self.bullet_client.GEOM_SPHERE, radius=0.01)
        # grasp_sphere_viz = self.bullet_client.createVisualShape(self.bullet_client.GEOM_SPHERE, radius=0.003, rgbaColor=[1, 0, 0, 1])
        # self.grasp_sphere = self.bullet_client.createMultiBody(
        #     baseMass=0.1,
        #     baseCollisionShapeIndex=grasp_sphere_col,
        #     baseVisualShapeIndex=grasp_sphere_viz,
        #     basePosition=self.bullet_client.getLinkState(self.psm.robot_id, self.psm.ee_link_index)[0],
        # )
        # c = self.bullet_client.createConstraint(
        #     parentBodyUniqueId=self.psm.robot_id,
        #     parentLinkIndex=self.psm.ee_link_index,
        #     childBodyUniqueId=self.grasp_sphere,
        #     childLinkIndex=-1,
        #     jointType=self.bullet_client.JOINT_POINT2POINT,
        #     # jointType=self.bullet_client.JOINT_FIXED,
        #     # jointAxis=[1, 1, 1],
        #     jointAxis=[0, 0, 0],
        #     parentFramePosition=[0, 0, 0],
        #     childFramePosition=[0, 0, 0],
        # )
        # # self.bullet_client.changeConstraint(c, maxForce=1_000_000, erp=5.0)
        # self.bullet_client.setCollisionFilterPair(self.psm.robot_id, self.grasp_sphere, 5, -1, 0)
        # self.bullet_client.setCollisionFilterPair(self.psm.robot_id, self.grasp_sphere, 6, -1, 0)
        # self.bullet_client.setCollisionFilterPair(self.psm.robot_id, self.grasp_sphere, 7, -1, 0)
        # self.bullet_client.setCollisionFilterPair(self.psm.robot_id, self.grasp_sphere, 8, -1, 0)
        # self.bullet_client.setCollisionFilterPair(self.psm.robot_id, self.grasp_sphere, 9, -1, 0)

        # self.bullet_client.createSoftBodyAnchor(
        #     softBodyBodyUniqueId=self.cloth_id,
        #     nodeIndex=11,
        #     bodyUniqueId=self.grasp_sphere,
        #     linkIndex=-1,
        # )
        self.vis = False
        # self.bullet_client.setPhysicsEngineParameter(sparseSdfVoxelSize=0.01)

    def check_gripper_contacts(self) -> tuple[bool, int | None]:
        contact_points = self.bullet_client.getContactPoints(bodyA=self.psm.robot_id, bodyB=self.cloth_id)

        # Check if the gripper is in contact with the cloth
        collision_with_gripper = False
        contact_position = None
        if len(contact_points) > 0:
            for contact in contact_points:
                # print(contact[3])
                # if contact[3] in [8, 9]:
                if contact[3] in [8, 9, 10]:
                    collision_with_gripper = True
                    contact_position = contact[6]
                    break

        # Find the closest vertex of the cloth to the gripper if there is a collision
        closest_vertex = None
        if collision_with_gripper:
            _, cloth_vertex_positions = self.bullet_client.getMeshData(self.cloth_id, -1, flags=self.bullet_client.MESH_DATA_SIMULATION_MESH)
            closest_vertex = int(np.argmin(np.linalg.norm(np.array(cloth_vertex_positions) - np.array(contact_position), axis=1)))

        return collision_with_gripper, closest_vertex

    def create_grasp_constraint(self, cloth_vertex_id: int):
        # Create a constraint between the gripper and the cloth
        constraint_id = self.bullet_client.createSoftBodyAnchor(
            softBodyBodyUniqueId=self.cloth_id,
            nodeIndex=cloth_vertex_id,
            bodyUniqueId=self.psm.robot_id,
            linkIndex=10,  # -> attached to a link with a spherical joint on the yaw link (7)
        )

        return constraint_id

    def check_contacts(self, body_1_id, body_2_id):
        contact_points = self.bullet_client.getContactPoints(bodyA=body_1_id, bodyB=body_2_id)

        if len(contact_points) > 0:
            _, vertex_positions = self.bullet_client.getMeshData(self.cloth_id, -1, flags=self.bullet_client.MESH_DATA_SIMULATION_MESH)
            if not self.vis:
                data = self.bullet_client.getMeshData(self.cloth_id, -1, flags=self.bullet_client.MESH_DATA_SIMULATION_MESH)
                print("--------------")
                print("data=", data)
                print(data[0])
                print(data[1])
                text_uid = []
                for i in range(data[0]):
                    pos = data[1][i]
                    uid = self.bullet_client.addUserDebugText(str(i), pos, textColorRGB=[1, 1, 1])
                    text_uid.append(uid)
                self.vis = True

        for contact in contact_points:
            for val, name in zip(
                contact,
                [
                    "contactFlag",
                    "bodyUniqueIdA",
                    "bodyUniqueIdB",
                    "linkIndexA",
                    "linkIndexB",
                    "positionOnA",
                    "positionOnB",
                    "contactNormalOnB",
                    "contactDistance",
                    "normalForce",
                    "lateralFriction1",
                    "lateralFrictionDir1",
                    "lateralFriction2",
                    "lateralFrictionDir2",
                ],
            ):
                print(f"{name}: {val}")

                position_on_a = contact[5]
                closest_vertex = np.argmin(np.linalg.norm(np.array(vertex_positions) - np.array(position_on_a), axis=1))
                # print(f"Closest vertex: {closest_vertex}")

        # print(contact_points)
        return len(contact_points)

    def add_cloth(self):
        attachements = [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33]

        mass = 0.03
        cloth_scale = 0.05
        # discretization = 10
        # cloth_length = 2 * cloth_scale
        # edge_length = 2 * cloth_scale / (discretization - 1)
        # collision_margin = edge_length / 5.0
        collision_margin = 0.001

        # Create the cloth
        self.cloth_id = self.bullet_client.loadSoftBody(
            fileName=f"{HERE}/meshes/cloth.obj",
            basePosition=[-0.05, 0.45, collision_margin / 2],
            baseOrientation=self.bullet_client.getQuaternionFromEuler([np.pi / 2, 0, 0]),
            collisionMargin=collision_margin,
            scale=cloth_scale,
            mass=mass,
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
        for attachment in attachements:
            self.bullet_client.createSoftBodyAnchor(
                softBodyBodyUniqueId=self.cloth_id,
                nodeIndex=attachment,
                bodyUniqueId=-1,
                linkIndex=-1,
            )

        # Visualize the attachment points
        for attachment in attachements:
            _, vertex_positions = self.bullet_client.getMeshData(self.cloth_id, -1, flags=self.bullet_client.MESH_DATA_SIMULATION_MESH)
            add_dummy_sphere(self.bullet_client, position=vertex_positions[attachment], radius=0.002, color=[0, 0, 1, 0.6], with_frame=False)


if __name__ == "__main__":
    env = DVRKEnv(render_mode="human")

    # env.psm.demo_motion()
    # Disable scientific notation for numpy
    np.set_printoptions(suppress=True)

    insertion_values = np.linspace(0.05, 0.145, 1000)
    current_joint_values = env.psm.get_joint_positions()
    grasped = False

    while True:
        for insertion_value in insertion_values:
            current_joint_values[2] = insertion_value
            env.psm.set_joint_positions(current_joint_values)
            env.bullet_client.stepSimulation()

            # positions, velocities, forces, torques = env.bullet_client.getJointStates(env.psm.robot_id, env.psm.joint_ids)
            positions = [val[0] for val in env.bullet_client.getJointStates(env.psm.robot_id, env.psm.joint_ids)]
            control_error = np.rad2deg(np.array(current_joint_values) - np.array(positions))
            print(control_error)

            if not grasped:
                contact, vertex = env.check_gripper_contacts()
                if contact:
                    assert vertex is not None
                    env.create_grasp_constraint(vertex)
            time.sleep(1.0 / env.simulation_hz)

        for insertion_value in insertion_values[::-1]:
            current_joint_values[2] = insertion_value
            env.psm.set_joint_positions(current_joint_values)
            env.bullet_client.stepSimulation()
            positions = [val[0] for val in env.bullet_client.getJointStates(env.psm.robot_id, env.psm.joint_ids)]
            control_error = np.rad2deg(np.array(current_joint_values) - np.array(positions))
            print(control_error)
            # contact, vertex = env.check_gripper_contacts()
            time.sleep(1.0 / env.simulation_hz)
