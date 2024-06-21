import pybullet as p
import pybullet_data
import time
import numpy as np

from utils import PSM, add_dummy_box, add_dummy_sphere

if __name__ == "__main__":
    simulation_hz = 500
    # Setup the simulation environment
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setRealTimeSimulation(0)
    p.setPhysicsEngineParameter(fixedTimeStep=1 / simulation_hz)
    p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
    p.setGravity(0, 0, -9.81)
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

    psm = PSM(
        bullet_client=p,
        urdf_path="dvrk.urdf",
        simulation_hz=simulation_hz,
        show_frames=True,
        base_position=[0.0, 0.0, 0.15],
        base_orientation=[0.0, 0.0, 0.0],
    )

    planeOrn = [0, 0, 0, 1]  # p.getQuaternionFromEuler([0.3,0,0])
    planeId = p.loadURDF("plane.urdf", [0, 0, -0.006], planeOrn)

    clothId = p.loadSoftBody(
        "cloth_z_up.obj",
        basePosition=[0, 0.5, 0],
        scale=0.03,
        mass=0.2,
        # useNeoHookean=1,
        useNeoHookean=0,
        useBendingSprings=1,
        useMassSpring=1,
        springElasticStiffness=40,
        springDampingStiffness=0.1,
        springDampingAllDirections=1,
        useSelfCollision=0,
        frictionCoeff=0.5,
        useFaceContact=1,
        collisionMargin=0.006,
        # collisionMargin=0.0,
    )

    p.changeVisualShape(clothId, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED)

    anchor_indices = [22, 4, 20, 3, 5, 21]
    for i in anchor_indices:
        p.createSoftBodyAnchor(clothId, i, -1, -1)

    # p.createSoftBodyAnchor(clothId, 15, boxId, -1, [0.5, -0.5, 0])
    # p.createSoftBodyAnchor(clothId, 19, boxId, -1, [-0.5, -0.5, 0])
    # p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
    data = p.getMeshData(clothId, -1, flags=p.MESH_DATA_SIMULATION_MESH)
    print("--------------")
    print("data=", data)
    print(data[0])
    print(data[1])
    text_uid = []
    for i in range(data[0]):
        pos = data[1][i]
        uid = p.addUserDebugText(str(i), pos, textColorRGB=[1, 1, 1])
        text_uid.append(uid)

    # target_pos = [0.1, 0.5, -0.15]
    # target_orn = [0.0, 0.0, 0.0]
    # target_sphere_id = add_dummy_sphere(radius=0.01, position=target_pos, orientation=target_orn, color=[0, 1, 0, 1.0])

    # # Use the inverse kinematics to find the joint angles that will move the end effector to the target position
    # joint_angles = p.calculateInverseKinematics(psm.robot_id, psm.ee_link_index, target_pos, target_orn)
    # print(joint_angles)
    #
    # psm.set_joint_positions(joint_angles)

    current_joint_values = psm.get_joint_positions()
    insertion_values = np.linspace(0.05, 0.12, 1000)
    grasped = False
    while True:
        for value in insertion_values:
            current_joint_values[2] = value
            psm.set_joint_positions(current_joint_values)
            p.stepSimulation()
            time.sleep(1 / simulation_hz)
        if not grasped:
            p.createSoftBodyAnchor(clothId, 15, psm.robot_id, 7)
            grasped = True
        for value in insertion_values[::-1]:
            current_joint_values[2] = value
            psm.set_joint_positions(current_joint_values)
            p.stepSimulation()
            time.sleep(1 / simulation_hz)
        # psm.set_joint_positions(current_joint_values)
        # p.stepSimulation()
        # time.sleep(1 / simulation_hz)

    # psm.demo_motion()

    p.disconnect()
