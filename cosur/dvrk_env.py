import pybullet as p

from utils import PSM

if __name__ == "__main__":
    simulation_hz = 500
    # Setup the simulation environment
    p.connect(p.GUI)
    p.setRealTimeSimulation(0)
    p.setPhysicsEngineParameter(fixedTimeStep=1 / simulation_hz)
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

    psm = PSM("dvrk.urdf", simulation_hz=simulation_hz, show_frames=True)
    psm.demo_motion()

    p.disconnect()
