import sys
import os

# 1. ur10.py가 있는 원본 경로 (절대 복사 금지, 여기서 끌어옴)
UR10_ORIGINAL_PATH = r"C:\isaacsim\process_test\scripts\test"
if UR10_ORIGINAL_PATH not in sys.path:
    sys.path.append(UR10_ORIGINAL_PATH)

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# [핵심 수정: 버전 충돌 없이 가장 확실하게 ROS 2 브릿지 강제 활성화]
import omni.kit.app
manager = omni.kit.app.get_app().get_extension_manager()
manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)

import asyncio
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

# 질문자님의 원본 씬 구성 요소 임포트
import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager
from isaacsim.core.experimental.objects import Cube, DistantLight, GroundPlane
from ur10 import UR10 

class IsaacROS2Node(Node):
    def __init__(self, robot):
        super().__init__('isaac_vla_node')
        self.robot = robot
        self.bridge = CvBridge()
        
        self.img_pub = self.create_publisher(RosImage, '/isaac_camera/image_raw', 1)
        self.action_sub = self.create_subscription(
            Float32MultiArray, '/openvla/action_cmd', self.action_callback, 1)

    def action_callback(self, msg):
        action = np.array(msg.data)
        if np.any(action != 0):
            current_pose = self.robot.get_end_effector_pose()
            target_pos = current_pose[0] + action[:3] # Delta 위치 적용
            # 역기구학(IK) 적용
            self.robot.set_end_effector_pose(position=target_pos, ik_method="damped-least-squares")

    def publish_image(self, img_array):
        msg = self.bridge.cv2_to_imgmsg(img_array, encoding="rgb8")
        self.img_pub.publish(msg)

class VLAControlExecutor:
    def __init__(self, usd_path: str):
        self.usd_path = usd_path
        self.robot = None
        self.ros_node = None
        # 질문자님 지정 홈 포즈
        self.home_pose = [0.0, -np.pi/2, np.pi/2, np.pi/2, np.pi/2, 0.0]
        self.cube_path = "/World/TargetCube"
        self.cube_pos = np.array([0.75, 0.0, 0.025])
        self.cube_size = np.array([0.05, 0.05, 0.05])
        self._physics_callback_id = None

    def setup_scene(self):
        # 1. 바닥 및 조명 세팅
        stage = stage_utils.get_current_stage()
        if not stage.GetPrimAtPath("/World/ground_plane"):
            GroundPlane("/World/ground_plane")
        if not stage.GetPrimAtPath("/World/DistantLight"):
            DistantLight("/World/DistantLight").set_intensities(1000)

        # 2. 로봇 소환
        stage_utils.add_reference_to_stage(usd_path=self.usd_path, path="/World/CustomAsset")
        self.robot = UR10(robot_path="/World/CustomAsset/ur10", create_robot=True)

        # 3. 큐브 소환
        Cube(paths=self.cube_path, positions=self.cube_pos, sizes=1.0, scales=self.cube_size, colors="red")
        
        print("[System] 로봇 및 큐브 씬 구성 완료.")

        # 4. ROS 2 통신 노드 연결
        rclpy.init()
        self.ros_node = IsaacROS2Node(self.robot)

    def _physics_callback(self, dt: float, context: object):
        if self.robot is None or self.ros_node is None: return

        # 카메라 캡처 (현재는 빈 이미지, 추후 실제 캡처 로직 교체)
        current_img = np.zeros((224, 224, 3), dtype=np.uint8) 
        self.ros_node.publish_image(current_img)
        
        # ROS 2 콜백 처리 (액션 수신 대기)
        rclpy.spin_once(self.ros_node, timeout_sec=0)

    async def start_async(self):
        app_utils.play()
        
        # 홈 위치로 관절 이동
        full_initial_pose = np.zeros(self.robot.num_dofs)
        full_initial_pose[:6] = self.home_pose
        self.robot.set_dof_positions(full_initial_pose)
        self.robot.set_dof_position_targets(full_initial_pose)
        
        print("[System] 홈 관절 값 초기화 완료. 대기 중...")
        for _ in range(60): await app_utils.update_app_async()
            
        print("[System] VLA 제어 루프 및 ROS 2 통신 시작.")
        self._physics_callback_id = SimulationManager.register_callback(
            self._physics_callback, event=SimulationEvent.PHYSICS_POST_STEP
        )

if __name__ == "__main__":
    usd_target = r"C:\isaacsim\process_test\assets\ur10_camera_gripper.usd"
    executor = VLAControlExecutor(usd_path=usd_target)
    executor.setup_scene()
    
    asyncio.ensure_future(executor.start_async())
    
    while simulation_app.is_running():
        simulation_app.update()
        
    if executor.ros_node:
        executor.ros_node.destroy_node()
    rclpy.shutdown()