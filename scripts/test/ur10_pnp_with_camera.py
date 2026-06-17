# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import sys
import asyncio
import numpy as np

import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.experimental.objects import Cube, DistantLight, GroundPlane
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager

from ur10 import UR10

class FullPickAndPlaceVerification:
    def __init__(self, usd_path: str):
        self.usd_path = usd_path
        self.robot = None
        
        self.cube_path = "/World/TargetCube"
        self.cube_pos = np.array([0.75, 0.0, 0.025])
        self.cube_size = np.array([0.05, 0.05, 0.05])
        
        self.gripper_length = 0.16   
        self.hover_margin = 0.08     
        self.grasp_margin = 0.01
        
        self.goal_orientation = np.array([0.707, 0.0, 0.707, 0.0])
        
        self.events_dt = [60, 40, 50, 80, 50, 40]
        self.wait_steps = 120 
        
        self.current_phase = 0
        self.step_count = 0
        self._physics_callback_id = None
        
        self.safe_working_pose = [0.0, -np.pi/2, np.pi/2, np.pi/2, np.pi/2, 0.0]

    def setup_scene(self):
        stage = stage_utils.get_current_stage()
        
        if not stage.GetPrimAtPath("/World/ground_plane"):
            GroundPlane("/World/ground_plane")
        if not stage.GetPrimAtPath("/World/DistantLight"):
            DistantLight("/World/DistantLight").set_intensities(1000)

        stage_utils.add_reference_to_stage(usd_path=self.usd_path, path="/World/CustomAsset")
        
        self.robot = UR10(robot_path="/World/CustomAsset/ur10", create_robot=False, attach_gripper=False)
        self.robot.reset_xform_op_properties()

        # 큐브 생성 및 기본 충돌체 활성화 (SDF 억지 코드 삭제)
        cube_shape = Cube(paths=self.cube_path, positions=self.cube_pos, sizes=1.0, scales=self.cube_size, colors="red")
        geom_prim = GeomPrim(paths=cube_shape.paths, apply_collision_apis=True)
        rigid_prim = RigidPrim(paths=cube_shape.paths)

    def _control_robotiq_joints(self, target_angle: float):
        """Robotiq 그리퍼 기구학 붕괴 방지: 구동축만 제어"""
        if self.robot is None: return
        dof_names = self.robot.dof_names
        
        # 🔥 [핵심] 'inner'가 포함된 종속 관절은 강제 제어에서 제외하여 물리 붕괴(관통) 원천 차단
        gripper_dof_indices = [
            i for i, name in enumerate(dof_names) 
            if ("finger" in name.lower() or "knuckle" in name.lower()) and "inner" not in name.lower()
        ]
        if gripper_dof_indices:
            targets = [target_angle] * len(gripper_dof_indices)
            self.robot.set_dof_position_targets(targets, dof_indices=gripper_dof_indices)

    def _physics_callback(self, dt: float, context: object):
        if self.robot is None: return

        cube_prim = RigidPrim(self.cube_path)
        cube_current_pos = cube_prim.get_world_poses()[0].numpy()[0]
        ik_method = "damped-least-squares"
        
        grasp_z = cube_current_pos[2] + self.gripper_length + self.grasp_margin
        hover_z = cube_current_pos[2] + self.gripper_length + self.hover_margin

        # Phase 1: 큐브 상공으로 이동
        if self.current_phase == 1:
            self._control_robotiq_joints(0.0)
            goal_pos = np.array([cube_current_pos[0], cube_current_pos[1], hover_z])
            self.robot.set_end_effector_pose(position=goal_pos, orientation=self.goal_orientation, ik_method=ik_method)
            
            self.step_count += 1
            if self.step_count >= (self.events_dt[0] + self.wait_steps):
                self.current_phase = 2
                self.step_count = 0

        # Phase 2: 정밀 하강
        elif self.current_phase == 2:
            goal_pos = np.array([cube_current_pos[0], cube_current_pos[1], grasp_z])
            self.robot.set_end_effector_pose(position=goal_pos, orientation=self.goal_orientation, ik_method=ik_method)
            
            self.step_count += 1
            if self.step_count >= (self.events_dt[1] + self.wait_steps):
                self.current_phase = 3
                self.step_count = 0

        # Phase 3: 파지 (Grasp) - 큐브 크기에 맞춰 0.4 각도로 단단하게 파지
        elif self.current_phase == 3:
            self._control_robotiq_joints(0.4)
            
            self.step_count += 1
            if self.step_count >= (self.events_dt[2] + self.wait_steps):
                self.current_phase = 4
                self.step_count = 0

        # Phase 4: 홈 포지션 리턴
        elif self.current_phase == 4:
            self._control_robotiq_joints(0.4)
            
            # 🔥 [수정된 핵심 코드] 0.0 배열로 덮어씌우지 말고, 정확히 로봇 팔(0번~5번) 관절 6개만 콕 집어서 홈 포지션 명령을 내립니다.
            self.robot.set_dof_position_targets(self.safe_working_pose, dof_indices=np.arange(6))
            
            self.step_count += 1
            if self.step_count >= (self.events_dt[3] + self.wait_steps):
                self.current_phase = 5
                self.step_count = 0

        # Phase 5: 원위치 복귀 및 해제
        elif self.current_phase == 5:
            goal_pos = np.array([self.cube_pos[0], self.cube_pos[1], grasp_z])
            self.robot.set_end_effector_pose(position=goal_pos, orientation=self.goal_orientation, ik_method=ik_method)
            
            if self.step_count > self.events_dt[4]:
                self._control_robotiq_joints(0.0)
            else:
                self._control_robotiq_joints(0.4)
            
            self.step_count += 1
            if self.step_count >= (self.events_dt[4] + self.wait_steps):
                self.current_phase = 6
                self.step_count = 0

        # Phase 6: 종료
        elif self.current_phase == 6:
            self._control_robotiq_joints(0.0)
            goal_pos = np.array([self.cube_pos[0], self.cube_pos[1], hover_z])
            self.robot.set_end_effector_pose(position=goal_pos, orientation=self.goal_orientation, ik_method=ik_method)
            
            self.step_count += 1
            if self.step_count >= (self.events_dt[5] + self.wait_steps):
                print("[Complete] 픽앤플레이스 및 제자리 원복 완료!")
                if self._physics_callback_id is not None:
                    SimulationManager.deregister_callback(self._physics_callback_id)
                    self._physics_callback_id = None

    async def start_async(self):
        app_utils.play()
        await app_utils.update_app_async()
        
        self.robot.set_world_poses(positions=[0.0, 0.0, 0.0], orientations=[1.0, 0.0, 0.0, 0.0])
        
        full_initial_pose = np.zeros(self.robot.num_dofs)
        full_initial_pose[:6] = self.safe_working_pose
        self.robot.set_dof_positions(full_initial_pose)
        self.robot.set_dof_position_targets(full_initial_pose)
        print("[System] 지정 홈 관절 값 초기화 완료. 5초간 대기합니다...")

        for _ in range(300):
            await app_utils.update_app_async()
            
        print("[System] 대기 종료. 풀 시퀀스 구동을 시작합니다.")
        self.current_phase = 1

        self._physics_callback_id = SimulationManager.register_callback(
            self._physics_callback, event=SimulationEvent.PHYSICS_POST_STEP
        )

if __name__ == "__main__":
    usd_target = r"C:\isaacsim\process_test\assets\ur10_camera_gripper.usd"
    executor = FullPickAndPlaceVerification(usd_path=usd_target)
    executor.setup_scene()
    
    asyncio.ensure_future(executor.start_async())
    
    while simulation_app.is_running():
        simulation_app.update()