# 1. Isaac Sim 가동
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

# 2. 익스텐션 활성화 (Deprecation 경고 해결: omni -> isaacsim 변경)
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.examples.interactive")
# 컨베이어 모듈 활성화
enable_extension("isaacsim.asset.gen.conveyor")
enable_extension("omni.isaac.conveyor") # 모터 제어용 API 익스텐션 추가 강제 활성화

import omni 
import numpy as np
import random
from isaacsim.cortex.framework.cortex_world import CortexWorld
from isaacsim.examples.interactive.ur10_palletizing.ur10_palletizing import Ur10Assets, BinStackingTask
from isaacsim.cortex.framework.cortex_rigid_prim import CortexRigidPrim
from isaacsim.cortex.behaviors.ur10 import bin_stacking_behavior as behavior
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.prims import create_prim
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.cortex.framework.robot import CortexUr10

# 방어막 모듈 임포트
from isaacsim.core.api.objects.capsule import VisualCapsule
from isaacsim.core.api.objects.sphere import VisualSphere
import isaacsim.cortex.framework.math_util as math_util

# 가이드 생성 위한 큐브 생성
from isaacsim.core.api.objects.cuboid import FixedCuboid

# --------------------------------------------------------------------------------
# [커스텀 로직] 박스 스폰 위치를 새 컨베이어 위치에 맞춰 조절
# --------------------------------------------------------------------------------
class CustomConveyorTask(BinStackingTask):
    def _spawn_bin(self, rigid_bin):
        spawn_x = random.uniform(-0.15, 0.15)
        spawn_y = 2.9  # 직접 맞추신 시작 Y 좌표
        spawn_z = 1.0  # 컨베이어 위쪽에서 투하 (직접 맞추신 Z 좌표)
        
        z = random.random() * 0.02 - 0.01
        w = random.random() * 0.02 - 0.01
        norm = np.sqrt(z**2 + w**2)
        quat = math_util.Quaternion([w / norm, 0, 0, z / norm])
        if random.random() > 0.5:
            quat = quat * math_util.Quaternion([0, 0, 1, 0])

        rigid_bin.set_world_pose(position=[spawn_x, spawn_y, spawn_z], orientation=quat.vals)
        # rigid_bin.set_linear_velocity(np.array([0, -0.30, 0]))
        rigid_bin.set_linear_velocity(np.array([0., 0., 0.])) 
        rigid_bin.set_visibility(True)

    def pre_step(self, time_step_index, simulation_time) -> None:
        spawn_new = False
        if self.on_conveyor is None:
            spawn_new = True
        else:
            (x, y, z), _ = self.on_conveyor.get_world_pose()
            # 로봇(X=0, Y=0) 쪽으로 굴러오는 궤적 감시
            is_on_conveyor = (y > 0.0) and (-0.4 < x < 0.4)
            if not is_on_conveyor:
                spawn_new = True

        if spawn_new:
            name = "bin_{}".format(len(self.bins))
            prim_path = self.env_path + "/bins/{}".format(name)
            add_reference_to_stage(usd_path=self.assets.small_klt_usd, prim_path=prim_path)
            self.on_conveyor = self.scene.add(CortexRigidPrim(name=name, prim_path=prim_path))

            self._spawn_bin(self.on_conveyor)
            self.bins.append(self.on_conveyor)


def main():
    print("-> 뷰포트 창 생성 및 스테이지 초기화 중...")
    omni.usd.get_context().new_stage()
    
    simulation_app.update()

    # 프레임 동기화
    world = CortexWorld(physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
    
    env_path = "/World/Ur10Table"
    ur10_assets = Ur10Assets()
    
    # 빛과 무대 기본 세팅
    create_prim("/World/defaultLight", "DistantLight")
    add_reference_to_stage(usd_path=ur10_assets.ur10_table_usd, prim_path=env_path)
    add_reference_to_stage(usd_path=ur10_assets.background_usd, prim_path="/World/Background")
    
    SingleXFormPrim(
        "/World/Background", position=[10.00, 2.00, -1.18180], orientation=[0.7071, 0, 0, 0.7071]
    )

    # 로봇 소환
    robot_obj = world.add_robot(CortexUr10(name="robot", prim_path="{}/ur10".format(env_path)))

    # --------------------------------------------------------------------------------
    # 오리지널 방어막 4개 복구
    # --------------------------------------------------------------------------------
    obs1 = world.scene.add(VisualSphere("/World/Ur10Table/Obstacles/FlipStationSphere",
                                        name="flip_station_sphere", position=np.array([0.73, 0.76, -0.13]),
                                        radius=0.2, visible=False))
    robot_obj.register_obstacle(obs1)
    
    obs2 = world.scene.add(VisualSphere("/World/Ur10Table/Obstacles/NavigationDome",
                                        name="navigation_dome_obs", position=[-0.031, -0.018, -1.086],
                                        radius=1.1, visible=False))
    robot_obj.register_obstacle(obs2)
    
    az = np.array([1.0, 0.0, -0.3])
    ax = np.array([0.0, 1.0, 0.0])
    ay = np.cross(az, ax)
    R = math_util.pack_R(ax, ay, az)
    quat = math_util.matrix_to_quat(R)
    obs3 = world.scene.add(VisualCapsule("/World/Ur10Table/Obstacles/NavigationBarrier",
                                         name="navigation_barrier_obs", position=[0.471, 0.276, -0.463 - 0.1],
                                         orientation=quat, radius=0.5, height=0.9, visible=False))
    robot_obj.register_obstacle(obs3)
    
    obs4 = world.scene.add(VisualCapsule("/World/Ur10Table/Obstacles/NavigationFlipStation",
                                         name="navigation_flip_station_obs", position=np.array([0.766, 0.755, -0.5]),
                                         radius=0.5, height=0.5, visible=False))
    robot_obj.register_obstacle(obs4)

    # ==================================================================
    # [씬 구성] 기존 컨베이어 지우고 내가 고른 에셋으로 배치하기
    # ==================================================================
    stage = omni.usd.get_context().get_stage()

    # 1. 구형 컨베이어 비활성화
    old_conveyor = stage.GetPrimAtPath(env_path + "/conveyor")
    if old_conveyor.IsValid():
        old_conveyor.SetActive(False)
        print("-> [성공] 기존 짧은 컨베이어 비활성화 완료")

    # 2. 새로운 에셋 로드 (A08로 변경 완료)
    nvidia_conveyor_usd = ur10_assets.assets_root_path + "/Isaac/Props/Conveyors/ConveyorBelt_A08.usd"
    new_conveyor_path = "/World/StraightConveyor"
    
    add_reference_to_stage(usd_path=nvidia_conveyor_usd, prim_path=new_conveyor_path)

    # 3. 위치 배치 및 Z축 회전
    world.scene.add(
        SingleXFormPrim(
            prim_path=new_conveyor_path,
            name="straight_conveyor",
            position=np.array([0.0, 0.4, -1.18]), # 지정하신 새 좌표
            # w, x, y, z 순서. 90도에서 180도 더 회전된 상태 (-90도)
            orientation=np.array([0.7071, 0.0, 0.0, -0.7071]) 
        )
    )
    print("-> [성공] 새 컨베이어 회전(-90도) 및 배치 완료")

    # 4. 컨베이어 모터(표면 이동 속도) 작동! (옴니그래프 직접 제어 방식)
    try:
        import omni.graph.core as og
        import pxr.Sdf as Sdf
        
        # 올려주신 사진 트리 구조에 있던 Rollers Xform을 타겟으로 지정합니다.
        target_path = new_conveyor_path + "/Rollers"
        
        graph_path = "/World/ConveyorGraph"
        keys = og.Controller.Keys
        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("Tick", "omni.graph.action.OnPlaybackTick"),
                    ("Conveyor", "isaacsim.asset.gen.conveyor.IsaacConveyor"),
                ],
                keys.CONNECT: [
                    ("Tick.outputs:tick", "Conveyor.inputs:onStep"),
                ],
                keys.SET_VALUES: [
                    ("Conveyor.inputs:velocity", 0.30),
                    # 방향을 유지합니다.
                    ("Conveyor.inputs:direction", [1.0, 0.0, 0.0]), 
                ]
            }
        )
        # 모터 노드에 Rollers 파츠 연결
        attr = og.Controller.attribute(graph_path + "/Conveyor.inputs:conveyorPrim")
        attr.set([Sdf.Path(target_path)])
        print(f"-> [성공] 옴니그래프(OmniGraph) 컨베이어 모터 구동 완료! (타겟: {target_path})")
    except Exception as e:
        print(f"-> [에러] 옴니그래프 컨베이어 생성 실패: {e}")
    # ==================================================================

    # --------------------------------------------------------------------------------
    # [Task 복구] 박스 생성 및 로봇 팔레타이징 동작 실행
    # --------------------------------------------------------------------------------
    task = CustomConveyorTask(env_path, ur10_assets)
    task.set_up_scene(world.scene)
    world.add_task(task)
    
    monitor_fn = getattr(task, "monitor", None)
    if monitor_fn is None:
        monitor_fn = getattr(task, "monitor_fn", None)
    if monitor_fn is None:
        def dummy_monitor(*args, **kwargs): pass
        monitor_fn = dummy_monitor
        
    decider_network = behavior.make_decider_network(robot_obj, monitor_fn)
    world.add_decider_network(decider_network)

    # ==================================================================
    # [물리 Stopper] 컨베이어 끝에서 박스가 보고대 쪽으로 더 밀리지 않게 막기
    # ==================================================================

    STOPPER_Y = 0.38       # 박스가 멈출 Y 위치. 너무 앞이면 키우고, 너무 늦으면 줄이세요.
    STOPPER_Z = -0.48      # 컨베이어 롤러보다 약간 위
    STOPPER_WIDTH_X = 0.035 # 컨베이어 폭 방향
    STOPPER_THICK_Y = 0.6
    STOPPER_HEIGHT_Z = 0.18

    end_stopper = world.scene.add(
        FixedCuboid(
            prim_path="/World/StraightConveyor/EndStopper",
            name="conveyor_end_stopper",
            position=np.array([0.0, STOPPER_Y, STOPPER_Z]),
            scale=np.array([STOPPER_WIDTH_X, STOPPER_THICK_Y, STOPPER_HEIGHT_Z]),
        )
    )

    print("-> [성공] 컨베이어 끝단 EndStopper 추가 완료")

    world.reset()
    world.reset_cortex()

    # 카메라가 로봇과 컨베이어를 잘 비추도록 설정 (Deprecation 경고 해결)
    try:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=np.array([3.0, 3.0, 2.0]), target=np.array([0.0, 0.0, 0.0]))
    except Exception:
        pass
    
    print("\n" + "="*50)
    print("=== [씬 세팅 모드] 로봇과 새 컨베이어 위치만 확인하세요 ===")
    print("="*50 + "\n")
    
    while simulation_app.is_running():
        world.step(render=True)

    simulation_app.close()

if __name__ == "__main__":
    main()
