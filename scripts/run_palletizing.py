from omni.isaac.kit import SimulationApp

# 1. Isaac Sim 가동
simulation_app = SimulationApp({"headless": False})

# 2. 익스텐션 활성화
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("isaacsim.examples.interactive")

import omni
import numpy as np
from isaacsim.cortex.framework.cortex_world import CortexWorld
from isaacsim.cortex.behaviors.ur10 import bin_stacking_behavior as behavior
from isaacsim.examples.interactive.ur10_palletizing.ur10_palletizing import BinStackingTask, Ur10Assets
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.prims import create_prim

# [추가] 방어막 생성을 위한 코어 모듈 임포트
from isaacsim.core.api.objects.capsule import VisualCapsule
from isaacsim.core.api.objects.sphere import VisualSphere
import isaacsim.cortex.framework.math_util as math_util

def main():
    print("-> 뷰포트 창 생성 및 스테이지 초기화 중...")
    omni.usd.get_context().new_stage()
    
    simulation_app.update()

    # 프레임 동기화
    world = CortexWorld(physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
    
    env_path = "/World/Ur10Table"
    ur10_assets = Ur10Assets()
    
    create_prim("/World/defaultLight", "DistantLight")
    add_reference_to_stage(usd_path=ur10_assets.ur10_table_usd, prim_path=env_path)
    add_reference_to_stage(usd_path=ur10_assets.background_usd, prim_path="/World/Background")
    
    from isaacsim.core.prims import SingleXFormPrim
    background_prim = SingleXFormPrim(
        "/World/Background", position=[10.00, 2.00, -1.18180], orientation=[0.7071, 0, 0, 0.7071]
    )
    
    from isaacsim.cortex.framework.robot import CortexUr10
    robot_obj = world.add_robot(CortexUr10(name="robot", prim_path="{}/ur10".format(env_path)))
    
    # --------------------------------------------------------------------------------
    # [핵심 수정] 원본 GUI와 동일한 궤적을 만들기 위해 4개의 투명 방어막(Obstacle)을 설치합니다.
    # --------------------------------------------------------------------------------
    # 1. 플립 스테이션 구체 방어막
    obs1 = world.scene.add(VisualSphere("/World/Ur10Table/Obstacles/FlipStationSphere",
                                        name="flip_station_sphere", position=np.array([0.73, 0.76, -0.13]),
                                        radius=0.2, visible=False))
    robot_obj.register_obstacle(obs1)
    
    # 2. 내비게이션 돔 방어막
    obs2 = world.scene.add(VisualSphere("/World/Ur10Table/Obstacles/NavigationDome",
                                        name="navigation_dome_obs", position=[-0.031, -0.018, -1.086],
                                        radius=1.1, visible=False))
    robot_obj.register_obstacle(obs2)
    
    # 3. 내비게이션 배리어 (회전값 적용)
    az = np.array([1.0, 0.0, -0.3])
    ax = np.array([0.0, 1.0, 0.0])
    ay = np.cross(az, ax)
    R = math_util.pack_R(ax, ay, az)
    quat = math_util.matrix_to_quat(R)
    obs3 = world.scene.add(VisualCapsule("/World/Ur10Table/Obstacles/NavigationBarrier",
                                         name="navigation_barrier_obs", position=[0.471, 0.276, -0.463 - 0.1],
                                         orientation=quat, radius=0.5, height=0.9, visible=False))
    robot_obj.register_obstacle(obs3)
    
    # 4. 플립 스테이션 캡슐 방어막
    obs4 = world.scene.add(VisualCapsule("/World/Ur10Table/Obstacles/NavigationFlipStation",
                                         name="navigation_flip_station_obs", position=np.array([0.766, 0.755, -0.5]),
                                         radius=0.5, height=0.5, visible=False))
    robot_obj.register_obstacle(obs4)
    # --------------------------------------------------------------------------------

    task = BinStackingTask(env_path, ur10_assets)
    task.set_up_scene(world.scene)
    world.add_task(task)
    
    # 오리지널 모니터 함수 바인딩
    monitor_fn = getattr(task, "monitor", None)
    if monitor_fn is None:
        monitor_fn = getattr(task, "monitor_fn", None)
    if monitor_fn is None:
        def dummy_monitor(*args, **kwargs): pass
        monitor_fn = dummy_monitor
        
    decider_network = behavior.make_decider_network(robot_obj, monitor_fn)
    world.add_decider_network(decider_network)
    
    world.reset()
    world.reset_cortex()
    
    print("\n" + "="*50)
    print("=== LPK 로보틱스 팔레타이징 독립형 스크립트 완벽 동기화 구동 ===")
    print("="*50 + "\n")
    
    while simulation_app.is_running():
        world.step(render=True)

    simulation_app.close()

if __name__ == "__main__":
    main()