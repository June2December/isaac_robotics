import zmq
import numpy as np
import time

def run_dummy_test():
    server_ip = "192.168.0.250" # 서버 IP (필요시 수정)
    
    print(f"1. 서버({server_ip})에 연결을 시도합니다...")
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{server_ip}:5555")
    
    # 2. 가짜 이미지(224x224 RGB) 및 프롬프트 생성
    print("2. 더미 데이터(가짜 이미지)를 생성합니다...")
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    prompt = "pick up the cube"
    
    data = {"image": dummy_image, "prompt": prompt}
    
    # 3. 데이터 전송 및 수신
    print(f"3. 서버로 데이터를 전송합니다. 프롬프트: '{prompt}'")
    start_time = time.time()
    socket.send_pyobj(data)
    
    print("4. 서버의 추론 결과를 기다리는 중...")
    action = socket.recv_pyobj()
    end_time = time.time()
    
    # 4. 결과 출력
    print("\n=== 테스트 결과 ===")
    print(f"수신된 Action 값: {action}")
    print(f"Action 배열 형태(Shape): {action.shape}")
    print(f"소요 시간: {end_time - start_time:.2f}초")

if __name__ == "__main__":
    run_dummy_test()