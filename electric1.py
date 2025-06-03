#헬륨-4 원자 모델 시뮬레이션
# 2-전자 원자 모델 시뮬레이션 (고전 역학 기반)

import pygame
import math
import numpy as np
from pygame.locals import K_q
import os
from datetime import datetime
import matplotlib.pyplot as plt

# 현재 날짜와 시간 기반 이름 생성
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_folder = os.path.join(os.path.expanduser("~/Desktop"), "He_atom")

# 폴더가 없으면 생성
os.makedirs(base_folder, exist_ok=True)

# 수소와 헬륨용 실험 폴더 생성

helium_folder = os.path.join(base_folder, f"try2_He_{timestamp}")
os.makedirs(helium_folder, exist_ok=True)

flatchart  = []


# 초기화
pygame.init()
width, height = 600, 600
# pygame.FULLSCREEN 오타 수정 확인
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
pygame.display.set_caption("2-전자 원자 모델 시뮬레이션 (고전 역학 기반)")
clock = pygame.time.Clock()

# 폰트 설정 (한글 폰트 지원)

font = pygame.font.SysFont(None, 18) # 기본 폰트
# print(f"DEBUG: 사용 중인 폰트: {font.get_name()}") # 어떤 폰트가 사용되는지 확인하고 싶을 때 주석 해제

# 색상
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (50, 50, 250)
RED  = (230, 30, 30)
GRAY = (150, 150, 150) # 전자 껍질 시각화용 색상
LIGHT_BLUE = (173, 216, 230) # 헬륨 크기 시각화용 색상 (연한 파랑)

# 시뮬레이션 단위계 설정 (자연 단위계 기반)
# 참고: 이 시뮬레이션은 실제 양자 역학적 원자 모델이 아닌,
#       고전 역학적 힘을 바탕으로 전자의 움직임을 '개념적으로' 시각화합니다.
COULOMB_CONSTANT = 1  # 쿨롱 상수 (k_e)
ELECTRON_CHARGE = -1  # 전자 전하 (q)
NUCLEUS_CHARGE_Z = 2  # 핵의 전하 (Z=2는 헬륨 원자에 해당)
ELECTRON_MASS  = 1  # 전자 질량
TIME_STEP    = 0.01 # 시간 간격 (dt)
VISUAL_SCALE   = 200 # 시뮬레이션 단위를 픽셀로 변환하는 배율

# 전자의 '껍질' 구속력 설정 (시뮬레이션 안정화 및 시각화 목적)
# 이 힘은 실제 원자의 양자 역학적 '전자 껍질'을 고전 역학적으로 근사한 것으로,
# 전자가 핵에서 너무 멀리 벗어나거나 너무 가까이 붙는 것을 방지합니다.
SHELL_TARGET_RADIUS = 1.0 # 전자가 머무르길 원하는 가상의 '껍질' 반지름
SHELL_SPRING_CONSTANT = 17.0 # 껍질 복원력 계수 (값이 클수록 구속 강함)

# 헬륨 원자의 대략적인 크기를 나타내는 반지름
# 이 값은 전자의 초기 위치와 유사하게 설정하여, 헬륨 원자의 '영역'을 시각화합니다.
HELIUM_SIZE_RADIUS = 1.2 # SHELL_TARGET_RADIUS보다 약간 크게 설정하여 구분

def calculate_shell_force(electron_pos, target_radius=SHELL_TARGET_RADIUS, spring_k=SHELL_SPRING_CONSTANT):
  """
  전자가 가상의 '껍질' 반지름으로 돌아오도록 하는 복원력을 계산합니다.
  """
  rx, ry = electron_pos[0], electron_pos[1]
  current_r = math.sqrt(rx**2 + ry**2) + 1e-10 # 0 나누기 방지
  delta_r = current_r - target_radius
  
  # 힘의 방향은 핵을 기준으로 전자가 있는 방향의 반대 또는 같은 방향
  fx = -spring_k * delta_r * (rx / current_r)
  fy = -spring_k * delta_r * (ry / current_r)
  return [fx, fy]

# 좌표 변환 함수
def convert_to_screen_pos(x, y):
  """ 시뮬레이션 좌표를 Pygame 화면 픽셀 좌표로 변환합니다. """
  return (int(x * VISUAL_SCALE + width / 2), int(height / 2 - y * VISUAL_SCALE))

# 텍스트 출력 함수
def draw_text(text, x, y, color=BLACK):
  """ 화면에 텍스트를 그립니다. """
  label = font.render(text, True, color)
  screen.blit(label, (x, y))

# 쿨롱 힘 함수 (발산 방지용 거리 보정 포함)
def calculate_softened_coulomb_force(charge1, charge2, pos1, pos2, softening_epsilon=0.05):
  """
  두 전하 사이의 쿨롱 힘을 계산합니다.
  softening_epsilon은 두 입자가 너무 가까워질 때 힘이 무한대로 발산하는 것을 방지합니다.
  """
  rx = pos1[0] - pos2[0]
  ry = pos1[1] - pos2[1]
  
  # 거리가 0에 가까워질 때 힘이 무한대로 발산하는 것을 방지하기 위해 epsilon 추가
  r_squared = rx**2 + ry**2 + softening_epsilon**2
  r = math.sqrt(r_squared)
  
  # 쿨롱 법칙: F = k * q1 * q2 / r^2 (방향 벡터 포함 시 r^3)
  fx = COULOMB_CONSTANT * charge1 * charge2 * rx / r**3
  fy = COULOMB_CONSTANT * charge1 * charge2 * ry / r**3
  return [fx, fy]

# 초기 조건 (비대칭 설정)
electron1_pos = [1.0, 0.0]
electron1_vel = [0.0, 0.6]

electron2_pos = [-0.9, 0.0]
electron2_vel = [0.0, -0.55]

nucleus_pos = [0.0, 0.0] # 핵은 고정

# 궤적 저장을 위한 리스트와 최대 점 개수 설정
electron1_trajectory = []
electron2_trajectory = []
MAX_TRAJECTORY_POINTS = 300 # 궤적에 표시할 최대 점의 개수

# 메인 시뮬레이션 루프
running = True
while running:
  # 이벤트 처리
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False

  keys = pygame.key.get_pressed()
  if keys[K_q]: # 'q' 키를 누르면 종료
    running = False

  # 화면 지우기
  screen.fill(WHITE)

  # --- 전자 1에 작용하는 힘 계산 ---
  F1_nucleus = calculate_softened_coulomb_force(ELECTRON_CHARGE, -NUCLEUS_CHARGE_Z * ELECTRON_CHARGE, electron1_pos, nucleus_pos)
  F1_electron2 = calculate_softened_coulomb_force(ELECTRON_CHARGE, ELECTRON_CHARGE, electron1_pos, electron2_pos)
  F1_shell = calculate_shell_force(electron1_pos)
  
  # 모든 힘을 합산
  total_F1 = [F1_nucleus[0] + F1_electron2[0] + F1_shell[0], 
        F1_nucleus[1] + F1_electron2[1] + F1_shell[1]]

  # --- 전자 2에 작용하는 힘 계산 ---
  F2_nucleus = calculate_softened_coulomb_force(ELECTRON_CHARGE, -NUCLEUS_CHARGE_Z * ELECTRON_CHARGE, electron2_pos, nucleus_pos)
  F2_electron1 = calculate_softened_coulomb_force(ELECTRON_CHARGE, ELECTRON_CHARGE, electron2_pos, electron1_pos)
  F2_shell = calculate_shell_force(electron2_pos)
  
  # 모든 힘을 합산
  total_F2 = [F2_nucleus[0] + F2_electron1[0] + F2_shell[0],
        F2_nucleus[1] + F2_electron1[1] + F2_shell[1]]

  # --- 가속도 계산 (F = ma => a = F/m) ---
  acc1 = [total_F1[0] / ELECTRON_MASS, total_F1[1] / ELECTRON_MASS]
  acc2 = [total_F2[0] / ELECTRON_MASS, total_F2[1] / ELECTRON_MASS]

  # --- 속도 업데이트 (v = v0 + at) ---
  electron1_vel[0] += acc1[0] * TIME_STEP
  electron1_vel[1] += acc1[1] * TIME_STEP
  electron2_vel[0] += acc2[0] * TIME_STEP
  electron2_vel[1] += acc2[1] * TIME_STEP

  # --- 위치 업데이트 (x = x0 + vt) ---
  electron1_pos[0] += electron1_vel[0] * TIME_STEP
  electron1_pos[1] += electron1_vel[1] * TIME_STEP
  electron2_pos[0] += electron2_vel[0] * TIME_STEP
  electron2_pos[1] += electron2_vel[1] * TIME_STEP

      # --- 전자 위치 업데이트 이후 거리 계산 ---
  distance = np.linalg.norm(np.array(electron1_pos) - np.array(electron2_pos))  # 두 전자의 거리
  log_distance = np.log(distance + 1e-10)  # 로그 계산 (0 방지용 epsilon)

  flatchart.append(log_distance)  # 리아프노프 지수 추정용 로그 거리 저장


  # --- 궤적 데이터 업데이트 ---
  electron1_trajectory.append(list(electron1_pos)) # 현재 위치를 복사하여 추가
  electron2_trajectory.append(list(electron2_pos))

  # 궤적 점 개수 제한
  if len(electron1_trajectory) > MAX_TRAJECTORY_POINTS:
    electron1_trajectory.pop(0) # 가장 오래된 점 제거
  if len(electron2_trajectory) > MAX_TRAJECTORY_POINTS:
    electron2_trajectory.pop(0)







  # --- 시각화 ---
  # 1. 헬륨 원자 '크기' 시각화: (가장 바깥에 그려짐)
  # 헬륨 원자의 대략적인 크기 또는 전자의 초기 위치를 나타내는 원
  pixel_helium_size_radius = int(HELIUM_SIZE_RADIUS * VISUAL_SCALE)
  pygame.draw.circle(
    screen, 
    LIGHT_BLUE, # 연한 파란색으로 표시
    convert_to_screen_pos(nucleus_pos[0], nucleus_pos[1]), # 핵을 중심으로
    pixel_helium_size_radius, 
    1 # 테두리 선만 (선 굵기 1)
  )

  # 2. '전자 껍질' 시각화: (헬륨 크기 선 안쪽에 그려짐)
  # 전자가 구속되는 가상의 '껍질' (SHELL_TARGET_RADIUS)
  pixel_shell_radius = int(SHELL_TARGET_RADIUS * VISUAL_SCALE)
  pygame.draw.circle(
    screen, 
    GRAY, # 연한 회색으로 표시
    convert_to_screen_pos(nucleus_pos[0], nucleus_pos[1]), # 핵을 중심으로
    pixel_shell_radius, 
    1 # 테두리 선만 (선 굵기 1)
  )

  # 3. 전자의 궤적 시각화: (가장 위에 그려져 전자가 움직이는 경로를 보여줌)
  if len(electron1_trajectory) > 1: 
    points1 = [convert_to_screen_pos(p[0], p[1]) for p in electron1_trajectory]
    pygame.draw.lines(screen, RED, False, points1, 1) # 전자1 궤적 (전자1과 같은 색)
  if len(electron2_trajectory) > 1:
    points2 = [convert_to_screen_pos(p[0], p[1]) for p in electron2_trajectory]
    pygame.draw.lines(screen, BLUE, False, points2, 1) # 전자2 궤적 (전자2와 같은 색)
  
  # 전자 및 핵 그리기 (궤적 위에 그려져서 가장 잘 보이도록)
  pygame.draw.circle(screen, RED, convert_to_screen_pos(electron1_pos[0], electron1_pos[1]), 6) # 전자 1
  pygame.draw.circle(screen, BLUE, convert_to_screen_pos(electron2_pos[0], electron2_pos[1]), 6) # 전자 2
  pygame.draw.circle(screen, BLACK, convert_to_screen_pos(nucleus_pos[0], nucleus_pos[1]), 10) # 핵


  # 정보 출력
  draw_text(f"elec 1 pos: ({electron1_pos[0]:.2f}, {electron1_pos[1]:.2f})", 10, 10)
  draw_text(f"elec 1 vel: ({electron1_vel[0]:.2f}, {electron1_vel[1]:.2f})", 10, 30)
  draw_text(f"elec 1 force:  ({total_F1[0]:.2e}, {total_F1[1]:.2e})", 10, 50)

  draw_text(f"elec 2 pos: ({electron2_pos[0]:.2f}, {electron2_pos[1]:.2f})", 10, 80)
  draw_text(f"elec 2 vel: ({electron2_vel[0]:.2f}, {electron2_vel[1]:.2f})", 10, 100)
  draw_text(f"elec 2 force:  ({total_F2[0]:.2e}, {total_F2[1]:.2e})", 10, 120)

  
  # 화면 업데이트
  pygame.display.flip()
  
  # 프레임 속도 제어
  clock.tick(60)

pygame.quit()

# 시뮬레이션 종료 후 궤적 데이터 저장 및 그래프 생성
plt.figure(figsize=(10, 4))
plt.plot(flatchart, color='purple')
plt.title("Log Distance between Two Electrons Over Time")
plt.xlabel("Frame")
plt.ylabel("log(distance)")
plt.grid(True)

# 그래프 저장 (헬륨 실험 기준)
graph_path = os.path.join(helium_folder, "log_distance_graph.png")
plt.savefig(graph_path)
plt.close()

