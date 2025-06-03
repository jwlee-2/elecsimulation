import pygame, math, os
from pygame.locals import K_q
import matplotlib.pyplot as plt
from datetime import datetime

# --- 바탕화면 경로 설정 ---------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_folder = os.path.join(os.path.expanduser("~/Desktop"), "H_atom")

os.makedirs(base_folder, exist_ok=True)

hydrogen_folder = os.path.join(base_folder, f"try2_H_{timestamp}")
os.makedirs(hydrogen_folder, exist_ok=True)
# --- 초기화 --------------------------------------------------------------
pygame.init()
width, height = 600, 600
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
pygame.display.set_caption("수소 원자 모델 (비혼돈 시각화)")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 18)

# --- 상수 ----------------------------------------------------------------
WHITE, BLACK = (255,255,255), (0,0,0)
BLUE  = (50,  50,250)
GRAY  = (150,150,150)

COULOMB_CONSTANT = 1
ELECTRON_CHARGE  = -1
NUCLEUS_CHARGE_Z = 1
ELECTRON_MASS    = 1
TIME_STEP        = 0.01
VISUAL_SCALE     = 200

SHELL_TARGET_RADIUS   = 1.0
SHELL_SPRING_CONSTANT = 10.0
ATOM_SIZE_RADIUS      = 1.2

# --- 함수 ----------------------------------------------------------------
def convert(x, y):
    return (int(x*VISUAL_SCALE + width/2), int(height/2 - y*VISUAL_SCALE))

def draw_text(text, x, y, c=BLACK):
    screen.blit(font.render(text, True, c), (x, y))

def coulomb(q1, q2, p1, p2, eps=0.05):
    rx, ry = p1[0]-p2[0], p1[1]-p2[1]
    r2 = rx*rx + ry*ry + eps*eps
    r  = math.sqrt(r2)
    f  = COULOMB_CONSTANT*q1*q2/r**3
    return [f*rx, f*ry]

def shell_force(pos):
    rx, ry = pos
    r  = math.hypot(rx, ry) + 1e-10
    dr = r - SHELL_TARGET_RADIUS
    k  = -SHELL_SPRING_CONSTANT*dr/r
    return [k*rx, k*ry]

# --- 초기 조건 ------------------------------------------------------------
electron_pos = [1.0, 0.0]
electron_vel = [0.0, 0.6]
electron_pos2 = [electron_pos[0] + 1e-5, electron_pos[1]]
electron_vel2 = electron_vel[:]
nucleus_pos  = [0.0, 0.0]

initial_distance = math.hypot(
    electron_pos2[0] - electron_pos[0],
    electron_pos2[1] - electron_pos[1]
)

traj, MAX_T = [], 300
lyapunov_vals = []
time_vals = []

# --- 메인 루프 ------------------------------------------------------------
running = True
frame = 0
MAX_FRAME = 3000

while running and frame < MAX_FRAME:
    for e in pygame.event.get():
        if e.type == pygame.QUIT: running = False
    if pygame.key.get_pressed()[K_q]: running = False

    # --- 힘 계산 (전자 1)
    F_nuc   = coulomb(ELECTRON_CHARGE, -NUCLEUS_CHARGE_Z*ELECTRON_CHARGE,
                      electron_pos, nucleus_pos)
    F_shell = shell_force(electron_pos)
    Fx, Fy  = F_nuc[0]+F_shell[0], F_nuc[1]+F_shell[1]
    acc = [Fx/ELECTRON_MASS, Fy/ELECTRON_MASS]
    electron_vel[0] += acc[0]*TIME_STEP
    electron_vel[1] += acc[1]*TIME_STEP
    electron_pos[0] += electron_vel[0]*TIME_STEP
    electron_pos[1] += electron_vel[1]*TIME_STEP

    # --- 힘 계산 (전자 2)
    F_nuc2 = coulomb(ELECTRON_CHARGE, -NUCLEUS_CHARGE_Z*ELECTRON_CHARGE,
                     electron_pos2, nucleus_pos)
    F_shell2 = shell_force(electron_pos2)
    Fx2, Fy2 = F_nuc2[0]+F_shell2[0], F_nuc2[1]+F_shell2[1]
    acc2 = [Fx2/ELECTRON_MASS, Fy2/ELECTRON_MASS]
    electron_vel2[0] += acc2[0]*TIME_STEP
    electron_vel2[1] += acc2[1]*TIME_STEP
    electron_pos2[0] += electron_vel2[0]*TIME_STEP
    electron_pos2[1] += electron_vel2[1]*TIME_STEP

    # --- 랴푸노프 지수 계산
    dx = electron_pos2[0] - electron_pos[0]
    dy = electron_pos2[1] - electron_pos[1]
    distance = math.hypot(dx, dy)
    if distance > 1e-10:
        lyapunov_vals.append(math.log(distance / initial_distance))
        time_vals.append(frame * TIME_STEP)

    # --- 궤적 저장
    traj.append(list(electron_pos))
    if len(traj) > MAX_T: traj.pop(0)

    # --- 화면 그리기
    screen.fill(WHITE)
    pygame.draw.circle(screen, (173,216,230), convert(*nucleus_pos),
                       int(ATOM_SIZE_RADIUS*VISUAL_SCALE), 1)
    pygame.draw.circle(screen, GRAY, convert(*nucleus_pos),
                       int(SHELL_TARGET_RADIUS*VISUAL_SCALE), 1)
    if len(traj) > 1:
        pygame.draw.lines(screen, BLUE, False,
                          [convert(*p) for p in traj], 1)
    pygame.draw.circle(screen, BLUE,  convert(*electron_pos), 6)
    pygame.draw.circle(screen, BLACK, convert(*nucleus_pos), 10)
    draw_text(f"pos: ({electron_pos[0]:.2f},{electron_pos[1]:.2f})", 10, 10)
    draw_text(f"vel: ({electron_vel[0]:.2f},{electron_vel[1]:.2f})", 10, 30)
    draw_text(f"F  : ({Fx:.2e},{Fy:.2e})",            10, 50)
    if lyapunov_vals:
        draw_text(f"Lyapunov: {lyapunov_vals[-1]:.4f}", 10, 70)

    pygame.display.flip()
    clock.tick(60)
    frame += 1

pygame.quit()

# --- 그래프 저장 (바탕화면) -----------------------------------------------
plt.figure(figsize=(6,4))
plt.plot(time_vals, lyapunov_vals, label="ln(d/d0)")
plt.xlabel("Time")
plt.ylabel("ln(distance ratio)")
plt.title("Liapunov Exponent (classical hydrogen model)")
plt.grid(True)
plt.tight_layout()

graph_path = os.path.join(hydrogen_folder, "log_distance_graph.png")
plt.savefig(graph_path)
plt.close()
