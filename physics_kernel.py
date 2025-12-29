import taichi as ti
import numpy as np
import logging

ti.init(arch=ti.gpu, default_fp=ti.f32, offline_cache=False, fast_math=False)

logger = logging.getLogger(__name__)

# --- GLOBAL CONSTANTS ---
G = 9.81
L1 = 1.0
L2 = 1.0
M1 = 1.0
M2 = 1.0

SCORE_DECAY = 15.0
SCORE_WARMUP = 50
WEIGHT_POS = 1.0
WEIGHT_VEL = 0.5
WEIGHT_ACC = 0.05

MAX_TRAJ_STEPS = 50000000 
traj_field_th1 = ti.field(dtype=ti.f32, shape=MAX_TRAJ_STEPS)
traj_field_th2 = ti.field(dtype=ti.f32, shape=MAX_TRAJ_STEPS)
traj_field_w1 = ti.field(dtype=ti.f32, shape=MAX_TRAJ_STEPS)
traj_field_w2 = ti.field(dtype=ti.f32, shape=MAX_TRAJ_STEPS)
traj_field_score = ti.field(dtype=ti.f32, shape=MAX_TRAJ_STEPS)

# Fields for coord conversion
coord_exchange = ti.field(dtype=ti.f32, shape=2)
coord_verify_result = ti.field(dtype=ti.f32, shape=())

# --- SHARED FUNCTIONS ---

@ti.func
def get_coordinate(i, j, x_min, x_max, y_min, y_max, res):
    # Map to pixel CENTERS for correct alignment
    x_val = x_min + (x_max - x_min) * (j + 0.5) / res
    y_val = y_min + (y_max - y_min) * (i + 0.5) / res
    return x_val, y_val

@ti.func
def get_derivs(th1, th2, w1, w2):
    delta = th1 - th2
    sin_d = ti.sin(delta); cos_d = ti.cos(delta); cos_2d = ti.cos(2 * delta)
    num1 = -G*(2*M1+M2)*ti.sin(th1) - M2*G*ti.sin(th1-2*th2) - 2*ti.sin(delta)*M2*(w2**2*L2 + w1**2*L1*ti.cos(delta))
    den1 = L1 * (2 * M1 + M2 - M2 * cos_2d)
    num2 = 2*ti.sin(delta)*(w1**2*L1*(M1+M2) + G*(M1+M2)*ti.cos(th1) + w2**2*L2*M2*ti.cos(delta))
    den2 = L2 * (2 * M1 + M2 - M2 * cos_2d)
    return w1, w2, num1/den1, num2/den2

@ti.func
def rk4_step(th1, th2, w1, w2, dt):
    k1_th1, k1_th2, k1_w1, k1_w2 = get_derivs(th1, th2, w1, w2)
    k2_th1, k2_th2, k2_w1, k2_w2 = get_derivs(th1 + k1_th1*dt/2, th2 + k1_th2*dt/2, w1 + k1_w1*dt/2, w2 + k1_w2*dt/2)
    k3_th1, k3_th2, k3_w1, k3_w2 = get_derivs(th1 + k2_th1*dt/2, th2 + k2_th2*dt/2, w1 + k2_w1*dt/2, w2 + k2_w2*dt/2)
    k4_th1, k4_th2, k4_w1, k4_w2 = get_derivs(th1 + k3_th1*dt, th2 + k3_th2*dt, w1 + k3_w1*dt, w2 + k3_w2*dt)
    
    return (th1 + dt/6*(k1_th1 + 2*k2_th1 + 2*k3_th1 + k4_th1),
            th2 + dt/6*(k1_th2 + 2*k2_th2 + 2*k3_th2 + k4_th2),
            w1 + dt/6*(k1_w1 + 2*k2_w1 + 2*k3_w1 + k4_w1),
            w2 + dt/6*(k1_w2 + 2*k2_w2 + 2*k3_w2 + k4_w2))

@ti.func
def calc_step_score(th1, th2, w1, w2, start_th1, start_th2, start_w1, start_w2):
    _, _, curr_dw1, curr_dw2 = get_derivs(th1, th2, w1, w2)
    _, _, start_dw1, start_dw2 = get_derivs(start_th1, start_th2, start_w1, start_w2)
    
    d_pos = 2.0 * (1.0 - ti.cos(th1 - start_th1)) + 2.0 * (1.0 - ti.cos(th2 - start_th2))
    d_vel = (w1 - start_w1)**2 + (w2 - start_w2)**2
    d_acc = (curr_dw1 - start_dw1)**2 + (curr_dw2 - start_dw2)**2
    
    dist_sq = (WEIGHT_POS * d_pos) + (WEIGHT_VEL * d_vel) + (WEIGHT_ACC * d_acc)
    return ti.exp(-SCORE_DECAY * dist_sq)

# --- KERNELS ---

@ti.kernel
def compute_trajectory_kernel(th1_init: float, th2_init: float, w1_init: float, w2_init: float, dt: float, steps: int):
    ti.loop_config(serialize=True)
    th1, th2, w1, w2 = th1_init, th2_init, w1_init, w2_init
    start_th1, start_th2, start_w1, start_w2 = th1, th2, w1, w2
    current_accumulated_score = 0.0
    
    for i in range(steps):
        traj_field_th1[i], traj_field_th2[i] = th1, th2
        traj_field_w1[i], traj_field_w2[i] = w1, w2
        
        step_score = 0.0
        if i > SCORE_WARMUP:
            quality = calc_step_score(th1, th2, w1, w2, start_th1, start_th2, start_w1, start_w2)
            time_weight = float(i) / float(steps)
            step_score = time_weight * quality

        current_accumulated_score += step_score
        traj_field_score[i] = ti.log(current_accumulated_score + 1e-10)
        
        th1, th2, w1, w2 = rk4_step(th1, th2, w1, w2, dt)

def run_trajectory(ic, duration, dt=0.01):
    steps = int(duration / dt)
    if steps > MAX_TRAJ_STEPS: 
        logger.warning(f"Trajectory too long: {steps} > {MAX_TRAJ_STEPS}")
        steps = MAX_TRAJ_STEPS
    # Cast to float32 to match GPU
    args = [float(np.float32(v)) for v in ic]
    compute_trajectory_kernel(args[0], args[1], args[2], args[3], float(dt), steps)
    ti.sync()
    return (traj_field_th1.to_numpy()[:steps], traj_field_th2.to_numpy()[:steps], 
            traj_field_w1.to_numpy()[:steps], traj_field_w2.to_numpy()[:steps])

def compute_recurrence_score_over_time(ic, t_max, dt=0.01):
    steps = int(t_max / dt)
    if steps > MAX_TRAJ_STEPS: 
        logger.warning(f"Trajectory too long: {steps} > {MAX_TRAJ_STEPS}")
        steps = MAX_TRAJ_STEPS
    run_trajectory(ic, t_max, dt) # Re-run to populate scores field
    scores = traj_field_score.to_numpy()[:steps]
    return np.zeros_like(scores), scores

@ti.kernel
def compute_poincare_map(
    result_grid: ti.types.ndarray(),
    x_min: float, x_max: float, y_min: float, y_max: float,
    res: int, steps: int, dt: float, mode: int
):
    for i, j in ti.ndrange(res, res):
        x_val, y_val = get_coordinate(i, j, x_min, x_max, y_min, y_max, res)
        
        th1, th2, w1, w2 = 0.0, 0.0, 0.0, 0.0
        if mode == 0: th1, th2 = x_val, y_val
        else: w1, w2 = x_val, y_val
            
        start_th1, start_th2, start_w1, start_w2 = th1, th2, w1, w2
        
        _, _, start_dw1, start_dw2 = get_derivs(start_th1, start_th2, start_w1, start_w2)
        
        recurrence_score = 0.0

        for step in range(steps):
            if step > SCORE_WARMUP:
                quality = calc_step_score(th1, th2, w1, w2, start_th1, start_th2, start_w1, start_w2)
                time_weight = float(step) / float(steps)
                recurrence_score += time_weight * quality
            
            th1, th2, w1, w2 = rk4_step(th1, th2, w1, w2, dt)

        result_grid[i, j] = ti.log(recurrence_score + 1e-10) if ti.log(recurrence_score + 1e-10) > 0.0 else 0.0

@ti.kernel
def compute_min_dist_map(
    result_grid: ti.types.ndarray(),
    x_min: float, x_max: float, y_min: float, y_max: float,
    res: int, steps: int, dt: float, mode: int
):
    for i, j in ti.ndrange(res, res):
        x_val, y_val = get_coordinate(i, j, x_min, x_max, y_min, y_max, res)
        
        th1, th2, w1, w2 = 0.0, 0.0, 0.0, 0.0
        if mode == 0: th1, th2 = x_val, y_val
        else: w1, w2 = x_val, y_val
            
        start_th1, start_th2, start_w1, start_w2 = th1, th2, w1, w2
        min_dist = 1.0e10
        
        for step in range(steps):
             # Standard RK4 (Inline)
            th1, th2, w1, w2 = rk4_step(th1, th2, w1, w2, dt)
            
            # Continuous Check (Every Step)
            if step > 1:
                dist = ti.sqrt((th1 - start_th1)**2 + (th2 - start_th2)**2 + WEIGHT_VEL*(w1 - start_w1)**2 + WEIGHT_VEL*(w2 - start_w2)**2)
                min_dist = ti.min(min_dist, dist)

        result_grid[i, j] = -ti.log(min_dist + 1e-10)

@ti.kernel
def compute_lyapunov_map(
    result_grid: ti.types.ndarray(),
    x_min: float, x_max: float, y_min: float, y_max: float,
    res: int, steps: int, dt: float, mode: int
):
    for i, j in ti.ndrange(res, res):
        x_val, y_val = get_coordinate(i, j, x_min, x_max, y_min, y_max, res)
        th1, th2, w1, w2 = 0.0, 0.0, 0.0, 0.0
        if mode == 0: th1, th2 = x_val, y_val
        else: w1, w2 = x_val, y_val
            
        th1_b, th2_b = th1 + 1e-5, th2 + 1e-5
        w1_b, w2_b = w1, w2
        lyapunov = 0.0
        
        for _ in range(steps):
            th1, th2, w1, w2 = rk4_step(th1, th2, w1, w2, dt)
            th1_b, th2_b, w1_b, w2_b = rk4_step(th1_b, th2_b, w1_b, w2_b, dt)
            x2_a = L1 * ti.sin(th1) + L2 * ti.sin(th2); y2_a = -L1 * ti.cos(th1) - L2 * ti.cos(th2)
            x2_b = L1 * ti.sin(th1_b) + L2 * ti.sin(th2_b); y2_b = -L1 * ti.cos(th1_b) - L2 * ti.cos(th2_b)
            dist = ti.sqrt((x2_a - x2_b)**2 + (y2_a - y2_b)**2)
            if dist > 0: lyapunov += ti.log(dist + 1e-10)
        result_grid[i, j] = -lyapunov

def run_fractal_gen(mode=0, res=512, t_max=10.0, dt=0.02, bounds=None, metric="lyapunov"):
    grid = np.zeros((res, res), dtype=np.float64)
    if bounds is None:
        if mode == 0: x_min, x_max, y_min, y_max = -np.pi, np.pi, -np.pi, np.pi
        else: x_min, x_max, y_min, y_max = -10.0, 10.0, -10.0, 10.0
    else: x_min, x_max, y_min, y_max = bounds

    steps = int(t_max / dt)
    if metric == "lyapunov":
        compute_lyapunov_map(grid, x_min, x_max, y_min, y_max, res, steps, dt, mode)
    elif metric == "poincare":
        compute_poincare_map(grid, x_min, x_max, y_min, y_max, res, steps, dt, mode)
    elif metric == "mindist":
        compute_min_dist_map(grid, x_min, x_max, y_min, y_max, res, steps, dt, mode)
    else: raise ValueError(f"Unknown metric: {metric}")
    return grid, (x_min, x_max, y_min, y_max)

# --- COORDINATES ---
@ti.kernel
def get_coords_from_index_kernel(i: int, j: int, x_min: float, x_max: float, y_min: float, y_max: float, res: int):
    x, y = get_coordinate(i, j, x_min, x_max, y_min, y_max, res)
    coord_exchange[0] = x
    coord_exchange[1] = y

@ti.kernel
def verify_point_score_kernel(th1_init: float, th2_init: float, w1_init: float, w2_init: float, dt: float, steps: int):
    ti.loop_config(serialize=True)
    th1, th2, w1, w2 = th1_init, th2_init, w1_init, w2_init
    start_th1, start_th2, start_w1, start_w2 = th1, th2, w1, w2
    _, _, start_dw1, start_dw2 = get_derivs(start_th1, start_th2, start_w1, start_w2)
    recurrence_score = 0.0
    
    for step in range(steps):
        if step > SCORE_WARMUP:
            # Optimized Score Calc
            quality = calc_step_score(th1, th2, w1, w2, start_th1, start_th2, start_w1, start_w2)
            
            time_weight = float(step) / float(steps)
            recurrence_score += time_weight * quality
        
        th1, th2, w1, w2 = rk4_step(th1, th2, w1, w2, dt)
        
    coord_verify_result[None] = ti.log(recurrence_score + 1e-10)

@ti.kernel
def verify_min_dist_point_kernel(th1_init: float, th2_init: float, w1_init: float, w2_init: float, dt: float, steps: int):
    ti.loop_config(serialize=True)
    th1, th2, w1, w2 = th1_init, th2_init, w1_init, w2_init
    start_th1, start_th2, start_w1, start_w2 = th1, th2, w1, w2
    min_dist = 1.0e10
    
    for step in range(steps):
        th1, th2, w1, w2 = rk4_step(th1, th2, w1, w2, dt)
        
        if step > SCORE_WARMUP:
            dist = ti.sqrt((th1 - start_th1)**2 + (th2 - start_th2)**2 + (w1 - start_w1)**2 + (w2 - start_w2)**2)
            if dist < min_dist:
                min_dist = dist
                
    coord_verify_result[None] = -ti.log(min_dist + 1e-10) # Match mapping logic

def find_global_max(grid, bounds, mode=0, t_max=10.0, dt=0.01, metric="poincare"):
    res = grid.shape[0]
    x_min, x_max, y_min, y_max = bounds
    
    # 1. Find Max Index
    flat_idx = np.argmax(grid)
    i_max, j_max = np.unravel_index(flat_idx, grid.shape)
    map_score = grid[i_max, j_max]
    
    logger.debug(f"===== FIND_GLOBAL_MAX DEBUG ({metric}) =====")
    logger.debug(f"Grid shape: {grid.shape}")
    logger.debug(f"Bounds: x=[{x_min:.6f}, {x_max:.6f}], y=[{y_min:.6f}, {y_max:.6f}]")
    logger.debug(f"Simulation params: t_max={t_max}s, dt={dt}")
    logger.debug(f"Coordinate formula: PIXEL CENTERS (j+0.5)/res")
    logger.debug(f"Max grid index: ({i_max}, {j_max})")
    logger.debug(f"Grid value at max index: {map_score:.6f}")
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Grid values around max:")
        for di in range(-2, 3):
            row_str = "  "
            for dj in range(-2, 3):
                ni, nj = i_max + di, j_max + dj
                if 0 <= ni < res and 0 <= nj < res:
                    val = grid[ni, nj]
                    marker = "*" if (di == 0 and dj == 0) else " "
                    row_str += f"{val:7.3f}{marker} "
                else:
                    row_str += "   ---    "
            logger.debug(row_str)
    
    # 2. Get Coords
    get_coords_from_index_kernel(int(i_max), int(j_max), float(x_min), float(x_max), float(y_min), float(y_max), int(res))
    ti.sync()
    coords = coord_exchange.to_numpy()
    x_val, y_val = coords[0], coords[1]
    
    logger.debug(f"Converted coordinates: x={x_val:.6f}, y={y_val:.6f}")
    
    if mode == 0: ic = [x_val, y_val, 0.0, 0.0]
    else: ic = [0.0, 0.0, x_val, y_val]
    
    logger.debug(f"Initial conditions: {ic}")
    
    # 3. VERIFY
    steps = int(t_max / dt)
    logger.debug(f"Verification steps: {steps}")
    args = [float(np.float32(v)) for v in ic]
    logger.debug(f"Args after float32 cast: {args}")
    
    if metric == "mindist":
        verify_min_dist_point_kernel(args[0], args[1], args[2], args[3], float(dt), steps)
    else:
        verify_point_score_kernel(args[0], args[1], args[2], args[3], float(dt), steps)
        
    ti.sync()
    verified_score = coord_verify_result[None]
    
    logger.debug(f"Grid Idx: ({i_max},{j_max}) -> Coords: {ic[:2]}")
    logger.debug(f"Map Score ({t_max}s): {map_score:.4f} | Trajectory Score ({t_max}s): {verified_score:.4f}")
    logger.debug(f"Discrepancy: {abs(map_score - verified_score):.6f}")
    logger.debug("===================================")
    
    return float(map_score), ic