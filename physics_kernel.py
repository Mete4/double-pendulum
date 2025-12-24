import taichi as ti
import numpy as np
from numba import njit

ti.init(arch=ti.gpu)

# Constant parameters
G = 9.81
L1 = 1.0
L2 = 1.0
M1 = 1.0
M2 = 1.0

# NUMBA (CPU) - Single Trajectory
@njit(cache=True, fastmath=True)
def _get_derivs_nb(th1, th2, w1, w2):
    delta = th1 - th2
    sin_d = np.sin(delta)
    cos_d = np.cos(delta)
    cos_2d = np.cos(2 * delta)
    
    num1 = -G*(2*M1+M2)*np.sin(th1) - M2*G*np.sin(th1-2*th2) - 2*sin_d*M2*(w2*w2*L2 + w1*w1*L1*cos_d)
    den1 = L1 * (2*M1 + M2 - M2*cos_2d)
    
    num2 = 2*sin_d*(w1*w1*L1*(M1+M2) + G*(M1+M2)*np.cos(th1) + w2*w2*L2*M2*cos_d)
    den2 = L2 * (2*M1 + M2 - M2*cos_2d)
    
    return w1, w2, num1/den1, num2/den2

@njit(cache=True, fastmath=True)
def _rk4_nb(th1, th2, w1, w2, dt):
    k1 = _get_derivs_nb(th1, th2, w1, w2)
    k2 = _get_derivs_nb(th1 + k1[0]*dt/2, th2 + k1[1]*dt/2, w1 + k1[2]*dt/2, w2 + k1[3]*dt/2)
    k3 = _get_derivs_nb(th1 + k2[0]*dt/2, th2 + k2[1]*dt/2, w1 + k2[2]*dt/2, w2 + k2[3]*dt/2)
    k4 = _get_derivs_nb(th1 + k3[0]*dt, th2 + k3[1]*dt, w1 + k3[2]*dt, w2 + k3[3]*dt)
    
    return (th1 + dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]),
            th2 + dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1]),
            w1 + dt/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2]),
            w2 + dt/6*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3]))

@njit(cache=True, fastmath=True)
def _run_traj_nb(th1, th2, w1, w2, dt, n):
    th1_out = np.empty(n, np.float64)
    th2_out = np.empty(n, np.float64)
    w1_out = np.empty(n, np.float64)
    w2_out = np.empty(n, np.float64)
    
    for i in range(n):
        th1_out[i], th2_out[i], w1_out[i], w2_out[i] = th1, th2, w1, w2
        th1, th2, w1, w2 = _rk4_nb(th1, th2, w1, w2, dt)
    
    return th1_out, th2_out, w1_out, w2_out

def run_trajectory(ic, duration, dt=0.01):
    """Run single trajectory - Numba CPU (fast for sequential)."""
    n_steps = int(duration / dt)
    return _run_traj_nb(float(ic[0]), float(ic[1]), float(ic[2]), float(ic[3]), dt, n_steps)

# Warm-up JIT
_dummy = run_trajectory([0.1, 0.1, 0, 0], 0.1, 0.01)

# TAICHI (GPU) - Fractal Map
@ti.func
def get_derivs(th1, th2, w1, w2):
    delta = th1 - th2
    num1 = -G * (2 * M1 + M2) * ti.sin(th1) - M2 * G * ti.sin(th1 - 2 * th2) - 2 * ti.sin(delta) * M2 * (w2**2 * L2 + w1**2 * L1 * ti.cos(delta))
    den1 = L1 * (2 * M1 + M2 - M2 * ti.cos(2 * delta))
    dw1 = num1 / den1

    num2 = 2 * ti.sin(delta) * (w1**2 * L1 * (M1 + M2) + G * (M1 + M2) * ti.cos(th1) + w2**2 * L2 * M2 * ti.cos(delta))
    den2 = L2 * (2 * M1 + M2 - M2 * ti.cos(2 * delta))
    dw2 = num2 / den2
    
    return w1, w2, dw1, dw2

@ti.kernel
def compute_poincare_map(
    result_grid: ti.types.ndarray(),
    x_min: float, x_max: float, y_min: float, y_max: float,
    res: int, t_max: float, dt: float, mode: int
):
    for i, j in ti.ndrange(res, res):
        th1, th2, w1, w2 = 0.0, 0.0, 0.0, 0.0

        x_val = x_min + (x_max - x_min) * j / res
        y_val = y_min + (y_max - y_min) * i / res

        if mode == 0:
            th1, th2 = x_val, y_val
        else:
            w1, w2 = x_val, y_val

        start_th1, start_th2, start_w1, start_w2 = th1, th2, w1, w2
        min_dist = 1e10
        steps = int(t_max / dt)

        for step in range(steps):
            th1, th2, w1, w2 = rk4_step(th1, th2, w1, w2, dt)
            t = step * dt
            if t > 1.0:
                dist = ti.sqrt(
                    (th1 - start_th1) ** 2 +
                    (th2 - start_th2) ** 2 +
                    (w1 - start_w1) ** 2 +
                    (w2 - start_w2) ** 2
                )
                min_dist = ti.min(min_dist, dist)

        result_grid[i, j] = ti.log(min_dist + 1e-10)
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

@ti.kernel
def compute_lyapunov_map(
    result_grid: ti.types.ndarray(),
    x_min: float, x_max: float, y_min: float, y_max: float,
    res: int, t_max: float, dt: float, mode: int
):
    for i, j in ti.ndrange(res, res):  # parallelizes - each pixel independent
        th1_a, th2_a, w1_a, w2_a = 0.0, 0.0, 0.0, 0.0
        
        x_val = x_min + (x_max - x_min) * j / res
        y_val = y_min + (y_max - y_min) * i / res
        
        if mode == 0:
            th1_a, th2_a = x_val, y_val
        else:
            w1_a, w2_a = x_val, y_val
            
        th1_b, th2_b = th1_a + 1e-5, th2_a + 1e-5
        w1_b, w2_b = w1_a, w2_a
        
        lyapunov = 0.0
        steps = int(t_max / dt)
        
        for _ in range(steps):
            th1_a, th2_a, w1_a, w2_a = rk4_step(th1_a, th2_a, w1_a, w2_a, dt)
            th1_b, th2_b, w1_b, w2_b = rk4_step(th1_b, th2_b, w1_b, w2_b, dt)
            
            x2_a = L1 * ti.sin(th1_a) + L2 * ti.sin(th2_a)
            y2_a = -L1 * ti.cos(th1_a) - L2 * ti.cos(th2_a)
            x2_b = L1 * ti.sin(th1_b) + L2 * ti.sin(th2_b)
            y2_b = -L1 * ti.cos(th1_b) - L2 * ti.cos(th2_b)
            
            dist = ti.sqrt((x2_a - x2_b)**2 + (y2_a - y2_b)**2)
            if dist > 0:
                lyapunov += ti.log(dist + 1e-10)
                
        result_grid[i, j] = lyapunov

def run_fractal_gen(mode=0, res=512, t_max=10.0, dt=0.02, bounds=None, metric="lyapunov"):
    grid = np.zeros((res, res), dtype=np.float32)

    if bounds is None:
        if mode == 0:
            x_min, x_max, y_min, y_max = -np.pi, np.pi, -np.pi, np.pi
        else:
            x_min, x_max, y_min, y_max = -10.0, 10.0, -10.0, 10.0
    else:
        x_min, x_max, y_min, y_max = bounds

    if metric == "lyapunov":
        compute_lyapunov_map(grid, x_min, x_max, y_min, y_max, res, t_max, dt, mode)
    elif metric == "poincare":
        compute_poincare_map(grid, x_min, x_max, y_min, y_max, res, t_max, dt, mode)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return grid, (x_min, x_max, y_min, y_max)

# scipy version for compatibility
def get_scipy_derivs(t, y):
    th1, th2, w1, w2 = y
    delta = th1 - th2
    num1 = -G * (2 * M1 + M2) * np.sin(th1) - M2 * G * np.sin(th1 - 2 * th2) - 2 * np.sin(delta) * M2 * (w2**2 * L2 + w1**2 * L1 * np.cos(delta))
    den1 = L1 * (2 * M1 + M2 - M2 * np.cos(2 * delta))
    d_w1 = num1 / den1

    num2 = 2 * np.sin(delta) * (w1**2 * L1 * (M1 + M2) + G * (M1 + M2) * np.cos(th1) + w2**2 * L2 * M2 * np.cos(delta))
    den2 = L2 * (2 * M1 + M2 - M2 * np.cos(2 * delta))
    d_w2 = num2 / den2
    
    return [w1, w2, d_w1, d_w2]