import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
import physics_kernel as pk
import audio as audio_lib
import os
import time

# Ensure directories exist
os.makedirs("visuals", exist_ok=True)
os.makedirs("audio", exist_ok=True)

# Synchronize with visualizer.py
# SOLVER_METHOD = 'DOP853'
# TOLERANCE = 1e-10
DT = 0.01 
VISUAL_DURATION = 20.0  
AUDIO_DURATION = VISUAL_DURATION * 750 # 750x pitch up
FRACTAL_RES = 1024 

VISUAL_ICS = {
    "Droopy_Pretzel": [0.5648, 2.8494, 0, 0],
    "Vase": [3.0564, 0.1714, 0, 0],
    "Fish": [2.7636, 0.0774, 0, 0],
    "Bird": [1.7634, -0.4750, 0, 0],
    "Swan": [2.2373, -0.4749, 0, 0],
    "Orc": [-0.8708, 2.0406, 0, 0],
    "Bowl": [1.7626, 2, 0, 0],
    "Seagull": [-0.0632, 1.9038, 0, 0],
    "Funky": [-0.5249, 2.0054, 0, 0],
    "Chaotic": [-2.1222, 1.9038, 0, 0],
}

def save_visual_map():
    print("Generating Stability Map...")
    # Use 240s for the map to match visualizer's chaotic depth
    fractal, extent = pk.run_fractal_gen(mode=0, t_max=240, res=FRACTAL_RES, dt=DT)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(fractal, extent=extent, origin='lower', cmap='magma')
    ax.set_title("Double Pendulum Stability Map", fontsize=16)
    
    for name, coords in VISUAL_ICS.items():
        ax.plot(coords[0], coords[1], 'o', color='white', markersize=6, markeredgecolor='black')
        ax.text(coords[0], coords[1] + 0.05, f" {name}", color='white', 
                fontsize=10, fontweight='bold', verticalalignment='bottom',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))
        
    ax.set_xlabel(r"$\theta_1$ (rad)")
    ax.set_ylabel(r"$\theta_2$ (rad)")
    
    map_path = "visuals/visuals_map.png"
    plt.savefig(map_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visual map to {map_path}")


import imageio

def generate_visual(name, ic):
    print(f"Generating visual and audio for {name}...")
    
    # 1. SOLVE FOR VIDEO 
    start_time = time.time()
    # t_eval_vid = np.linspace(0, VISUAL_DURATION, int(VISUAL_DURATION / DT))
    # sol_vid = solve_ivp(pk.get_scipy_derivs, (0, VISUAL_DURATION), ic, t_eval=t_eval_vid, 
    #                     method=SOLVER_METHOD, rtol=TOLERANCE, atol=TOLERANCE)
    # th1_v, th2_v = sol_vid.y[0], sol_vid.y[1]
    th1_v, th2_v, w1_v, w2_v = pk.run_trajectory(ic, VISUAL_DURATION, DT)
    print(f"Video solved in {time.time() - start_time:.2f} seconds")
    
    # 2. SOLVE FOR AUDIO
    start_time = time.time()
    # t_eval_aud = np.linspace(0, AUDIO_DURATION, int(AUDIO_DURATION / DT))
    # sol_aud = solve_ivp(pk.get_scipy_derivs, (0, AUDIO_DURATION), ic, t_eval=t_eval_aud, 
    #                     method=SOLVER_METHOD, rtol=TOLERANCE, atol=TOLERANCE)
    # th1_a, th2_a = sol_aud.y[0], sol_aud.y[1]
    th1_a, th2_a, w1_a, w2_a = pk.run_trajectory(ic, AUDIO_DURATION, DT)
    print(f"Audio solved in {time.time() - start_time:.2f} seconds")
    
    # VIDEO PRE-PROCESSING
    fps = 30
    total_frames = int(VISUAL_DURATION * fps) 
    sub = int((VISUAL_DURATION / DT) / total_frames)
    if sub < 1: sub = 1
    
    th1_sub = th1_v[::sub][:total_frames]
    th2_sub = th2_v[::sub][:total_frames]
    
    x1 = pk.L1 * np.sin(th1_sub)
    y1 = -pk.L1 * np.cos(th1_sub)
    x2 = x1 + pk.L2 * np.sin(th2_sub)
    y2 = y1 - pk.L2 * np.cos(th2_sub)
    
    # Setup Figure
    plt.ioff()
    fig, (ax_phys, ax_phase) = plt.subplots(1, 2, figsize=(10, 4), dpi=72)
    
    ax_phys.set_xlim(-2.2, 2.2)
    ax_phys.set_ylim(-2.2, 2.2)
    ax_phys.set_aspect('equal')
    ax_phys.set_title(f"Real Space: {name}")
    line, = ax_phys.plot([], [], 'o-', lw=2, color='#2c3e50')
    trace, = ax_phys.plot([], [], '-', lw=0.8, alpha=0.3, color='#3498db')
    
    ax_phase.set_xlim(-np.pi, np.pi)
    ax_phase.set_ylim(-np.pi, np.pi)
    ax_phase.set_title(r"Angle Space ($\theta_1$ vs $\theta_2$)")
    phase_trace, = ax_phase.plot([], [], '-', lw=0.8, color='#8e44ad')
    plt.tight_layout()
    
    temp_vid = f"visuals/temp_{name.lower()}.mp4"
    start_render = time.time()

    # Draw static elements once
    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(fig.bbox)

    frames = []
    max_trace = 600

    # Animation loop
    for i in range(len(th1_sub)):
        fig.canvas.restore_region(bg)
        
        line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
        start = max(0, i - max_trace)
        trace.set_data(x2[start:i+1], y2[start:i+1])
        phase_trace.set_data(th1_sub[start:i+1], th2_sub[start:i+1])
        
        ax_phys.draw_artist(line)
        ax_phys.draw_artist(trace)
        ax_phase.draw_artist(phase_trace)
        
        fig.canvas.blit(fig.bbox)
        
        # Capture frame
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)[:, :, :3]
        frames.append(frame.copy())

    plt.close(fig)
    imageio.mimwrite(temp_vid, frames, fps=fps, codec='libx264')
    print(f"Video rendered in {time.time() - start_render:.2f} seconds")

    
    # Audio
    audio_engine = audio_lib.AudioEngine()
    chunk = audio_engine.process_data(th1_a, th2_a, target_duration=VISUAL_DURATION) 
    audio_engine.save(chunk, f"audio/{name.lower()}.wav")
    sr = audio_engine.sample_rate
    target_samples = int((len(frames) / fps) * sr)
    chunk_len = len(chunk)
    
    # Ensure stereo float32 in [-1, 1]
    chunk_float = chunk.astype(np.float32) / 32767.0
    chunk_len = len(chunk_float)

    if chunk_len >= target_samples:
        final_float = chunk_float[:target_samples]
    else:
        final_float = np.zeros((target_samples, 2), dtype=np.float32)
        final_float[:chunk_len] = chunk_float

    final_audio = (np.clip(final_float, -1, 1) * 32767).astype(np.int16)
    temp_aud = f"audio/temp_{name.lower()}.wav"
    audio_engine.save(final_audio, temp_aud)
    
    # Merge using ffmpeg
    final_output = f"visuals/{name.lower()}.mp4"
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_vid,
        "-i", temp_aud,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        final_output
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    
    # Cleanup
    if os.path.exists(temp_vid): os.remove(temp_vid)
    if os.path.exists(temp_aud): os.remove(temp_aud)
    
    print(f"Done: {final_output} in {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    save_visual_map()
    
    for name, ic in VISUAL_ICS.items():
        generate_visual(name, ic)
