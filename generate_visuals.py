import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
import physics_kernel as pk
import audio as audio_lib
import os
import time
import imageio as iio
import subprocess

# Ensure directories exist
os.makedirs("visuals", exist_ok=True)
os.makedirs("audio", exist_ok=True)

DT = 0.0005
VIDEO_DURATION = 20.0
VISUAL_DURATION_MAP = 60.0
AUDIO_DURATION = VIDEO_DURATION * 250.0
AUDIO_ANALYSIS_DURATION = 100.0 
FRACTAL_RES = 1024 

VISUAL_ICS = {
    "Droopy_Pretzel": [0.59467864, 2.8615718, 0, 0],
    "Vase": [3.0619299348742754, 0.168055422438968, 0, 0],
    "Fish": [2.759155743665746, 0.07674522687195183, 0, 0], # Look at Angle Space
    "Bird": [1.7355003, -0.5604707, 0, 0],
    "Swan": [2.2360219580994816, -0.4737880207726114, 0, 0],
    "Orc": [-0.8730425984196533, 2.0532762372871023, 0, 0],
    "Bowl": [1.7521011966811415, 2.0084663140884893, 0, 0],
    "Seagull": [-0.06029538916329304, 1.9089072838647736, 0, 0],
    "Funky": [-0.5045871831228763, 1.9851873314760473, 0, 0],
    "Flower": [3.0364986202188264, 0.6095652665478357, 0, 0],
    "Shoe": [2.3925353573635637, 0.8989729850276332, 0, 0],
    "Guitars": [2.6646864539452366, 1.6114292719597023, 0, 0], # Look at Angle Space
    "Chaotic": [-2.1222, 1.9038, 0, 0],
}


def save_stability_map():
    print("Generating Stability Map...")
    fractal, extent = pk.run_fractal_gen(mode=0, t_max=VISUAL_DURATION_MAP, res=FRACTAL_RES, dt=DT, metric="lyapunov")
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(fractal, extent=extent, origin='lower', cmap='magma_r')
    ax.set_title("Double Pendulum Stability Map", fontsize=16)
    
    for name, coords in VISUAL_ICS.items():
        ax.plot(coords[0], coords[1], 'o', color='white', markersize=6, markeredgecolor='black')
        ax.text(coords[0], coords[1] + 0.05, f" {name}", color='white', 
                fontsize=10, fontweight='bold', verticalalignment='bottom',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))
        
    ax.set_xlabel(r"$\theta_1$ (rad)")
    ax.set_ylabel(r"$\theta_2$ (rad)")
    
    map_path = "visuals/stability_map.png"
    plt.savefig(map_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved stability map to {map_path}")

def save_min_dist_map():
    print("Generating Min Dist Map...")
    fractal, extent = pk.run_fractal_gen(mode=0, t_max=VISUAL_DURATION_MAP, res=FRACTAL_RES, dt=DT, metric="mindist")
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(fractal, extent=extent, origin='lower', cmap='twilight_shifted_r')
    ax.set_title("Double Pendulum Min Distance Map", fontsize=16)
    
    for name, coords in VISUAL_ICS.items():
        ax.plot(coords[0], coords[1], 'o', color='white', markersize=6, markeredgecolor='black')
        ax.text(coords[0], coords[1] + 0.05, f" {name}", color='white', 
                fontsize=10, fontweight='bold', verticalalignment='bottom',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))
        
    ax.set_xlabel(r"$\theta_1$ (rad)")
    ax.set_ylabel(r"$\theta_2$ (rad)")
    
    map_path = "visuals/min_dist_map.png"
    plt.savefig(map_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved min dist map to {map_path}")

def save_poincare_map():
    print("Generating Poincare Map...")
    fractal, extent = pk.run_fractal_gen(mode=0, res=FRACTAL_RES, t_max=VISUAL_DURATION_MAP, dt=DT, metric="poincare")
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(fractal, extent=extent, origin='lower', cmap='magma')
    ax.set_title("Double Pendulum Periodicity Map (Angle Space)", fontsize=16)
    
    for name, coords in VISUAL_ICS.items():
        ax.plot(coords[0], coords[1], 'o', color='white', markersize=6, markeredgecolor='black', zorder=10)
        ax.text(coords[0], coords[1] + 0.05, f" {name}", color='white',
                fontsize=10, fontweight='bold', verticalalignment='bottom',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2), zorder=11)
        
    ax.set_xlabel(r"$\theta_1$ (rad)")
    ax.set_ylabel(r"$\theta_2$ (rad)")

    map_path = "visuals/poincare_map.png"
    plt.savefig(map_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved poincare map to {map_path}")

def save_trajectory_visual(name, ic):
    print(f"Generating visual and audio for {name}...")
    
    # 1. SOLVE FOR VIDEO
    start_time = time.time()
    th1_v, th2_v, w1_v, w2_v = pk.run_trajectory(ic, VIDEO_DURATION, DT)
    print(f"Video solved in {time.time() - start_time:.2f} seconds")
    
    # 2. SOLVE FOR AUDIO
    start_time = time.time()
    th1_a, th2_a, w1_a, w2_a = pk.run_trajectory(ic, AUDIO_DURATION, DT)
    # th1_a, th2_a, w1_a, w2_a = pk.run_trajectory(ic, AUDIO_ANALYSIS_DURATION, DT)
    print(f"Audio solved in {time.time() - start_time:.2f} seconds")
    
    # VIDEO PRE-PROCESSING
    fps = 30
    total_frames = int(VIDEO_DURATION * fps) 
    sub = int((VIDEO_DURATION / DT) / total_frames)
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
    
    # Calculate limits with padding
    def get_limits(data, padding=0.1):
        dmin, dmax = np.min(data), np.max(data)
        margin = max((dmax - dmin) * padding, 0.1)
        return dmin - margin, dmax + margin

    th1_min, th1_max = get_limits(th1_v)
    th2_min, th2_max = get_limits(th2_v)
    
    ax_phase.set_xlim(th1_min, th1_max)
    ax_phase.set_ylim(th2_min, th2_max)
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
    iio.mimwrite(temp_vid, frames, fps=fps, codec='libx264')
    print(f"Video rendered in {time.time() - start_render:.2f} seconds")

    
    # Audio
    audio_engine = audio_lib.AudioEngine()
    # final_audio = audio_engine.synthesize_spectral(th1_a, th2_a, output_duration=VIDEO_DURATION, input_dt=DT)
    final_audio = audio_engine.process_data(th1_a, th2_a, target_duration=VIDEO_DURATION) 

    audio_engine.save(final_audio, f"audio/{name.lower()}.wav")

    # Merge using ffmpeg
    final_output = f"visuals/{name.lower()}.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_vid,
        "-i", f"audio/{name.lower()}.wav",
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
    
    # Save full path PNG (Real + Angle Space)
    print(f"Saving full path graph for {name}...")
    
    # Downsample for static plot 
    skip_p = max(1, len(th1_v) // 100000)
    th1_p = th1_v[::skip_p]
    th2_p = th2_v[::skip_p]
    
    x1_p = pk.L1 * np.sin(th1_p)
    y1_p = -pk.L1 * np.cos(th1_p)
    x2_p = x1_p + pk.L2 * np.sin(th2_p)
    y2_p = y1_p - pk.L2 * np.cos(th2_p)
    
    fig_p, (ax_phys_p, ax_phase_p) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Real Space Path
    ax_phys_p.plot(x2_p, y2_p, lw=0.4, alpha=0.6, color='#3498db')
    ax_phys_p.set_xlim(-2.2, 2.2); ax_phys_p.set_ylim(-2.2, 2.2)
    ax_phys_p.set_aspect('equal')
    ax_phys_p.set_title(f"Real Space Path ({VIDEO_DURATION}s): {name}")
    ax_phys_p.set_xlabel("X"); ax_phys_p.set_ylabel("Y")
    
    # Angle Space Path
    ax_phase_p.plot(th1_p, th2_p, lw=0.4, alpha=0.7, color='#8e44ad')
    p1min, p1max = get_limits(th1_p); p2min, p2max = get_limits(th2_p)
    ax_phase_p.set_xlim(p1min, p1max); ax_phase_p.set_ylim(p2min, p2max)
    ax_phase_p.set_title(r"Angle Space ($\theta_1$ vs $\theta_2$)")
    ax_phase_p.set_xlabel(r"$\theta_1$"); ax_phase_p.set_ylabel(r"$\theta_2$")
    
    plt.tight_layout()
    path_png = f"visuals/{name.lower()}_path.png"
    plt.savefig(path_png, dpi=200)
    plt.close(fig_p)
    
    print(f"Done: {final_output} and {path_png} in {time.time() - start_time:.2f}s")

def save_overlay_map():
    print("Generating Overlay Map...")
    # Paths to the images
    periodicity_path = "visuals/min_dist_map.png"
    stability_map_path = "visuals/stability_map.png"
    output_path = "visuals/overlay_periodicity_stability.png"

    # Load images as arrays
    periodicity_img = iio.imread(periodicity_path)
    stability_map_img = iio.imread(stability_map_path)


    fig, ax = plt.subplots(figsize=(12, 12))

    # Cover any previous titles
    height = periodicity_img.shape[0]
    width = periodicity_img.shape[1]
    white_height = int(0.03 * height)  
    white_block = np.ones((white_height, width, 3), dtype=np.uint8) * 255

    if periodicity_img.shape[-1] == 4:
        white_block = np.ones((white_height, width, 4), dtype=np.uint8) * 255

    periodicity_img_clean = np.vstack([white_block, periodicity_img[white_height:]])
    stability_map_img_clean = np.vstack([white_block, stability_map_img[white_height:]])

    ax.imshow(periodicity_img_clean)
    ax.imshow(stability_map_img_clean, alpha=0.5)  
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    fig.suptitle("Overlay: Periodicity and Stability (Angle Space)", fontsize=18, y=0.985)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Saved overlay map to {output_path}")

if __name__ == "__main__":
    save_stability_map()
    save_min_dist_map()
    save_poincare_map()
    save_overlay_map()
    
    for name, ic in VISUAL_ICS.items():
        save_trajectory_visual(name, ic)
        