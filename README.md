
# Double Pendulum Chaos Explorer

This project gives a GPU-accelerated environment for analyzing the chaotic dynamics of the Double Pendulum. It allows for the generation of high-resolution Lyapunov stability and Poincaré recurrence fractals, interactive exploration of phase space, and the sonification of chaotic trajectories through simulation.

The project implements findings from the video ["Double Pendulums are Chaoticn't"](https://www.youtube.com/watch?v=dtjb2OhEQcU) by [2swap](https://www.youtube.com/@twoswap)

## Features


### GPU-Accelerated Fractal Engines
The application uses **Taichi Lang** to simulate hundreds of thousands of pendulums in parallel:
*   **Lyapunov Stability Map:** Measures sensitivity to initial conditions (Chaos). This replicates the method used in the 2swap video.
*   **Poincaré Recurrence Map:** Measures how closely a trajectory returns to its own starting state (Periodicity), to find resonant loops that might be hidden in the Lyapunov gradients.

### Interactive Phase Space Explorer (`visualizer.py`)
A graphical interface built with Matplotlib to view the mathematical properties of the system:
*   **Fractal Rendering:** Zooming into the stability map maintains high resolution at any depth
*   **Trajectory Solving:** Can toggle animation off to compute and display the full path of a pendulum without waiting for real-time playback
*   **Dual-Space Visualization:** Trajectories are plotted in Real Space (Cartesian) and Angle Space ($\theta_1$ vs $\theta_2$), visualizing the system
*   **Different Modes:** View both Angle Space (initial position variations) and Momentum Space (initial velocity variations)

### Batch Rendering & Visualization (`generate_visuals.py`)
Generates videos for specific points of interest 
*   **Video Generation:** Renders `.mp4` video files combining real-space physics and angle-space phase plots.
*   **Stability Map Export:** Makes a high-resolution PNG of the fractal with labeled visuals

### Physics Sonification (`audio.py`)
Interprets the geometry of the pendulum's motion as sound:
*   **Raw Data Mapping:** The angular position of the top arm ($\theta_1$) acts as the Left audio channel, and the bottom arm ($\theta_2$) as the Right audio channel
*   **Time Compression (Pitch Shift):** To bring the low-frequency mechanical oscillations into the audible range, the simulation data is solved for a duration 750 times longer than the video length and compressed which pitch-shifts the signal

## Installation

The project was tested with Python 3.12 on Windows 10 and a GPU-capable environment for Taichi (supports CUDA, Vulkan, Metal, and DirectX).

1.  Clone the repository.
2.  Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage


### Interactive Exploration

To launch the GUI explorer:

```bash
python visualizer.py
```

The application will ask for a **Metric** (Lyapunov vs Poincaré) and a **Space** (Angle vs Momentum) in the terminal:

*   **Angle Space:** Varying initial positions ($\theta_1, \theta_2$) with zero velocity
*   **Momentum Space:** Varying initial velocity ($\omega_1, \omega_2$) with zero displacement

### Generating Media

To render the stability map and video for interesting orbits:

```bash
python generate_visuals.py
```

This will create two directories:
*   `visuals/`: Contains the `visuals_map.png` and `.mp4` video files for each point of interest
*   `audio/`: Contains the corresponding `.wav` audio files.

To generate the Poincaré Recurrence Map:

```bash
python generate_periodicity.py
```
## Technical Details


### Physics Implementation
The system solves the explicit Lagrangian equations of motion for a double pendulum with unit mass ($M=1$) and unit length ($L=1$). 
*   **Single Trajectories:** Computed using `numba` (JIT-compiled CPU) with a fixed-step Runge-Kutta 4 (RK4) integrator. This approach allows for the rapid generation of long-duration (240s+) trajectories and high-framerate animations while maintaining double-precision floating-point accuracy.

### Fractal Algorithms
Original video measures the divergence between a pendulum and a "shadow" pendulum offset by $\epsilon$, this app implements two  kernels for fractal generation:

1.  **Lyapunov Divergence (Stability Map):**
  *   **Logic:** Simulates two systems ($S_A, S_B$) separated by $10^{-5}$ radians.
  *   **Metric:** $\log(\sum \|S_A(t) - S_B(t)\|)$.
  *   **Result:** Bright regions indicate Chaos (butterfly effect); dark regions indicate Stability.

2.  **Poincaré Recurrence (Periodicity Map):**
  *   **Logic:** Simulates a single system $S_A$ and compares its current state $S_A(t)$ against its initial state $S_A(0)$ after a minimum duration.
  *   **Metric:** $\log(\min \|S_A(t) - S_A(0)\|)$.
  *   **Result:** Bright regions indicate resonant, repeating orbits. Darker regions indicate quasi-periodic orbits that drift

### Gallery

#### Stability Map
![Stability Map](visuals/visuals_map.png)
#### Periodicity Map
![Periodicity Map](visuals/periodicity_angle_label.png)
#### Overlay of Visuals & Periodicity
![Overlay Map](visuals/overlay_visual_periodicity.png)
#### Rendered Videos
*More examples of generated MP4 outputs can be found in the [visuals/](visuals/) directory*

<video controls>
  <source src="visuals/vase.mp4" type="video/mp4">
</video>
<video controls>
  <source src="visuals/funky.mp4" type="video/mp4">
</video>
<video controls>
  <source src="visuals/fish.mp4" type="video/mp4">
</video>

## Credits

* **Original Concept:** [2swap](https://www.youtube.com/@twoswap)
* **Helpful 2swap Resources:** [swaptube](https://github.com/2swap/swaptube)
