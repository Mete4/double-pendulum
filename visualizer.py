import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import TextBox, Button
import physics_kernel as pk
import logging

# Configure Logging
DEBUG_LOGGING = False
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if DEBUG_LOGGING:
    logging.getLogger().setLevel(logging.DEBUG) 

# Simulation Parameters
DT = 0.01
DURATION = 60 * 1 

class DoublePendulumApp:
    def __init__(self, mode=0, metric="lyapunov"):
        self.mode = mode  # 0: Angle Space, 1: Momentum Space
        self.metric = metric
        if self.metric == "lyapunov": self.metric_name = "Stability"
        elif self.metric == "poincare": self.metric_name = "Recurrence"
        else: self.metric_name = "Min Dist"
        
        # UI State
        self.show_animation = False
        self.ani = None
        self._busy = False
        self.zoom_timer = None
        
        # Initial Generation
        logger.info(f"Generating {self.metric_name} fractal for {'Angle' if mode==0 else 'Momentum'} space...")
        self.fractal, self.extent = pk.run_fractal_gen(mode=mode, t_max=DURATION, res=512, dt=DT, metric=metric)
        
        # Find Global Max
        self.max_val, self.max_ic = pk.find_global_max(self.fractal, self.extent, mode, DURATION, DT, metric=metric)
        
        x_min, x_max, y_min, y_max = self.extent
        self._last_zoom = [x_min, x_max, y_min, y_max]

        logger.info(f"Global Max found at {self.max_ic} with score {self.max_val:.4f}")

        
        # Setup UI Layout
        self.fig, (self.ax_map, self.ax_phys, self.ax_phase) = plt.subplots(1, 3, figsize=(18, 6))
        plt.subplots_adjust(bottom=0.15) # Make room for controls

        # --- Fractal Map ---
        self.cmap = ''
        if self.metric == "lyapunov": self.cmap = 'magma_r'
        elif self.metric == "poincare": self.cmap = 'magma'
        else: self.cmap = 'twilight_shifted_r'
        
        self.im = self.ax_map.imshow(self.fractal, extent=self.extent, origin='lower', cmap=self.cmap)
        self.ax_map.set_title(f"{self.metric_name} Map\n(Max Score: {self.max_val:.2f})")
        self.ax_map.set_xlabel("Theta 1" if mode==0 else "Omega 1")
        self.ax_map.set_ylabel("Theta 2" if mode==0 else "Omega 2")
        self.click_point, = self.ax_map.plot([], [], 'ro', markersize=8, markeredgecolor='white', zorder=10)
        
        # --- Real Space Pendulum ---
        self.ax_phys.set_xlim(-2.2, 2.2); self.ax_phys.set_ylim(-2.2, 2.2)
        self.ax_phys.set_aspect('equal')
        self.ax_phys.set_title("Real Space Pendulum")
        self.line, = self.ax_phys.plot([], [], 'o-', lw=3, color='#2c3e50')
        self.trace, = self.ax_phys.plot([], [], '-', lw=1, alpha=0.3, color='#3498db')
        
        # --- Phase Space ---
        self.ax_phase.set_title("Angle Space (Theta 1 vs Theta 2)")
        self.phase_trace, = self.ax_phase.plot([], [], '-', lw=1, color='#8e44ad')
        self.phase_dot, = self.ax_phase.plot([], [], 'ko', markersize=5)
        
        # Interactive Controls
        axbox1 = plt.axes([0.10, 0.05, 0.08, 0.04])
        axbox2 = plt.axes([0.20, 0.05, 0.08, 0.04])
        ax_toggle = plt.axes([0.30, 0.05, 0.12, 0.04])
        ax_max = plt.axes([0.44, 0.05, 0.12, 0.04])
        ax_cmap = plt.axes([0.58, 0.05, 0.12, 0.04])
        
        self.text_box1 = TextBox(axbox1, ("Th1" if mode == 0 else "W1") + ": ", initial="0.0")
        self.text_box2 = TextBox(axbox2, ("Th2" if mode == 0 else "W2") + ": ", initial="0.0")
        
        self.btn_toggle = Button(ax_toggle, "Animate: OFF", color="#ecf0f1", hovercolor="#bdc3c7")
        self.btn_max = Button(ax_max, "Best Point", color="#ecf0f1", hovercolor="#bdc3c7")
        self.btn_cmap = Button(ax_cmap, "Invert Color", color="#ecf0f1", hovercolor="#bdc3c7")
        
        # Connect Events
        self.text_box1.on_submit(self.on_submit)
        self.text_box2.on_submit(self.on_submit)
        self.btn_toggle.on_clicked(self.toggle_animation)
        self.btn_max.on_clicked(self.on_show_max)
        self.btn_cmap.on_clicked(self.toggle_cmap)
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.ax_map.callbacks.connect('xlim_changed', self.on_zoom)
        self.ax_map.callbacks.connect('ylim_changed', self.on_zoom)
        
        self.ax_map.callbacks.connect('ylim_changed', self.on_zoom)
        

        logger.info("UI Initialized.")
        # Simulation of the best point
        self.on_show_max(None)
        plt.show()

    def toggle_cmap(self, event):
        if self.cmap.endswith("_r"):
            self.cmap = self.cmap[:-2]
        else:
            self.cmap = self.cmap + "_r"
        self.im.set_cmap(self.cmap)
        self.fig.canvas.draw_idle()

    def _update_toggle_button(self):
        if self.show_animation:
            self.btn_toggle.label.set_text("Animate: ON")
            self.btn_toggle.color = "#2ecc71"
        else:
            self.btn_toggle.label.set_text("Animate: OFF")
            self.btn_toggle.color = "#ecf0f1"
        self.fig.canvas.draw_idle()

    def toggle_animation(self, event):
        # Update State
        self.show_animation = not self.show_animation
        
        self._update_toggle_button()
        
        # Switch Mode Without Recalculating Physics
        if self.show_animation:
            if not hasattr(self, 'th1') or self.th1 is None:
                logger.warning("No trajectory data available to animate.")
                self.show_animation = False
                self._update_toggle_button()
                return

            # Start Animation
            self._setup_artists(animated=True)
            
            skip = max(1, len(self.th1) // 1000) 
            
            def update(frame_idx):
                # Update data for current frame
                self.line.set_data([0, self.x1[frame_idx], self.x2[frame_idx]], [0, self.y1[frame_idx], self.y2[frame_idx]])
                
                trace_start = max(0, frame_idx - 500)
                self.trace.set_data(self.x2[trace_start:frame_idx:2], self.y2[trace_start:frame_idx:2])
                
                self.phase_trace.set_data(self.th1[:frame_idx:5], self.th2[:frame_idx:5])
                self.phase_dot.set_data([self.th1[frame_idx]], [self.th2[frame_idx]])
                
                self.phase_dot.set_data([self.th1[frame_idx]], [self.th2[frame_idx]])
                
                return self.line, self.trace, self.phase_trace, self.phase_dot

            # Create animation
            self.ani = FuncAnimation(self.fig, update, frames=range(0, len(self.th1), skip), 
                                     interval=20, blit=True, repeat=False)
            self.fig.canvas.draw_idle()
            
        else:
            # Stop Animation (Static Mode)
            if self.ani:
                if self.ani.event_source:
                    self.ani.event_source.stop()
                self.ani = None
            
            self._setup_artists(animated=False)
            
            # Show FULL path
            self.line.set_data([0, self.x1[-1], self.x2[-1]], [0, self.y1[-1], self.y2[-1]])
            self.trace.set_data(self.x2, self.y2)
            self.phase_trace.set_data(self.th1, self.th2)
            self.phase_dot.set_data([self.th1[-1]], [self.th2[-1]])
            
            self.fig.canvas.draw_idle()

    def _setup_artists(self, animated):
        """Helper to toggle Matplotlib animation state"""
        self.line.set_animated(animated)
        self.trace.set_animated(animated)
        self.phase_trace.set_animated(animated)
        self.phase_trace.set_animated(animated)
        self.phase_dot.set_animated(animated)

    def on_zoom(self, event_ax):
        if event_ax != self.ax_map: return
        
        # 200ms debounce
        if self.zoom_timer:
            self.zoom_timer.stop()
        self.zoom_timer = self.fig.canvas.new_timer(interval=200)
        self.zoom_timer.add_callback(self._perform_zoom)
        self.zoom_timer.start()

    def _perform_zoom(self):
        if self.zoom_timer:
            self.zoom_timer.stop()
            self.zoom_timer = None
            
        x_min, x_max = self.ax_map.get_xlim()
        y_min, y_max = self.ax_map.get_ylim()

        if hasattr(self, '_last_zoom') and np.allclose(self._last_zoom, [x_min, x_max, y_min, y_max], rtol=1e-3):
            return
        self._last_zoom = [x_min, x_max, y_min, y_max]

        logger.debug(f"Zoom detected. Re-rendering fractal...")
        self.fractal, self.extent = pk.run_fractal_gen(
            mode=self.mode, t_max=DURATION, res=512, dt=DT, 
            bounds=(x_min, x_max, y_min, y_max), metric=self.metric
        )
        
        self.max_val, self.max_ic = pk.find_global_max(self.fractal, self.extent, self.mode, DURATION, DT, metric=self.metric)
        
        self.im.set_data(self.fractal)
        self.im.set_extent(self.extent)
        self.im.set_clim(vmin=np.min(self.fractal), vmax=np.max(self.fractal))
        self.ax_map.set_title(f"{self.metric_name} Map\n(Max Score: {self.max_val:.2f})")
        self.fig.canvas.draw_idle()

    def on_show_max(self, event):
        logger.info(f"Best point in view: {self.max_ic} (Score: {self.max_val:.4f})")
        # Update Text Boxes
        self.text_box1.eventson = False
        self.text_box2.eventson = False
        if self.mode == 0:
            self.text_box1.set_val(f"{self.max_ic[0]:.4f}")
            self.text_box2.set_val(f"{self.max_ic[1]:.4f}")
        else:
            self.text_box1.set_val(f"{self.max_ic[2]:.4f}")
            self.text_box2.set_val(f"{self.max_ic[3]:.4f}")
        self.text_box1.eventson = True
        self.text_box2.eventson = True
        
        self.simulate_and_animate(self.max_ic)

    def on_click(self, event):
        if self.fig.canvas.manager.toolbar.mode != '' or event.inaxes != self.ax_map: return
        
        self.text_box1.eventson = False
        self.text_box2.eventson = False
        self.text_box1.set_val(f"{event.xdata:.4f}")
        self.text_box2.set_val(f"{event.ydata:.4f}")
        self.text_box1.eventson = True
        self.text_box2.eventson = True
        
        if self.mode == 0:
            ic = [event.xdata, event.ydata, 0.0, 0.0]
        else:
            ic = [0.0, 0.0, event.xdata, event.ydata]
            
        self.simulate_and_animate(ic)

    def on_submit(self, text):
        try:
            v1 = float(self.text_box1.text)
            v2 = float(self.text_box2.text)
            if self.mode == 0:
                ic = [v1, v2, 0.0, 0.0]
            else:
                ic = [0.0, 0.0, v1, v2]
            self.simulate_and_animate(ic)
        except ValueError:
            logger.error("Invalid input.")

    def simulate_and_animate(self, ic):
        if self._busy: return
        self._busy = True
        
        if self.ani and self.ani.event_source:
            self.ani.event_source.stop()
        self.ani = None
        
        try:
            self.current_ic = ic
            
            # Update Map Marker
            if self.mode == 0:
                self.click_point.set_data([ic[0]], [ic[1]])
            else:
                self.click_point.set_data([ic[2]], [ic[3]])

            # 1. Run Physics
            self.th1, self.th2, self.w1, self.w2 = pk.run_trajectory(ic, DURATION, DT)
            
            # Check for NaN/Inf in trajectory
            if not np.all(np.isfinite(self.th1)):
                logger.error(f"Simulation Diverged (NaN detected) for point: {ic}")
                self.th1 = self.th2 = self.w1 = self.w2 = None
                return

            # 2. Compute Score
            _, self.recurrence_scores = pk.compute_recurrence_score_over_time(ic, DURATION, DT)
            final_score = self.recurrence_scores[-1]
            logger.info(f">>> Trajectory Final Score: {final_score:.4f} for point: {ic}")

            # 3. Prepare Data
            self.t_eval = np.linspace(0, DURATION, len(self.th1))
            self.x1 = pk.L1 * np.sin(self.th1); self.y1 = -pk.L1 * np.cos(self.th1)
            self.x2 = self.x1 + pk.L2 * np.sin(self.th2); self.y2 = self.y1 - pk.L2 * np.cos(self.th2)
            
            # 4. Set Limits
            lim = 2.2
            self.ax_phys.set_xlim(-lim, lim); self.ax_phys.set_ylim(-lim, lim)
            self.ax_phase.set_xlim(np.min(self.th1)-0.1, np.max(self.th1)+0.1)
            self.ax_phase.set_ylim(np.min(self.th2)-0.1, np.max(self.th2)+0.1)

            # 5. Trigger Initial Draw
            self.show_animation = not self.show_animation 
            self.toggle_animation(None)
                
        except Exception as e:
            logger.error(f"Simulation Error: {e}", exc_info=True)
        finally:
            self._busy = False

if __name__ == "__main__":
    print("Select Mode and Metric:")
    print("1: Angle Space (Lyapunov)")
    print("2: Momentum Space (Lyapunov)")
    print("3: Angle Space (Poincare)")
    print("4: Momentum Space (Poincare)")
    print("5: Angle Space (Min Dist - Hybrid)")
    print("6: Momentum Space (Min Dist - Hybrid)")
    
    choice = input("Select option (1-6): ")
    
    if choice == "1":
        DoublePendulumApp(mode=0, metric="lyapunov")
    elif choice == "2":
        DoublePendulumApp(mode=1, metric="lyapunov")
    elif choice == "3":
        DoublePendulumApp(mode=0, metric="poincare")
    elif choice == "4":
        DoublePendulumApp(mode=1, metric="poincare")
    elif choice == "5":
        DoublePendulumApp(mode=0, metric="mindist")
    elif choice == "6":
        DoublePendulumApp(mode=1, metric="mindist")
    else:
        print("Invalid selection.")