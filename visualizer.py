import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import physics_kernel as pk


# Scipy precision parameters
SOLVER_METHOD = 'DOP853'
TOLERANCE = 1e-10
DT = 0.01 
DURATION = 60*1


class DoublePendulumApp:
    def __init__(self, mode=0, metric="lyapunov"):
        self.mode = mode # 0: Angle, 1: Momentum
        self.metric = metric
        metric_name = "Stability" if metric == "lyapunov" else "poincare"
        print(f"Generating {metric_name} fractal for {'Angle' if mode==0 else 'Momentum'} space...")
        self.fractal, self.extent = pk.run_fractal_gen(mode=mode, t_max=DURATION, res=512, dt=DT, metric=metric)

        # UI Setup: 3 Subplots
        self.fig, (self.ax_map, self.ax_phys, self.ax_phase) = plt.subplots(1, 3, figsize=(18, 6))

        # Plot fractal (Map)
        cmap = 'magma' if metric == "lyapunov" else 'twilight_shifted'
        self.im = self.ax_map.imshow(self.fractal, extent=self.extent, origin='lower', cmap=cmap)
        self.ax_map.set_title(f"{metric_name} Map\n(Zoom & Click to Explore)")
        self.click_point, = self.ax_map.plot([], [], 'ro', markersize=8, markeredgecolor='white', zorder=5)
        
        # Physical display
        self.ax_phys.set_xlim(-2.2, 2.2); self.ax_phys.set_ylim(-2.2, 2.2)
        self.ax_phys.set_aspect('equal')
        self.line, = self.ax_phys.plot([], [], 'o-', lw=3, color='#2c3e50')
        self.trace, = self.ax_phys.plot([], [], '-', lw=1, alpha=0.3, color='#3498db')
        self.ax_phys.set_title("Real Space Pendulum")
        
        # Phase Space Display
        self.ax_phase.set_xlim(-6*np.pi, 6*np.pi); self.ax_phase.set_ylim(-6*np.pi, 6*np.pi)
        self.ax_phase.set_title("Angle Space (theta_1 vs theta_2)")
        self.phase_trace, = self.ax_phase.plot([], [], '-', lw=1, color='#8e44ad')
        
        self.ani = None
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.ax_map.callbacks.connect('xlim_changed', self.on_zoom)
        self.ax_map.callbacks.connect('ylim_changed', self.on_zoom)
        
        # Add Text Boxes and Toggle Button
        from matplotlib.widgets import TextBox, Button
        axbox1 = plt.axes([0.1, 0.02, 0.08, 0.04])
        axbox2 = plt.axes([0.22, 0.02, 0.08, 0.04])
        ax_toggle = plt.axes([0.35, 0.015, 0.15, 0.05])
        
        self.text_box1 = TextBox(axbox1, ("theta1" if mode == 0 else "omega1") + ": ", initial="0.0")
        self.text_box2 = TextBox(axbox2, ("theta2" if mode == 0 else "omega2") + ": ", initial="0.0")
        
        self.show_animation = False
        self.btn_toggle = Button(ax_toggle, "Animate: OFF", color="#ecf0f1", hovercolor="#bdc3c7")
        self._set_toggle_style()
        
        self.text_box1.on_submit(self.on_submit); self.text_box2.on_submit(self.on_submit)
        self.btn_toggle.on_clicked(self.toggle_animation)
        
        print("UI Initialized. Zoom triggers high-res re-render.")
        plt.show()

    def _set_toggle_style(self):
        if self.show_animation:
            self.btn_toggle.label.set_text("Animate: ON")
            self.btn_toggle.color = "#2ecc71" # Green
        else:
            self.btn_toggle.label.set_text("Animate: OFF")
            self.btn_toggle.color = "#ecf0f1" # Light grey

    def toggle_animation(self, event):
        self.show_animation = not self.show_animation
        self._set_toggle_style()

        # If toggling ON: start animation or create it
        if self.show_animation:
            if getattr(self, 't_eval', None) is None:
                print("No simulation data to animate. Run a simulation first.")
            else:
                if self.ani and getattr(self.ani, 'event_source', None):
                    try:
                        self.ani.event_source.start()
                    except Exception:
                        pass
                else:
                    def update(i):
                        self.line.set_data([0, self.x1[i], self.x2[i]], [0, self.y1[i], self.y2[i]])
                        self.trace.set_data(self.x2[:i+1], self.y2[:i+1])
                        self.phase_trace.set_data(self.th1[:i+1], self.th2[:i+1])
                        return self.line, self.trace, self.phase_trace

                    self.line.set_animated(True)
                    self.trace.set_animated(True)
                    self.phase_trace.set_animated(True)
                    self.ani = FuncAnimation(self.fig, update, frames=len(self.t_eval),
                                             blit=True, interval=16, repeat=False)

        # If toggling OFF: stop animation and show full path
        else:
            if self.ani and getattr(self.ani, 'event_source', None):
                try:
                    self.ani.event_source.stop()
                except Exception:
                    pass
                self.ani = None

            if getattr(self, 'th1', None) is not None:
                max_points = 10000
                step = max(1, len(self.th1) // max_points)

                self.line.set_animated(False)
                self.trace.set_animated(False)
                self.phase_trace.set_animated(False)

                self.line.set_data([0, self.x1[-1], self.x2[-1]], [0, self.y1[-1], self.y2[-1]])
                self.trace.set_data(self.x2[::step], self.y2[::step])
                self.phase_trace.set_data(self.th1[::step], self.th2[::step])

                self.ax_phys.relim(); self.ax_phys.autoscale_view()
                self.ax_phase.relim(); self.ax_phase.autoscale_view()
                self.fig.canvas.draw_idle()

        self.fig.canvas.draw_idle()

    def on_zoom(self, event_ax):
        if event_ax != self.ax_map: return
        x_min, x_max = self.ax_map.get_xlim()
        y_min, y_max = self.ax_map.get_ylim()

        # Avoid recursive calls or tiny updates
        if hasattr(self, '_last_zoom') and np.allclose(self._last_zoom, [x_min, x_max, y_min, y_max], rtol=1e-3):
            return
        self._last_zoom = [x_min, x_max, y_min, y_max]

        metric_name = "Stability" if self.metric == "lyapunov" else "poincare"
        print(f"Re-rendering {metric_name} fractal for zoomed region: x[{x_min:.2f}, {x_max:.2f}], y[{y_min:.2f}, {y_max:.2f}]")
        self.fractal, self.extent = pk.run_fractal_gen(mode=self.mode, t_max=DURATION, res=512, dt=DT, bounds=(x_min, x_max, y_min, y_max), metric=self.metric)
        self.im.set_data(self.fractal)
        self.im.set_extent(self.extent)
        self.fig.canvas.draw_idle()

    def on_submit(self, text):
        try:
            v1, v2 = float(self.text_box1.text), float(self.text_box2.text)
            self.click_point.set_data([v1], [v2])
            ic = [v1, v2, 0, 0] if self.mode == 0 else [0, 0, v1, v2]
            self.simulate_and_animate(ic)
            self.fig.canvas.draw_idle()
        except ValueError: print("Invalid input")

    def on_click(self, event):
        if self.fig.canvas.manager.toolbar.mode != '' or event.inaxes != self.ax_map: return
        print(f"Click detected at {event.xdata:.4f}, {event.ydata:.4f}")
        
        # Block text box events to prevent triple-simulation (since set_val triggers on_submit)
        self.text_box1.eventson = False
        self.text_box2.eventson = False
        self.text_box1.set_val(f"{event.xdata:.4f}")
        self.text_box2.set_val(f"{event.ydata:.4f}")
        self.text_box1.eventson = True
        self.text_box2.eventson = True
        
        self.click_point.set_data([event.xdata], [event.ydata])
        ic = [event.xdata, event.ydata, 0.0, 0.0] if self.mode == 0 else [0.0, 0.0, event.xdata, event.ydata]
        self.simulate_and_animate(ic)

    def simulate_and_animate(self, ic):
        if hasattr(self, '_busy') and self._busy: return
        self._busy = True
        try:
            if self.ani and self.ani.event_source:
                self.ani.event_source.stop()
            self.ani = None

            print(f"Integrating for {DURATION}s...")
            import time
            start_t = time.time()
            
            t_eval = np.arange(int(DURATION / DT))

            # Run the trajectory and cache results for toggling 
            th1, th2, w1, w2 = pk.run_trajectory(ic, DURATION, DT)

            print(f"Sim finished in {time.time()-start_t:.2f}s.")

            th1, th2 = th1, th2
            x1, y1 = pk.L1 * np.sin(th1), -pk.L1 * np.cos(th1)
            x2, y2 = x1 + pk.L2 * np.sin(th2), y1 - pk.L2 * np.cos(th2)

            # Cache simulation arrays so toggle can start/stop animation later
            self.t_eval = t_eval
            self.th1 = th1
            self.th2 = th2
            self.w1 = w1
            self.w2 = w2
            self.x1 = x1
            self.y1 = y1
            self.x2 = x2
            self.y2 = y2
            
            # Calculate axis limits with padding
            def get_limits(data, padding=0.1):
                dmin, dmax = np.min(data), np.max(data)
                margin = (dmax - dmin) * padding
                if margin < 0.01:  
                    margin = 0.01
                return dmin - margin, dmax + margin

            th1_min, th1_max = get_limits(th1)
            th2_min, th2_max = get_limits(th2)
            self.ax_phase.set_xlim(th1_min, th1_max)
            self.ax_phase.set_ylim(th2_min, th2_max)
            
            x2_min, x2_max = get_limits(x2)
            y2_min, y2_max = get_limits(y2)

            phys_lim = max(2.2, abs(x2_min), abs(x2_max), abs(y2_min), abs(y2_max)) * 1.1
            self.ax_phys.set_xlim(-phys_lim, phys_lim)
            self.ax_phys.set_ylim(-phys_lim, phys_lim)

            start_t = time.time()
            if self.show_animation:
                def update(i):
                    self.line.set_data([0, self.x1[i], self.x2[i]], [0, self.y1[i], self.y2[i]])
                    self.trace.set_data(self.x2[:i+1], self.y2[:i+1])
                    self.phase_trace.set_data(self.th1[:i+1], self.th2[:i+1])
                    return self.line, self.trace, self.phase_trace
                self.line.set_animated(True)
                self.trace.set_animated(True)
                self.phase_trace.set_animated(True)
                self.ani = FuncAnimation(self.fig, update, frames=len(self.t_eval), 
                                         blit=True, interval=16, repeat=False)
            else:
                self.line.set_animated(False)
                self.trace.set_animated(False)
                self.phase_trace.set_animated(False)
                
                # Downsample for fast plotting 
                max_points = 10000
                step = max(1, len(th1) // max_points)
                
                self.line.set_data([0, x1[-1], x2[-1]], [0, y1[-1], y2[-1]])
                self.trace.set_data(x2[::step], y2[::step])
                self.phase_trace.set_data(th1[::step], th2[::step])
                
                self.ax_phys.relim(); self.ax_phys.autoscale_view()
                self.ax_phase.relim(); self.ax_phase.autoscale_view()
                self.fig.canvas.draw()
            
            self.fig.canvas.draw_idle()
            print(f"Animation finished in {time.time()-start_t:.2f}s.")
        finally:
            self._busy = False

if __name__ == "__main__":
    print("Select Mode and Metric:")
    print("1: Angle Space (Lyapunov)")
    print("2: Momentum Space (Lyapunov)")
    print("3: Angle Space (Poincare)")
    print("4: Momentum Space (Poincare)")
    choice = input("Select option (1-4): ")
    if choice == "1":
        DoublePendulumApp(mode=0, metric="lyapunov")
    elif choice == "2":
        DoublePendulumApp(mode=1, metric="lyapunov")
    elif choice == "3":
        DoublePendulumApp(mode=0, metric="poincare")
    elif choice == "4":
        DoublePendulumApp(mode=1, metric="poincare")
    else:
        print("Invalid selection.")
