import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
import numpy as np

# GLOBAL PARAMS
TIMER = 0
SET_POINTS = 10000
TIME_STEP = 0.001


class Simulator:
    def __init__(self, system, num_steps=SET_POINTS, display_plots=False):
        self.sim = True
        self.sys = system
        self.total_steps = num_steps
        self.display_plots = display_plots

        # Simulation Animation Setup
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, autoscale_on=False, xlim=(-6.5, 6.5), ylim=(-1, 2))
        self.ax.set_aspect('equal')
        self.ax.grid()
        self.xdata, self.ydata = [], []
        self.pend_xdata, self.pend_ydata = [], []
        self.time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)
        self.cart_patch = self.ax.add_patch(Rectangle((0, 0), 0, 0, linewidth=1, edgecolor='k', facecolor='g'))
        self.pend_patch = self.ax.add_patch(ConnectionPatch((0, 0), (0, 0), 'data', arrowstyle='-'))
        self.ln, = self.ax.plot([], [], 'ro')
        self.cart_width = 0.7
        self.cart_height = 0.2

        return

    def run(self):
        num_steps = 0
        integral = 0

        self.sys.update_states(K=integral)
        previous_error = self.sys.get_error()

        while num_steps <= self.total_steps:
            num_steps += 1
            error = self.sys.get_error()

            K, integral = self.get_pid_controls(previous_error, error, integral)

            self.sys.update_states(K=K)
            self.sys.update_tracked_states(K=K, integral=integral, error=error)

            previous_error = error

        return

    def get_pid_controls(self, prev_error, error, integral):

        # PID Coefficients
        Kp = -10
        Kd = -10
        Ki = -5

        derivative = (error - prev_error) / TIME_STEP
        integral += error * TIME_STEP
        K = Kp*error + Kd*derivative + Ki*integral

        return K, integral

    def sim_init_fn(self,):
        self.ln.set_data([], [])
        self.time_text.set_text('')
        self.cart_patch.set_xy((-self.cart_width / 2, -self.cart_height / 2))
        self.cart_patch.set_width(self.cart_width)
        self.cart_patch.set_height(self.cart_height)
        self.pend_patch.xy1 = self.cart_patch.get_center()

        # Set solution data
        self.xdata = self.sys.cart_x_tracking
        self.pend_xdata = np.sin(self.sys.theta_tracking) * 0.9
        self.pend_ydata = np.cos(self.sys.theta_tracking) * 0.9
        # Offset values with cart x values
        self.pend_xdata = self.pend_xdata + self.xdata

        self.pend_patch.xy2 = self.pend_xdata, self.pend_ydata

        return self.ln, self.time_text, self.cart_patch, self.pend_patch

    def sim_update_fn(self, frame_idx):
        thisx = [self.xdata[frame_idx], self.pend_xdata[frame_idx]]
        thisy = [0, self.pend_ydata[frame_idx]]
        self.ln.set_data(thisx, thisy)
        self.time_text.set_text(f'idx: {frame_idx} -- Theta: {self.sys.theta_tracking[frame_idx]:.2f} -- Cart X: {self.sys.cart_x_tracking[frame_idx]:.2f}')
        self.cart_patch.set_x(self.xdata[frame_idx] - self.cart_width / 2)
        self.pend_patch.xy1 = self.cart_patch.get_center()
        self.pend_patch.xy2 = self.pend_xdata[frame_idx], self.pend_ydata[frame_idx]
        return self.ln, self.time_text, self.cart_patch, self.pend_patch

    def display_simulation(self, solution_data):
        xdata = solution_data
        ani = animation.FuncAnimation(self.fig, self.sim_update_fn,
                                      frames=np.arange(1, len(xdata)),
                                      interval=25, init_func=self.sim_init_fn, blit=True,
                                      repeat=False)
        plt.show()
        return

