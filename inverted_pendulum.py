import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import StateSpace
from simulator import TIME_STEP

# CONSTANTS
g = 9.81
l_m = 0.3
m_Kg = 0.2
M_Kg = 0.5
I = 0.006
b_fric = 0.9


class CartPendSys:
    def __init__(self, theta=0.05 * np.pi):
        # State Space Y
        # Y = [x, theta, x_dot, theta_dot]
        # Y_dot = Ax + Bu
        den = I * (M_Kg + m_Kg) + m_Kg * M_Kg * l_m**2
        A_1 = (m_Kg**2 * g * l_m**2) / den
        A_2 = -(I + m_Kg * l_m**2) * b_fric / den
        A_3 = m_Kg * g * l_m * (M_Kg + m_Kg) / den
        A_4 = -(m_Kg * l_m * b_fric) / den
        A = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, A_1, A_2, 0],
                      [0, A_3, A_4, 0]])
        B_1 = (I + m_Kg * l_m**2) / den
        B_2 = -m_Kg * l_m / den
        B = np.array([[0],
                      [0],
                      [B_1],
                      [B_2]])
        C = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
        D = np.array([[0], [0]])

        ss = StateSpace(A, B, C, D)

        self.ss_discrete = ss.to_discrete(TIME_STEP)
        self.A_discrete = self.ss_discrete.A
        self.B_discrete = self.ss_discrete.B

        # States: [x, theta, x_dot, theta_dot]
        # **2pi theta is straight down**
        '''
             0
        3pi/2    pi/2
            pi
        '''
        self.states = np.array([[0], [theta], [0], [0]])
        self.previous_states = self.states
        self.pend_target_rads = 0.

        # Track stats for plotting
        self.K_tracking = []
        self.integral_tracking = []
        self.theta_tracking = []
        self.cart_x_tracking = []
        self.error_tracking = []

    def get_error(self):
        error = self.pend_target_rads - self.previous_states[1]
        return error

    def update_states(self, K):
        states = np.matmul(self.A_discrete, self.previous_states) + self.B_discrete * K

        # Ensure pendulum limits not exceeded
        states[0] = np.clip(states[0], -5, 5)
        states[1] = states[1] % (2*np.pi)
        if states[1] > np.pi:
            states[1] = -2 * np.pi + states[1]

        self.previous_states = states
        return

    def update_tracked_states(self, K, integral, error):
        self.K_tracking.append(K)
        self.integral_tracking.append(integral)
        self.theta_tracking.append(self.previous_states[1][0])
        self.cart_x_tracking.append(self.previous_states[0][0])
        self.error_tracking.append(error)
        return

    def plot_sim_results(self):
        for item, title in zip([self.K_tracking, self.integral_tracking, self.theta_tracking, self.cart_x_tracking, self.error_tracking],
                               ['K', 'Integral', 'theta', 'Cart X', 'Error']):
            plt.figure()
            plt.plot(range(len(item)-1), item[1:])
            plt.title(title)

        plt.show()
        return
