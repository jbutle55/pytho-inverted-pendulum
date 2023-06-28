from inverted_pendulum import CartPendSys
from simulator import Simulator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('macosx')


def main():
    sys = CartPendSys()
    sim = Simulator(sys, display_plots=True)

    sim.run()

    print('Running Simulator...')
    sim.display_simulation(sim.sys.cart_x_tracking)
    print('Simulation Complete.')

    if sim.display_plots:
        sim.sys.plot_sim_results()

    return


if __name__ == '__main__':
    main()
