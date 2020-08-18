from pydrake.examples.pendulum import (
    PendulumGeometry, PendulumInput, PendulumParams, PendulumPlant,
    PendulumState)
from pydrake.geometry import SceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import (DiagramBuilder, LeafSystem, BasicVector)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

plt.ion()


def pendulum_energy(pendulum_params, theta, theta_dot):
    m = pendulum_params.mass()
    l = pendulum_params.length()
    g = pendulum_params.gravity()
    return m * g * l * (1 - np.cos(theta)) + 0.5 * m * (l * theta_dot) ** 2


class LyapunovVisualizer(LeafSystem):
    def __init__(self, damping, theta0, thetadot0, save_folder=None):
        LeafSystem.__init__(self)
        self.DeclareVectorInputPort("state", BasicVector(2))
        self.save_folder = save_folder
        self.visualization_count = 0
        self.pendulum_params = PendulumParams()
        self.pendulum_params.set_damping(damping=damping)
        self.pendulum_states_energy = np.empty((3, 0))
        self.fig = plt.figure(figsize=(10, 4))
        self.lyapunov_ax = self.fig.add_subplot(122, projection='3d')
        theta_grid = np.arange(-np.pi/2, np.pi/2, 0.05)
        thetadot_grid = np.arange(-3, 3, 0.01)
        theta_grid, thetadot_grid = np.meshgrid(theta_grid, thetadot_grid)
        energy_grid = np.empty_like(theta_grid)
        for i in range(theta_grid.shape[0]):
            for j in range(theta_grid.shape[1]):
                energy_grid[i, j] = pendulum_energy(
                    self.pendulum_params, theta_grid[i, j],
                    thetadot_grid[i, j])
        self.lyapunov_ax.plot_surface(theta_grid, thetadot_grid, energy_grid)
        self.lyapunov_circle, = self.lyapunov_ax.plot(
            [theta0], [thetadot0], [pendulum_energy(
                self.pendulum_params, theta0, thetadot0)], marker='o',
            markersize=10, color='r')
        self.lyapunov_ax.set_xlabel(r"$\theta$")
        self.lyapunov_ax.set_ylabel(r"$\dot{\theta}$")
        self.lyapunov_ax.set_zlabel("V")
        self.lyapunov_ax.view_init(elev=5, azim=-41)

        self.pendulum_ax = self.fig.add_subplot(121)
        l = self.pendulum_params.length()
        self.pendulum_arm, = self.pendulum_ax.plot(
            np.array([0, l * np.sin(theta0)]),
            np.array([0, -l * np.cos(theta0)]), linewidth=5)
        self.pendulum_sphere, = self.pendulum_ax.plot(
            l*np.sin(theta0), -l*np.cos(theta0), marker='o', markersize=15)
        #self.pendulum_ax.axis('equal')
        self.pendulum_ax.set_xlim(-l, l)
        self.pendulum_ax.set_ylim(-1.1 * l, l)
        self.pendulum_title = self.pendulum_ax.set_title("t=0s")
        self.lyapunov_traj = None

        self.fig.canvas.draw()
        if self.save_folder is not None:
            if not os.path.isdir(self.save_folder):
                os.mkdir(self.save_folder)
            self.fig.savefig(f"{self.save_folder}/figure{self.visualization_count:03d}.png")

    def DoPublish(self, context, event):
        LeafSystem.DoPublish(self, context, event)
        pendulum_state = self.EvalVectorInput(context, 0)
        time = context.get_time()

        theta = pendulum_state.theta()
        thetadot = pendulum_state.thetadot()
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        energy = pendulum_energy(self.pendulum_params, theta, thetadot)
        self.pendulum_states_energy = np.hstack((
            self.pendulum_states_energy,
            np.array([theta, thetadot, energy]).reshape((3, 1))))

        l = self.pendulum_params.length()
        self.pendulum_arm.set_xdata(np.array([0, l * sin_theta]))
        self.pendulum_arm.set_ydata(np.array([0, -l * cos_theta]))
        self.pendulum_sphere.set_xdata(l * sin_theta)
        self.pendulum_sphere.set_ydata(-l * cos_theta)
        self.pendulum_title.set_text(f"t={time:.2f}s")

        self.lyapunov_circle.set_data([theta], [thetadot])
        self.lyapunov_circle.set_3d_properties([energy])

        if self.lyapunov_traj is None:
            self.lyapunov_traj, = self.lyapunov_ax.plot(
                self.pendulum_states_energy[0, :],
                self.pendulum_states_energy[1, :],
                self.pendulum_states_energy[2, :], 'r')
        else:
            self.lyapunov_traj.set_data(
                self.pendulum_states_energy[0, :],
                self.pendulum_states_energy[1, :])
            self.lyapunov_traj.set_3d_properties(
                self.pendulum_states_energy[2, :])

        self.fig.canvas.draw()
        self.visualization_count += 1
        if self.save_folder is not None:
            self.fig.savefig(f"{self.save_folder}/figure{self.visualization_count:03d}.png")


if __name__ == "__main__":
    theta0 = 0.4 * np.pi
    thetadot0 = 0.5
    builder = DiagramBuilder()
    pendulum = builder.AddSystem(PendulumPlant())
    damping = 0.7
    visualizer = builder.AddSystem(LyapunovVisualizer(damping, theta0, thetadot0, "/home/hongkaidai/Desktop/lyapunov"))
    builder.Connect(pendulum.get_state_output_port(), visualizer.get_input_port(0))
    builder.ExportInput(pendulum.get_input_port())

    diagram = builder.Build()

    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()

    pendulum_context = diagram.GetMutableSubsystemContext(pendulum, context)
    pendulum_context.get_mutable_continuous_state_vector().SetFromVector([theta0, thetadot0])
    pendulum_params = pendulum.get_mutable_parameters(context=pendulum_context)
    pendulum_params.set_damping(damping=damping)
    context.FixInputPort(0, [0])

    simulator.set_publish_every_time_step(True)
    simulator.AdvanceTo(4.)
