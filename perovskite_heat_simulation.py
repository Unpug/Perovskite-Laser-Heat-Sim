#!/usr/bin/env python3
"""
Finite-difference simulation of transient heat conduction with a moving Gaussian heat source.

This script is adapted from the code and model described in AJ's research report on
ultrafast pulsed laser-induced degradation in multilayer perovskite photovoltaics.

Model:
    rho * c_p * dT/dt = k * ∇²T + Q

The simulation is performed on a 2D square domain using an explicit finite-difference scheme
with fixed-temperature (Dirichlet) boundary conditions.

Notes:
- The report mixes micrometer-scale geometry with SI material parameters. This script keeps
  the geometry in micrometers for readability, then converts all spatial quantities to meters
  internally so the PDE update is dimensionally consistent.
- The Gaussian width and scan speed are interpreted in micrometer-based geometry, then
  converted to SI units before use.

Dependencies:
    numpy, matplotlib

Examples:
    python perovskite_heat_simulation.py
    python perovskite_heat_simulation.py --no-animate --output-prefix run1
    python perovskite_heat_simulation.py --nx 150 --ny 150 --t-total 5e-4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


@dataclass
class SimulationConfig:
    # Domain size in micrometers
    lx_um: float = 1000.0
    ly_um: float = 1000.0

    # Grid resolution
    nx: int = 100
    ny: int = 100

    # Material properties (SI)
    density: float = 7800.0            # kg/m^3
    specific_heat: float = 500.0       # J/(kg*K)
    conductivity: float = 50.0         # W/(m*K)

    # Laser / source parameters
    q0: float = 2.0e12                 # W/m^3
    sigma_um: float = 100.0            # Gaussian std dev in micrometers
    speed_um_per_s: float = 1.5e6      # scan speed in micrometers / second

    # Time integration
    t_total: float = 1.0e-3            # total time in seconds
    cfl_factor: float = 0.20           # explicit scheme safety factor
    ambient_temperature: float = 300.0 # K

    # Animation / plotting
    steps_per_frame: int = 10
    interval_ms: int = 50


class HeatConductionSimulation:
    def __init__(self, config: SimulationConfig):
        self.cfg = config

        # Geometry in micrometers for presentation
        self.x_um = np.linspace(0.0, self.cfg.lx_um, self.cfg.nx)
        self.y_um = np.linspace(0.0, self.cfg.ly_um, self.cfg.ny)
        self.X_um, self.Y_um = np.meshgrid(self.x_um, self.y_um)

        # Convert domain to SI units for PDE calculations
        self.x_m = self.x_um * 1e-6
        self.y_m = self.y_um * 1e-6
        self.X_m, self.Y_m = np.meshgrid(self.x_m, self.y_m)
        self.dx_m = self.x_m[1] - self.x_m[0]
        self.dy_m = self.y_m[1] - self.y_m[0]

        # Material diffusivity
        self.alpha = self.cfg.conductivity / (self.cfg.density * self.cfg.specific_heat)

        # Stable explicit time step for 2D heat equation
        denom = (1.0 / self.dx_m**2) + (1.0 / self.dy_m**2)
        self.dt = self.cfg.cfl_factor / (2.0 * self.alpha * denom)
        self.n_steps = max(1, int(np.ceil(self.cfg.t_total / self.dt)))

        # Temperature field and history
        self.t = 0.0
        self.T = np.full((self.cfg.ny, self.cfg.nx), self.cfg.ambient_temperature, dtype=float)
        self.times_us: list[float] = []
        self.max_temps: list[float] = []
        self.avg_temps: list[float] = []

        # Precompute SI source parameters
        self.sigma_m = self.cfg.sigma_um * 1e-6
        self.speed_m_per_s = self.cfg.speed_um_per_s * 1e-6
        self.y_center_m = 0.5 * self.cfg.ly_um * 1e-6

    def heat_source(self, t: float) -> np.ndarray:
        """Moving Gaussian volumetric heat source Q(x, y, t) in W/m^3."""
        x_center_m = self.speed_m_per_s * t
        return self.cfg.q0 * np.exp(
            -(
                (self.X_m[1:-1, 1:-1] - x_center_m) ** 2
                + (self.Y_m[1:-1, 1:-1] - self.y_center_m) ** 2
            ) / (2.0 * self.sigma_m**2)
        )

    def step(self) -> None:
        """Advance the solution by one explicit time step."""
        T_new = self.T.copy()

        laplacian = (
            (self.T[1:-1, 2:] - 2.0 * self.T[1:-1, 1:-1] + self.T[1:-1, :-2]) / self.dx_m**2
            + (self.T[2:, 1:-1] - 2.0 * self.T[1:-1, 1:-1] + self.T[:-2, 1:-1]) / self.dy_m**2
        )

        source_term = self.heat_source(self.t) / (self.cfg.density * self.cfg.specific_heat)

        T_new[1:-1, 1:-1] = self.T[1:-1, 1:-1] + self.dt * (
            self.alpha * laplacian + source_term
        )

        # Dirichlet boundary conditions: fixed ambient temperature on all edges.
        T_new[0, :] = self.cfg.ambient_temperature
        T_new[-1, :] = self.cfg.ambient_temperature
        T_new[:, 0] = self.cfg.ambient_temperature
        T_new[:, -1] = self.cfg.ambient_temperature

        self.T = T_new
        self.t += self.dt
        self._record_observables()

    def run(self) -> None:
        for _ in range(self.n_steps):
            self.step()

    def _record_observables(self) -> None:
        self.times_us.append(self.t * 1e6)
        self.max_temps.append(float(np.max(self.T)))
        self.avg_temps.append(float(np.mean(self.T)))

    def save_history_csv(self, path: str | Path) -> None:
        data = np.column_stack([self.times_us, self.max_temps, self.avg_temps])
        header = "time_us,max_temperature_K,avg_temperature_K"
        np.savetxt(path, data, delimiter=",", header=header, comments="")

    def save_final_field_png(self, path: str | Path) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            self.T,
            extent=[0, self.cfg.lx_um, 0, self.cfg.ly_um],
            origin="lower",
            aspect="auto",
        )
        ax.set_title("Final Temperature Field")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Temperature (K)")
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def animate(self) -> None:
        fig, (ax_field, ax_max, ax_avg) = plt.subplots(1, 3, figsize=(18, 6))

        vmax = max(self.cfg.ambient_temperature + 1.0, self.cfg.ambient_temperature + 50.0)
        im = ax_field.imshow(
            self.T,
            extent=[0, self.cfg.lx_um, 0, self.cfg.ly_um],
            origin="lower",
            cmap="inferno",
            vmin=self.cfg.ambient_temperature,
            vmax=vmax,
            animated=True,
            aspect="auto",
        )
        ax_field.set_title("Temperature Distribution")
        ax_field.set_xlabel("x (µm)")
        ax_field.set_ylabel("y (µm)")
        cbar = fig.colorbar(im, ax=ax_field)
        cbar.set_label("Temperature (K)")

        line_max, = ax_max.plot([], [], lw=2)
        ax_max.set_title("Max Temperature vs Time")
        ax_max.set_xlabel("Time (µs)")
        ax_max.set_ylabel("Max Temperature (K)")
        ax_max.set_xlim(0.0, self.cfg.t_total * 1e6)
        ax_max.set_ylim(self.cfg.ambient_temperature, self.cfg.ambient_temperature + 100.0)
        time_text_max = ax_max.text(0.05, 0.92, "", transform=ax_max.transAxes)

        line_avg, = ax_avg.plot([], [], lw=2)
        ax_avg.set_title("Average Temperature vs Time")
        ax_avg.set_xlabel("Time (µs)")
        ax_avg.set_ylabel("Average Temperature (K)")
        ax_avg.set_xlim(0.0, self.cfg.t_total * 1e6)
        ax_avg.set_ylim(self.cfg.ambient_temperature, self.cfg.ambient_temperature + 30.0)
        time_text_avg = ax_avg.text(0.05, 0.92, "", transform=ax_avg.transAxes)

        def init():
            im.set_data(self.T)
            line_max.set_data([], [])
            line_avg.set_data([], [])
            time_text_max.set_text("")
            time_text_avg.set_text("")
            return im, line_max, line_avg, time_text_max, time_text_avg

        frames = max(1, int(np.ceil(self.n_steps / self.cfg.steps_per_frame)))

        def update(_frame_index: int):
            for _ in range(self.cfg.steps_per_frame):
                if self.t >= self.cfg.t_total:
                    break
                self.step()

            im.set_data(self.T)
            line_max.set_data(self.times_us, self.max_temps)
            line_avg.set_data(self.times_us, self.avg_temps)
            time_label = f"t = {self.t * 1e6:.2f} µs"
            time_text_max.set_text(time_label)
            time_text_avg.set_text(time_label)
            return im, line_max, line_avg, time_text_max, time_text_avg

        animation.FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=frames,
            interval=self.cfg.interval_ms,
            blit=True,
            repeat=False,
        )

        fig.tight_layout()
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D transient heat conduction simulation")
    parser.add_argument("--nx", type=int, default=100, help="Number of grid points in x")
    parser.add_argument("--ny", type=int, default=100, help="Number of grid points in y")
    parser.add_argument("--lx-um", type=float, default=1000.0, help="Domain length in x (µm)")
    parser.add_argument("--ly-um", type=float, default=1000.0, help="Domain length in y (µm)")
    parser.add_argument("--t-total", type=float, default=1.0e-3, help="Total simulation time (s)")
    parser.add_argument("--q0", type=float, default=2.0e12, help="Heat source amplitude (W/m^3)")
    parser.add_argument("--sigma-um", type=float, default=100.0, help="Gaussian width (µm)")
    parser.add_argument("--speed-um-per-s", type=float, default=1.5e6, help="Source speed (µm/s)")
    parser.add_argument("--no-animate", action="store_true", help="Run without live animation")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="simulation_output",
        help="Prefix for saved CSV and PNG outputs in non-animated mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimulationConfig(
        lx_um=args.lx_um,
        ly_um=args.ly_um,
        nx=args.nx,
        ny=args.ny,
        q0=args.q0,
        sigma_um=args.sigma_um,
        speed_um_per_s=args.speed_um_per_s,
        t_total=args.t_total,
    )

    sim = HeatConductionSimulation(cfg)

    if args.no_animate:
        sim.run()
        prefix = Path(args.output_prefix)
        csv_path = prefix.with_suffix(".csv")
        png_path = prefix.with_suffix(".png")
        sim.save_history_csv(csv_path)
        sim.save_final_field_png(png_path)
        print("Simulation completed.")
        print(f"Time step dt = {sim.dt:.3e} s")
        print(f"Number of steps = {sim.n_steps}")
        print(f"Final max temperature = {np.max(sim.T):.3f} K")
        print(f"Saved history to: {csv_path}")
        print(f"Saved final field image to: {png_path}")
    else:
        sim.animate()


if __name__ == "__main__":
    main()
