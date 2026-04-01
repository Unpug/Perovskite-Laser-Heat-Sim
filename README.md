# Perovskite Heat Conduction Simulation

A Python implementation of the finite-difference heat-conduction model described in the research report **“Ultrafast Pulsed Laser-Induced Degradation and Charge Carrier Dynamics in Multilayer Perovskite Photovoltaics via Cross-sectional Multimodal Atomic Force Microscopy.”**

This repository focuses on the computational component of the report: a 2D transient heat-conduction simulation with a moving Gaussian heat source used to model nanosecond laser heating in a perovskite photovoltaic structure.

## Scientific basis

The simulation is based on the transient heat equation

```text
rho * c_p * dT/dt = k * nabla^2 T + Q
```

with a moving Gaussian volumetric heat source of the form

```text
Q(x, y, t) = Q0 * exp(-((x - v t)^2 + (y - yc)^2) / (2 sigma^2))
```

In the report, this model is used to simulate laser-induced heating on a 1000 um x 1000 um domain and compare predicted thermal behavior with experimental AFM observations of deformation and conductivity loss in laser-exposed perovskite layers.

## What this code does

- solves the 2D transient heat equation with an explicit finite-difference method
- applies fixed-temperature boundary conditions
- models a moving Gaussian laser source
- supports live animation of the temperature field and temperature statistics
- supports headless execution that saves:
  - a CSV file of time, maximum temperature, and average temperature
  - a PNG image of the final temperature field

## Files

- `perovskite_heat_simulation.py` — main simulation script
- `requirements.txt` — minimal Python dependencies
- `STS-report.pdf` — for additional reading

## Installation

It is recommended to use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick start

Run with live animation:

```bash
python perovskite_heat_simulation.py
```

Run without animation and save outputs:

```bash
python perovskite_heat_simulation.py --no-animate --output-prefix run1
```

This will generate:

- `run1.csv`
- `run1.png`

## Parameters

You can change the main physical and numerical settings from the command line.

```bash
python perovskite_heat_simulation.py \
  --nx 150 \
  --ny 150 \
  --lx-um 1000 \
  --ly-um 1000 \
  --t-total 1e-3 \
  --q0 2e12 \
  --sigma-um 100 \
  --speed-um-per-s 1.5e6
```

### Main command-line options

- `--nx` : number of grid points in x
- `--ny` : number of grid points in y
- `--lx-um` : domain size in x, in micrometers
- `--ly-um` : domain size in y, in micrometers
- `--t-total` : total simulation time in seconds
- `--q0` : volumetric heat-source amplitude in W/m^3
- `--sigma-um` : Gaussian source width in micrometers
- `--speed-um-per-s` : source scan speed in micrometers per second
- `--no-animate` : disable live animation and save outputs instead
- `--output-prefix` : prefix for output files in non-animated mode

## Notes on units and implementation

The original report presents the domain in micrometers while material properties are given in SI units. In this implementation:

- user-facing geometry is kept in micrometers for readability
- all PDE calculations are carried out internally in meters
- the explicit time step is chosen from a 2D diffusion stability condition

This makes the script easier to use in a repo while keeping the numerical update dimensionally consistent.

## Example workflow for a repo

A simple reproducible workflow is:

```bash
python perovskite_heat_simulation.py --no-animate --output-prefix baseline
```

Then commit:

- the Python source
- `requirements.txt`
- selected generated output figures
- a short note in the repo explaining what parameter set was used

## Interpreting outputs

- The temperature-field image shows the final thermal distribution across the 2D domain.
- The CSV lets you plot temperature metrics later or compare multiple parameter sweeps.
- Increasing `q0` or decreasing scan speed generally causes stronger local heating.
- Narrower `sigma` values concentrate heating more strongly around the laser path.

## Suggested future extensions

To keep the repo aligned with the report, useful next steps would be:

- parameter sweeps across scan speed and Gaussian width
- saving full animations as GIF or MP4
- coupling thermal fields to a deformation proxy
- adding multilayer or depth-dependent material properties
- comparing nanosecond and femtosecond-inspired source models

## Citation / provenance

This implementation is derived from the computational model and code appendix contained in the accompanying research report.
