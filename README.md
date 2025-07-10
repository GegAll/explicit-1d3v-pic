# 1D3V Explicit Particle-In-Cell (PIC) Code

_A modular Python implementation for one-dimensional, three-velocity-component plasma simulations_

---

## Overview

This repository provides a fully-featured **explicit** 1D3V PIC code in pure Python, suitable for studying a range of standard plasma benchmarks:

- **Vacuum pulse propagation** (FDTD in vacuum)  
- **Cyclotron gyration** of a test electron in a uniform magnetic field  
- **Langmuir oscillations** (electrostatic plasma waves)  
- **Two-stream instability** (electrostatic beam–plasma instability)  
- **Weibel instability** (magnetic filamentation from temperature anisotropy)  

Each test is driven by a dedicated `main_<test>.py` script and shares a common core of grid, particle, field-solver, and diagnostics modules.

---

```text
.
├── README.md
├── params.py
├── grid.py
├── particles.py
├── field_solver.py
├── diagnostics.py
├── main_vacuum_propagation.py
├── main_cyclotron.py
├── main_langmuir.py
├── main_two_stream_instability.py
└── main_weibel_instability.py
```


---

## Core Modules

### `params.py`
Defines normalized constants:
- Charge `q`, mass `m`, speed of light `c`
- Permittivity `eps0`, permeability `mu0`
- Domain length `L`, grid resolution `Nx`, particles per cell `npp`, timestep `dt`, etc.

### `grid.py`
- **`Grid1D`**: Yee-type 1D grid with ghost cells for periodic boundaries.  
- Stores **E**-fields at cell centers (`Ex`, `Ey`, `Ez`) and **B**-fields at half-nodes (`By`, `Bz`).  
- Holds charge density `rho` and currents `Jx`, `Jy`, `Jz`.  
- Methods: `apply_field_periodic_BC()`, `apply_charge_periodic_BC()`.

### `particles.py`
- **`Particles`**: Structure-of-arrays for particle positions and velocities (\(v_x, v_y, v_z\)).  
- **`gather_fields(grid)`**: Linear interpolation of grid fields to particle locations.  
- **`boris_push(dt)`**: Standard Boris algorithm for updating velocities and positions.  
- **`deposit_charge_and_current(grid, dt)`**: Cloud-In-Cell (CIC) scatter of charge & current to grid.  
- **`seed_langmuir(delta, mode)`**: Utility to impose sinusoidal density perturbations.

### `field_solver.py`
- **`initialize_pulse(grid, amplitude, mode)`**: Seed a vacuum pulse in `Ez`.  
- **`update_B(grid, dt)`**: Yee update for magnetic field.  
- **`update_E(grid, dt)`**: Yee update for electric field including current drive.  
- **`gauss_correction(grid)`**: FFT-based Poisson solve to enforce Gauss’s law on `Ex`.

### `diagnostics.py`
- **Energy diagnostics**: `field_energy()`, `kinetic_energy()`, `plot_energies()`.  
- **Spectral & growth-rate fits**: `fit_growth_and_plot()` (two-stream), `plot_weibel_growth()`.  
- **Oscillation fits**: `cos_fit()`, `omega_fit()`, `circle_center()`.  
- **Phase-space & trajectory plots**: `plot_vxvy_plane()`, `plot_tvx_plane()`.  
- **I/O helpers**: CSV writers for time-series data (`save_weibel_diagnostics()`, etc.).

---

## Test Scripts

Each `main_<test>.py` script follows this pattern:

1. **Argument parsing** via `argparse`.  
2. **Output directory** creation and parameter file dump (`params.txt`).  
3. **Initialization** of `Grid1D` and `Particles`, plus specific IC seeding.  
4. **Yee leap-frog loop**: deposit → fields → push → diagnostics.  
5. **Post-processing**: save CSV, compute fits, generate & save plots.

---

## Script Descriptions

### main.py  
A **general-purpose 1D 3V PIC driver** that unifies all built-in test problems and a custom IC mode.  
- **Test selection**: Use `--ic` flag to choose between `vacuum`, `cyclotron`, `langmuir`, `two_stream`, `weibel`, or `custom`.  
- **Common arguments**: grid size (`--Nx`, `--L`), time stepping (`--CFL`, `--nsteps`), particle settings (`--npp`, `--q`, `--m`), output directory (`--outdir`), and more.  
- **Workflow**:  
  1. Parse arguments, write `params.txt`.  
  2. Initialize `Grid1D` + `Particles`.  
  3. Apply IC (field or particle seeding).  
  4. Advance fields & particles in a single loop.  
  5. Collect diagnostics, save CSV, generate plots.  

### main_vacuum_propagation.py  
Simulates **electromagnetic pulse propagation in vacuum** using the Yee FDTD scheme (no charges).  
- **Initial condition**: Sinusoidal `Ez` pulse seeded via `initialize_pulse()`.  
- **Diagnostics**:  
  - Track electric (`EE`) and magnetic (`ME`) field energies.  
  - Plot total vs. individual energies to verify energy conservation.  

### main_cyclotron.py  
Tracks **single-electron gyration** in uniform `B0` (and optional `E0`) fields using the Boris pusher.  
- **Initial condition**: One particle at center with specified `v_y`.  
- **Diagnostics**:  
  - Record `(v_x, v_y)` trajectory over time.  
  - Compute kinetic energy and field energy (constant `E0` contribution).  
  - Fit simulated gyrofrequency (`omega_sim`) via `omega_fit()` and compare to theoretical `omega_c = |q| B0 / m`.  
  - Plot `v_x` vs. `t` and `v_x`–`v_y` plane with fitted circle center.  

### main_langmuir.py  
Runs the **Langmuir (electrostatic) oscillation** benchmark by perturbing the density.  
- **Initial condition**: Particles seeded with sinusoidal density `delta * sin(kx)`.  
- **Solver specifics**:  
  - Pure electrostatic: push on `E_x` only.  
  - Periodic **Gauss correction** every `--poisson_interval` steps to enforce `∇·E = ρ/ε0`.  
- **Diagnostics**:  
  - Track mode amplitude `A(k)`, kinetic energy `KE`, electric energy `EE`, total energy `TE`.  
  - Fit plasma frequency `ω_p = sqrt(n0 q^2/(m ε0))` via `omega_fit()`.  
  - Save CSV and plot `A(k)` vs. time.  

### main_two_stream_instability.py  
Executes the **two-stream instability** in an electrostatic limit.  
- **Initial condition**: Two counter-streaming beams with speeds ±`v0`.  
- **Seeding**: Small sinusoidal perturbation in `E_x` at mode `--mode`.  
- **Diagnostics**:  
  - Compute Fourier amplitude `|E_x(k)|` over time.  
  - Fit exponential growth rate `γ_sim` via `fit_growth_and_plot()`.  
  - Compare to theoretical fluid result `γ_theo = k v0`.  

### main_weibel_instability.py  
Implements the **Weibel (filamentation) instability** in a magnetized 1D3V plasma.  
- **Initial condition**: Anisotropic Maxwellian with `vth_para != vth_perp`.  
- **Seeding**: Small sinusoidal perturbation in `B_z` at mode `--mode`.  
- **Diagnostics**:  
  - Track magnetic mode amplitude `|B_z(k)|`.  
  - Fit linear growth rate `γ_sim` via `plot_weibel_growth()`.  
  - Compare to fluid‐Weibel estimate `γ_theo = vth_perp/(√2 c) ω_p`.  

---

## Usage Examples

```bash
# Two‐stream instability
python main_two_stream_instability.py --Nx 400 --npp 500 --v0 0.2 --mode 2 --amplitude 1e-4

# Weibel instability
python main_weibel_instability.py --npp 1000 --vth_para 0.05 --vth_perp 0.3 --mode 1

# Vacuum pulse propagation
python main_vacuum_propagation.py --Nx 200 --nsteps 5000 --vac_amp 1e-3 --vac_mode 3

# Cyclotron gyration
python main_cyclotron.py --nsteps 1000 --B0 1e-4 --E0 0.0

# Langmuir oscillations
python main_langmuir.py --Nx 200 --Np 200000 --delta 0.05 --lm_mode 3
```

## Extending the Code
New instabilities / ICs: add scripts following existing templates.

Boundary conditions: only periodic; extend grid.py for open or reflective.

Solvers: currently explicit Yee + Gauss correction; semi-implicit or spectral may be integrated.

Performance: vectorized NumPy; consider Numba JIT for speedups.
