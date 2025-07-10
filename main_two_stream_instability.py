import params
from grid import Grid1D
from particles import Particles
from field_solver import update_E, update_B
from diagnostics import fit_growth_and_plot
import time
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Command-line argument parsing
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Benchmark Two-Stream instability growth rate against theory"
)
# Simulation parameters
parser.add_argument('--n0_target',     type=int,    default=1.0,              help='Target density')
parser.add_argument('--Nx',            type=int,    default=400,              help='Grid points')
parser.add_argument('--npp',           type=int,    default=200,              help='Particles per cell')
parser.add_argument('--L',             type=float,  default=2*np.pi,          help='Domain length')
parser.add_argument('--c',             type=float,  default=1.0,              help='Speed of light')
parser.add_argument('--m',             type=float,  default=1.0,              help='Particle mass')
parser.add_argument('--v0',            type=float,  default=0.2,             help='two-stream initial velocity')
parser.add_argument('--eps0',          type=float,  default=1.0,              help='Permittivity')
parser.add_argument('--CFL',           type=float,  default=0.5,              help='Courant number')
parser.add_argument('--nsteps',        type=int,    default=2000,             help='Total time steps')
parser.add_argument('--save_interval', type=int,    default=1,                help='Diagnostic save interval')
parser.add_argument('--amplitude',     type=float,  default=1e-4,             help='Perturbation amplitude')
parser.add_argument('--mode',          type=int,    default=2,                help='Modes for Ex seed')
parser.add_argument('--outdir',        type=str,    default='twostream_results', help='Output directory')
args = parser.parse_args()

# Extract parameters
n0_target     = args.n0_target
Nx            = args.Nx
npp           = args.npp
L             = args.L
c             = args.c
v0            = args.v0 * c
m             = args.m
eps0          = args.eps0
CFL           = args.CFL
nsteps        = args.nsteps
save_int      = args.save_interval
amplitude     = args.amplitude
mode          = args.mode
outdir        = args.outdir
save_interval = args.save_interval

dx   = L / Nx
dt   = CFL * dx / c
Np   = Nx * npp
n0_real = Np / L
q    = -np.sqrt(1.0 / n0_real) # normalized

# Prepare output directory
os.makedirs(outdir, exist_ok=True)
param_fn = os.path.join(outdir, f"twostream_params_n{nsteps}_CFL{CFL:.3f}_npp{npp}.txt")
with open(param_fn, 'w') as pf:
    pf.write("# Two-stream benchmark parameters\n")
    for k,v in vars(args).items(): pf.write(f"{k} = {v}\n")
print(f"Saved parameters to {param_fn}")

# -----------------------------------------------------------------------------
# Main driver for two-stream
# -----------------------------------------------------------------------------
def main_twostream(Nx, Np, q, m, eps0, L, dx, dt, v0,
                   nsteps, save_interval, amplitude, mode):
    # Initialize Grid & Particles
    grid      = Grid1D(Nx, dx)
    particles = Particles('electrons', Np, q, m, L, dx)

    # neutralizing ion background
    n0 = particles.Np / L
    grid.rho_background = -particles.q * n0

    # two‐stream initial condition
    half = particles.Np // 2
    particles.vx[:half] = +v0
    particles.vx[half:] = -v0

    # initial deposit
    particles.deposit_charge_and_current(grid, dt)
    grid.apply_charge_periodic_BC()

    # seed Ex mode
    k = 2*np.pi * mode / L
    x_centers = (np.arange(Nx)+0.5)*dx
    grid.Ex[1:-1] = amplitude * np.sin(k * x_centers)
    grid.apply_field_periodic_BC()

    # half‐step B for Yee
    update_B(grid, 0.5*dt)

    # diagnostics prep
    mode_shape = np.exp(-1j * k * x_centers)
    times, amp_E = [], []

    # time loop
    for step in range(nsteps):
        # deposit + BC
        particles.deposit_charge_and_current(grid, dt)
        grid.apply_charge_periodic_BC()

        # fields push
        update_E(grid, dt)
        update_B(grid, dt)

        # Boris push
        particles.gather_fields(grid)
        particles.boris_push(dt)

        if step % save_interval == 0:
            t = step * dt
            Ex_phys = grid.Ex[1:-1]
            A_k = np.dot(Ex_phys, mode_shape) * dx
            times.append(t)
            amp_E.append(np.abs(A_k))

    return np.array(times), np.array(amp_E)

# -----------------------------------------------------------------------------
# Script entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    start = time.time()
    times, amp_E = main_twostream(
        Nx, Np, q, m, eps0, L, dx, dt, v0,
        nsteps, save_interval, amplitude, mode
    )
    end = time.time()
    print(f"Simulation Time : {end - start:.2f} s")

    plot_fn = os.path.join(outdir, f"twostream_lnamp_n{nsteps}_CFL{CFL:.3f}_npp{npp}.png")
    title_fn = f'nsteps: {nsteps} | CFL: {CFL} | npp: {npp}'
    gamma_sim, A0 = fit_growth_and_plot(amp_E, times, dt, mode, L, v0, title=title_fn, filename=plot_fn)
    print(f"Measured growth rate γ_sim = {gamma_sim:.4f}")

    