import params
from grid import Grid1D
from diagnostics import kinetic_energy, field_energy, plot_energies
from field_solver import update_B, update_E, initialize_pulse
import time
import os
import argparse
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Command-line argument parsing
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Propagate a vacuum pulse and plot field energies with record of parameters"
)
# Simulation parameters (defaults from params.py)
parser.add_argument('--Nx',   type=int,   default=params.Nx, help='Number of grid points')
parser.add_argument('--dx',   type=float, default=params.dx, help='Grid spacing')
parser.add_argument('--L',    type=float, default=params.L,  help='Box size')
parser.add_argument('--c',    type=float, default=params.c,  help='Speed of light')
parser.add_argument('--CFL',  type=float, default=0.99,     help='Courant number')
parser.add_argument('--nsteps', type=int, default=10000,   help='Number of time steps')
parser.add_argument('--nsave', type=int, default=100,   help='Number of time steps after which the measurement is saved')
parser.add_argument('--amp',  type=float, default=1e-3,      help='Initial pulse amplitude')
parser.add_argument('--mode', type=int,   default=3,         help='Initial pulse mode')
parser.add_argument('--outdir', type=str, default='results', help='Output directory')
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Extract parameters from args
# -----------------------------------------------------------------------------
Nx     = args.Nx
dx     = args.dx
L      = args.L
c      = args.c
CFL    = args.CFL
dt     = CFL * dx / c
nsteps = args.nsteps
nsave  = args.nsave
amp    = args.amp
mode   = args.mode
outdir = args.outdir

# -----------------------------------------------------------------------------
# Write parameters to a .txt file for record
# -----------------------------------------------------------------------------
os.makedirs(outdir, exist_ok=True)
param_fn = os.path.join(outdir, f"vacuum_params_n{nsteps}_CFL_{CFL:.3f}.txt")
with open(param_fn, 'w') as pf:
    pf.write(f"# Simulation parameters for vacuum propagator\n")
    pf.write(f"Nx      = {Nx}\n")
    pf.write(f"dx      = {dx}\n")
    pf.write(f"L       = {L}\n")
    pf.write(f"c       = {c}\n")
    pf.write(f"CFL     = {CFL}\n")
    pf.write(f"dt      = {dt}\n")
    pf.write(f"nsteps  = {nsteps}\n")
    pf.write(f"amp     = {amp}\n")
    pf.write(f"mode    = {mode}\n")
print(f"Saved parameters to {param_fn}")

# -----------------------------------------------------------------------------
# Main driver: propagate a pulse in vacuum and record field energies
# -----------------------------------------------------------------------------
def main_vacuum(Nx, dx, L, dt, nsteps, amp, mode):
    # Initialize grid and pulse
    grid = Grid1D(Nx, dx)
    initialize_pulse(grid, amplitude=amp, mode=mode)
    
    # Initial half-step for B-field (to t = dt/2)
    update_B(grid, 0.5 * dt)

    # Storage for energies and times
    total_energy = []
    electric     = []
    magnetic     = []
    times        = []

    # Time-stepping loop
    for n in range(nsteps):
        # update E from Ez^n to Ez^(n+1)
        update_E(grid, dt)
        # update B from By^(n+1/2) to By^(n+3/2)
        update_B(grid, dt)
        
        # record every 100 steps
        if n % nsave == 0:
            EE, ME = field_energy(grid)
            total_energy.append(EE + ME)
            electric.append(EE)
            magnetic.append(ME)
            times.append((n + 1) * dt)

    return total_energy, electric, magnetic, centroids, times

# -----------------------------------------------------------------------------
# Script entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()
    TE, EE_data, ME_data, centroids, t_data = main_vacuum(Nx, dx, L, dt, nsteps, amp, mode)
    end_time   = time.time()
    print(f"Simulation Time: {end_time - start_time:.2f} s")

    # Construct a descriptive filename for the plot
    png_fn = os.path.join(outdir, f"energies_n{nsteps}_CFL_{CFL:.3f}.png")
    
    # Plot and save energies
    plot_energies(t_data, TE, EE_data, ME_data, save=True, filename=png_fn, title=f"nsteps: {nsteps} | CFL: {CFL:.3f}")
    print(f"Saved energy plot to {png_fn}")