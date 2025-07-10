"""
Nice values for Langmuir Oscillations

CFL = 0.07
poisson_interval = 50
delta = 0.05
mode = 3
"""

import params
from grid import Grid1D
from diagnostics import cos_fit, omega_fit
from diagnostics import plot_energies
from field_solver import update_E, initialize_pulse, gauss_correction
from particles import Particles
import time
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Command-line argument parsing
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Benchmark Langmuir oscillation frequency against theory"
)
# Simulation parameters
parser.add_argument('--Nx',               type=int,    default=params.Nx,             help='Grid points')
parser.add_argument('--Np',               type=int,    default=params.Np,             help='Number of particles')
parser.add_argument('--dx',               type=float,  default=params.dx,             help='Grid spacing')
parser.add_argument('--L',                type=float,  default=params.L,              help='Domain length')
parser.add_argument('--c',                type=float,  default=params.c,              help='Speed of light')
parser.add_argument('--q',                type=float,  default=params.q,              help='Particle charge')
parser.add_argument('--m',                type=float,  default=params.m,              help='Particle mass')
parser.add_argument('--eps0',             type=float,  default=params.eps0,           help='Permittivity')
parser.add_argument('--CFL',              type=float,  default=params.CFL,            help='Courant number')
parser.add_argument('--dt',               type=float,  default=params.dt,             help='Time step')
parser.add_argument('--nsteps',           type=int,    default=params.nsteps,         help='Total time steps')
parser.add_argument('--poisson_interval', type=int,    default=params.poisson_interval, help='Gauss correction interval')
parser.add_argument('--delta',            type=float,  default=params.delta,          help='Perturbation amplitude')
parser.add_argument('--mode',             type=int,    default=params.mode,           help='Mode number')
parser.add_argument('--save_interval',    type=int,    default=params.save_interval,  help='Diagnostic save interval')
parser.add_argument('--outdir',           type=str,    default='langmuir_results',    help='Output directory')
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Extract parameters
# -----------------------------------------------------------------------------
Nx, Np    = args.Nx, args.Np
dx, L     = args.dx, args.L
c          = args.c
q, m       = args.q, args.m
eps0       = args.eps0
CFL        = args.CFL
dt         = CFL * dx
nsteps     = args.nsteps
poisson_int= args.poisson_interval
delta      = args.delta
mode       = args.mode
save_int   = args.save_interval
outdir     = args.outdir

# Prepare output
os.makedirs(outdir, exist_ok=True)
param_fn = os.path.join(outdir, f"langmuir_params_n{nsteps}_CFL_{CFL:.3f}_delta_{delta:.3f}_mode{mode}_ngauss_{poisson_int}.txt")
with open(param_fn, 'w') as pf:
    pf.write("# Langmuir oscillation benchmark parameters\n")
    pf.write(f"Nx                = {Nx}\n")
    pf.write(f"Np                = {Np}\n")
    pf.write(f"dx                = {dx}\n")
    pf.write(f"L                 = {L}\n")
    pf.write(f"c                 = {c}\n")
    pf.write(f"q                 = {q}\n")
    pf.write(f"m                 = {m}\n")
    pf.write(f"eps0              = {eps0}\n")
    pf.write(f"CFL               = {CFL}\n")
    pf.write(f"dt                = {dt}\n")
    pf.write(f"nsteps            = {nsteps}\n")
    pf.write(f"poisson_interval  = {poisson_int}\n")
    pf.write(f"delta             = {delta}\n")
    pf.write(f"mode              = {mode}\n")
    pf.write(f"save_interval     = {save_int}\n")
print(f"Saved parameters to {param_fn}")

# -----------------------------------------------------------------------------
# Main driver for Langmuir oscillations
# -----------------------------------------------------------------------------
def main_langmuir(Nx, Np, q, m, eps0, L, dx, dt, nsteps,
                   poisson_interval, delta, mode, save_interval):
    # Initialize grid
    grid = Grid1D(Nx, dx)
    # Initalize electrons with a position shift
    electrons = Particles('electrons', Np, q, m, L, dx)
    electrons.seed_langmuir(delta, mode)

    # Ion background
    n0 = Np / L
    grid.rho_background = - q * n0

    # Initial deposit + Gauss correction
    electrons.deposit_charge_and_current(grid, dt)
    grid.apply_charge_periodic_BC()
    gauss_correction(grid)

    # Precompute mode shape
    x_centers = (np.arange(Nx)+0.5)*dx
    mode_shape = np.sin(2*np.pi*mode * x_centers / L)

    # Storage for energies, time and amplitueds
    amplitudes = []
    energy     = []
    kinetic    = []
    electric   = []
    times      = []

    # Time-stepping loop
    for n in range(nsteps):
        electrons.deposit_charge_and_current(grid, dt)
        grid.apply_charge_periodic_BC()

        # E-kick
        update_E(grid, dt)
        electrons.gather_fields(grid)
        # Boris push
        electrons.boris_push(dt)

        # Enforce gauss correction
        if (n+1) % poisson_interval == 0:
            gauss_correction(grid)

        # Record data every n steps
        if n % save_interval == 0:
            Ex_phys = grid.Ex[1:-1]
            A_n = (2/Nx) * np.dot(Ex_phys, mode_shape) * dx
            amplitudes.append(A_n)
            times.append((n+1)*dt)
            EE_val = 0.5*eps0 * np.sum(Ex_phys**2) * dx
            KE_val = 0.5*m * np.sum(electrons.vx**2)
            electric.append(EE_val)
            kinetic.append(KE_val)
            energy.append(KE_val + EE_val)
    

    return np.array(amplitudes), np.array(energy), np.array(kinetic), np.array(electric), np.array(times)

# -----------------------------------------------------------------------------
# Script entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    start = time.time()
    A, TE, KE, EE_data, t = main_langmuir(
        Nx, Np, q, m, eps0, L, dx, dt, nsteps,
        poisson_int, delta, mode, save_int
    )
    end = time.time()
    print(f"Simulation Time: {end - start:.2f} s")

    # Theoretical plasma frequency
    n0 = Np / L
    omega_p = np.sqrt(n0 * q**2 / (m * eps0))

    # Simulation frequency fit
    omega_sim, omega_err = omega_fit(t, A, omega_p, 0.0)
    rel_err = (omega_sim/omega_p - 1.0)
    rel_err_err = omega_err / omega_p

    # Print results
    print("Δt / (0.1 / ω_p) = " + f"{dt*omega_p/0.1:.6f}")
    print(f"Simulated ω_sim      = {omega_sim:.6f} ± {omega_err:.6f}")
    print(f"Theoretical ω_p      = {omega_p:.6f}")
    print(f"Relative error       = {rel_err*100:.2f}% ± {rel_err_err*100:.2f}%")

    # Plot amplitude vs time
    amp_fn = os.path.join(outdir, f"amplitude_n{nsteps}_CFL_{CFL:.3f}_delta_{delta:.3f}_mode{mode}_ngauss_{poisson_int}.png")
    plt.figure()
    plt.plot(t, A, 'o-', markersize=2, label=r"$\omega_{p,th}$ " + f"= {omega_p:.3f} \n" + r"$\omega_{p,sim}$ " + f" = {omega_sim:.3f} \nΔω = {abs(rel_err)*100:.3f}%")
    plt.xlabel('t', fontsize=14)
    plt.ylabel('Mode amplitude', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Langmuir Mode Amplitude', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=14, loc="upper right")
    plt.tight_layout()
    plt.savefig(amp_fn)
    plt.show()
    print(f"Saved amplitude plot to {amp_fn}")

    # Plot total energy vs time
    energy_fn = os.path.join(outdir, f"energy_n{nsteps}_CFL_{CFL:.3f}_delta_{delta:.3f}_mode{mode}_ngauss_{poisson_int}.png")
    plt.figure()
    plt.plot(t, TE, label='Total')
    plt.plot(t, KE, label='Kinetic')
    plt.plot(t, EE_data, label='Electric')
    plt.xlabel('t', fontsize=14)
    plt.ylabel('Energy', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=14, loc="upper right")
    plt.title(f'nsteps: {nsteps} | CFL: {CFL:.3f} | d: {delta:.3f} | m: {mode}', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(energy_fn)
    plt.show()
    print(f"Saved energy plot to {energy_fn}")