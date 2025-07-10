#!/usr/bin/env python3
import params
from grid import Grid1D
from particles import Particles
from field_solver import update_E, update_B
from diagnostics import save_weibel_diagnostics, plot_weibel_growth
import time
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import csv

# -----------------------------------------------------------------------------
# Command-line argument parsing
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Benchmark Weibel instability growth rate against theory"
)
# grid / geometry
parser.add_argument('--Nx',            type=int,    default=256,              help='Number of grid cells')
parser.add_argument('--L',             type=float,  default=2*np.pi,          help='Domain length (in c/ω_p units)')
# particles
parser.add_argument('--npp',           type=int,    default=2000,             help='Particles per cell')
parser.add_argument('--m',             type=float,  default=1.0,              help='Particle mass')
parser.add_argument('--c',             type=float,  default=1.0,              help='Speed of light')
parser.add_argument('--eps0',          type=float,  default=1.0,              help='Permittivity')
parser.add_argument('--mu0',           type=float,  default=1.0,              help='Permeability')
parser.add_argument('--kB',            type=float,  default=1.0,              help='Boltzmann constant')
# anisotropy
parser.add_argument('--vth_para',      type=float,  default=0.05,             help='Parallel thermal speed (as fraction of c)')
parser.add_argument('--vth_perp',      type=float,  default=0.25,             help='Perpendicular thermal speed (as fraction of c)')
# timestep & runtime
parser.add_argument('--CFL',           type=float,  default=0.1,              help='Courant number')
parser.add_argument('--t_end',         type=float,  default=50.0,             help='End time (in 1/ω_p units)')
parser.add_argument('--save_interval', type=int,    default=1,                help='Diagnostic save interval')
# seeding
parser.add_argument('--mode',          type=int,    default=1,                help='Mode number to seed B_z')
parser.add_argument('--amplitude',     type=float,  default=1e-4,             help='Seed perturbation amplitude')
# outputs
parser.add_argument('--outdir',        type=str,    default='weibel_results', help='Output directory')
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Extract and compute derived parameters
# -----------------------------------------------------------------------------
Nx            = args.Nx
L             = args.L
dx            = L / Nx

npp           = args.npp
Np            = Nx * npp
n0            = Np / L
q             = -np.sqrt(1.0 / n0)    # set ω_p = 1

m             = args.m
c             = args.c
eps0          = args.eps0
mu0           = args.mu0
kB            = args.kB
omega_p       = np.sqrt((n0*q**2)/(m*eps0))

vth_para      = args.vth_para * c
vth_perp      = args.vth_perp * c

CFL           = args.CFL
dt            = CFL * dx / c
t_end         = args.t_end
nsteps        = int(t_end / dt)

save_interval = args.save_interval
mode          = args.mode
amplitude     = args.amplitude
outdir        = args.outdir

# -----------------------------------------------------------------------------
# Prepare output directory & parameter file
# -----------------------------------------------------------------------------
os.makedirs(outdir, exist_ok=True)
param_fn = os.path.join(
    outdir,
    f"weibel_params_mode{mode}_n{nsteps}_npp{npp}_CFL{CFL}.txt"
)
with open(param_fn, 'w') as pf:
    pf.write("# Weibel benchmark parameters\n")
    for key, val in vars(args).items():
        pf.write(f"{key} = {val}\n")
print(f"Saved parameters to {param_fn}")

# -----------------------------------------------------------------------------
# Main driver for Weibel
# -----------------------------------------------------------------------------
def main_weibel(Nx, Np, q, m, eps0, mu0, kB, L, dx, dt,
                vth_para, vth_perp, nsteps, save_interval,
                mode, amplitude):
    # Initialize grid & particles
    grid      = Grid1D(Nx, dx)
    particles = Particles('electrons', Np, q, m, L, dx)

    # neutralizing ion background
    n0 = Np / L
    grid.rho_background = -particles.q * n0

    # Initialize anisotropy velocities
    particles.vx = np.random.normal(0, vth_para, Np)
    particles.vy = np.random.normal(0, vth_perp, Np)
    particles.vz = np.random.normal(0, vth_perp, Np)

    # Initial deposit
    particles.deposit_charge_and_current(grid, dt)
    grid.apply_charge_periodic_BC()

    # seed B-Field mode
    k = 2*np.pi * mode / L
    x_centers = (np.arange(Nx)+0.5)*dx
    grid.Bz[1:-1] = amplitude * np.sin(k * x_centers)
    grid.apply_field_periodic_BC()

    # half-step E for Yee
    update_E(grid, 0.5*dt)

    # Storage for time and amplitude
    times, amp_B = [], []

    # Time-stepping loop
    for step in range(nsteps):
        # deposit + BC
        particles.deposit_charge_and_current(grid, dt)
        grid.apply_charge_periodic_BC()

        # field push
        update_E(grid, dt)
        update_B(grid, dt)

        # particle push
        particles.gather_fields(grid)
        particles.boris_push(dt)

        # diagnostics
        if step % save_interval == 0:
            t = step * dt
            Bz_phys = grid.Bz[1:-1]
            A_k = np.dot(Bz_phys, np.exp(-1j*k*x_centers)) * dx
            times.append(t)
            amp_B.append(np.abs(A_k))

    return np.array(times), np.array(amp_B)

# -----------------------------------------------------------------------------
# Script entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting Weibel simulation...")
    t0 = time.time()
    times, amp_B = main_weibel(Nx, Np, q, m, eps0, mu0, kB, L, dx, dt,
                vth_para, vth_perp, nsteps, save_interval,
                mode, amplitude)
    t1 = time.time()
    print(f"Simulation wall-time: {t1 - t0:.2f} s")

    # Compute lnB
    lnB = np.log(amp_B)

    # Compute theoretical growth
    gamma_theo = omega_p*vth_perp/np.sqrt(2*c)
    print(f"Theoretical growth rate γ_theo = {gamma_theo:.4f}")

    # Save time-series to CSV
    csv_fn = os.path.join(outdir, f"weibel_data_mode{mode}_n{nsteps}_npp{npp}_CFL{CFL}.csv")
    save_weibel_diagnostics(csv_fn, times, amp_B, lnB)
    
    # Fit exponential in linear window
    plot_fn = os.path.join(outdir, f"weibel_lnB_mode{mode}_n{nsteps}_npp{npp}_CFL{CFL}.png")
    title_fn = f'nsteps: {nsteps} | CFL: {CFL} | npp: {npp}'
    gamma_sim, A0 = plot_weibel_growth(times, lnB, gamma_theo, title=title_fn, filename=plot_fn)