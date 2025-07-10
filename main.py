#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
import csv
import matplotlib.pyplot as plt

from grid import Grid1D
from particles import Particles
from field_solver import update_E, update_B
from diagnostics import (
    fit_growth_and_plot, plot_weibel_growth, save_weibel_diagnostics,
    kinetic_energy, field_energy, plot_energies,
    cos_fit, omega_fit, circle_center, plot_tvx_plane, plot_vxvy_plane
)

# -----------------------------------------------------------------------------
# Command‐line arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="General 1D3V PIC driver")
parser.add_argument('--ic',
                    choices=['two_stream','weibel','vacuum','cyclotron','langmuir','custom'],
                    required=True, help='Which test or IC to run')
parser.add_argument('--Nx',            type=int,   default=256,     help='Grid cells')
parser.add_argument('--L',             type=float, default=2*np.pi, help='Box length')
parser.add_argument('--CFL',           type=float, default=0.1,     help='Courant number')
parser.add_argument('--c',             type=float, default=1.0,     help='Speed of light')
parser.add_argument('--nsteps',        type=int,   default=1000,    help='Time steps')
parser.add_argument('--save_interval', type=int,   default=1,       help='Diagnostics interval')
parser.add_argument('--outdir',        type=str,   default='results', help='Output directory')

# two-stream
parser.add_argument('--npp',   type=int,   default=1000, help='Particles per cell')
parser.add_argument('--v0',    type=float, default=0.2,  help='Drift speed (fraction of c)')
parser.add_argument('--ts_mode', type=int, default=2,    help='Mode for Ex seed')
parser.add_argument('--ts_amp',  type=float, default=1e-4, help='Amplitude for Ex seed')

# weibel
parser.add_argument('--vth_para', type=float, default=0.1, help='v_th,∥ (fraction of c)')
parser.add_argument('--vth_perp', type=float, default=0.3, help='v_th,⊥ (fraction of c)')
parser.add_argument('--wb_mode',  type=int,   default=1,   help='Mode for Bz seed')
parser.add_argument('--wb_amp',   type=float, default=1e-4, help='Amplitude for Bz seed')

# vacuum
parser.add_argument('--vac_amp',  type=float, default=1e-3, help='Pulse amplitude')
parser.add_argument('--vac_mode', type=int,   default=3,    help='Pulse mode')
parser.add_argument('--nsave',    type=int,   default=100,  help='Save every n steps (vacuum)')

# cyclotron
parser.add_argument('--B0', type=float, default=1e-4, help='Uniform B field')
parser.add_argument('--E0', type=float, default=0.0,  help='Uniform E field')

# langmuir
parser.add_argument('--Np',              type=int,   default=100000, help='Total particles')
parser.add_argument('--delta',           type=float, default=0.05,   help='Perturbation amplitude')
parser.add_argument('--lm_mode',         type=int,   default=3,      help='Mode number')
parser.add_argument('--poisson_interval',type=int,   default=50,     help='Gauss‐corr interval')

# custom
parser.add_argument('--custom_mode', type=int,   default=3,    help='Mode for custom density perturbation')
parser.add_argument('--delta_rho',   type=float, default=0.2,  help='Density perturbation amplitude')

args = parser.parse_args()


# -----------------------------------------------------------------------------
# Derived parameters
# -----------------------------------------------------------------------------
Nx, L, cfl, c = args.Nx, args.L, args.CFL, args.c
dx = L / Nx
dt = cfl * dx / c

# prepare output directory
os.makedirs(args.outdir, exist_ok=True)
with open(os.path.join(args.outdir, 'params.txt'), 'w') as pf:
    for key, val in vars(args).items():
        pf.write(f"{key} = {val}\n")

# common x-centers for modes
x_centers = (np.arange(Nx) + 0.5) * dx

# -----------------------------------------------------------------------------
# Test‐specific setup and main PIC loop
# -----------------------------------------------------------------------------
if args.ic == 'two_stream':
    Np = Nx * args.npp
    grid = Grid1D(Nx, dx)
    particles = Particles('electrons', Np, -np.sqrt(1.0/(Np/L)), 1.0, L, dx)

    # Two‐stream IC
    half = Np // 2
    v0 = args.v0 * c
    particles.vx[:half] = +v0
    particles.vx[half:] = -v0

    particles.deposit_charge_and_current(grid, dt)
    grid.apply_charge_periodic_BC()

    k = 2*np.pi * args.ts_mode / L
    grid.Ex[1:-1] = args.ts_amp * np.sin(k * x_centers)
    grid.apply_field_periodic_BC()
    update_B(grid, 0.5*dt)

    times, amp = [], []
    for step in range(args.nsteps):
        particles.deposit_charge_and_current(grid, dt)
        grid.apply_charge_periodic_BC()
        update_E(grid, dt)
        update_B(grid, dt)
        particles.gather_fields(grid)
        particles.boris_push(dt)

        if step % args.save_interval == 0:
            t = step * dt
            Ex_phys = grid.Ex[1:-1]
            A_k = np.dot(Ex_phys, np.exp(-1j*k*x_centers)) * dx
            times.append(t)
            amp.append(np.abs(A_k))

    times = np.array(times); amp = np.array(amp)
    # CSV
    csv_fn = os.path.join(args.outdir, 'two_stream_data.csv')
    with open(csv_fn, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t','amp','lnamp'])
        for t,a in zip(times, amp):
            w.writerow([t, a, np.log(a)])
    print("Saved two-stream data to", csv_fn)

    gamma, A0 = fit_growth_and_plot(
        amp, times, dt,
        args.ts_mode, L, v0,
        title='two_stream',
        filename=os.path.join(args.outdir, 'two_stream_fit.png')
    )
    print("γ_sim =", gamma)


elif args.ic == 'weibel':
    Np = Nx * args.npp
    grid = Grid1D(Nx, dx)
    particles = Particles('electrons', Np, -np.sqrt(1.0/(Np/L)), 1.0, L, dx)

    # Weibel IC
    particles.vx = np.random.normal(0, args.vth_para*c, Np)
    particles.vy = np.random.normal(0, args.vth_perp*c, Np)
    particles.vz = np.random.normal(0, args.vth_perp*c, Np)

    particles.deposit_charge_and_current(grid, dt)
    grid.apply_charge_periodic_BC()

    k = 2*np.pi * args.wb_mode / L
    grid.Bz[1:-1] = args.wb_amp * np.sin(k * x_centers)
    grid.apply_field_periodic_BC()
    update_E(grid, 0.5*dt)

    times, amp = [], []
    for step in range(args.nsteps):
        particles.deposit_charge_and_current(grid, dt)
        grid.apply_charge_periodic_BC()
        update_E(grid, dt)
        update_B(grid, dt)
        particles.gather_fields(grid)
        particles.boris_push(dt)

        if step % args.save_interval == 0:
            t = step * dt
            Bz_phys = grid.Bz[1:-1]
            A_k = np.dot(Bz_phys, np.exp(-1j*k*x_centers)) * dx
            times.append(t)
            amp.append(np.abs(A_k))

    times = np.array(times); amp = np.array(amp)
    save_weibel_diagnostics(
        os.path.join(args.outdir, 'weibel_data.csv'),
        times, amp, np.log(amp)
    )
    gamma = plot_weibel_growth(
        times, np.log(amp),
        title='weibel',
        filename=os.path.join(args.outdir, 'weibel_fit.png')
    )
    print("γ_sim =", gamma)


elif args.ic == 'vacuum':
    grid = Grid1D(Nx, dx)
    from field_solver import initialize_pulse
    initialize_pulse(grid, amplitude=args.vac_amp, mode=args.vac_mode)
    update_B(grid, 0.5*dt)

    TE, EE, ME, times = [], [], [], []
    for n in range(args.nsteps):
        update_E(grid, dt)
        update_B(grid, dt)
        if n % args.nsave == 0:
            eE, mE = field_energy(grid)
            TE.append(eE+mE); EE.append(eE); ME.append(mE)
            times.append((n+1)*dt)

    TE, EE, ME, times = map(np.array, (TE, EE, ME, times))
    csv_fn = os.path.join(args.outdir, 'vacuum_energy.csv')
    with open(csv_fn, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t','Etot','E','B'])
        for t,e_tot,e_e,e_b in zip(times, TE, EE, ME):
            w.writerow([t, e_tot, e_e, e_b])
    print("Saved vacuum energy to", csv_fn)
    plot_energies(times, TE, EE, ME, save=True,
                  filename=os.path.join(args.outdir,'vacuum_energies.png'))


elif args.ic == 'cyclotron':
    grid = Grid1D(Nx, dx)
    grid.Bz[:] = args.B0
    grid.Ex[:] = args.E0
    particles = Particles('electron', 1, args.q, args.m, L, dx)
    particles.x[:] = 0.0
    particles.vy[:] = 1.0

    times, vx_l, vy_l, KE = [], [], [], []
    for n in range(args.nsteps):
        particles.gather_fields(grid)
        particles.boris_push(dt)
        t = (n+1)*dt
        times.append(t)
        vx_l.append(particles.vx[0])
        vy_l.append(particles.vy[0])
        KE.append(0.5*args.m*(particles.vx[0]**2 + particles.vy[0]**2))

    times = np.array(times); vx_l = np.array(vx_l); vy_l = np.array(vy_l)
    csv_fn = os.path.join(args.outdir, 'cyclotron_traj.csv')
    with open(csv_fn, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t','vx','vy','KE'])
        for t,vx,vy,ke in zip(times, vx_l, vy_l, KE):
            w.writerow([t, vx, vy, ke])
    print("Saved cyclotron trajectory to", csv_fn)

    omega_c = abs(args.q)*args.B0/args.m
    omega_sim, _ = omega_fit(times, vx_l, omega_c, 0.0)
    plot_tvx_plane(times, vx_l, 2*np.pi/omega_c, 2*np.pi/omega_sim,
                   save=True, filename=os.path.join(args.outdir,'cyclotron_tvx.png'))
    circle_center(vx_l, vy_l, 2*np.pi/omega_c, dt, periods=1)
    plot_vxvy_plane(vx_l, vy_l,
                    *circle_center(vx_l, vy_l, 2*np.pi/omega_c, dt, periods=1),
                    save=True, filename=os.path.join(args.outdir,'cyclotron_vxvy.png'))
    print("ω_sim =", omega_sim, "ω_c =", omega_c)


elif args.ic == 'langmuir':
    grid = Grid1D(Nx, dx)
    particles = Particles('electrons', args.Np, args.q, args.m, L, dx)
    particles.seed_langmuir(args.delta, args.lm_mode)
    grid.rho_background = -args.q*(args.Np/L)

    particles.deposit_charge_and_current(grid, dt)
    grid.apply_charge_periodic_BC()
    from field_solver import gauss_correction
    gauss_correction(grid)
    mode_shape = np.sin(2*np.pi*args.lm_mode * x_centers / L)

    A, TE, KE, EE, times = [], [], [], [], []
    for n in range(args.nsteps):
        particles.deposit_charge_and_current(grid, dt)
        grid.apply_charge_periodic_BC()
        update_E(grid, dt)
        particles.gather_fields(grid)
        particles.boris_push(dt)
        if (n+1) % args.poisson_interval == 0:
            gauss_correction(grid)
        if n % args.save_interval == 0:
            t = (n+1)*dt
            Exp = grid.Ex[1:-1]
            A.append((2/Nx)*np.dot(Exp, mode_shape)*dx)
            KE.append(0.5*args.m*np.sum(particles.vx**2))
            EE.append(0.5*args.eps0*np.sum(Exp**2)*dx)
            times.append(t)

    A, TE, KE, EE, times = map(np.array, (A, TE, KE, EE, times))
    csv_fn = os.path.join(args.outdir, 'langmuir_data.csv')
    with open(csv_fn, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t','A','KE','EE','TE'])
        for t,a,ke,ee,te in zip(times, A, KE, EE, TE):
            w.writerow([t, a, ke, ee, te])
    print("Saved langmuir data to", csv_fn)
    omega_p = np.sqrt((args.Np/L)*args.q**2/(args.m*args.eps0))
    omega_sim, _ = omega_fit(times, A, omega_p, 0.0)
    plt.figure(); plt.plot(times, A, 'o-')
    plt.savefig(os.path.join(args.outdir,'langmuir_amplitude.png'))
    print("ω_sim =", omega_sim, "ω_p =", omega_p)


elif args.ic == 'custom':
    # custom density‐perturbation IC
    Np = Nx * args.npp
    grid = Grid1D(Nx, dx)
    particles = Particles('electrons', Np, -np.sqrt(1.0/(Np/L)), 1.0, L, dx)
    grid.rho_background = -particles.q * (Np / L)

    # sinusoidal density perturbation
    k_c = args.custom_mode
    delta = args.delta_rho
    xp = np.linspace(0, L, Np, endpoint=False)
    xp += (delta/(2*np.pi*k_c)) * np.cos(2*np.pi*k_c * xp / L)
    particles.x = np.mod(xp, L)

    # zero initial fields, then half‐step B
    grid.Ex[:] = grid.Ey[:] = grid.Ez[:] = 0.0
    grid.Bx[:] = grid.By[:] = grid.Bz[:] = 0.0
    particles.deposit_charge_and_current(grid, dt)
    grid.apply_charge_periodic_BC()
    update_B(grid, 0.5*dt)

    times, amp = [], []
    for step in range(args.nsteps):
        particles.deposit_charge_and_current(grid, dt)
        grid.apply_charge_periodic_BC()
        update_E(grid, dt)
        update_B(grid, dt)
        particles.gather_fields(grid)
        particles.boris_push(dt)

        if step % args.save_interval == 0:
            t = step * dt
            rho_phys = grid.rho[1:-1] - np.mean(grid.rho[1:-1])
            A_k = np.dot(rho_phys, np.exp(-1j*k_c*x_centers)) * dx
            times.append(t)
            amp.append(np.abs(A_k))

    times = np.array(times); amp = np.array(amp)
    # save CSV
    csv_fn = os.path.join(args.outdir, 'custom_data.csv')
    with open(csv_fn, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t','amp','lnamp'])
        for t,a in zip(times, amp):
            w.writerow([t, a, np.log(a)])
    print("Saved custom data to", csv_fn)
    # plot
    plt.figure()
    plt.plot(times, np.log(amp), '-o')
    plt.xlabel('t [1/ωp]'); plt.ylabel('ln |ρ_k|')
    plt.title(f'Custom density mode k={k_c}')
    plt.savefig(os.path.join(args.outdir,'custom_growth.png'))
    print("Saved custom growth plot")
else:
    raise ValueError(f"Unknown test: {args.ic}")