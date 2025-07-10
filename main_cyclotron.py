import matplotlib.pyplot as plt
import params
from grid import Grid1D
from particles import Particles
from diagnostics import cos_fit, omega_fit, circle_center, plot_tvx_plane, plot_vxvy_plane
from field_solver import gauss_correction, update_B
import numpy as np
import time
import os
import argparse

# -----------------------------------------------------------------------------
# Command-line argument parsing
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Simulate cyclotron motion of a test electron in a uniform B field"
)
# Geometry & physics
parser.add_argument('--Nx',     type=int,    default=params.Nx,    help='Number of grid points')
parser.add_argument('--dx',     type=float,  default=params.dx,    help='Grid spacing')
parser.add_argument('--L',      type=float,  default=params.L,     help='Domain length')
parser.add_argument('--c',      type=float,  default=params.c,     help='Speed of light')
parser.add_argument('--q',      type=float,  default=params.q,     help='Particle charge')
parser.add_argument('--m',      type=float,  default=params.m,     help='Particle mass')
parser.add_argument('--B0',     type=float,  default=params.B0,    help='Uniform B-field magnitude')
parser.add_argument('--E0',     type=float,  default=0.0,          help='Uniform E-field magnitude')
# Time-stepping
parser.add_argument('--CFL',     type=float,  default=0.99,       help='Couriant Number')
parser.add_argument('--nsteps', type=int,    default=500,         help='Number of time steps')
# Output
parser.add_argument('--outdir', type=str,    default='cyclotron_results', help='Output directory')
args = parser.parse_args()

# Extract parameters
Nx, dx, L = args.Nx, args.dx, args.L
c, q, m   = args.c, args.q, args.m
CFL       = args.CFL
B0        = args.B0
E0        = args.E0
dt        = CFL * dx
eps0      = params.eps0
nsteps    = args.nsteps
outdir    = args.outdir

# Prepare output folder and parameter file
os.makedirs(outdir, exist_ok=True)
param_fn = os.path.join(outdir, f"cyclotron_params_n{nsteps}_CFL_{CFL:.3f}_E0{E0:.2f}_B0{B0:.2f}.txt")
with open(param_fn, 'w') as pf:
    pf.write("# Cyclotron motion simulation parameters\n")
    pf.write(f"Nx      = {Nx}\n")
    pf.write(f"dx      = {dx}\n")
    pf.write(f"L       = {L}\n")
    pf.write(f"c       = {c}\n")
    pf.write(f"q       = {q}\n")
    pf.write(f"m       = {m}\n")
    pf.write(f"B0      = {B0}\n")
    pf.write(f"E0      = {E0}\n")
    pf.write(f"dt      = {dt}\n")
    pf.write(f"nsteps  = {nsteps}\n")
print(f"Saved parameters to {param_fn}")

# -----------------------------------------------------------------------------
# Main cyclotron driver
# -----------------------------------------------------------------------------
def main_cyclotron(Nx, dx, L, q, m, B0, dt, nsteps):
    # Build grid and uniform B-field
    grid = Grid1D(Nx, dx)
    grid.Bz[:] = B0
    grid.Ex[:] = E0

    # Single test electron
    particles = Particles(name='electron', Np=1, q=q, m=m,
                          domain_length=L, grid_dx=dx)
    particles.x[:]  = 0.0
    particles.vy[:] = 1.0

    # Storage for energies, times and velocities
    times, vx_list, vy_list = [], [], []
    kinetic, electric, total_energy = [], [], []
    
    # Time-stepping loop
    for n in range(nsteps):
        # Interpolate the fields onto the particle
        particles.gather_fields(grid)
        
        # Boris push
        particles.boris_push(dt)

        # Compute kinetic energy
        KE = 0.5 * particles.m * (particles.vx[0]**2 + particles.vy[0]**2)
        # Copmute field energy
        FE = 0.5 * eps0 * (E0**2) * L
        # Total energy
        EE = KE + FE
        
        # Record data at each step
        kinetic.append(KE)
        electric.append(FE)
        total_energy.append(EE)
        vx_list.append(particles.vx[0])
        vy_list.append(particles.vy[0])
        times.append((n+1)*dt)

    return np.array(times), np.array(vx_list), np.array(vy_list), np.array(kinetic), np.array(electric), np.array(total_energy)
    #return kinetic_x, kinetic_y, kinetic, vx_list, vy_list, times 


# -----------------------------------------------------------------------------
# Script entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()
    t, vx, vy, kinetic, electric, total_energy = main_cyclotron(Nx, dx, L, q, m, B0, dt, nsteps)
    end_time   = time.time()
    print(f"Simulation Time: {end_time - start_time:.2f} s")

    # Theoretical gyrofrequency & period
    omega_c = abs(q) * B0 / m
    T_c     = 2 * np.pi / omega_c

    # Fit simulated data
    omega_sim, omega_err = omega_fit(t, vx, omega_c, 0.0)
    T_sim = 2 * np.pi / omega_sim

    # Print summary
    print(f"Simulated gyrofrequency     ω_sim = {omega_sim:.6f} ± {omega_err:.6f}")
    print(f"Theoretical gyrofrequency   ω_c   = {omega_c:.6f}")
    print(f"Relative error in ω:        {(omega_sim/omega_c - 1)*100:.2f}%")
    print(f"Simulated gyro-period       T_sim = {T_sim:.6f}")
    print(f"Theoretical gyro-period     T_c   = {T_c:.6f}")

    # -------------------------------------------------------------------------
    # Plotting: include parameters in filenames
    # -------------------------------------------------------------------------

    cycle = int(round(2*2*np.pi / dt))
    # 1) time vs vx
    tvx_fn = os.path.join(outdir, f"tvx_n{nsteps}_CFL_{CFL:.3f}_E0{E0:.2f}_B0{B0:.2f}.png")
    plot_tvx_plane(t, vx, T_c, T_sim, thd=int(0.1*cycle), save=True, filename=tvx_fn)
    print(f"Saved t-vx plot to {tvx_fn}")

    # Fit the circle to get the velocity drift
    xc, yc  = circle_center(vx, vy, T_c, dt, periods=1)
    print(f"Fitted center: ({xc:.1f}, {yc:.1f})")

    print("expected v_d = ", (0.0, -E0/B0))

    # 2) vx vs vy
    vxy_fn = os.path.join(outdir, f"vxvy_n{nsteps}_CFL_{CFL:.3f}_E0{E0:.2f}_B0{B0:.2f}.png")
    plot_vxvy_plane(vx, vy, xc, yc, save=True, filename=vxy_fn)
    print(f"Saved vx-vy plot to {vxy_fn}")

    # 3) Energy conservation
    plt.figure()
    plt.plot(t[:int(0.175*cycle)], kinetic[:int(0.175*cycle)], '-', label="Kinetic")
    #plt.ylim([0.4, 0.6])
    #plt.plot(t, electric, '-', label="Electric")
    #plt.plot(t, total_energy, '-', label="Total")
    plt.xlabel('t', fontsize=16)
    plt.ylabel('Kinetic Energy', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(f"nsteps: {nsteps} | CFL: {CFL:.3f} | B0: {B0:.0f} | E0: {E0:.0f}", fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{outdir}/energy_n{nsteps}_CFL_{CFL:.3f}_E0{E0:.2f}_B0{B0:.2f}.png")
    plt.show()