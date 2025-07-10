#-------------------------
# Particles parameters
#-------------------------
q     = -1.0      # electron's charge
m     = 1.0       # electron's mass
c     = 1.0       # speed of light (normalized)
eps0  = 1.0       # permittivity (normalized)
mu0   = 1.0       # permeability (normalized)
delta = 0.05      # charge density perturbation in units of dx
mode  = 3         # perturbation k mode

#-------------------------
# Grid & Time parameters
#-------------------------
L    = 1.0       # domain length
Nx   = 200       # number of **physical** cells
npp  = 1000      # number of particles per cell
Np   = npp * Nx  # number of particles
dx   = L / Nx
B0   = 1.0       # constant magnetic field
CFL  = 0.07      # Courant number < 1 for stability
dt   = CFL * dx / c
nsteps = 1000   # total time steps
poisson_interval = 50 # Steps between Gauss Law enforcement

save_interval = 1 # for diagnostics