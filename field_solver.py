import numpy as np
import params

# Field solvers

def initialize_pulse(grid, amplitude=1e-3, mode=1):
    """
    Seed a sinusoidal electric pulse into the Ez field at t=0.

    Parameters
    ----------
    grid : Grid1D
        The simulation grid containing field arrays and geometry.
    amplitude : float, optional
        Amplitude of the initial Ez perturbation (default: 1e-3).
    mode : int, optional
        Fourier mode number to seed (k = 2*pi*mode/L, default: 1).
    """
    x = (np.arange(grid.Nx) + 0.5)*grid.dx
    k = 2*np.pi * mode / params.L
    grid.Ez[1:-1] = amplitude * np.sin(k * x)
    grid.apply_field_periodic_BC()

# -----------------------------------------------------------------------------
# Half‐step B (to t=dt/2), full‐step E and B updates
# -----------------------------------------------------------------------------

def update_B(grid, dt):
    """
    Advance the magnetic field by one time-step (Yee scheme).
    This updates B_y and B_z using finite differences of the electric field.

    Parameters
    ----------
    grid : Grid1D
        Simulation grid with Ez, Ey, By, Bz arrays.
    dt : float
        Time-step increment.
    """
    # B_y at half‐nodes  i=1..Nx
    #   ∂t B_y =  + ∂x E_z
    grid.By[1:-1] += (dt/grid.dx) * (grid.Ez[2:]   - grid.Ez[1:-1])
    # B_z at half‐nodes
    #   ∂t B_z =  - ∂x E_y
    grid.Bz[1:-1] -= (dt/grid.dx) * (grid.Ey[2:]   - grid.Ey[1:-1])

    grid.apply_field_periodic_BC()


def update_E(grid, dt):
    """
    Advance the electric field by one time-step (Yee scheme).

    Updates Ex, Ey, and Ez using the current density and spatial derivatives of B.

    Parameters
    ----------
    grid : Grid1D
        Simulation grid with Jx, Jy, Jz and Bz, By arrays.
    dt : float
        Time-step increment.
    """
    # E_x at centers
    #   ∂t E_x = - J_x/ε0
    grid.Ex[1:-1] -= dt * grid.Jx[1:-1] / params.eps0

    # E_y at centers
    #   ∂t E_y = -c^2 ∂x B_z - J_y/ε0
    grid.Ey[1:-1] -= (params.c**2 * dt/grid.dx) * \
                     (grid.Bz[1:-1] - grid.Bz[:-2]) \
                     + dt * grid.Jy[1:-1] / params.eps0

    # E_z at centers
    #   ∂t E_z = +c^2 ∂x B_y - J_z/ε0
    grid.Ez[1:-1] += (params.c**2 * dt/grid.dx) * \
                     (grid.By[1:-1] - grid.By[:-2]) \
                     - dt * grid.Jz[1:-1] / params.eps0

    grid.apply_field_periodic_BC()


def gauss_correction(grid):
    """
    Enforce Gauss's law by recomputing Ex from the charge density via FFT Poisson solve.

    This projects out any divergence errors in the electric field.

    Parameters
    ----------
    grid : Grid1D
        Simulation grid with rho and Ex arrays.
    """
    rho_phys = grid.rho[1:-1] - np.mean(grid.rho[1:-1])
    rho_hat  = np.fft.rfft(rho_phys)
    k        = 2*np.pi * np.fft.rfftfreq(grid.Nx, grid.dx)
    Ex_hat   = rho_hat / (1j * k * params.eps0)
    Ex_hat[0] = 0.0
    grid.Ex[1:-1] = np.fft.irfft(Ex_hat, n=grid.Nx)
    grid.apply_field_periodic_BC()