import numpy as np
import params

class Particles:
    def __init__(self, name, Np, q, m, domain_length, grid_dx):
        """
        Initialize a particle species in a 1D periodic domain.

        Parameters
        ----------
        name : str
            Identifier for the particle species (e.g., 'electrons').
        Np : int
            Total number of simulation particles.
        q : float
            Charge of each macro-particle.
        m : float
            Mass of each macro-particle.
        domain_length : float
            Physical length of the 1D domain.
        grid_dx : float
            Grid spacing (dx) for mapping particles to the grid.
        """
        self.name = name
        self.Np = Np
        self.q  = q
        self.m = m
        self.L = domain_length
        self.dx = grid_dx

        # Initialize particle positions uniformly in [0, L)
        self.x = np.random.rand(Np) * self.L
        # Initialize velocities to zero for each component
        self.vx = np.zeros(Np)
        self.vy = np.zeros(Np)
        self.vz = np.zeros(Np)
        # Allocate arrays for interpolated fields at particle locations
        self.Ex_interp = np.zeros(Np)
        self.Ey_interp = np.zeros(Np)
        self.Ez_interp = np.zeros(Np)
        self.By_interp = np.zeros(Np)
        self.Bz_interp = np.zeros(Np)

    def gather_fields(self, grid):
        """
        Interpolate grid-based electromagnetic fields to particle positions
        using linear (cloud-in-cell) weighting.

        Parameters
        ----------
        grid : Grid1D
            The simulation grid containing field arrays and spacing dx.
        """
        xi = (self.x / self.dx)
        i0 = np.floor(xi).astype(int)
        w1 = xi - i0
        w0 = 1 - w1

        Nx = grid.Nx
        # Map indices into periodic domain with ghost cells offset
        i0_mod = (i0 % Nx) + 1
        i1_mod = ((i0 + 1) % Nx) + 1

        # Interpolate electric field components
        self.Ex_interp = w0 * grid.Ex[i0_mod] + w1 * grid.Ex[i1_mod]
        self.Ey_interp = w0 * grid.Ey[i0_mod] + w1 * grid.Ey[i1_mod]
        self.Ez_interp = w0 * grid.Ez[i0_mod] + w1 * grid.Ez[i1_mod]
        # Interpolate magnetic field components
        self.By_interp = w0 * grid.By[i0_mod] + w1 * grid.By[i1_mod]
        self.Bz_interp = w0 * grid.Bz[i0_mod] + w1 * grid.Bz[i1_mod]

    def boris_push(self, dt):
        """
        Advance particle velocities and positions by one time-step using the
        Boris algorithm (also known as the Vay or Buneman push).

        Performs a half-electric kick, a magnetic rotation, then another half
        electric kick, and updates positions with periodic wrap.

        Parameters
        ----------
        dt : float
            Time-step duration.
        """
        qom = self.q / self.m

        # Half-step acceleration by electric field
        vx_minus = self.vx + qom * self.Ex_interp * (0.5 * dt)
        vy_minus = self.vy + qom * self.Ey_interp * (0.5 * dt)
        vz_minus = self.vz + qom * self.Ez_interp * (0.5 * dt)

        # Magnetic rotation vector components
        ty = qom * self.By_interp * (0.5 * dt)
        tz = qom * self.Bz_interp * (0.5 * dt)
        t2 = ty*ty + tz*tz

        # First rotation (v')
        vx_prime = vx_minus + (vy_minus * tz - vz_minus * ty)
        vy_prime = vy_minus + (vz_minus * 0 - vx_minus * tz)
        vz_prime = vz_minus + (vx_minus * ty - vy_minus * 0)

        # Compute s = 2 t / (1 + |t|^2)
        sy = 2 * ty / (1 + t2)
        sz = 2 * tz / (1 + t2)

        # Second rotation (v_plus)
        vx_plus = vx_minus + (vy_prime * sz - vz_prime * sy)
        vy_plus = vy_minus + (vz_prime * 0 - vx_prime * sz)
        vz_plus = vz_minus + (vx_prime * sy - vy_prime * 0)

        # Final half-step electric kick
        self.vx = vx_plus + qom * self.Ex_interp * (0.5 * dt)
        self.vy = vy_plus + qom * self.Ey_interp * (0.5 * dt)
        self.vz = vz_plus + qom * self.Ez_interp * (0.5 * dt)

        # Update positions with periodic boundary wrap
        self.x = (self.x + self.vx * dt) % self.L

    def deposit_charge_and_current(self, grid, dt):
        """
        Scatter particle charge and current density onto the grid using the
        Cloud-In-Cell (CIC) scheme.

        This zeroth-moment and first-moment deposit ensures smooth
        mapping from particles to grid densities.

        Parameters
        ----------
        grid : Grid1D
            The simulation grid to accumulate rho and J components.
        dt : float
            Time-step duration (unused for static charge deposit but
            included for consistency).
        """
        # Reset grid charge and currents
        grid.rho.fill(grid.rho_background)
        grid.Jx.fill(0.0)
        grid.Jy.fill(0.0)
        grid.Jz.fill(0.0)

        xi = self.x / grid.dx
        i0 = np.floor(xi).astype(int)
        w1 = xi - i0
        w0 = 1 - w1

        Nx = grid.Nx
        i0_mod = (i0 % Nx) + 1
        i1_mod = ((i0 + 1) % Nx) + 1

        # Deposit charge density ρ
        np.add.at(grid.rho, i0_mod, w0 * self.q / grid.dx)
        np.add.at(grid.rho, i1_mod, w1 * self.q / grid.dx)
        # Deposit current density J = q v
        np.add.at(grid.Jx, i0_mod, w0 * self.vx * self.q / grid.dx)
        np.add.at(grid.Jx, i1_mod, w1 * self.vx * self.q / grid.dx)
        np.add.at(grid.Jy, i0_mod, w0 * self.vy * self.q / grid.dx)
        np.add.at(grid.Jy, i1_mod, w1 * self.vy * self.q / grid.dx)
        np.add.at(grid.Jz, i0_mod, w0 * self.vz * self.q / grid.dx)
        np.add.at(grid.Jz, i1_mod, w1 * self.vz * self.q / grid.dx)

    def seed_langmuir(self, delta, mode):
        """
        Impose a sinusoidal density perturbation on particle positions
        to initialize a Langmuir oscillation test.

        Parameters
        ----------
        delta : float
            Perturbation amplitude in units of grid spacing.
        mode : int
            Fourier mode number for the density wave.
        """
        # Shift positions by a sine wave of amplitude delta*dx
        self.x += (delta * self.dx) * np.sin(2 * np.pi * mode * self.x / self.L)
        # Enforce periodic wrap
        self.x %= self.L