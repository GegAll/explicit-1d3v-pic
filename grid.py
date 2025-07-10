import numpy as np
import params


class Grid1D:
    """
    1D Yee grid with ghost cells for periodic boundary conditions.

    Attributes
    ----------
    Nx : int
        Number of physical grid cells.
    dx : float
        Grid spacing.
    Ex, Ey, Ez : ndarray
        Electric field components at cell centers, length Nx+2 (includes ghosts).
    By, Bz : ndarray
        Magnetic field components at half-nodes, length Nx+2 (includes ghosts).
    rho : ndarray
        Charge density at cell centers, length Nx+2.
    Jx, Jy, Jz : ndarray
        Current density components at cell centers, length Nx+2.
    rho_background : float
        Uniform background charge density for neutrality.
    """
    def __init__(self, Nx, dx):
        """
        Initialize the 1D grid with ghost cells for periodic BCs.

        Parameters
        ----------
        Nx : int
            Number of physical cells.
        dx : float
            Grid spacing.
        """
        self.Nx = Nx
        self.dx = dx
        # add two ghost cells (indices 0 and Nx+1) for periodic wrap
        self.Ex = np.zeros(Nx+2)
        self.Ey = np.zeros(Nx+2)
        self.Ez = np.zeros(Nx+2)
        self.By = np.zeros(Nx+2)
        self.Bz = np.zeros(Nx+2)

        # Instantiate charge and current arrays
        self.rho = np.zeros(Nx+2)
        self.Jx  = np.zeros(Nx+2)
        self.Jy  = np.zeros(Nx+2)
        self.Jz  = np.zeros(Nx+2)

        # Uniform background density for neutrality
        self.rho_background = 0.0

    def apply_field_periodic_BC(self):
        """
        Enforce periodic boundary conditions on electromagnetic fields.

        Wrap ghost cells of Ex, Ey, Ez, By, and Bz so that
        field[0] = field[Nx] and field[Nx+1] = field[1].
        """
        for arr in [self.Ex, self.Ey, self.Ez, self.By, self.Bz]:
            arr[0]    = arr[self.Nx]
            arr[-1]   = arr[1]

    def apply_charge_periodic_BC(self):
        """
        Enforce periodic boundary conditions on charge and current densities.

        Wrap ghost cells of rho, Jx, Jy, and Jz so that
        arr[0] = arr[Nx] and arr[Nx+1] = arr[1].
        """
        for arr in [self.rho, self.Jx, self.Jy, self.Jz]:
            arr[0]    = arr[self.Nx]
            arr[-1]   = arr[1]