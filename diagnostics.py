import numpy as np
import params
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import csv

def field_energy(g):
    """
    Compute electromagnetic field energy in the grid.

    Parameters
    ----------
    g : Grid1D
        Grid object containing fields Ez and By and grid spacing dx.

    Returns
    -------
    EE : float
        Electric field energy: 0.5 * eps0 * \int Ez^2 dx.
    ME : float
        Magnetic field energy: 0.5 * mu0  * \int By^2 dx.
    """
    EE = 0.5*params.eps0 * np.sum(g.Ez[1:-1]**2) * g.dx
    ME = 0.5*params.mu0  * np.sum(g.By[1:-1]**2) * g.dx
    return EE, ME

def kinetic_energy(p):
    """
    Compute total kinetic energy of particles.

    Parameters
    ----------
    p : Particles
        Particle object with velocities vx and mass m.

    Returns
    -------
    KE : float
        Kinetic energy: 0.5 * m * \sum vx^2 over all particles.
    """
    KE = 0.5 * p.m * np.sum(p.vx**2)
    return KE

def cos_fit(t, A, omega, phi):
    """
    Cosine model function for curve fitting.

    Parameters
    ----------
    t : array_like
        Time array.
    A : float
        Amplitude of cosine.
    omega : float
        Angular frequency.
    phi : float
        Phase shift.

    Returns
    -------
    y : array_like
        A * cos(omega * t + phi)
    """
    return A * np.cos(omega*t + phi)

def omega_fit(t, A, omega, phi):
    """
    Fit a cosine to data to extract oscillation frequency.

    Uses scipy.optimize.curve_fit on the cos_fit model.

    Parameters
    ----------
    t : array_like
        Time points.
    A : array_like
        Data values to fit.
    omega : float
        Initial guess for angular frequency.
    phi : float
        Initial guess for phase.

    Returns
    -------
    omega_sim : float
        Fitted angular frequency.
    omega_sim_err : float
        Standard error of the fitted frequency.
    """
    popt, pcov = curve_fit(cos_fit, t, A, p0=[A[0], omega, phi])
    omega_sim, omega_sim_err = popt[1], np.sqrt(pcov[1][1])
    return omega_sim, omega_sim_err

def circle_center(x, y, T_c, dt, periods=2):
    """
    Determine center of circular trajectory in (x,y) via least squares.

    Clips the trajectory to a specified number of periods
    and fits x^2+y^2 + A*x + B*y + C = 0 to solve for circle center.

    Parameters
    ----------
    x : array_like
        x-coordinates over time.
    y : array_like
        y-coordinates over time.
    T_c : float
        Theoretical cyclotron period.
    dt : float
        Time step.
    periods : int, optional
        Number of periods to include in fit (default=2).

    Returns
    -------
    xc : float
        x-coordinate of circle center.
    yc : float
        y-coordinate of circle center.
    """
    steps_per_period = int(round(T_c/dt))
    N = periods*steps_per_period
    i0 = steps_per_period + N//2

    i_start = i0 - N//2
    i_end   = i0 + N//2
    x  = x[i_start:i_end]
    y  = y[i_start:i_end]
    # Build the linear system: M @ [A, B, C] = b
    # where b = -(x^2 + y^2)
    A_mat = np.vstack([x, y, np.ones_like(x)]).T
    b_vec = -(x**2 + y**2)

    # Solve the overdetermined system in the least-squares sense
    coeffs, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    A, B, C = coeffs

    # Compute center
    xc = -A / 2
    yc = -B / 2
    return xc, yc

def plot_energies(times, energy, electric, magnetic, save=False, title='Energy conservation in vacuum FDTD', filename=None, fontsize=16):
    """
    Plot total, electric, and magnetic field energies over time.

    Parameters
    ----------
    times : array_like
        Time points.
    energy : array_like
        Total field energy at each time.
    electric : array_like
        Electric field energy at each time.
    magnetic : array_like
        Magnetic field energy at each time.
    save : bool, optional
        Whether to save the plot to file.
    title : str, optional
        Plot title.
    filename : str, optional
        Path to save the figure if save=True.
    fontsize : int, optional
        Font size for labels and ticks.
    """
    plt.plot(times, energy, 'o-', label="total energy")
    plt.plot(times, electric, 'o-', label="electric energy")
    plt.plot(times, magnetic, 'o-', label="magnetic energy")
    plt.xlabel('t', fontsize=fontsize)
    plt.ylabel('Total field energy', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.grid(True)
    plt.tight_layout()
    plt.legend(fontsize=fontsize)
    if save:
        plt.savefig(filename)
    plt.show()

def plot_vxvy_plane(vx , vy, xc, yc, save=True, filename=None, fontsize=16, title=r"Cyclotron Oscillations in $v_x$-$v_y$ plane"):
    """
    Plot particle velocity trajectory in the vx-vy plane and mark center.

    Parameters
    ----------
    vx : array_like
        Velocity component in x.
    vy : array_like
        Velocity component in y.
    xc : float
        x-coordinate of drift velocity center.
    yc : float
        y-coordinate of drift velocity center.
    save : bool, optional
        Whether to save the plot to file.
    filename : str, optional
        Path to save the figure if save=True.
    fontsize : int, optional
        Font size for labels and ticks.
    title : str, optional
        Plot title.
    """
    plt.plot(vx, vy)
    plt.scatter(xc, yc, marker='x', color='r', label=r'v$_d$ = ' + f'({xc:.1f}, {yc:.1f})')
    plt.xlabel(r"$v_x$", fontsize=fontsize)
    plt.ylabel(r"$v_y$", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=fontsize)
    plt.tight_layout()
    if save:
        plt.savefig(filename)
    plt.show()

def plot_tvx_plane(t, vx, T_c, T_sim, thd=250, save=True, filename=None, fontsize=16, title=r"Cyclotron Oscillations in t-$v_x$ plane"):
    """
    Plot vx versus time for one cyclotron period and annotate period error.

    Parameters
    ----------
    t : array_like
        Time points.
    vx : array_like
        Velocity in x over time.
    T_c : float
        Theoretical cyclotron period.
    T_sim : float
        Simulated cyclotron period.
    thd : int, optional
        Number of time steps to display.
    save : bool, optional
        Whether to save the plot to file.
    filename : str, optional
        Path to save the figure if save=True.
    fontsize : int, optional
        Font size for labels and ticks.
    title : str, optional
        Plot title.
    """
    dtau = 100*(abs(T_sim/T_c) - 1)
    plt.plot(t[:thd], vx[:thd], label=r"$\Delta\tau_c$" + f" = {dtau:.2f} %")
    plt.xlabel("t", fontsize=fontsize)
    plt.ylabel(r"$v_x$", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.legend(loc='upper right', fontsize=fontsize)
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(filename)
    plt.show()

def fit_growth_and_plot(amp_E, times, dt, mode, L, v0, title='', filename=None):
    """
    Fit exponential growth rate of a two-stream instability and plot comparison.

    Parameters
    ----------
    amp_E : array_like
        Electric field mode amplitude |E_k| over time.
    times : array_like
        Time points corresponding to amp_E.
    dt : float
        Time step size.
    mode : int
        Seeded Fourier mode number.
    L : float
        Domain length.
    v0 : float
        Beam drift speed.
    title : str, optional
        Plot title.
    filename : str, optional
        Path to save the plot.

    Returns
    -------
    gamma_sim : float
        Measured exponential growth rate.
    A0 : float
        Fitted initial amplitude.
    """
    # theoretical growth (in the non-relativistic regime)
    k = 2*np.pi*mode/L
    gamma_theo = k*v0
    print(f"Measured growth rate γ_theo = {gamma_theo:.4f}")
    # fit growth
    ln_amp = np.log(amp_E)
    mask = (times>0) & (times<30000000*dt)
    p = np.polyfit(times[mask], ln_amp[mask], 1)
    gamma_sim = p[0]; A0 = np.exp(p[1])
    gamma_error = abs(1 - gamma_sim/gamma_theo)*100
    print(f"Measured growth rate γ_sim = {gamma_sim:.4f}")
    # plot
    plt.figure()
    plt.plot(times[mask], p[0]*times[mask]+p[1], 'r-',
             label=r"γ$_{theo}$ = " + f"{gamma_theo:.2f} \n" +
                   r"γ$_{sim}$ = " + f"{gamma_sim:.2f}")
    plt.plot(times, ln_amp, alpha=0.5,
             label=f"Δγ={gamma_error:.2f} %")
    plt.title(title)
    plt.xlabel('t'); plt.ylabel('ln|E_k|')
    plt.legend(); plt.grid(True)
    plt.savefig(filename)
    plt.show()
    print(f"Saved plot to {filename}")
    return gamma_sim, A0

def plot_weibel_growth(times, lnB, gamma_theo, title=None, filename=None, t_start=5.0, t_stop=30.0):
    """
    Fit and plot exponential growth for the Weibel instability.

    Parameters
    ----------
    times : array_like
        Time points.
    lnB : array_like
        Logarithm of magnetic field mode amplitude ln|B_k|.
    gamma_theo : float
        Theoretical growth rate.
    title : str, optional
        Plot title.
    filename : str, optional
        Path to save the plot.
    t_start : float, optional
        Start time for linear fit window.
    t_stop : float, optional
        End time for linear fit window.

    Returns
    -------
    gamma_sim : float
        Measured growth rate.
    A0 : float
        Intercept from linear fit.
    """
    mask = (times>=t_start) & (times<=t_stop)
    p = np.polyfit(times[mask], lnB[mask], 1)
    gamma_sim, A0 = p[0], p[1]
    gamma_error = abs(1 - gamma_sim/gamma_theo)*100
    print(f"Measured growth rate γ_sim = {gamma_sim:.4f}")
    print(f"Growth rate error Δγ = {gamma_error:.2f}%")
    plt.figure()
    plt.plot(times[mask], gamma_sim*times[mask] + A0,
             'r--', label=r"γ$_{theo}$ = " + f"{gamma_theo:.2f} \n" +
                       r"γ$_{sim}$ = " + f"{gamma_sim:.2f}")
    plt.plot(times, lnB, alpha=0.5, label=f"Δγ={gamma_error:.2f}%")
    plt.xlabel('t'); plt.ylabel('ln|B_k|'); plt.title(title)
    plt.grid(True); plt.legend(loc='lower right')
    plt.savefig(filename); plt.show()
    print(f"Saved plot to {filename}")
    return gamma_sim, A0

def save_weibel_diagnostics(csv_fn, times, amp_B, lnB):
    """
    Save Weibel diagnostics to CSV file.

    Parameters
    ----------
    csv_fn : str
        Filename for CSV output.
    times : array_like
        Time points.
    amp_B : array_like
        Magnetic field mode amplitudes.
    lnB : array_like
        Logarithm of amp_B.
    """
    with open(csv_fn, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time','amp_B','lnB'])
        for t, b, lnb in zip(times, amp_B, lnB):
            writer.writerow([t, b, lnb])
    print(f"Saved data to {csv_fn}")